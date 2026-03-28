import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterator, Optional, Tuple

from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,
)
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.utils import get_available_gpu_memory, is_npu

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker import EAGLEWorker

logger = logging.getLogger(__name__)
_IS_NPU = is_npu()

_DEVICE_TO_DRAFT_RUNNER = {
    "npu": EAGLEDraftNpuGraphRunner,
    "cuda": EAGLEDraftCudaGraphRunner,
}


@dataclass
class EAGLERuntimeState:
    speculative_num_steps: int
    speculative_num_draft_tokens: int
    draft_attn_backend: Optional[object]
    draft_extend_attn_backend: Optional[object]
    cuda_graph_runner: Optional[object]
    cuda_graph_runner_for_draft_extend: Optional[object]
    target_attn_backend: object
    target_graph_runner: Optional[object]


class AdaptiveRuntimeStateManager:
    def __init__(self, worker: "EAGLEWorker"):
        self.worker = worker
        self.runtime_states: Dict[int, EAGLERuntimeState] = {}

    def register(self, state: EAGLERuntimeState):
        self.runtime_states[state.speculative_num_steps] = state

    def build_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
    ) -> EAGLERuntimeState:
        worker = self.worker
        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(worker.device, worker.gpu_id)

        with self._override_worker_state(
            speculative_num_steps, speculative_num_draft_tokens
        ):
            draft_backend_factory = DraftBackendFactory(
                worker.server_args,
                worker.draft_model_runner,
                worker.topk,
                speculative_num_steps,
            )
            draft_attn_backend = draft_backend_factory.create_decode_backend()
            draft_extend_attn_backend = (
                draft_backend_factory.create_draft_extend_backend()
            )
            # Draft graph capture calls back into eagle_worker.draft_forward(),
            # which reads worker-local backends instead of the backend passed to
            # the graph runner constructor. Keep the worker and draft runner in
            # sync with the runtime state being built.
            worker.draft_attn_backend = draft_attn_backend
            worker.draft_extend_attn_backend = draft_extend_attn_backend
            worker.draft_model_runner.draft_attn_backend = draft_attn_backend
            cuda_graph_runner, cuda_graph_runner_for_draft_extend = (
                self._capture_draft_cuda_graphs(
                    draft_attn_backend,
                    draft_extend_attn_backend,
                    speculative_num_steps,
                    speculative_num_draft_tokens,
                )
            )
            target_attn_backend = self._build_target_attn_backend()
            target_graph_runner = self._capture_target_cuda_graph(
                target_attn_backend,
                speculative_num_steps,
                speculative_num_draft_tokens,
            )

        after_mem = get_available_gpu_memory(worker.device, worker.gpu_id)
        logger.info(
            f"Built adaptive runtime state steps={speculative_num_steps}: "
            f"elapsed={time.perf_counter() - tic:.2f}s, "
            f"mem={(before_mem - after_mem):.2f}GB"
        )

        return EAGLERuntimeState(
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
            draft_attn_backend=draft_attn_backend,
            draft_extend_attn_backend=draft_extend_attn_backend,
            cuda_graph_runner=cuda_graph_runner,
            cuda_graph_runner_for_draft_extend=cuda_graph_runner_for_draft_extend,
            target_attn_backend=target_attn_backend,
            target_graph_runner=target_graph_runner,
        )

    def activate_runtime_state(self, speculative_num_steps: int):
        worker = self.worker
        runtime_state = self.runtime_states.get(speculative_num_steps)
        if runtime_state is None:
            raise ValueError(
                f"Missing adaptive runtime state for steps={speculative_num_steps}"
            )

        if worker.speculative_num_steps != runtime_state.speculative_num_steps:
            logger.info(
                "Switch adaptive runtime state: "
                f"steps {worker.speculative_num_steps} -> {runtime_state.speculative_num_steps}, "
                f"draft_tokens {worker.speculative_num_draft_tokens} -> "
                f"{runtime_state.speculative_num_draft_tokens}"
            )

        worker.speculative_num_steps = runtime_state.speculative_num_steps
        worker.speculative_num_draft_tokens = runtime_state.speculative_num_draft_tokens
        worker.cuda_graph_runner = runtime_state.cuda_graph_runner
        worker.cuda_graph_runner_for_draft_extend = (
            runtime_state.cuda_graph_runner_for_draft_extend
        )
        # draft_attn_backend lives on both worker and draft_model_runner;
        # set via worker property so both stay in sync.
        worker.draft_attn_backend = runtime_state.draft_attn_backend
        worker.draft_extend_attn_backend = runtime_state.draft_extend_attn_backend
        worker.draft_model_runner.draft_attn_backend = runtime_state.draft_attn_backend
        worker.target_worker.model_runner.attn_backend = (
            runtime_state.target_attn_backend
        )
        worker.target_worker.model_runner.graph_runner = (
            runtime_state.target_graph_runner
        )
        self._sync_server_args(
            runtime_state.speculative_num_steps,
            runtime_state.speculative_num_draft_tokens,
        )

    def _build_target_attn_backend(self):
        model_runner = self.worker.target_worker.model_runner
        backup = model_runner.init_new_workspace
        try:
            return model_runner._get_attention_backend(init_new_workspace=True)
        finally:
            model_runner.init_new_workspace = backup

    def _capture_target_cuda_graph(
        self,
        target_attn_backend,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
    ) -> Optional[CudaGraphRunner]:
        worker = self.worker
        if worker.server_args.disable_cuda_graph:
            return None

        return CudaGraphRunner(
            worker.target_worker.model_runner,
            attn_backend=target_attn_backend,
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )

    def _capture_draft_cuda_graphs(
        self,
        draft_attn_backend,
        draft_extend_attn_backend,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
    ) -> Tuple[Optional[object], Optional[object]]:
        """Capture draft decode and draft extend cuda graphs.

        Worker state (speculative_num_steps, draft_attn_backend, etc.) is
        already overridden by the caller via ``_override_worker_state``, so
        no manual backup/restore is needed here.
        """
        worker = self.worker
        cuda_graph_runner = None
        cuda_graph_runner_for_draft_extend = None

        if worker.server_args.disable_cuda_graph:
            return cuda_graph_runner, cuda_graph_runner_for_draft_extend

        if speculative_num_steps > 1:
            draft_runner_cls = _DEVICE_TO_DRAFT_RUNNER[worker.target_worker.device]
            cuda_graph_runner = draft_runner_cls(
                worker,
                draft_attn_backend=draft_attn_backend,
                speculative_num_steps=speculative_num_steps,
            )

        if draft_extend_attn_backend and not _IS_NPU:
            cuda_graph_runner_for_draft_extend = EAGLEDraftExtendCudaGraphRunner(
                worker,
                draft_extend_attn_backend=draft_extend_attn_backend,
                speculative_num_steps=speculative_num_steps,
            )

        return cuda_graph_runner, cuda_graph_runner_for_draft_extend

    def _iter_server_args(self) -> Iterator[object]:
        worker = self.worker
        candidates = [
            worker.server_args,
            getattr(worker.draft_model_runner, "server_args", None),
            getattr(worker.target_worker, "server_args", None),
            getattr(worker.target_worker.model_runner, "server_args", None),
        ]
        seen = set()
        for server_args in candidates:
            if server_args is None or id(server_args) in seen:
                continue
            seen.add(id(server_args))
            yield server_args

    @contextmanager
    def _override_worker_state(
        self, speculative_num_steps: int, speculative_num_draft_tokens: int
    ):
        """Temporarily override both server_args and worker attributes.

        This is a single unified backup/restore so that ``build_runtime_state``
        and ``_capture_draft_cuda_graphs`` don't need separate backup logic.
        """
        worker = self.worker

        # Backup worker attributes
        backup_steps = worker.speculative_num_steps
        backup_draft_tokens = worker.speculative_num_draft_tokens
        backup_draft_attn_backend = worker.draft_attn_backend
        backup_draft_extend_attn_backend = worker.draft_extend_attn_backend
        backup_draft_model_runner_attn_backend = getattr(
            worker.draft_model_runner, "draft_attn_backend", None
        )

        # Backup server_args
        server_args_backups = []
        for server_args in self._iter_server_args():
            server_args_backups.append(
                (
                    server_args,
                    server_args.speculative_num_steps,
                    server_args.speculative_num_draft_tokens,
                )
            )
            server_args.speculative_num_steps = speculative_num_steps
            server_args.speculative_num_draft_tokens = speculative_num_draft_tokens

        worker.speculative_num_steps = speculative_num_steps
        worker.speculative_num_draft_tokens = speculative_num_draft_tokens

        try:
            yield
        finally:
            worker.speculative_num_steps = backup_steps
            worker.speculative_num_draft_tokens = backup_draft_tokens
            worker.draft_attn_backend = backup_draft_attn_backend
            worker.draft_extend_attn_backend = backup_draft_extend_attn_backend
            worker.draft_model_runner.draft_attn_backend = (
                backup_draft_model_runner_attn_backend
            )
            for sa, prev_steps, prev_tokens in reversed(server_args_backups):
                sa.speculative_num_steps = prev_steps
                sa.speculative_num_draft_tokens = prev_tokens

    def _sync_server_args(
        self, speculative_num_steps: int, speculative_num_draft_tokens: int
    ):
        for server_args in self._iter_server_args():
            server_args.speculative_num_steps = speculative_num_steps
            server_args.speculative_num_draft_tokens = speculative_num_draft_tokens
