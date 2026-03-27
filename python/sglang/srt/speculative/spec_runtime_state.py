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
        *,
        draft_backend_factory_cls=DraftBackendFactory,
        capture_draft_cuda_graphs_fn=None,
    ) -> EAGLERuntimeState:
        worker = self.worker
        if capture_draft_cuda_graphs_fn is None:
            capture_draft_cuda_graphs_fn = self._capture_draft_cuda_graphs_for_state
        with self._override_runtime_server_args(
            speculative_num_steps, speculative_num_draft_tokens
        ):
            draft_backend_factory = draft_backend_factory_cls(
                worker.server_args,
                worker.draft_model_runner,
                worker.topk,
                speculative_num_steps,
            )
            draft_attn_backend = draft_backend_factory.create_decode_backend()
            draft_extend_attn_backend = (
                draft_backend_factory.create_draft_extend_backend()
            )
            cuda_graph_runner, cuda_graph_runner_for_draft_extend = (
                capture_draft_cuda_graphs_fn(
                    draft_attn_backend,
                    draft_extend_attn_backend,
                    speculative_num_steps,
                    speculative_num_draft_tokens,
                )
            )

            backup_init_new_workspace = (
                worker.target_worker.model_runner.init_new_workspace
            )
            try:
                target_attn_backend = (
                    worker.target_worker.model_runner._get_attention_backend(
                        init_new_workspace=True
                    )
                )
            finally:
                worker.target_worker.model_runner.init_new_workspace = (
                    backup_init_new_workspace
                )

            target_graph_runner = None
            if not worker.server_args.disable_cuda_graph:
                tic = time.perf_counter()
                before_mem = get_available_gpu_memory(worker.device, worker.gpu_id)
                logger.info(
                    "Capture target verify cuda graph for adaptive state "
                    f"steps={speculative_num_steps}, draft_tokens={speculative_num_draft_tokens} "
                    f"begin. avail mem={before_mem:.2f} GB"
                )
                target_graph_runner = CudaGraphRunner(
                    worker.target_worker.model_runner,
                    attn_backend=target_attn_backend,
                    speculative_num_steps=speculative_num_steps,
                    speculative_num_draft_tokens=speculative_num_draft_tokens,
                )
                after_mem = get_available_gpu_memory(worker.device, worker.gpu_id)
                logger.info(
                    "Capture target verify cuda graph for adaptive state "
                    f"steps={speculative_num_steps}, draft_tokens={speculative_num_draft_tokens} "
                    f"end. elapsed={time.perf_counter() - tic:.2f} s, "
                    f"mem usage={(before_mem - after_mem):.2f} GB, avail mem={after_mem:.2f} GB."
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
        worker.draft_attn_backend = runtime_state.draft_attn_backend
        worker.draft_extend_attn_backend = runtime_state.draft_extend_attn_backend
        worker.cuda_graph_runner = runtime_state.cuda_graph_runner
        worker.cuda_graph_runner_for_draft_extend = (
            runtime_state.cuda_graph_runner_for_draft_extend
        )
        self._sync_runtime_server_args(
            runtime_state.speculative_num_steps,
            runtime_state.speculative_num_draft_tokens,
        )
        worker.draft_model_runner.draft_attn_backend = runtime_state.draft_attn_backend
        worker.target_worker.model_runner.attn_backend = (
            runtime_state.target_attn_backend
        )
        worker.target_worker.model_runner.graph_runner = (
            runtime_state.target_graph_runner
        )

    def _capture_draft_cuda_graphs_for_state(
        self,
        draft_attn_backend,
        draft_extend_attn_backend,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
    ) -> Tuple[Optional[object], Optional[object]]:
        worker = self.worker
        cuda_graph_runner = None
        cuda_graph_runner_for_draft_extend = None

        if worker.server_args.disable_cuda_graph:
            return cuda_graph_runner, cuda_graph_runner_for_draft_extend

        device_to_draft_runner = {
            "npu": EAGLEDraftNpuGraphRunner,
            "cuda": EAGLEDraftCudaGraphRunner,
        }

        if speculative_num_steps > 1:
            backup_steps = worker.speculative_num_steps
            backup_draft_tokens = worker.speculative_num_draft_tokens
            backup_draft_attn_backend = worker.draft_attn_backend
            try:
                worker.speculative_num_steps = speculative_num_steps
                worker.speculative_num_draft_tokens = speculative_num_draft_tokens
                worker.draft_attn_backend = draft_attn_backend

                tic = time.perf_counter()
                before_mem = get_available_gpu_memory(worker.device, worker.gpu_id)
                logger.info(
                    "Capture draft cuda graph for adaptive state "
                    f"steps={speculative_num_steps}, draft_tokens={speculative_num_draft_tokens} "
                    f"begin. avail mem={before_mem:.2f} GB"
                )
                cuda_graph_runner = device_to_draft_runner[worker.target_worker.device](
                    worker,
                    draft_attn_backend=draft_attn_backend,
                    speculative_num_steps=speculative_num_steps,
                )
                after_mem = get_available_gpu_memory(worker.device, worker.gpu_id)
                logger.info(
                    "Capture draft cuda graph for adaptive state "
                    f"steps={speculative_num_steps}, draft_tokens={speculative_num_draft_tokens} "
                    f"end. elapsed={time.perf_counter() - tic:.2f} s, "
                    f"mem usage={(before_mem - after_mem):.2f} GB, avail mem={after_mem:.2f} GB."
                )
            finally:
                worker.speculative_num_steps = backup_steps
                worker.speculative_num_draft_tokens = backup_draft_tokens
                worker.draft_attn_backend = backup_draft_attn_backend

        if draft_extend_attn_backend and not _IS_NPU:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(worker.device, worker.gpu_id)
            logger.info(
                "Capture draft extend cuda graph for adaptive state "
                f"steps={speculative_num_steps}, draft_tokens={speculative_num_draft_tokens} "
                f"begin. avail mem={before_mem:.2f} GB"
            )
            cuda_graph_runner_for_draft_extend = EAGLEDraftExtendCudaGraphRunner(
                worker,
                draft_extend_attn_backend=draft_extend_attn_backend,
                speculative_num_steps=speculative_num_steps,
            )
            after_mem = get_available_gpu_memory(worker.device, worker.gpu_id)
            logger.info(
                "Capture draft extend cuda graph for adaptive state "
                f"steps={speculative_num_steps}, draft_tokens={speculative_num_draft_tokens} "
                f"end. elapsed={time.perf_counter() - tic:.2f} s, "
                f"mem usage={(before_mem - after_mem):.2f} GB, avail mem={after_mem:.2f} GB."
                )

        return cuda_graph_runner, cuda_graph_runner_for_draft_extend

    def _iter_runtime_server_args(self) -> Iterator[object]:
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
    def _override_runtime_server_args(
        self, speculative_num_steps: int, speculative_num_draft_tokens: int
    ):
        backups = []
        for server_args in self._iter_runtime_server_args():
            backups.append(
                (
                    server_args,
                    server_args.speculative_num_steps,
                    server_args.speculative_num_draft_tokens,
                )
            )
            server_args.speculative_num_steps = speculative_num_steps
            server_args.speculative_num_draft_tokens = speculative_num_draft_tokens
        try:
            yield
        finally:
            for (
                server_args,
                previous_steps,
                previous_draft_tokens,
            ) in reversed(backups):
                server_args.speculative_num_steps = previous_steps
                server_args.speculative_num_draft_tokens = previous_draft_tokens

    def _sync_runtime_server_args(
        self, speculative_num_steps: int, speculative_num_draft_tokens: int
    ):
        for server_args in self._iter_runtime_server_args():
            server_args.speculative_num_steps = speculative_num_steps
            server_args.speculative_num_draft_tokens = speculative_num_draft_tokens
