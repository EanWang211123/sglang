import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Union

from sglang.srt.speculative.adaptive_spec_params import (
    AdaptiveSpeculativeParams,
    load_adaptive_config,
)
from sglang.srt.speculative.batch_size_aware_spec_params import (
    BatchSizeAwareSpecParams,
    load_batch_size_aware_config,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
        EAGLEDraftExtendCudaGraphRunner,
    )

logger = logging.getLogger(__name__)


@dataclass
class SpecRuntimeState:
    """A complete set of runtime resources bound to a specific speculative
    decoding configuration.

    Each decode round runs three stages — draft, verify, extend — and every
    stage has shape-dependent resources (attention backends and CUDA graphs)
    that must match the current configuration.  Switching adaptive steps
    means swapping the entire state atomically.
    """

    # -- Configuration (determines shapes for all stages) --
    speculative_num_steps: int
    speculative_num_draft_tokens: int

    # -- Draft stage: draft model multi-step autoregressive generation --
    draft_attn_backend: "AttentionBackend | None"
    cuda_graph_runner: "EAGLEDraftCudaGraphRunner | None"

    # -- Verify stage: target model one-pass tree verification --
    target_attn_backend: "AttentionBackend"
    target_graph_runner: "CudaGraphRunner | CPUGraphRunner | None"

    # -- Extend stage: draft model KV cache catch-up after verify --
    draft_extend_attn_backend: "AttentionBackend | None"
    cuda_graph_runner_for_draft_extend: "EAGLEDraftExtendCudaGraphRunner | None"


class AdaptiveSpecWorker(Protocol):
    """Protocol that a worker must implement to use AdaptiveController."""

    speculative_num_steps: int
    server_args: "ServerArgs"

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: list[int] | None = None,
    ) -> SpecRuntimeState: ...

    def apply_runtime_state(self, state: SpecRuntimeState) -> None: ...


_STRATEGY_EMA = "ema"
_STRATEGY_BATCH_SIZE_AWARE = "batch_size_aware"
_VALID_STRATEGIES = (_STRATEGY_EMA, _STRATEGY_BATCH_SIZE_AWARE)


class AdaptiveController:
    """Facade that owns adaptive decision-making and runtime state switching.

    Works with any worker that implements ``AdaptiveSpecWorker`` protocol:
      - ``build_adaptive_runtime_state(steps, draft_tokens)`` → runtime state
      - ``apply_runtime_state(state)`` → apply it to the worker

    The worker only needs to:
      1. Call ``register()`` for the initial state, then ``init_states()``
         once during startup.
      2. Call ``on_verify_complete(accept_lengths, batch_size)`` after each
         decode verify.

    Supported strategies
    --------------------
    ``"ema"`` (default)
        Uses EMA of observed acceptance lengths to adapt num_steps.
        Config is loaded from *config_path* via :func:`load_adaptive_config`.

    ``"batch_size_aware"``
        Resolves num_steps directly from the current batch size via a
        user-supplied lookup table.
        Config is loaded from *config_path* via
        :func:`load_batch_size_aware_config`.
    """

    def __init__(
        self,
        worker: AdaptiveSpecWorker,
        config_path: str | None = None,
        strategy: str = _STRATEGY_EMA,
    ):
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown adaptive speculative strategy {strategy!r}. "
                f"Valid choices: {_VALID_STRATEGIES}"
            )
        self.worker = worker
        self._strategy = strategy

        if strategy == _STRATEGY_BATCH_SIZE_AWARE:
            cfg = load_batch_size_aware_config(config_path)
            self.params: Union[AdaptiveSpeculativeParams, BatchSizeAwareSpecParams] = (
                BatchSizeAwareSpecParams(
                    default_steps=worker.speculative_num_steps,
                    config=cfg,
                )
            )
        else:
            cfg = load_adaptive_config(config_path)
            self.params = AdaptiveSpeculativeParams(
                initial_steps=worker.speculative_num_steps,
                config=cfg,
            )

        self._states: dict[int, SpecRuntimeState] = {}

    @property
    def candidate_steps(self) -> list[int]:
        return self.params.candidate_steps

    def register(self, state: SpecRuntimeState, steps: int | None = None) -> None:
        """Register a pre-built runtime state.

        *steps* defaults to ``state.speculative_num_steps`` when not given.
        """
        key = steps if steps is not None else state.speculative_num_steps
        self._states[key] = state

    def init_states(self) -> None:
        """Build and register runtime states for all candidate steps."""
        if self._strategy == _STRATEGY_BATCH_SIZE_AWARE:
            self._init_states_batch_size_aware()
        else:
            self._init_states_ema()
        self._activate(self.params.current_steps)

    def _init_states_ema(self) -> None:
        """Build states for the EMA strategy.

        The initial default state was already captured during worker init and
        registered via ``register()``.  It covers all batch sizes, which is
        exactly what EMA needs, so we skip rebuilding it and only build the
        remaining candidate steps.
        """
        for steps in self.params.candidate_steps:
            if steps in self._states:
                continue
            self._states[steps] = self.worker.build_adaptive_runtime_state(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=steps + 1,
            )

    def _init_states_batch_size_aware(self) -> None:
        """Build states for the batch_size_aware strategy.

        Every candidate step — including the default one that was pre-registered
        during worker init — must be rebuilt so that each step only captures the
        batch sizes actually routed to it.  After all states are replaced the
        old full-batch-size CUDA graphs are dereferenced; an explicit GC pass
        reclaims the GPU memory before the server starts serving requests.
        """
        assert isinstance(self.params, BatchSizeAwareSpecParams)
        all_bs: list[int] = self.worker.server_args.cuda_graph_bs or []
        for steps in self.params.candidate_steps:
            cuda_graph_bs = self.params.batch_sizes_for_step(steps, all_bs)
            if not cuda_graph_bs:
                logger.warning(
                    f"batch_size_aware: no batch sizes in cuda_graph_bs={all_bs} "
                    f"are routed to steps={steps}; skipping state build."
                )
                continue
            self._states[steps] = self.worker.build_adaptive_runtime_state(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=steps + 1,
                cuda_graph_bs=cuda_graph_bs,
            )

    def on_verify_complete(
        self,
        accept_lengths: list[int] | None = None,
        batch_size: int | None = None,
    ) -> None:
        """Call this method in one of two ways depending on the active strategy"""
        if self._strategy == _STRATEGY_BATCH_SIZE_AWARE:
            assert isinstance(self.params, BatchSizeAwareSpecParams)
            if batch_size is not None and self.params.update(batch_size):
                self._activate(self.params.current_steps)
        else:
            assert isinstance(self.params, AdaptiveSpeculativeParams)
            if accept_lengths is not None and self.params.update(accept_lengths):
                self._activate(self.params.current_steps)

    def _activate(self, speculative_num_steps: int) -> None:
        state = self._states.get(speculative_num_steps)
        if state is None:
            raise ValueError(
                f"Missing adaptive runtime state for steps={speculative_num_steps}"
            )
        self.worker.apply_runtime_state(state)
