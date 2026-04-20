import logging
import os
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple

import torch

from sglang.srt.distributed import get_tp_group, tensor_model_parallel_all_gather
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,
)
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.cuda_graph_runner import (
    CudaGraphRunner,
    get_batch_sizes_to_capture,
)
from sglang.srt.model_executor.input_buffers import _forward_input_buffer_pool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
)
from sglang.srt.speculative.eagle_utils import (
    build_tree_kernel_efficient,
    organize_draft_results,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    assign_draft_cache_locs,
    draft_tp_context,
    fast_topk,
    generate_token_bitmask,
    get_last_loc_large_page_size_large_top_k,
    load_token_map,
    maybe_detect_nan,
    maybe_detect_oob,
    select_top_k_tokens,
)
from sglang.srt.speculative.spec_timing_warmup_sim_recorder import (
    SpecTimingWarmupSimRecorder,
)
from sglang.srt.utils import (
    MultiprocessingSerializer,
    empty_context,
    get_available_gpu_memory,
    is_cuda,
    is_npu,
    next_power_of_2,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

_is_npu = is_npu()

if is_cuda():
    from sgl_kernel import segment_packbits  # noqa: F401

logger = logging.getLogger(__name__)


class EAGLEWorker(TpModelWorker):

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.gpu_id = gpu_id
        self.device = server_args.device
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.spec_timing_warmup_sim_recorder: Optional[SpecTimingWarmupSimRecorder] = None

        # Adaptive speculative
        self.adaptive_controller: Optional[AdaptiveController] = None
        if server_args.speculative_adaptive:
            self.adaptive_controller = AdaptiveController(
                self,
                config_path=server_args.speculative_adaptive_config,
                strategy=server_args.speculative_adaptive_strategy,
            )

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Do not capture cuda graph in `super().__init__()`
        # It will be captured later.
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():
            if server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(server_args.speculative_token_map)
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

        # Init draft worker
        if server_args.enable_dp_attention and self.speculative_algorithm.is_eagle3():
            ctx = draft_tp_context(get_attention_tp_group())
        else:
            ctx = empty_context()
        with (
            ctx
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            super().__init__(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # FIXME
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=target_worker.model_runner.memory_pool_config,
            )

        spec_timing_warmup_sim_results = os.getenv("SPEC_TIMING_WARMUP_SIM_RESULTS")
        if spec_timing_warmup_sim_results and self.tp_rank == 0 and (
            self.dp_rank is None or self.dp_rank == 0
        ):
            self.spec_timing_warmup_sim_recorder = SpecTimingWarmupSimRecorder(
                spec_timing_warmup_sim_results
            )

        embed, head = self.target_worker.model_runner.model.get_embed_and_head()

        if self.speculative_algorithm.is_eagle3():
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_model_runner.model, "load_lm_head_from_target")
                and self.draft_model_runner.model.load_lm_head_from_target
            ):
                self.draft_model_runner.model.set_embed_and_head(embed, head)
            else:
                self.draft_model_runner.model.set_embed(embed)

            # grab hot token ids
            if self.draft_model_runner.model.hot_token_id is not None:
                self.hot_token_id = self.draft_model_runner.model.hot_token_id.to(
                    embed.device
                )

        else:
            if self.hot_token_id is not None:
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            self.draft_model_runner.model.set_embed_and_head(embed, head)

        # Single packed (max, id-as-fp32) tensor for the cross-rank all_gather
        # so we issue one NCCL call per draft step instead of two.
        self._draft_top1_local_pack_buf: Optional[torch.Tensor] = None
        self._draft_top1_gathered_pack_buf: Optional[torch.Tensor] = None
        self._draft_top1_best_rank_buf: Optional[torch.Tensor] = None
        self._draft_top1_rank_index_buf: Optional[torch.Tensor] = None
        self._draft_top1_winner_f32_buf: Optional[torch.Tensor] = None
        # Pre-filled `ones` returned as `topk_p` so we skip a per-call alloc.
        self._draft_top1_ones_buf: Optional[torch.Tensor] = None
        self._draft_top1_token_cap: int = 0
        self._draft_top1_gather_cap: int = 0
        # Pre-built `local_pos -> global_vocab_id` lookup table for the
        # local-top1 path; see `_build_local_pos_to_global_id_lut`.
        self._draft_top1_local_to_global_id_lut: Optional[torch.Tensor] = None

        # Pre-allocate local-top1 scratch buffers ONCE, up-front. They are
        # referenced by captured CUDA graphs (draft decode and draft extend);
        # re-allocating later via torch.empty would drop the old tensor's
        # Python reference and let another tensor silently reuse the memory
        # while captured graphs still write/read through the baked-in
        # pointer. Size them for the largest bs any subsequent graph capture
        # can request and forbid re-allocation afterwards.
        self._preallocate_draft_top1_buffers(server_args)

        # Init attention backend and cuda graphs
        self.draft_model_runner.server_args.disable_cuda_graph = (
            backup_disable_cuda_graph
        )
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        self.eagle_use_aux_hidden_state = False
        if self.speculative_algorithm.is_eagle3():
            self.eagle_use_aux_hidden_state = True
            eagle_config = getattr(
                self.draft_model_runner.model_config.hf_config, "eagle_config", {}
            )
            self.eagle_use_aux_hidden_state = eagle_config.get(
                "use_aux_hidden_state", True
            )
        with self.draft_tp_context(
            self.draft_model_runner.tp_group
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.init_attention_backend()
            self.init_cuda_graphs()
            if self.adaptive_controller is not None:
                self.adaptive_controller.register(
                    SpecRuntimeState(
                        speculative_num_steps=self.speculative_num_steps,
                        speculative_num_draft_tokens=self.speculative_num_draft_tokens,
                        draft_attn_backend=self.draft_attn_backend,
                        cuda_graph_runner=self.cuda_graph_runner,
                        target_attn_backend=self.target_worker.model_runner.attn_backend,
                        target_graph_runner=self.target_worker.model_runner.graph_runner,
                        draft_extend_attn_backend=self.draft_extend_attn_backend,
                        cuda_graph_runner_for_draft_extend=self.cuda_graph_runner_for_draft_extend,
                    )
                )
                self.adaptive_controller.init_states()

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

    def init_attention_backend(self):
        # Create multi-step attn backends and cuda graph runners
        draft_backend_factory = DraftBackendFactory(
            self.server_args,
            self.draft_model_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Initialize decode attention backend
        self.draft_attn_backend = draft_backend_factory.create_decode_backend()

        # Initialize draft extend attention backend (respects speculative_attention_mode setting)
        self.draft_extend_attn_backend = (
            draft_backend_factory.create_draft_extend_backend()
        )

        self.draft_model_runner.draft_attn_backend = self.draft_attn_backend

    def init_cuda_graphs(self):
        """Capture cuda graphs."""
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if self.server_args.disable_cuda_graph:
            return

        Device2DraftCudaGraphRunner = {
            "npu": EAGLEDraftNpuGraphRunner,
            "cuda": EAGLEDraftCudaGraphRunner,
        }
        # Capture draft
        if self.speculative_num_steps > 1:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner = Device2DraftCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

        # Capture extend
        if self.draft_extend_attn_backend and not _is_npu:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
            )
            self.cuda_graph_runner_for_draft_extend = EAGLEDraftExtendCudaGraphRunner(
                self
            )
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            logger.info(
                f"Capture draft extend cuda graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. mem usage={(before_mem - after_mem):.2f} GB. avail mem={after_mem:.2f} GB."
            )

    def apply_runtime_state(self, state: SpecRuntimeState):
        """Apply a pre-built runtime state to this worker."""
        if self.speculative_num_steps == state.speculative_num_steps:
            return

        logger.info(
            "Switch adaptive runtime state: "
            f"steps {self.speculative_num_steps} -> {state.speculative_num_steps}, "
            f"draft_tokens {self.speculative_num_draft_tokens} -> "
            f"{state.speculative_num_draft_tokens}"
        )

        self.speculative_num_steps = state.speculative_num_steps
        self.speculative_num_draft_tokens = state.speculative_num_draft_tokens
        # Draft stage
        self.draft_attn_backend = state.draft_attn_backend
        self.draft_model_runner.draft_attn_backend = state.draft_attn_backend
        self.cuda_graph_runner = state.cuda_graph_runner
        # Verify stage
        self.target_worker.model_runner.attn_backend = state.target_attn_backend
        self.target_worker.model_runner.graph_runner = state.target_graph_runner
        # Extend stage
        self.draft_extend_attn_backend = state.draft_extend_attn_backend
        self.cuda_graph_runner_for_draft_extend = (
            state.cuda_graph_runner_for_draft_extend
        )
        # Sync server_args
        self.server_args.speculative_num_steps = state.speculative_num_steps
        self.server_args.speculative_num_draft_tokens = (
            state.speculative_num_draft_tokens
        )

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: list[int] | None = None,
    ) -> SpecRuntimeState:
        """Build a SpecRuntimeState for the given step configuration.

        Args:
            speculative_num_steps: Number of draft steps for this state.
            speculative_num_draft_tokens: Total draft tokens (= steps + 1).
            cuda_graph_bs: When provided, only these batch sizes are captured
                for all three CUDA graph runners (draft / extend / verify).
                ``None`` falls back to the full ``server_args.cuda_graph_bs``.
        """
        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)

        with self._override_worker_state(
            speculative_num_steps, speculative_num_draft_tokens, cuda_graph_bs
        ):
            capture_bs, _ = get_batch_sizes_to_capture(self.draft_model_runner)
            logger.info(
                f"Adaptive cuda graph capture (steps={speculative_num_steps}): "
                f"batch_sizes={capture_bs}"
            )

            # Reuse existing init methods for draft attention backend and cuda graphs
            self.init_attention_backend()
            self.init_cuda_graphs()

            # Capture target attention backend and CUDA graph
            target_model_runner = self.target_worker.model_runner
            backup_init = target_model_runner.init_new_workspace
            try:
                target_attn_backend = target_model_runner._get_attention_backend(
                    init_new_workspace=True
                )
            finally:
                target_model_runner.init_new_workspace = backup_init

            target_graph_runner = None
            if not self.server_args.disable_cuda_graph:
                target_graph_runner = CudaGraphRunner(
                    target_model_runner,
                    attn_backend=target_attn_backend,
                    speculative_num_steps=speculative_num_steps,
                    speculative_num_draft_tokens=speculative_num_draft_tokens,
                )

            state = SpecRuntimeState(
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
                # Draft stage
                draft_attn_backend=self.draft_attn_backend,
                cuda_graph_runner=self.cuda_graph_runner,
                # Verify stage
                target_attn_backend=target_attn_backend,
                target_graph_runner=target_graph_runner,
                # Extend stage
                draft_extend_attn_backend=self.draft_extend_attn_backend,
                cuda_graph_runner_for_draft_extend=self.cuda_graph_runner_for_draft_extend,
            )

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Built adaptive runtime state steps={speculative_num_steps}: "
            f"elapsed={time.perf_counter() - tic:.2f}s, "
            f"mem={(before_mem - after_mem):.2f}GB"
        )

        return state

    @contextmanager
    def _override_worker_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: list[int] | None = None,
    ):
        """Temporarily override server_args and worker attributes for graph capture.

        Args:
            cuda_graph_bs: When not ``None``, temporarily replaces
                ``server_args.cuda_graph_bs`` so that all three graph runners
                (draft / extend / verify) capture only this subset of batch sizes.
        """
        sa = self.server_args
        backup = (
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            self.draft_attn_backend,
            self.draft_extend_attn_backend,
            getattr(self.draft_model_runner, "draft_attn_backend", None),
            getattr(self, "cuda_graph_runner", None),
            getattr(self, "cuda_graph_runner_for_draft_extend", None),
            sa.speculative_num_steps,
            sa.speculative_num_draft_tokens,
            sa.cuda_graph_bs,
        )
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        sa.speculative_num_steps = speculative_num_steps
        sa.speculative_num_draft_tokens = speculative_num_draft_tokens
        if cuda_graph_bs is not None:
            sa.cuda_graph_bs = cuda_graph_bs
        # Each adaptive build must start with a clean buffer pool so that
        # runners with different num_tokens_per_bs (= speculative_num_steps+1)
        # do NOT accidentally reuse each other's "accept_length" / "extend_seq_lens"
        # buffers (share_buffers picks the largest old buffer, which would carry
        # the wrong fill-value from a previous build).
        pool_snapshot = dict(_forward_input_buffer_pool)
        _forward_input_buffer_pool.clear()
        try:
            yield
        finally:
            (
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
                self.draft_attn_backend,
                self.draft_extend_attn_backend,
                self.draft_model_runner.draft_attn_backend,
                self.cuda_graph_runner,
                self.cuda_graph_runner_for_draft_extend,
                sa.speculative_num_steps,
                sa.speculative_num_draft_tokens,
                sa.cuda_graph_bs,
            ) = backup
            # Restore the buffer pool to its pre-build state so subsequent
            # builds always start clean.
            _forward_input_buffer_pool.clear()
            _forward_input_buffer_pool.update(pool_snapshot)

    @property
    def draft_model_runner(self):
        return self.model_runner

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Run speculative decoding forward.

        NOTE: Many states of batch is modified as you go through. It is not guaranteed that
        the final output batch have the same state as the input.

        Args:
            batch: The batch to run forward. The state of the batch is modified as it runs.
        Returns:
            A tuple of the final logit output of the target model, next tokens accepted,
            the batch id (used for overlap schedule), and number of accepted tokens.
        """
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            (
                logits_output,
                next_token_ids,
                seq_lens_cpu,
                can_run_cuda_graph,
            ) = self.forward_target_extend(batch)
            with self.draft_tp_context(
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                self.forward_draft_extend(
                    batch,
                    logits_output.hidden_states,
                    next_token_ids,
                    seq_lens_cpu,
                    logits_output.mm_input_embeds,
                )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=can_run_cuda_graph,
            )
        else:
            # For batch_size_aware strategy: resolve steps before drafting.
            # For ema strategy: this call is a no-op (accept_lengths not given).
            controller = getattr(self, "adaptive_controller", None)
            if controller is not None:
                controller.on_verify_complete(batch_size=batch.batch_size())

            enable_log_time = self.server_args.enable_speculative_time_logging
            should_record_time = (
                enable_log_time or self.spec_timing_warmup_sim_recorder is not None
            )
            should_log_time = enable_log_time and self.tp_rank == 0 and (
                self.dp_rank is None or self.dp_rank == 0
            )
            device_module = (
                torch.get_device_module(self.device) if should_record_time else None
            )
            draft_elapsed_ms = None
            verify_elapsed_ms = None
            draft_extend_elapsed_ms = None

            with self.draft_tp_context(
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                if should_record_time:
                    device_module.synchronize()
                    draft_start = time.perf_counter()
                spec_info = self.draft(batch)
                if should_record_time:
                    device_module.synchronize()
                    draft_elapsed_ms = (time.perf_counter() - draft_start) * 1000.0

            if should_record_time:
                device_module.synchronize()
                verify_start = time.perf_counter()
            logits_output, verify_output, model_worker_batch, can_run_cuda_graph = (
                self.verify(batch, spec_info)
            )
            if should_record_time:
                device_module.synchronize()
                verify_elapsed_ms = (time.perf_counter() - verify_start) * 1000.0

            with self.draft_tp_context(
                self.draft_model_runner.tp_group
            ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
                # NOTE: We should use `check_forward_draft_extend_after_decode`
                # when DP attention is enabled, but it is slow. Skip it for now.
                should_run_draft_extend = (
                    self.server_args.enable_dp_attention
                    or batch.spec_info.verified_id.shape[0] > 0
                )
                if should_run_draft_extend:
                    # decode is not finished
                    if should_record_time:
                        device_module.synchronize()
                        draft_extend_start = time.perf_counter()
                    self.forward_draft_extend_after_decode(batch)
                    if should_record_time:
                        device_module.synchronize()
                        draft_extend_elapsed_ms = (
                            time.perf_counter() - draft_extend_start
                        ) * 1000.0

            if self.spec_timing_warmup_sim_recorder is not None:
                draft_bs = batch.batch_size()
                draft_steps = self.speculative_num_steps
                verify_bs = batch.batch_size()
                verify_query_len = (
                    spec_info.draft_token.numel() // verify_bs if verify_bs > 0 else 0
                )
                draft_extend_bs = batch.batch_size()
                draft_extend_query_len = self.speculative_num_steps + 1
                self.spec_timing_warmup_sim_recorder.update(
                    "draft", draft_bs, draft_steps, draft_elapsed_ms
                )
                self.spec_timing_warmup_sim_recorder.update(
                    "verify", verify_bs, verify_query_len, verify_elapsed_ms
                )
                if draft_extend_elapsed_ms is not None:
                    self.spec_timing_warmup_sim_recorder.update(
                        "draft_extend",
                        draft_extend_bs,
                        draft_extend_query_len,
                        draft_extend_elapsed_ms,
                    )

            if should_log_time:
                draft_bs = batch.batch_size()
                draft_steps = self.speculative_num_steps
                verify_bs = batch.batch_size()
                verify_query_len = (
                    spec_info.draft_token.numel() // verify_bs if verify_bs > 0 else 0
                )
                draft_extend_bs = batch.batch_size()
                draft_extend_query_len = self.speculative_num_steps + 1
                logger.info(
                    "run_batch speculative timing: "
                    "draft(bs=%s, draft_steps=%s, time_ms=%.3f) "
                    "verify(bs=%s, query_len=%s, time_ms=%.3f) "
                    "draft_extend(bs=%s, query_len=%s, time_ms=%s)",
                    draft_bs,
                    draft_steps,
                    draft_elapsed_ms,
                    verify_bs,
                    verify_query_len,
                    verify_elapsed_ms,
                    draft_extend_bs,
                    draft_extend_query_len,
                    (
                        f"{draft_extend_elapsed_ms:.3f}"
                        if draft_extend_elapsed_ms is not None
                        else "skipped"
                    ),
                )

            # For ema strategy: update EMA after verify results are available.
            # For batch_size_aware strategy: this call is a no-op (batch_size not given).
            controller = getattr(self, "adaptive_controller", None)
            if controller is not None:
                controller.on_verify_complete(verify_output.accept_length_per_req_cpu)

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=verify_output.verified_id,
                num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
                accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
                can_run_cuda_graph=can_run_cuda_graph,
            )

    def check_forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        local_need_forward = batch.spec_info.verified_id.shape[0] > 0
        if not self.server_args.enable_dp_attention:
            return local_need_forward

        global_need_forward = torch.tensor(
            [
                (local_need_forward),
            ],
            dtype=torch.int64,
        )
        torch.distributed.all_reduce(
            global_need_forward, group=get_tp_group().cpu_group
        )
        global_need_forward_cnt = global_need_forward[0].item()
        need_forward = global_need_forward_cnt > 0
        return need_forward

    def forward_target_extend(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, Optional[torch.Tensor], bool]:
        """Run the target extend.

        Args:
            batch: The batch to run. States could be modified.

        Returns:
            logits_output: The output of logits. It will contain the full hidden states.
            next_token_ids: Next token ids generated.
            seq_lens_cpu: CPU copy of sequence lengths for the draft prefill path.
            can_run_cuda_graph: Whether the target prefill ran with cuda graph.
        """
        # Forward with the target model and get hidden states.
        # We need the full hidden states to prefill the KV cache of the draft model.
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
        logits_output, next_token_ids = (
            batch_result.logits_output,
            batch_result.next_token_ids,
        )
        return (
            logits_output,
            next_token_ids,
            model_worker_batch.seq_lens_cpu,
            batch_result.can_run_cuda_graph,
        )

    def _draft_preprocess_decode(self, batch: ScheduleBatch):
        batch.maybe_evict_swa()
        for req in batch.reqs:
            req.decode_batch_idx += 1

        # Parse args
        num_seqs = batch.batch_size()
        spec_info = batch.spec_info

        # Accumulate penalty
        if batch.sampling_info.penalizer_orchestrator.is_required:
            # This is a relaxed version of penalties for speculative decoding.
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                spec_info.verified_id.to(torch.int64)
            )

        # Allocate cache locations
        # Layout of the out_cache_loc
        # [       topk 0         ] [       topk 1         ]
        # [iter=0, iter=1, iter=2] [iter=0, iter=1, iter=2]
        if self.page_size == 1:
            alloc_len_per_decode = self.speculative_num_steps * self.topk
            # TODO: We only need self.speculative_num_steps - 1 * topk cache loc
            out_cache_loc, token_to_kv_pool_state_backup = alloc_token_slots(
                batch.tree_cache,
                num_seqs * alloc_len_per_decode,
                backup_state=True,
            )
        else:
            if self.topk == 1:
                prefix_lens, seq_lens, last_loc = get_last_loc_large_page_size_top_k_1(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                )
                prefix_lens_cpu = batch.seq_lens_cpu
                seq_lens_cpu = batch.seq_lens_cpu + self.speculative_num_steps
                extend_num_tokens = num_seqs * self.speculative_num_steps
            else:
                # In this case, the last partial page needs to be duplicated.
                # KV cache layout in batch.req_to_token_pool.req_to_token:
                #
                # | -------- | -- xxxx .. | -- xxxx .. | -- xxxx .. |
                #    prefix     top-k = 0    tok-k = 1    top-k = 2
                #
                #  "-" means prefix tokens
                #  "x" means speculative draft tokens
                #  "." means padded tokens

                (
                    prefix_lens,
                    seq_lens,
                    last_loc,
                    self.num_new_pages_per_topk,
                    self.extend_lens,
                    last_page_lens,
                ) = get_last_loc_large_page_size_large_top_k(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                    self.topk,
                    self.page_size,
                )
                prefix_lens_cpu = batch.seq_lens_cpu
                last_page_lens_cpu = prefix_lens_cpu % self.page_size
                num_new_pages_per_topk = (
                    last_page_lens_cpu + self.speculative_num_steps + self.page_size - 1
                ) // self.page_size
                seq_lens_cpu = (
                    prefix_lens_cpu // self.page_size * self.page_size
                    + num_new_pages_per_topk * (self.page_size * self.topk)
                )
                extend_num_tokens = torch.sum((seq_lens_cpu - prefix_lens_cpu)).item()

            out_cache_loc, token_to_kv_pool_state_backup = (
                alloc_paged_token_slots_extend(
                    batch.tree_cache,
                    prefix_lens,
                    prefix_lens_cpu,
                    seq_lens,
                    seq_lens_cpu,
                    last_loc,
                    extend_num_tokens,
                    backup_state=True,
                )
            )

        if self.page_size > 1 and self.topk > 1:
            last_page_lens_cumsum = torch.cumsum(last_page_lens, dim=0)
            duplicate_cache_len = torch.sum(last_page_lens_cpu).item() * (self.topk - 1)
            target_cache_loc = torch.zeros(
                duplicate_cache_len, dtype=torch.int32, device=self.device
            )
            source_cache_loc = torch.zeros(
                duplicate_cache_len, dtype=torch.int32, device=self.device
            )
        else:
            # When source_cache_loc is not needed, simply skip
            duplicate_cache_len = 0
            source_cache_loc, target_cache_loc, last_page_lens_cumsum = None, None, None

        assign_draft_cache_locs[(num_seqs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            self.extend_lens,
            self.num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            batch.req_to_token_pool.req_to_token.shape[1],
            self.topk,
            self.speculative_num_steps,
            self.page_size,
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps + self.page_size),
        )

        if self.page_size > 1 and self.topk > 1:
            if duplicate_cache_len > 0:
                self.draft_model_runner.token_to_kv_pool.move_kv_cache(
                    target_cache_loc, source_cache_loc
                )
            # Remove padded slots
            # TODO: We only need self.speculative_num_steps - 1 cache loc
            out_cache_loc = out_cache_loc[
                : num_seqs * self.topk * self.speculative_num_steps
            ]

        batch.out_cache_loc = out_cache_loc
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        batch.return_hidden_states = False
        spec_info.positions = batch.seq_lens.repeat_interleave(self.topk, dim=0)
        self.token_to_kv_pool_allocator.restore_state(token_to_kv_pool_state_backup)

    def _draft_preprocess_idle(self, batch: ScheduleBatch):
        batch.spec_info = EagleDraftInput.create_idle_input(
            device=self.device,
            hidden_size=self.model_config.hidden_size,
            dtype=self.model_config.dtype,
            topk=self.topk,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

    def draft(self, batch: ScheduleBatch):
        # Parse args
        if batch.forward_mode.is_idle():
            self._draft_preprocess_idle(batch)
        else:
            self._draft_preprocess_decode(batch)

        spec_info = batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)

        spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        spec_info.num_tokens_per_req = self.topk
        spec_info.num_tokens_for_logprob_per_req = self.topk
        batch.return_hidden_states = False

        # Get forward batch
        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.capture_hidden_mode == CaptureHiddenMode.LAST
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )
        can_cuda_graph = self.cuda_graph_runner and self.cuda_graph_runner.can_run(
            forward_batch
        )
        if can_cuda_graph:
            parent_list, top_scores_index, draft_tokens = self.cuda_graph_runner.replay(
                forward_batch
            )
        else:
            forward_batch.can_run_dp_cuda_graph = False
            if (
                not forward_batch.forward_mode.is_idle()
                and self.speculative_num_steps > 1
            ):
                # Skip attention backend init for idle mode or 1-step draft
                self.draft_attn_backend.init_forward_metadata(forward_batch)
            # Run forward steps
            parent_list, top_scores_index, draft_tokens = self.draft_forward(
                forward_batch
            )

        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            spec_info.verified_id,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            batch.seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
        )

        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=forward_batch.seq_lens_sum,
            seq_lens_cpu=forward_batch.seq_lens_cpu,
        )

    def _should_use_local_spec_draft_top1(self) -> bool:
        if not self.server_args.speculative_local_draft_top1:
            return False
        if self.topk != 1:
            return False
        # When speculative_token_map/hot_token_id is active, the shared lm_head can be
        # remapped to a hot-vocab subset, so lm_head.shard_indices no longer describe the
        # actual local logits layout. The local-top1 optimization would decode wrong ids.
        if self.hot_token_id is not None:
            return False
        lm_head = getattr(self.draft_model_runner.model, "lm_head", None)
        if lm_head is None or not hasattr(lm_head, "shard_indices"):
            return False
        return get_tp_group().world_size > 1

    def _preallocate_draft_top1_buffers(self, server_args: ServerArgs) -> None:
        """Allocate local-top1 scratch buffers with a safe upper bound.

        Called exactly once before any CUDA graph is captured. The capacity
        must cover the largest num_tokens any draft-decode / draft-extend
        CUDA graph will pass in (for topk=1 this is the capture-time bs,
        bounded by cuda_graph_max_bs / max_running_requests).
        """
        if not server_args.speculative_local_draft_top1:
            return

        tp_size = int(get_tp_group().world_size)
        if tp_size <= 1:
            return

        max_bs_candidates = [
            server_args.cuda_graph_max_bs,
            server_args.max_running_requests,
        ]
        max_bs_candidates = [x for x in max_bs_candidates if x is not None and x > 0]
        max_num_tokens = max(max_bs_candidates) if max_bs_candidates else 0
        # Hard safety floor so that we always have enough room.
        max_num_tokens = max(max_num_tokens, 1024)

        device = torch.device(self.device)
        gather_cap = max_num_tokens * tp_size

        # (N, 2) fp32 pack: [:, 0] = local_max, [:, 1] = global_id-as-fp32.
        self._draft_top1_local_pack_buf = torch.empty(
            (max_num_tokens, 2), dtype=torch.float32, device=device
        )
        self._draft_top1_gathered_pack_buf = torch.empty(
            (gather_cap, 2), dtype=torch.float32, device=device
        )
        self._draft_top1_gather_cap = gather_cap

        self._draft_top1_best_rank_buf = torch.empty(
            (max_num_tokens,), dtype=torch.int64, device=device
        )
        self._draft_top1_rank_index_buf = torch.empty(
            (1, max_num_tokens), dtype=torch.int64, device=device
        )
        self._draft_top1_winner_f32_buf = torch.empty(
            (1, max_num_tokens), dtype=torch.float32, device=device
        )
        self._draft_top1_ones_buf = torch.ones(
            (max_num_tokens, 1), dtype=torch.float32, device=device
        )
        self._draft_top1_token_cap = max_num_tokens

        self._build_local_pos_to_global_id_lut(device=device)

    def _build_local_pos_to_global_id_lut(self, device: torch.device) -> None:
        """Precompute local-shard-position -> global-vocab-id for local top-1.

        `_distributed_draft_top1_from_local_logits` takes argmax within the
        two valid regions of the padded local shard (base: [0, num_org),
        added: [num_org_padded, num_org_padded + num_added)). A flat LUT of
        length `num_org_padded + num_added_padded` lets us map the winning
        local position to its global vocab id with a single gather, removing
        the runtime base/added branch.
        """
        lm_head = getattr(self.draft_model_runner.model, "lm_head", None)
        if lm_head is None or not hasattr(lm_head, "shard_indices"):
            self._draft_top1_local_to_global_id_lut = None
            return
        shard = lm_head.shard_indices
        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        num_added_padded = int(shard.num_added_elements_padded)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)

        total = num_org_padded + num_added_padded
        # Padding slots are filled with 0 for safety — argmax is restricted to
        # the valid regions above, so these entries are never read.
        lut = torch.zeros((total,), dtype=torch.int64, device=device)
        if num_org > 0:
            lut[:num_org] = (
                torch.arange(num_org, dtype=torch.int64, device=device)
                + org_vocab_start
            )
        if num_added > 0:
            lut[num_org_padded : num_org_padded + num_added] = (
                torch.arange(num_added, dtype=torch.int64, device=device)
                + added_vocab_start
            )
        # The cross-rank all_gather packs ids into fp32 alongside the max
        # value to halve NCCL calls. fp32 has 24 mantissa bits, so ids must
        # fit in [0, 2**24) for a lossless round-trip. This covers every
        # real-world LLM vocabulary (the largest known is ~256K).
        _FP32_INT_EXACT_MAX = 1 << 24
        assert int(lut.max().item()) < _FP32_INT_EXACT_MAX, (
            f"Global vocab id >= 2**24 is not exactly representable in fp32; "
            f"local-top1 all_gather packing would be lossy. "
            f"Max id = {int(lut.max().item())}."
        )
        self._draft_top1_local_to_global_id_lut = lut

    def _ensure_draft_top1_buffers(
        self, *, num_tokens: int, tp_size: int, device: torch.device
    ) -> None:
        # Buffers MUST have been preallocated in __init__ (before any CUDA
        # graph capture). If we ever need more capacity than was preallocated
        # it means the preallocation upper bound was wrong — we fail loudly
        # rather than silently re-allocating, because re-allocating here would
        # invalidate already-captured CUDA graphs (see the note at the
        # _preallocate_draft_top1_buffers call site).
        needed_gather_cap = num_tokens * tp_size
        assert (
            self._draft_top1_local_pack_buf is not None
            and self._draft_top1_gathered_pack_buf is not None
            and self._draft_top1_best_rank_buf is not None
            and self._draft_top1_rank_index_buf is not None
            and self._draft_top1_winner_f32_buf is not None
            and self._draft_top1_ones_buf is not None
        ), (
            "Local-top1 scratch buffers were not preallocated. "
            "speculative_local_draft_top1 should have triggered "
            "_preallocate_draft_top1_buffers in __init__."
        )
        assert self._draft_top1_gather_cap >= needed_gather_cap, (
            f"Local-top1 gather buffer too small: need {needed_gather_cap}, "
            f"have {self._draft_top1_gather_cap}. Increase --cuda-graph-max-bs "
            f"or the internal safety floor in _preallocate_draft_top1_buffers."
        )
        assert self._draft_top1_token_cap >= num_tokens, (
            f"Local-top1 token buffer too small: need {num_tokens}, "
            f"have {self._draft_top1_token_cap}."
        )
        assert self._draft_top1_local_pack_buf.device == device, (
            f"Local-top1 buffer device mismatch: "
            f"buffer on {self._draft_top1_local_pack_buf.device}, request on {device}."
        )

    def _distributed_draft_top1_from_local_logits(
        self, local_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Global draft top-1 computed from per-rank sharded logits.

        Skips the vocab-size all_gather by taking argmax within each rank's
        local shard, gathering only the (max, global_id) pair, and selecting
        the winning rank per token. The caller is expected to pass
        `logits_output.next_token_logits`, which — when
        `forward_batch.use_local_spec_draft_top1` is set — is the untouched
        local shard produced by `LogitsProcessor._get_logits` (no fp32 copy,
        no softcap; both are argmax-invariant).
        """
        lm_head = self.draft_model_runner.model.lm_head
        if not hasattr(lm_head, "weight") or not hasattr(lm_head, "shard_indices"):
            raise RuntimeError(
                "EAGLE local draft top1 requires a vocab-parallel lm_head with "
                "`weight` and `shard_indices`."
            )
        shard = lm_head.shard_indices
        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)
        num_tokens = int(local_logits.shape[0])
        device = local_logits.device

        if num_tokens == 0:
            empty_probs = torch.empty((0, 1), dtype=torch.float32, device=device)
            empty_ids = torch.empty((0, 1), dtype=torch.int64, device=device)
            return empty_probs, empty_ids

        self._ensure_draft_top1_buffers(
            num_tokens=num_tokens, tp_size=tp_size, device=device
        )

        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)

        # Argmax is restricted to the two valid regions of the padded local
        # shard so `local_arg` always indexes a real vocab entry.
        if num_added > 0:
            base_max, base_arg = torch.max(local_logits[:, :num_org], dim=-1)
            added_max, added_arg = torch.max(
                local_logits[:, num_org_padded : num_org_padded + num_added], dim=-1
            )
            use_added = added_max > base_max
            local_max = torch.where(use_added, added_max, base_max)
            local_arg = torch.where(use_added, added_arg + num_org_padded, base_arg)
        elif num_org > 0:
            local_max, local_arg = torch.max(local_logits[:, :num_org], dim=-1)
        else:
            # This rank owns no real vocab entries — force -inf so every other
            # rank wins during the cross-rank argmax.
            local_max = torch.full(
                (num_tokens,),
                torch.finfo(local_logits.dtype).min,
                dtype=local_logits.dtype,
                device=device,
            )
            local_arg = torch.zeros((num_tokens,), dtype=torch.int64, device=device)

        # Single gather via a pre-built LUT instead of runtime base/added branching.
        assert self._draft_top1_local_to_global_id_lut is not None, (
            "Local top-1 LUT was not built. _preallocate_draft_top1_buffers "
            "must be called in EAGLEWorker.__init__ before this path is used."
        )
        global_ids = self._draft_top1_local_to_global_id_lut[local_arg]

        if tp_size > 1:
            # Single NCCL call instead of two: pack `(local_max, global_id)`
            # into one (N, 2) fp32 tensor and all_gather it whole. This is the
            # latency-dominant cost of the local-top1 path, so halving the
            # call count matters more than trimming volume.
            local_pack = self._draft_top1_local_pack_buf[:num_tokens]
            local_pack[:, 0].copy_(local_max)
            local_pack[:, 1].copy_(global_ids)

            gathered_pack = self._draft_top1_gathered_pack_buf[: num_tokens * tp_size]
            tp_group.all_gather_into_tensor(gathered_pack, local_pack)

            packed = gathered_pack.view(tp_size, num_tokens, 2)
            best_rank = self._draft_top1_best_rank_buf[:num_tokens]
            torch.argmax(packed[..., 0], dim=0, out=best_rank)

            rank_index = self._draft_top1_rank_index_buf[:, :num_tokens]
            rank_index[0].copy_(best_rank)
            winner_f32 = self._draft_top1_winner_f32_buf[:, :num_tokens]
            torch.gather(packed[..., 1], 0, rank_index, out=winner_f32)

            # `.to(int64)` always returns a fresh tensor (different dtype),
            # which both casts back from the fp32-packed id and breaks any
            # aliasing with the reused scratch buffers above — no extra clone.
            selected_ids_out = winner_f32.to(torch.int64).view(-1, 1)
        else:
            # Unreachable: `_should_use_local_spec_draft_top1` gates this path
            # on tp_size > 1. Kept as a minimal correctness fallback.
            selected_ids_out = global_ids.view(-1, 1).clone()

        # Pre-filled ones, shared read-only across calls.
        top1_p = self._draft_top1_ones_buf[:num_tokens]
        return top1_p, selected_ids_out

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info = forward_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]
        # TODO: We only need self.speculative_num_steps - 1 cache loc
        out_cache_loc = out_cache_loc.reshape(
            forward_batch.batch_size, self.topk, self.speculative_num_steps
        )
        out_cache_loc = out_cache_loc.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []

        # Forward multiple steps
        scores = None
        use_local_spec_draft_top1 = self._should_use_local_spec_draft_top1()
        draft_vocab_size = self.draft_model_runner.model.lm_head.num_embeddings
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            forward_batch.input_ids = input_ids
            # This is a temporary fix for the case that the user is using standalone
            # speculative decoding and the draft model architecture is gpt-oss. gpt-oss
            # rope kernel needs cache_loc to be contiguous.
            if (
                self.server_args.speculative_algorithm == "STANDALONE"
                and self.model_config.hf_config.architectures[0] == "GptOssForCausalLM"
            ):
                out_cache_loc = out_cache_loc.contiguous()
            forward_batch.out_cache_loc = out_cache_loc[i]
            forward_batch.positions.add_(1)
            forward_batch.attn_backend = self.draft_attn_backend.attn_backends[i]
            spec_info.hidden_states = hidden_states
            forward_batch.use_local_spec_draft_top1 = use_local_spec_draft_top1

            # Run forward
            logits_output = self.draft_model_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
            maybe_detect_nan(logits_output.next_token_logits, f"draft_forward step {i}")
            if use_local_spec_draft_top1:
                topk_p, topk_index = self._distributed_draft_top1_from_local_logits(
                    logits_output.next_token_logits
                )
            else:
                probs = torch.softmax(logits_output.next_token_logits, dim=-1)
                topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            maybe_detect_oob(
                topk_index,
                0,
                draft_vocab_size,
                f"draft_forward step {i}: topk_index OOB vs vocab_size={draft_vocab_size}",
            )
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

        return parent_list, top_scores_index, draft_tokens

    def clear_cache_pool(self):
        # allocator and kv cache pool are shared with target worker
        pass

    def verify(self, batch: ScheduleBatch, spec_info: EagleVerifyInput):
        seq_lens_pre_verify = batch.seq_lens.clone()
        spec_info.prepare_for_verify(batch, self.page_size)
        spec_info.num_tokens_per_req = self.speculative_num_steps + 1
        batch.return_hidden_states = False
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = spec_info

        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=spec_info.seq_lens_cpu
        )
        assert model_worker_batch.capture_hidden_mode == spec_info.capture_hidden_mode

        if batch.has_grammar:
            retrieve_next_token_cpu = spec_info.retrive_next_token.cpu()
            retrieve_next_sibling_cpu = spec_info.retrive_next_sibling.cpu()
            draft_tokens_cpu = spec_info.draft_token.view(
                spec_info.retrive_next_token.shape
            ).cpu()

        # Forward
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        vocab_mask = None
        if batch.has_grammar:
            # Generate the logit mask for structured output.
            # Overlap the CPU operations for bitmask generation with the forward pass.
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                spec_info,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert spec_info.grammar is not None
                vocab_mask = vocab_mask.to(spec_info.retrive_next_token.device)
                # NOTE (sk): otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                batch.sampling_info.vocab_mask = None

        maybe_detect_nan(logits_output.next_token_logits, "verify: target model logits")

        spec_info.hidden_states = logits_output.hidden_states
        res: EagleVerifyOutput = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask,
        )

        # Post process based on verified outputs.
        # Pick indices that we care (accepted)
        logits_output.next_token_logits = logits_output.next_token_logits[
            res.accepted_indices
        ]
        logits_output.hidden_states = logits_output.hidden_states[res.accepted_indices]

        if (
            self.target_worker.model_runner.hybrid_gdn_config is not None
            or self.target_worker.model_runner.mamba2_config is not None
            or self.target_worker.model_runner.hybrid_lightning_config is not None
        ):
            self._mamba_verify_update(
                batch, res, logits_output, spec_info, seq_lens_pre_verify
            )

        if batch.return_logprob:
            add_output_logprobs_for_spec_v1(batch, res, logits_output)

        # Prepare the batch for the next draft forwards.
        batch.forward_mode = (
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        )
        batch.spec_info = res.draft_input

        return logits_output, res, model_worker_batch, can_run_cuda_graph

    def _mamba_verify_update(
        self,
        batch: ScheduleBatch,
        res: EagleVerifyOutput,
        logits_output: LogitsProcessorOutput,
        spec_info: EagleVerifyInput,
        seq_lens_pre_verify: torch.Tensor,
    ):
        # Under DP attention, some ranks can be IDLE during target verify and never
        # initialize mamba forward metadata for this step.
        if batch.forward_mode.is_idle():
            return

        accepted_length = (
            torch.tensor(
                res.accept_length_per_req_cpu,
                device=logits_output.hidden_states.device,
                dtype=torch.int64,
            )
            + 1
        )
        cumulative_accepted_lengths = torch.cumsum(accepted_length, dim=0)
        # prepend 0 to the cumulative_accepted_lengths
        accepted_indices_start = torch.cat(
            [
                torch.zeros(
                    1,
                    dtype=cumulative_accepted_lengths.dtype,
                    device=cumulative_accepted_lengths.device,
                ),
                cumulative_accepted_lengths[:-1],
            ]
        )
        accepted_indices_offset = torch.arange(
            0,
            len(batch.seq_lens) * batch.spec_info.draft_token_num,
            step=batch.spec_info.draft_token_num,
            dtype=accepted_indices_start.dtype,
            device=accepted_indices_start.device,
        )

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        # res.accepted_indices.shape[0] > 0 skips DP attn idle batch
        if spec_info.topk > 1 and res.accepted_indices.shape[0] > 0:
            # accepted_indices=[0,2,3,4,5,7,9,10,11], accepted_length=[4, 3, 2], cumulative_accepted_lengths=[4, 7, 9]
            # first_token_indices_per_req=prepend(0, accepted_indices[cumulative_accepted_lengths[:-1]]) = [0, 5, 10]
            # last_token_indices_per_req=accepted_indices[cumulative_accepted_lengths - 1] = [4, 9, 11] (last token ID of each req)
            # max_relative_indices_per_req = [4,4,1]; those are the per-req spec-decoding step offsets that contain the correct mamba caches
            # first_token_indices_per_req = res.accepted_indices[accepted_indices_start]
            accepted_steps = (
                res.accepted_indices[cumulative_accepted_lengths - 1]
                - accepted_indices_offset
            )
        else:
            accepted_steps = accepted_length - 1

        if batch.mamba_track_indices is not None:
            # If after verify, the request's seq_lens has crossed a mamba track interval,
            # we need to update the mamba state for the request at the crossing point.
            mamba_track_interval = self.server_args.mamba_track_interval
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != batch.seq_lens // mamba_track_interval
            )
            tracking_point = (
                batch.seq_lens // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            mamba_steps_to_track = torch.where(
                to_track_mask,
                res.accepted_indices[to_track_ith + accepted_indices_start]
                - accepted_indices_offset,
                -1,
            )
        else:
            mamba_steps_to_track = None

        self.target_worker.model_runner.attn_backend.update_mamba_state_after_mtp_verify(
            accepted_steps=accepted_steps,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
        )

    def forward_draft_extend(
        self,
        batch: ScheduleBatch,
        hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        mm_input_embeds: Optional[torch.Tensor] = None,
    ):
        """Run draft model extend. This API modifies the states of the batch.

        Args:
            batch: The batch to run.
            hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        batch.spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            verified_id=next_token_ids,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )
        batch.return_hidden_states = False
        batch.spec_info.prepare_for_extend(batch)
        batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=seq_lens_cpu
        )
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )
        forward_batch.return_logprob = False
        forward_batch.use_local_spec_draft_top1 = self._should_use_local_spec_draft_top1()
        if mm_input_embeds is not None:
            forward_batch.mm_input_embeds = mm_input_embeds
        logits_output = self.draft_model_runner.forward(forward_batch).logits_output
        maybe_detect_nan(logits_output.next_token_logits, "draft_extend_for_prefill")
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        assert forward_batch.spec_info is batch.spec_info
        self.capture_for_decode(logits_output, forward_batch.spec_info)

    def forward_draft_extend_after_decode(self, batch: ScheduleBatch):
        assert isinstance(batch.spec_info, EagleDraftInput)
        # Backup fields that will be modified in-place
        seq_lens_backup = batch.seq_lens.clone()
        seq_lens_cpu_backup = batch.seq_lens_cpu.clone()
        req_pool_indices_backup = batch.req_pool_indices
        accept_length_backup = batch.spec_info.accept_length.clone()
        return_logprob_backup = batch.return_logprob

        input_is_idle = batch.forward_mode.is_idle()

        if not input_is_idle and batch.spec_info.verified_id.numel() == 0:
            batch = batch.copy()
            batch.prepare_for_idle()
            hidden_size = (
                self.model_config.hidden_size * 3
                if self.speculative_algorithm.is_eagle3()
                and self.eagle_use_aux_hidden_state
                else self.model_config.hidden_size
            )
            batch.spec_info = EagleDraftInput.create_idle_input(
                device=self.device,
                hidden_size=hidden_size,
                dtype=self.model_config.dtype,
                topk=self.topk,
                capture_hidden_mode=CaptureHiddenMode.LAST,
            )

        batch.spec_info.num_tokens_per_req = self.speculative_num_steps + 1
        batch.spec_info.num_tokens_for_logprob_per_req = 1
        batch.spec_info.prepare_extend_after_decode(
            batch,
            self.speculative_num_steps,
        )
        batch.forward_mode = (
            ForwardMode.DRAFT_EXTEND
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )

        batch.return_hidden_states = False
        model_worker_batch = batch.get_model_worker_batch()
        assert model_worker_batch.capture_hidden_mode == CaptureHiddenMode.LAST
        forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.draft_model_runner
        )
        forward_batch.use_local_spec_draft_top1 = self._should_use_local_spec_draft_top1()
        if forward_batch.seq_lens_cpu is not None:
            forward_batch.seq_lens_sum = forward_batch.seq_lens_cpu.sum().item()
        else:
            forward_batch.seq_lens_sum = batch.seq_lens.sum().item()

        # Run
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run(forward_batch)
        )
        if can_cuda_graph:
            logits_output = self.cuda_graph_runner_for_draft_extend.replay(
                forward_batch
            )
            forward_batch.spec_info.topk_p, forward_batch.spec_info.topk_index = (
                logits_output.topk_p,
                logits_output.topk_index,
            )
            forward_batch.spec_info.hidden_states = logits_output.hidden_states
        else:
            forward_batch.can_run_dp_cuda_graph = False
            if not forward_batch.forward_mode.is_idle():
                attn_backend = (
                    self.draft_extend_attn_backend
                    or self.draft_model_runner.attn_backend
                )
                attn_backend.init_forward_metadata(forward_batch)
                forward_batch.attn_backend = attn_backend
            logits_output = self.draft_model_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
            self.capture_for_decode(logits_output, forward_batch.spec_info)

        maybe_detect_nan(
            logits_output.next_token_logits,
            f"draft_extend_after_decode (cuda_graph={can_cuda_graph})",
        )

        # Restore backup.
        # This is because `seq_lens` can be modified in `prepare_extend_after_decode`
        batch.forward_mode = (
            ForwardMode.DECODE if not input_is_idle else ForwardMode.IDLE
        )
        batch.seq_lens = seq_lens_backup
        batch.seq_lens_cpu = seq_lens_cpu_backup
        batch.req_pool_indices = req_pool_indices_backup
        batch.spec_info.accept_length = accept_length_backup
        batch.return_logprob = return_logprob_backup

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, draft_input: EagleDraftInput
    ):
        if self._should_use_local_spec_draft_top1():
            draft_input.topk_p, draft_input.topk_index = (
                self._distributed_draft_top1_from_local_logits(
                    logits_output.next_token_logits
                )
            )
        else:
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            draft_input.topk_p, draft_input.topk_index = fast_topk(
                probs, self.topk, dim=-1
            )
        draft_input.hidden_states = logits_output.hidden_states

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.tp_rank]
        )
        success, message = self.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        if not success:
            return success, message

        success, message = self.target_worker.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        return success, message


@torch.compile(dynamic=True, disable=_is_npu)
def get_last_loc_large_page_size_top_k_1(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens,
    speculative_num_steps: int,
):
    prefix_lens = seq_lens
    seq_lens = prefix_lens + speculative_num_steps
    last_loc = get_last_loc(
        req_to_token,
        req_pool_indices,
        prefix_lens,
    )
    return prefix_lens, seq_lens, last_loc
