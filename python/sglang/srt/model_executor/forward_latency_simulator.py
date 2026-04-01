"""Forward-pass latency simulator for pre-traffic profiling.

After CUDA-graph capture completes and before real requests arrive, this module
constructs realistic ForwardBatch instances—with per-sequence ``seq_lens``—
and replays the captured CUDA graphs to measure attention computation latency
for different context-length distributions and batch sizes.

Key differences vs ``_dummy_run``
----------------------------------
* ``_dummy_run`` uses a *uniform* fill value (typically 1 for FlashAttention)
  for every sequence, which is required for CUDA-graph warmup but cannot
  reflect realistic KV-cache attend lengths.
* This simulator lets callers specify *per-sequence* seq_lens such as
  ``[1024, 2048, 4096]``.  The CUDA graph is replayed with those lengths
  written into the buffers, so the FlashAttention kernel really does attend
  to N tokens per sequence.

Design constraints
------------------
* All data construction and forward-pass invocations are contained here.
  External callers only import :class:`ForwardLatencySimulator` and call
  :meth:`run`.
* Uses ``req_pool_indices=0`` and ``out_cache_loc=0`` as dummy KV-cache
  references.  The attention kernel executes with the correct *shape* but
  operates on garbage KV values, which is intentional: only timing matters.
* Automatically detects whether speculative decoding is active and simulates
  ``TARGET_VERIFY`` (verify step) instead of plain ``DECODE``.
* Falls back to the non-graph path when the requested batch size exceeds the
  captured graph's ``max_bs``.

Configuration (server args)
---------------------------
  --forward-latency-sim-batch-sizes "1,4,8,16"
  --forward-latency-sim-seq-lens '{"4": [1024, 2048, 4096, 512]}'
  --forward-latency-sim-warmup 3        (default)
  --forward-latency-sim-repeat 10       (default)

Environment
-----------
  SGLANG_FORWARD_LATENCY_SIM_STATS_DIR: when set, appends one JSONL line per
  (batch_size, seq_lens) config to querylen_<N>_batchsize_<M>.jsonl (same layout
  as SGLANG_SPEC_TIMING_STATS_DIR). draft_times and draft_extend_times are 0.
"""

from __future__ import annotations

# Env var for writing timing-style stats (compatible with SPEC_DECODE_STATS format)
SGLANG_FORWARD_LATENCY_SIM_STATS_DIR = "SGLANG_FORWARD_LATENCY_SIM_STATS_DIR"

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.utils import require_mlp_tp_gather

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class LatencySimResult:
    """Latency measurement result for one (batch_size, seq_lens) combination."""

    batch_size: int
    seq_lens: List[int]
    #: Per-sequence query length: ``num_tokens_per_bs`` (TARGET_VERIFY = verify
    #: tokens per seq; DECODE = 1).
    query_len: int
    latency_mean_ms: float
    latency_std_ms: float
    latency_min_ms: float
    latency_max_ms: float
    used_cuda_graph: bool

    def to_dict(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "seq_lens": self.seq_lens,
            "query_len": self.query_len,
            "latency_mean_ms": self.latency_mean_ms,
            "latency_std_ms": self.latency_std_ms,
            "latency_min_ms": self.latency_min_ms,
            "latency_max_ms": self.latency_max_ms,
            "used_cuda_graph": self.used_cuda_graph,
        }


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class ForwardLatencySimulator:
    """Simulates decode / verify forward passes with custom per-sequence seq_lens.

    Constructs a realistic :class:`ForwardBatch` for each (batch_size, seq_lens)
    configuration and replays it through the CUDA graph (or falls back to the
    eager path) to measure attention latency.

    Parameters
    ----------
    model_runner:
        The :class:`ModelRunner` instance whose model and attention backend will
        be used for timing.  Must be fully initialised (weights loaded, KV pool
        allocated, CUDA graphs captured) before calling :meth:`run`.
    """

    # L2 cache size upper bound (bytes).  H100 has ~50 MB, GH200 ~60 MB.
    # Using 128 MB guarantees a full flush on all current NVIDIA GPUs.
    _L2_FLUSH_BYTES = 128 * 1024 * 1024

    def __init__(self, model_runner: "ModelRunner") -> None:
        self.mr = model_runner
        # Pre-allocate a scratch buffer for L2 cache flushing.  Writing to
        # this buffer before each timing measurement evicts all cached KV
        # pages from L2, ensuring each run sees realistic HBM-bandwidth
        # pressure (no carry-over from previous iterations or aliased pages).
        try:
            self._l2_flush_buf: Optional[torch.Tensor] = torch.empty(
                self._L2_FLUSH_BYTES // 4,  # float32 elements
                dtype=torch.float32,
                device=model_runner.device,
            )
            logger.info(
                "[ForwardLatencySimulator] L2 flush buffer allocated: %.1f MB on %s",
                self._L2_FLUSH_BYTES / 1024 / 1024,
                model_runner.device,
            )
        except Exception as e:
            self._l2_flush_buf = None
            logger.warning(
                "[ForwardLatencySimulator] Failed to allocate L2 flush buffer (%s); "
                "measurements may be overly optimistic when KV aliasing occurs.",
                e,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        batch_sizes: List[int],
        seq_lens_config: Optional[Dict[int, List[int]]] = None,
        fixed_seq_len: Optional[int] = None,
        num_warmup: int = 3,
        num_repeat: int = 10,
    ) -> List[LatencySimResult]:
        """Run latency simulation for each requested batch size.

        Parameters
        ----------
        batch_sizes:
            List of batch sizes (number of concurrent sequences) to profile.
        seq_lens_config:
            Optional mapping ``{batch_size: [seq_len_per_seq]}``.  Missing entries
            are filled with :meth:`_max_safe_seq_len` (``context_len - num_tokens_per_bs``
            for TARGET_VERIFY) unless *fixed_seq_len* is set.  Keys can be :class:`int`
            or string (JSON keys).
        fixed_seq_len:
            When set, all sequences that are **not** explicitly specified in
            *seq_lens_config* use this value as their KV-cache length (clamped to
            ``_max_safe_seq_len()``).  This allows simulating a fixed context length
            across all batch sizes without writing a full JSON *seq_lens_config*.
            E.g. ``fixed_seq_len=300`` simulates every sequence with seq_len=300
            (or the safe maximum if 300 > context_len - num_tokens_per_bs).
        num_warmup:
            Number of forward passes for GPU warmup before timing starts.
        num_repeat:
            Number of measured forward passes per (batch_size, seq_lens) config.

        Returns
        -------
        List[LatencySimResult]
            One result per batch size, in the same order as *batch_sizes*.
        """
        mr = self.mr
        is_spec = not mr.spec_algorithm.is_none()
        query_len = self._get_num_tokens_per_bs()
        safe_max = self._max_safe_seq_len()

        # If fixed_seq_len is given, clamp it and warn if it was reduced.
        effective_fixed_seq_len: Optional[int] = None
        if fixed_seq_len is not None:
            effective_fixed_seq_len = min(fixed_seq_len, safe_max)
            if effective_fixed_seq_len != fixed_seq_len:
                logger.warning(
                    "[ForwardLatencySimulator] --forward-latency-sim-fixed-seq-len=%d "
                    "was clamped to %d (= context_len %d - num_tokens_per_bs %d) to "
                    "avoid KV-pool out-of-bounds. The attention backend computes "
                    "max_seq_len_k = seq_len + num_tokens_per_bs, so seq_len must be "
                    "<= context_len - num_tokens_per_bs.",
                    fixed_seq_len,
                    effective_fixed_seq_len,
                    mr.model_config.context_len,
                    query_len,
                )

        logger.info(
            "[ForwardLatencySimulator] begin: n_configs=%d is_spec=%s mode=%s "
            "query_len=%d (tokens per sequence in this forward; total_tokens=bs*query_len) "
            "default_seq_len=%d (fixed=%s, safe_max=%d, context_len=%d)",
            len(batch_sizes),
            is_spec,
            "TARGET_VERIFY" if is_spec else "DECODE",
            query_len,
            effective_fixed_seq_len if effective_fixed_seq_len is not None else safe_max,
            str(effective_fixed_seq_len),
            safe_max,
            mr.model_config.context_len,
        )

        results: List[LatencySimResult] = []
        for bs in batch_sizes:
            seq_lens = self._resolve_seq_lens(bs, seq_lens_config, effective_fixed_seq_len)

            # --- Aliasing pre-check: skip this bs if KV pool is too small ---
            skip_reason = self._aliasing_skip_reason(bs, seq_lens)
            if skip_reason is not None:
                logger.warning(
                    "[ForwardLatencySimulator] SKIP bs=%d: %s",
                    bs,
                    skip_reason,
                )
                continue

            # Avoid ambiguous ``...`` in logs: show prefix + explicit ``(+N more)``.
            _max_show = 8
            if len(seq_lens) <= _max_show:
                seq_lens_repr = str(seq_lens)
            else:
                seq_lens_repr = (
                    f"{seq_lens[:_max_show]!r} (+{len(seq_lens) - _max_show} more)"
                )
            logger.info(
                "[ForwardLatencySimulator] bs=%d query_len=%d total_tokens=%d "
                "seq_lens=%s",
                bs,
                query_len,
                bs * query_len,
                seq_lens_repr,
            )
            forward_batch = self._build_forward_batch(bs, seq_lens)
            mean_ms, std_ms, min_ms, max_ms, used_graph = self._measure(
                forward_batch, num_warmup, num_repeat
            )
            result = LatencySimResult(
                batch_size=bs,
                seq_lens=seq_lens,
                query_len=query_len,
                latency_mean_ms=round(mean_ms, 4),
                latency_std_ms=round(std_ms, 4),
                latency_min_ms=round(min_ms, 4),
                latency_max_ms=round(max_ms, 4),
                used_cuda_graph=used_graph,
            )
            results.append(result)
            self._maybe_write_sim_stats(
                seq_lens=seq_lens,
                mean_ms=mean_ms,
                batch_size=bs,
                query_len=query_len,
            )
            logger.info(
                "[ForwardLatencySimulator] bs=%d query_len=%d mean=%.3f ms ± %.3f ms  "
                "[min=%.3f max=%.3f]  cuda_graph=%s",
                bs,
                query_len,
                mean_ms,
                std_ms,
                min_ms,
                max_ms,
                used_graph,
            )

        logger.info("[ForwardLatencySimulator] done.")
        return results

    def _maybe_write_sim_stats(
        self,
        *,
        seq_lens: List[int],
        mean_ms: float,
        batch_size: int,
        query_len: int,
    ) -> None:
        """Append one JSONL line to querylen_<N>_batchsize_<M>.jsonl when env var set.
        Format matches SGLANG_SPEC_TIMING_STATS_DIR; draft_times/draft_extend_times=0.
        """
        output_dir = os.environ.get(SGLANG_FORWARD_LATENCY_SIM_STATS_DIR, "").strip()
        if not output_dir:
            return
        try:
            from sglang.srt.distributed import get_tensor_model_parallel_rank

            if get_tensor_model_parallel_rank() != 0:
                return
        except ImportError:
            pass
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(
            output_dir,
            f"querylen_{query_len}_batchsize_{batch_size}.jsonl",
        )
        avg_lens = sum(seq_lens) / len(seq_lens) if seq_lens else 0.0
        record = {
            "seq_lens": seq_lens,
            "avg_lens": round(avg_lens, 6),
            "draft_times": 0.0,
            "draft_extend_times": 0.0,
            "verify_times": round(mean_ms, 6),
            "batch_size": batch_size,
            "query_len": query_len,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Sequence-length resolution
    # ------------------------------------------------------------------

    def _max_safe_seq_len(self) -> int:
        """Return the largest seq_len that will not overflow the KV page structures.

        In TARGET_VERIFY mode the attention backend computes::

            max_seq_len_k = max(seq_lens) + num_tokens_per_bs

        and then indexes ``strided_indices[:max_seq_pages]`` / ``page_table[:, :max_seq_pages]``
        which were allocated with ``max_num_pages = ceil(context_len / page_size)`` entries.
        If ``seq_len = context_len``, ``max_seq_len_k > context_len`` causes an out-of-bounds
        GPU read → ``cudaErrorIllegalAddress``.

        For DECODE mode ``num_tokens_per_bs == 1`` so the same arithmetic applies:
        ``max_seq_len_k = seq_len + 0`` (no draft delta), but we still subtract 1 to stay safe.
        """
        num_tokens_per_bs = self._get_num_tokens_per_bs()
        context_len = self.mr.model_config.context_len
        # Reserve num_tokens_per_bs positions so seq_len + delta <= context_len.
        return max(1, context_len - num_tokens_per_bs)

    def _resolve_seq_lens(
        self,
        bs: int,
        seq_lens_config: Optional[Dict],
        fixed_seq_len: Optional[int] = None,
    ) -> List[int]:
        """Return a list of ``bs`` seq_lens for the given batch size.

        Priority order:
        1. Explicit per-sequence values from ``seq_lens_config[bs]``.
        2. ``fixed_seq_len`` if provided (already clamped by caller to safe_max).
        3. ``_max_safe_seq_len()`` (context_len − num_tokens_per_bs).

        Both int and str keys are tried for *seq_lens_config* (JSON deserialisation
        produces string keys).  The result is always truncated / padded to exactly
        *bs* entries.
        """
        max_len = self._max_safe_seq_len()
        default_len = fixed_seq_len if fixed_seq_len is not None else max_len
        specified: List[int] = []

        if seq_lens_config:
            # JSON deserialisation produces string keys; try both.
            val = seq_lens_config.get(bs) or seq_lens_config.get(str(bs))
            if val is not None:
                specified = list(val)

        if len(specified) < bs:
            specified.extend([default_len] * (bs - len(specified)))
        return specified[:bs]

    def _aliasing_skip_reason(
        self, bs: int, seq_lens: List[int]
    ) -> Optional[str]:
        """Return a human-readable reason string if running bs would cause KV
        page aliasing (multiple sequences sharing physical pages), or None if
        the batch is safe to run.

        Called before each batch to gate simulation: aliased measurements are
        misleading (intra-kernel L2 hits make latency look unrealistically
        low) so we skip rather than produce bad data.
        """
        mr = self.mr
        page_size = max(1, mr.page_size)
        try:
            kv_total_slots = mr.token_to_kv_pool.size
        except AttributeError:
            kv_total_slots = getattr(mr, "max_total_num_tokens", 0)
        if kv_total_slots <= 0:
            return None  # can't determine pool size; let _prepare handle it

        try:
            pool_rows = mr.req_to_token_pool.req_to_token.shape[0]
        except AttributeError:
            pool_rows = bs
        effective_bs = min(bs, pool_rows)

        num_tokens_per_bs = self._get_num_tokens_per_bs()
        max_seq_len = max(seq_lens) if seq_lens else 1
        max_kv_len = max_seq_len + num_tokens_per_bs
        max_pages = (max_kv_len + page_size - 1) // page_size
        kv_total_pages = max(1, kv_total_slots // page_size)
        pages_needed = effective_bs * max_pages

        if pages_needed > kv_total_pages:
            safe_pages_per_seq = max(1, kv_total_pages // effective_bs)
            safe_seq_len = max(1, safe_pages_per_seq * page_size - num_tokens_per_bs)
            return (
                f"KV pool aliasing would occur: bs={effective_bs} needs "
                f"{pages_needed} pages ({max_pages} pages/seq × {effective_bs} seqs) "
                f"but kv_pool has only {kv_total_pages} pages. "
                f"Multiple sequences would share physical KV pages → "
                f"intra-kernel L2 hits → latency underestimated. "
                f"Fix: use --forward-latency-sim-fixed-seq-len {safe_seq_len} "
                f"(max non-aliased seq_len at bs={effective_bs}), "
                f"or increase --mem-fraction-static."
            )
        return None

    # ------------------------------------------------------------------
    # Forward-mode / spec helpers
    # ------------------------------------------------------------------

    def _get_num_tokens_per_bs(self) -> int:
        mr = self.mr
        if mr.spec_algorithm.is_dflash():
            return mr.server_args.speculative_num_draft_tokens or 1
        if (
            mr.spec_algorithm.is_eagle()
            or mr.spec_algorithm.is_standalone()
            or mr.spec_algorithm.is_ngram()
        ):
            return mr.server_args.speculative_num_draft_tokens or 1
        return 1

    def _get_forward_mode(self) -> ForwardMode:
        if not self.mr.spec_algorithm.is_none():
            return ForwardMode.TARGET_VERIFY
        return ForwardMode.DECODE

    def _build_spec_info(self, num_tokens_per_bs: int):
        """Build a minimal spec_info matching what was used during graph capture."""
        mr = self.mr

        if mr.spec_algorithm.is_eagle() or mr.spec_algorithm.is_standalone():
            from sglang.srt.speculative.eagle_info import EagleVerifyInput

            return EagleVerifyInput(
                draft_token=None,
                custom_mask=None,
                positions=None,
                retrive_index=None,
                retrive_next_token=None,
                retrive_next_sibling=None,
                retrive_cum_len=None,
                spec_steps=mr.server_args.speculative_num_steps,
                topk=mr.server_args.speculative_eagle_topk,
                draft_token_num=mr.server_args.speculative_num_draft_tokens,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                seq_lens_sum=None,
                seq_lens_cpu=None,
            )

        if mr.spec_algorithm.is_dflash():
            from sglang.srt.speculative.dflash_info import DFlashVerifyInput

            return DFlashVerifyInput(
                draft_token=None,
                positions=None,
                draft_token_num=num_tokens_per_bs,
                custom_mask=None,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )

        if mr.spec_algorithm.is_ngram():
            from sglang.srt.speculative.ngram_info import NgramVerifyInput

            info = NgramVerifyInput(
                draft_token=None,
                tree_mask=None,
                positions=None,
                retrive_index=None,
                retrive_next_token=None,
                retrive_next_sibling=None,
                draft_token_num=num_tokens_per_bs,
            )
            info.capture_hidden_mode = CaptureHiddenMode.NULL
            return info

        return None

    # ------------------------------------------------------------------
    # KV-mapping helper
    # ------------------------------------------------------------------

    def _prepare_scattered_kv_mapping(
        self, bs: int, seq_lens_clamped: List[int]
    ) -> torch.Tensor:
        """Assign random, non-overlapping KV pages to each simulated sequence.

        Why sequential slabs are wrong
        --------------------------------
        The old implementation gave sequence i a contiguous slab starting at
        ``i * slab``.  This caused two problems:

        1. **Sequential prefetch**: the GPU's memory prefetcher handles
           contiguous address streams efficiently, giving lower latency than
           real workloads where pages are scattered across HBM.
        2. **Intra-batch aliasing**: when ``bs * slab > kv_pool_size`` the
           modulo wrapped around, so sequence *k* and sequence *k + N* pointed
           to the **same physical pages**.  FlashAttention then read identical
           K/V data for those sequences, which hit L2 on the second access →
           artificially low latency *within* a single kernel call (L2 flush
           between runs does **not** help here).

        This implementation fixes both problems by drawing a random permutation
        of all available KV pages and assigning disjoint subsets to each
        sequence.  The resulting access pattern matches production: requests
        arrive in arbitrary order and their KV pages end up scattered randomly
        throughout the pool.

        When the pool cannot fit all sequences without overlap (large bs ×
        seq_len), a cyclic extension of the permutation is used.  Pages are
        still randomly ordered, so at least the sequential-prefetch bias is
        eliminated, even though some physical pages are shared.

        Note
        ----
        KV *values* at the assigned locations are still zeros (never populated
        by a real forward pass).  Reading zeros from HBM has the same bandwidth
        cost as reading real values, so timing is still representative.
        """
        mr = self.mr
        device = mr.device
        page_size = max(1, mr.page_size)

        try:
            req_pool = mr.req_to_token_pool.req_to_token  # GPU [pool_rows, max_ctx]
        except AttributeError:
            return torch.zeros(bs, dtype=torch.int32, device=device)

        pool_rows, max_col = req_pool.shape
        effective_bs = min(bs, pool_rows)

        try:
            kv_total_slots = mr.token_to_kv_pool.size
        except AttributeError:
            kv_total_slots = getattr(mr, "max_total_num_tokens", 0)
        if kv_total_slots <= 0:
            logger.warning(
                "[ForwardLatencySimulator] Cannot determine KV pool size; "
                "falling back to req_pool_indices=0 (less realistic timing)."
            )
            return torch.zeros(bs, dtype=torch.int32, device=device)

        num_tokens_per_bs = self._get_num_tokens_per_bs()
        max_seq_len = max(seq_lens_clamped) if seq_lens_clamped else 1
        max_kv_len = max_seq_len + num_tokens_per_bs
        max_pages = (max_kv_len + page_size - 1) // page_size

        # Total number of KV pages available in the pool.
        kv_total_pages = max(1, kv_total_slots // page_size)
        pages_needed = effective_bs * max_pages

        # ---- Random page assignment ----------------------------------------
        # Aliasing is already ruled out by _aliasing_skip_reason() in run().
        # Use a random permutation for non-overlapping assignment: each
        # sequence gets a disjoint, randomly-scattered subset of KV pages,
        # matching production access patterns.
        perm = torch.randperm(kv_total_pages, dtype=torch.int64)[:pages_needed]
        page_assignments = perm.view(effective_bs, max_pages)

        # Convert page indices → token-slot values stored in req_to_token.
        # req_to_token[i, j*page_size] = page_index * page_size
        slots = page_assignments * page_size  # [effective_bs, max_pages]

        # Column positions in req_to_token that FlashAttention reads
        # (strided_indices = [0, page_size, 2*page_size, ...]).
        j_idx = torch.arange(max_pages, dtype=torch.int64)
        strided_cols = j_idx * page_size                     # CPU [max_pages]
        valid_mask = strided_cols < max_col
        valid_cols_cpu = strided_cols[valid_mask].tolist()
        valid_slots = slots[:, valid_mask].to(req_pool.dtype).to(device)

        for k, col in enumerate(valid_cols_cpu):
            req_pool[:effective_bs, col] = valid_slots[:, k]

        logger.info(
            "[ForwardLatencySimulator] random KV mapping applied: "
            "bs=%d, max_pages=%d, kv_total_pages=%d (no aliasing)",
            effective_bs,
            max_pages,
            kv_total_pages,
        )
        return torch.arange(effective_bs, dtype=torch.int32, device=device)

    # ------------------------------------------------------------------
    # ForwardBatch construction
    # ------------------------------------------------------------------

    def _build_forward_batch(self, bs: int, seq_lens: List[int]) -> ForwardBatch:
        """Construct a ForwardBatch with realistic per-sequence seq_lens.

        Key properties
        --------------
        * ``seq_lens`` / ``seq_lens_cpu`` are set to the caller-specified values.
          These flow through ``populate_from_forward_batch`` →
          ``init_forward_metadata_replay_cuda_graph`` so FlashAttention attends
          to the correct number of tokens per sequence.
        * ``req_pool_indices = arange(bs)``; ``req_to_token_pool`` rows 0..bs-1
          are pre-populated with non-overlapping scattered page indices (via
          ``_prepare_scattered_kv_mapping``), so FlashAttention reads from
          distinct HBM locations—much closer to real multi-request workloads.
        * Positions are set to ``seq_len[i] + j`` for j = 0 .. num_tokens_per_bs-1
          (i.e., the first new token for each sequence is at position seq_len[i]).
        """
        mr = self.mr
        device = mr.device

        num_tokens_per_bs = self._get_num_tokens_per_bs()
        forward_mode = self._get_forward_mode()
        num_tokens = bs * num_tokens_per_bs

        # Clamp seq_lens to [1, max_safe_seq_len].
        # The upper bound is context_len - num_tokens_per_bs rather than context_len
        # because the attention backend computes:
        #   max_seq_len_k = max(seq_lens) + num_tokens_per_bs
        # and then indexes strided_indices / page_table which are allocated with
        # max_num_pages = ceil(context_len / page_size) entries.
        # If seq_len == context_len, max_seq_len_k > context_len → OOB GPU access.
        safe_max = self._max_safe_seq_len()
        seq_lens_clamped = [min(max(int(s), 1), safe_max) for s in seq_lens]

        # ---- Tensors --------------------------------------------------
        seq_lens_gpu = torch.tensor(seq_lens_clamped, dtype=torch.int32, device=device)
        seq_lens_cpu_t = torch.tensor(seq_lens_clamped, dtype=torch.int32, device="cpu")

        # Random token IDs give diverse MoE expert routing per token,
        # which loads distinct expert weight matrices from HBM—much closer to
        # real workloads than all-zeros (which collapses routing to same experts,
        # leaving their weights L2-cached and hiding weight-bandwidth pressure).
        vocab_size = mr.model_config.vocab_size
        input_ids = torch.randint(
            0, vocab_size, (num_tokens,), dtype=torch.int64, device=device
        )

        # Use distinct req-pool rows (0..bs-1) with scattered KV page mappings.
        # This ensures FlashAttention reads from distinct HBM locations rather
        # than repeatedly hitting the same L2-cached page-0 data.
        req_pool_indices = self._prepare_scattered_kv_mapping(bs, seq_lens_clamped)

        # Write new tokens to kv slot 0 (dummy, discarded after timing).
        out_cache_loc = torch.zeros(num_tokens, dtype=torch.int64, device=device)

        # Positions: sequence i starts from seq_len[i], one position per token.
        positions = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        for i, slen in enumerate(seq_lens_clamped):
            start = i * num_tokens_per_bs
            end = start + num_tokens_per_bs
            positions[start:end] = torch.arange(
                slen, slen + num_tokens_per_bs, dtype=torch.int64, device=device
            )

        next_token_logits_buffer = torch.zeros(
            (num_tokens, mr.model_config.vocab_size), dtype=torch.float32, device=device
        )

        # DP-attention / MLP-sync global token counts
        use_mlp_tp_gather = require_mlp_tp_gather(mr.server_args)
        if use_mlp_tp_gather:
            dp_size = mr.server_args.dp_size
            global_num_tokens_gpu = torch.full(
                (dp_size,), num_tokens, dtype=torch.int32, device=device
            )
            global_num_tokens_for_logprob_gpu = torch.full(
                (dp_size,), num_tokens, dtype=torch.int32, device=device
            )
            global_num_tokens_cpu: Optional[List[int]] = [num_tokens] * dp_size
            global_dp_buffer_len: Optional[int] = num_tokens * dp_size
        else:
            global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32, device=device)
            global_num_tokens_for_logprob_gpu = torch.zeros(
                (1,), dtype=torch.int32, device=device
            )
            global_num_tokens_cpu = None
            global_dp_buffer_len = None

        num_token_non_padded = torch.tensor([num_tokens], dtype=torch.int32, device=device)

        # ---- Spec info ------------------------------------------------
        spec_info = self._build_spec_info(num_tokens_per_bs)
        capture_hidden_mode = (
            spec_info.capture_hidden_mode
            if spec_info is not None
            and getattr(spec_info, "capture_hidden_mode", None) is not None
            else CaptureHiddenMode.NULL
        )
        if mr.server_args.enable_return_hidden_states:
            capture_hidden_mode = CaptureHiddenMode.FULL

        # ---- Assemble -------------------------------------------------
        forward_batch = ForwardBatch(
            forward_mode=forward_mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens_gpu,
            seq_lens_cpu=seq_lens_cpu_t,
            orig_seq_lens=seq_lens_gpu.clone(),
            out_cache_loc=out_cache_loc,
            seq_lens_sum=int(seq_lens_gpu.sum().item()),
            return_logprob=False,
            positions=positions,
            next_token_logits_buffer=next_token_logits_buffer,
            global_num_tokens_gpu=global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
            global_num_tokens_cpu=global_num_tokens_cpu,
            global_dp_buffer_len=global_dp_buffer_len,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            req_to_token_pool=mr.req_to_token_pool,
            token_to_kv_pool=mr.token_to_kv_pool,
            attn_backend=mr.attn_backend,
            spec_algorithm=mr.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=capture_hidden_mode,
            global_forward_mode=forward_mode,
            num_token_non_padded=num_token_non_padded,
        )
        return forward_batch

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def _flush_l2_cache(self) -> None:
        """Flush GPU L2 cache by writing to a buffer larger than the cache.

        Why this is needed
        ------------------
        Without flushing, repeated iterations over the same forward batch
        cause KV-cache pages to accumulate in L2.  On the 2nd+ iteration the
        FlashAttention kernel's memory reads are served from L2 rather than
        HBM, making measured latency unrealistically low—especially when KV
        aliasing is present (multiple sequences share physical pages).

        In production each decode step processes a *different* batch whose KV
        pages are not cached, so every read goes to HBM.  Writing _L2_FLUSH_BYTES
        of data to a scratch buffer before each measurement evicts those pages
        and restores the "cold-cache" condition that matches production.

        Implementation
        --------------
        A single ``fill_(0.0)`` issues a streaming write to every cache line
        in *_l2_flush_buf* (128 MB > any current GPU L2), which the cache
        controller must evict all prior occupants to accommodate.  The
        subsequent ``synchronize()`` in the caller ensures the write completes
        before the timed kernel starts.
        """
        if self._l2_flush_buf is not None:
            self._l2_flush_buf.fill_(0.0)

    def _run_once(self, forward_batch: ForwardBatch) -> bool:
        """Execute one forward pass; returns True when the CUDA graph was used."""
        mr = self.mr

        # Graph path matches ModelRunner._forward_raw: single graph_runner; DFLASH
        # dynamic verify qlen is handled inside CudaGraphRunner.replay/can_run.

        use_graph = (
            forward_batch.forward_mode.is_cuda_graph()
            and mr.graph_runner is not None
            and mr.graph_runner.can_run(forward_batch)
        )

        if use_graph:
            mr.graph_runner.replay(forward_batch)
        else:
            # Eager (non-graph) fallback — still exercises the full attention kernel.
            num_tokens = int(forward_batch.input_ids.numel())
            forward_batch.dp_local_start_pos = None
            forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                forward_batch.global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)
            mr.attn_backend.init_forward_metadata(forward_batch)
            mr.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
            )

        return use_graph

    def _measure(
        self,
        forward_batch: ForwardBatch,
        num_warmup: int,
        num_repeat: int,
    ) -> Tuple[float, float, float, float, bool]:
        """Warmup then measure latency.

        Returns
        -------
        Tuple of (mean_ms, std_ms, min_ms, max_ms, used_cuda_graph).
        """
        device_module = torch.get_device_module(self.mr.device)
        used_graph = False

        with torch.inference_mode():
            # --- Warmup ---
            for _ in range(num_warmup):
                self._flush_l2_cache()
                used_graph = self._run_once(forward_batch)
            device_module.synchronize()

            # --- Measure ---
            # Flush L2 before every timed iteration so each run starts with a
            # "cold cache" that matches production (each batch has distinct KV
            # pages that are not pre-loaded in L2).  Without the flush, KV
            # page aliasing or repeated access to the same physical pages would
            # give unrealistically low latency.
            latencies: List[float] = []
            for _ in range(num_repeat):
                self._flush_l2_cache()
                device_module.synchronize()
                t0 = time.perf_counter()
                self._run_once(forward_batch)
                device_module.synchronize()
                latencies.append((time.perf_counter() - t0) * 1000.0)

        if not latencies:
            return 0.0, 0.0, 0.0, 0.0, used_graph

        mean_ms = sum(latencies) / len(latencies)
        min_ms = min(latencies)
        max_ms = max(latencies)
        if len(latencies) > 1:
            variance = sum((x - mean_ms) ** 2 for x in latencies) / (len(latencies) - 1)
            std_ms = variance**0.5
        else:
            std_ms = 0.0

        return mean_ms, std_ms, min_ms, max_ms, used_graph


# ---------------------------------------------------------------------------
# Config parsing utilities (used by model_runner)
# ---------------------------------------------------------------------------


def parse_sim_batch_sizes(raw: Optional[str]) -> Optional[List[int]]:
    """Parse a comma-separated string of batch sizes, e.g. ``"1,4,8,16"``."""
    if not raw:
        return None
    try:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError as exc:
        logger.warning(
            "[ForwardLatencySimulator] Cannot parse --forward-latency-sim-batch-sizes %r: %s",
            raw,
            exc,
        )
        return None


def parse_sim_seq_lens(raw: Optional[str]) -> Optional[Dict[int, List[int]]]:
    """Parse a JSON string mapping batch_size → seq_lens list.

    Examples::

        '{"4": [1024, 2048, 4096, 512]}'
        '{"8": [2048, 2048, 4096, 4096, 1024, 1024, 512, 512], "4": [1024, 2048, 4096, 512]}'
    """
    if not raw:
        return None
    try:
        raw_dict = json.loads(raw)
        return {int(k): [int(v) for v in vals] for k, vals in raw_dict.items()}
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning(
            "[ForwardLatencySimulator] Cannot parse --forward-latency-sim-seq-lens %r: %s",
            raw,
            exc,
        )
        return None
