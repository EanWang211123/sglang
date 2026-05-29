"""Throughput-aware adaptive speculative decoding controller.

Inherits the full CUDA-graph / runtime-state machinery from
:class:`AdaptiveController` but replaces the decision logic with a
throughput score: ``E[accepted_tokens | S steps] / itl_cost(seqlen, bs, S)``.

Key differences from the EMA-based parent:
  - Acceptance rates are tracked **per position** (not as a scalar EMA of
    accepted-token count) and are **shared across all batch sizes**.
  - The ITL cost is looked up from an offline profiling table.
  - Step changes are governed by a **cooldown** (``update_interval`` batches
    since the last change), not a global periodic counter.
  - A single ``_current_steps`` value is maintained rather than one
    ``current_steps`` per BS slot.

Also contains :class:`DraftProbAdaptiveController`, which uses the draft
model's per-step greedy softmax probabilities (captured via a GPU side-buffer
during the draft CUDA graph) instead of a sliding-window acceptance tracker.
The verify step count is chosen *per batch* to maximise batch-level throughput.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import torch

from sglang.srt.speculative.adaptive_runtime_state import AdaptiveController
from sglang.srt.speculative.adaptive_spec_params import build_per_bs_params
from sglang.srt.speculative.throughput_aware_params import (
    ITLCostTable,
    PositionAcceptanceTracker,
    ThroughputScorer,
)
from sglang.srt.utils import log_info_on_rank0

logger = logging.getLogger(__name__)


def _format_position_rates(rates: list[Optional[float]]) -> str:
    parts = []
    for k, rate in enumerate(rates):
        if rate is None:
            parts.append(f"p{k}=?")
        else:
            parts.append(f"p{k}={rate:.3f}")
    return "[" + ", ".join(parts) + "]"


def _format_score_breakdown(rows: list[dict]) -> str:
    parts = []
    for row in rows:
        steps = int(row["steps"])
        expected = row["expected"]
        itl_cost = row["itl_cost"]
        score = row["score"]
        if score is not None and expected is not None and itl_cost is not None:
            parts.append(
                f"S={steps}:E={expected:.2f}/itl={itl_cost:.2f}→{score:.4f}"
            )
        elif expected is not None:
            parts.append(f"S={steps}:E={expected:.2f}/itl=?")
        else:
            parts.append(f"S={steps}:?")
    return "[" + ", ".join(parts) + "]"


def load_throughput_config(path: Optional[str]) -> dict:
    """Load the throughput-aware config JSON.

    Expected format (integer-string keys are BS lower bounds, same as the
    standard adaptive config; extra non-integer keys are throughput-specific)::

        {
            "itl_cost_path": "/path/to/profiling.jsonl",
            "warmup_per_pos": 5,
            "ema_alpha": 0.2,
            "update_interval": 5,
            "1":   {"steps": [1, 3, 7]},
            "64":  {"steps": [1, 2, 5]},
            "128": {"steps": [1, 2, 4]}
        }

    Returns an empty dict when *path* is ``None``.
    """
    if path is None:
        return {}
    with open(path) as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(
            f"speculative_adaptive_throughput_config must be a JSON object, "
            f"got {type(cfg).__name__}"
        )
    return cfg


class ThroughputAwareAdaptiveController(AdaptiveController):
    """Adaptive controller driven by throughput score rather than EMA hysteresis.

    Inherits all CUDA-graph capture / runtime-state switching machinery from
    :class:`AdaptiveController`.  The following methods are overridden:

    * ``get_steps_for_batch`` — **sole decision point**: fires every
      ``update_interval`` batches (periodic, not cooldown) **and only when all
      active positions have completed warmup**.  Scores every candidate step via
      :class:`ThroughputScorer` and returns the winner.  At all other times it
      returns the cached ``_current_steps``.  A single value is shared across
      all BS ranges.
    * ``on_verify_complete`` — **data collection only**: updates the per-position
      acceptance tracker and advances the batch counter.  No step selection
      happens here.

    Warmup gate: when the step count increases, the new upper positions start
    their warmup phase.  Step changes are blocked until every position in
    ``[0, current_steps)`` has seen at least ``warmup_per_pos`` observations.
    This prevents premature decisions based on noisy early data and avoids the
    oscillation pattern that comes from jumping to a higher step before the
    upper positions have reliable estimates.

    The ``_bs_params`` dict inherited from the parent is still populated (so
    that ``candidate_steps``, ``init_states``, and CUDA-graph pruning all work
    unchanged), but its EMA ``update()`` method is never called.
    """

    def __init__(
        self,
        worker,
        throughput_config_path: Optional[str] = None,
    ):
        # --- Load config -------------------------------------------------
        cfg = load_throughput_config(throughput_config_path)

        # --- Initialize parent attributes directly -----------------------
        # build_per_bs_params ignores non-integer keys (itl_cost_path, etc.),
        # so the throughput config path can be passed directly.
        self.worker = worker
        self._bs_list, self._bs_params = build_per_bs_params(throughput_config_path)
        self._states: dict = {}
        self._cuda_graph_bs = None

        # --- Throughput-specific state -----------------------------------
        max_steps = max(self.candidate_steps)
        # window_size: number of verify batches averaged per position.
        # The warmup gate blocks step changes until every active position
        # has accumulated this many observations (full window).
        window_size: int = int(cfg.get("window_size", 20))
        self._update_interval: int = int(cfg.get("update_interval", 10))

        self._tracker = PositionAcceptanceTracker(
            max_steps=max_steps,
            window_size=window_size,
        )

        itl_cost_path: Optional[str] = cfg.get("itl_cost_path")
        if itl_cost_path:
            self._itl_table: Optional[ITLCostTable] = ITLCostTable(itl_cost_path)
            self._scorer: Optional[ThroughputScorer] = ThroughputScorer(
                self._tracker, self._itl_table
            )
        else:
            self._itl_table = None
            self._scorer = None
            logger.warning(
                "ThroughputAwareAdaptiveController: no 'itl_cost_path' in config; "
                "throughput scoring disabled — will hold the initial step."
            )

        # Current active step (single value shared across all BS slots).
        # Set to the middle candidate of the smallest BS slot as a safe default;
        # will be overwritten by init_states() → activate().
        first_bs = self._bs_list[0]
        first_candidates = self._bs_params[first_bs].candidate_steps
        self._current_steps: int = first_candidates[len(first_candidates) // 2]

        # Monotonically increasing batch counter.  Evaluation fires whenever
        # this is divisible by ``_update_interval`` (periodic, not cooldown).
        self._batch_count: int = 0

        log_info_on_rank0(
            logger,
            f"ThroughputAwareAdaptiveController initialized: "
            f"bs_list={self._bs_list}, candidate_steps={self.candidate_steps}, "
            f"update_interval={self._update_interval}, "
            f"window_size={window_size}, "
            f"itl_cost_path={itl_cost_path!r}",
        )

    # ------------------------------------------------------------------
    # Overridden decision interface
    # ------------------------------------------------------------------

    def activate_step_by_batch(
        self, batch_size: int, current_steps: int, avg_seqlen: float = 0.0
    ) -> None:
        """Activate the optimal throughput-aware step if it differs from *current_steps*.

        Overrides the base class to pass *avg_seqlen* ``(avg + max) / 2`` to
        :meth:`get_steps_for_batch` so that the ITL cost table is looked up
        with a representative sequence length rather than the default 0.
        """
        target = self.get_steps_for_batch(batch_size, avg_seqlen=avg_seqlen)
        if target != current_steps:
            self._activate(target)

    def get_steps_for_batch(self, batch_size: int, avg_seqlen: float = 0.0) -> int:
        """Score candidate steps and return the optimal one.

        This method is the sole decision point.  It is called **before** each
        draft so the caller can activate the correct runtime state.

        Two conditions must both be met before an evaluation fires:

        1. **Warmup gate**: every position in ``[0, _current_steps)`` has
           completed warmup (``warmup_count >= warmup_per_pos``).  This blocks
           changes while new positions are still accumulating initial data,
           preventing oscillation caused by premature decisions on noisy
           warmup estimates.

        2. **Periodic trigger**: ``_batch_count`` is divisible by
           ``update_interval``.  Evaluation fires at a fixed cadence regardless
           of whether a step change occurred, consistent with the original
           EMA-based strategy.

        If either condition is not met, the cached ``_current_steps`` is
        returned unchanged.

        Args:
            batch_size: Number of requests in the current batch.
            avg_seqlen: Pre-computed index seqlen ``(avg + max) / 2`` used to
                look up the ITL cost for each candidate step.
        """
        if self._scorer is None:
            return self._current_steps

        # Gate 1: warmup lock — all active positions must have real data.
        if not self._tracker.all_positions_warmed(self._current_steps):
            return self._current_steps

        # Gate 2: periodic trigger — only evaluate every update_interval batches.
        if self._batch_count % self._update_interval != 0:
            return self._current_steps

        # Both gates cleared — score every candidate and pick the best.
        padded_bs = self._pad_to_cuda_graph_bs(batch_size)
        bs_key = self._find_closest_bs(padded_bs)
        candidate_steps = self._bs_params[bs_key].candidate_steps

        max_pos = max(candidate_steps) if candidate_steps else 0
        pos_rates = self._tracker.snapshot_position_rates(max_pos)
        score_rows = self._scorer.score_breakdown(
            candidate_steps, batch_size, avg_seqlen
        )
        new_steps = self._scorer.pick_best_from_breakdown(candidate_steps, score_rows)

        if new_steps != self._current_steps:
            old_steps = self._current_steps

            # When stepping down, clear positions no longer active so they
            # don't bias scores the next time those positions are explored.
            if new_steps < old_steps:
                self._tracker.clear_positions_above(new_steps)

            self._current_steps = new_steps
            logger.info(
                f"ThroughputAwareAdaptiveController: BS slot {bs_key} "
                f"(actual bs={batch_size}, seqlen={avg_seqlen:.0f}) "
                f"steps {old_steps} -> {new_steps}; "
                f"pos_rates={_format_position_rates(pos_rates)}; "
                f"scores={_format_score_breakdown(score_rows)}"
            )

        return self._current_steps

    def on_verify_complete(
        self,
        accept_lengths: list[int],
        batch_size: int = 0,
    ) -> None:
        """Update the per-position acceptance tracker and advance the batch counter.

        Step selection is handled entirely in ``get_steps_for_batch``; this
        method only records acceptance observations and advances ``_batch_count``
        so that the periodic trigger in ``get_steps_for_batch`` fires at the
        correct cadence.

        Args:
            accept_lengths: ``num_correct_drafts_per_req`` list for the batch.
            batch_size: Number of requests in the batch.
        """
        if batch_size <= 0:
            return

        self._tracker.update(accept_lengths, self._current_steps)
        self._batch_count += 1


class DraftProbAdaptiveController(AdaptiveController):
    """Adaptive verify-step controller driven by per-step draft softmax probs.

    Unlike :class:`ThroughputAwareAdaptiveController` (which maintains a
    sliding-window acceptance tracker fed from historical verify results),
    this controller reads the draft model's **greedy softmax probability at
    each draft position** captured during the current batch's draft forward
    pass, and uses those probabilities to estimate the expected acceptance
    length for every candidate verify step count *k*.

    Algorithm (per batch):
      1. Always draft ``fixed_draft_steps`` tokens.
      2. During the draft CUDA graph, per-step greedy probs are written
         in-place to ``_per_step_draft_probs`` [bs, fixed_draft_steps].
      3. After draft, for each candidate k compute::

             total_expected(k) = sum_seq(1 + p[0] + p[0]*p[1] + ... + prod(p[:k]))
             score(k) = total_expected(k) / itl_cost(seqlen, bs, k)

      4. Pick k* = argmax score(k) and switch the verify CUDA graph to k*.
      5. After verify + draft-extend, restore STS to ``fixed_draft_steps``.

    Config JSON keys (same file as throughput-aware config):
      - ``"enable_draft_prob_adaptive": true``   — activates this controller.
      - ``"fixed_draft_steps": 5``               — always draft this many steps.
      - ``"itl_cost_path": "..."``               — offline profiling JSONL.
      - ``"1": {"steps": [1,2,3,4,5]}, ...``    — candidate verify steps per BS.

    Constraints:
      - Requires ``speculative_eagle_topk = 1`` (greedy / linear chain).
      - Requires spec-v1 (non-overlap) scheduler so that the verify graph can
        be changed after the draft and before the verify call.
    """

    def __init__(
        self,
        worker,
        throughput_config_path: Optional[str] = None,
    ):
        cfg = load_throughput_config(throughput_config_path)

        # Inherit candidate_steps, _states, _cuda_graph_bs machinery from parent.
        self.worker = worker
        self._bs_list, self._bs_params = build_per_bs_params(throughput_config_path)
        self._states: dict = {}
        self._cuda_graph_bs = None

        # Fixed number of draft steps (draft CUDA graph is always for this many steps).
        self._fixed_draft_steps: int = int(
            cfg.get("fixed_draft_steps", max(self.candidate_steps))
        )

        # ITL cost table for throughput scoring.
        itl_cost_path: Optional[str] = cfg.get("itl_cost_path")
        if itl_cost_path:
            self._itl_table: Optional[ITLCostTable] = ITLCostTable(itl_cost_path)
        else:
            self._itl_table = None
            logger.warning(
                "DraftProbAdaptiveController: no 'itl_cost_path' in config; "
                "verify steps will always equal fixed_draft_steps."
            )

        # GPU side-buffer for per-step greedy probs; shape [max_bs, fixed_draft_steps].
        # Allocated lazily in init_side_buffer() after the draft CUDA graph is captured
        # (so max_bs is known).  draft_forward() writes to this buffer in-place;
        # the CUDA graph captures those writes and replays them on every forward pass.
        self._per_step_draft_probs: Optional[torch.Tensor] = None

        log_info_on_rank0(
            logger,
            f"DraftProbAdaptiveController initialized: "
            f"fixed_draft_steps={self._fixed_draft_steps}, "
            f"candidate_steps={self.candidate_steps}, "
            f"itl_cost_path={itl_cost_path!r}",
        )

    # ------------------------------------------------------------------
    # Side-buffer lifecycle
    # ------------------------------------------------------------------

    def init_side_buffer(self, max_bs: int, device: str) -> None:
        """Allocate the GPU tensor that draft_forward() writes per-step probs to.

        Must be called **after** the draft CUDA graph is captured (so max_bs
        is known) and **before** any replay.  The CUDA graph records the
        in-place copy operations into this buffer; reading it after replay
        gives the correct per-step greedy probabilities for the current batch.
        """
        self._per_step_draft_probs = torch.zeros(
            (max_bs, self._fixed_draft_steps),
            dtype=torch.float32,
            device=device,
        )
        log_info_on_rank0(
            logger,
            f"DraftProbAdaptiveController: side buffer allocated "
            f"[{max_bs}, {self._fixed_draft_steps}] on {device}",
        )

    # ------------------------------------------------------------------
    # Pre-draft interface (always return fixed_draft_steps)
    # ------------------------------------------------------------------

    def get_steps_for_batch(self, batch_size: int, avg_seqlen: float = 0.0) -> int:
        """Pre-draft: always return fixed_draft_steps so the draft always runs fully."""
        return self._fixed_draft_steps

    def activate_step_by_batch(
        self, batch_size: int, current_steps: int, avg_seqlen: float = 0.0
    ) -> None:
        """Ensure STS = fixed_draft_steps before each draft round."""
        if current_steps != self._fixed_draft_steps:
            self._activate(self._fixed_draft_steps)

    # ------------------------------------------------------------------
    # Post-draft interface (compute optimal verify steps from probs)
    # ------------------------------------------------------------------

    def get_optimal_verify_steps(self, batch_size: int, avg_seqlen: float) -> int:
        """After draft: read per-step probs and return the optimal verify step count.

        For each candidate k, computes::

            total_expected(k) = sum_{seq} (1 + p[0] + p[0]*p[1] + ... + prod(p[:k]))
            score(k)          = total_expected(k) / itl_cost(seqlen, bs, k)

        Returns the k with the highest score.  Falls back to ``fixed_draft_steps``
        if the side buffer is not yet ready or the ITL table is missing.

        Args:
            batch_size: Actual (unpadded) number of requests in the batch.
            avg_seqlen: (avg + max) / 2 of current sequence lengths, used to
                look up ITL cost.
        """
        if self._per_step_draft_probs is None or self._itl_table is None:
            return self._fixed_draft_steps

        probs = self._per_step_draft_probs[:batch_size]  # [bs, fixed_draft_steps]

        candidates = [
            k for k in self.candidate_steps if 1 <= k <= self._fixed_draft_steps
        ]
        if not candidates:
            return self._fixed_draft_steps

        best_k = candidates[0]
        best_score = -float("inf")

        for k in candidates:
            # Cumulative products along draft steps: [bs, k]
            # cum_prods[s, j] = prod(probs[s, 0:j+1])
            k_probs = probs[:, :k]
            cum_prods = torch.cumprod(k_probs, dim=1)
            # total_expected = sum_seq(1) + sum_seq_j(cum_prods[s, j])
            total_expected = float(batch_size) + float(cum_prods.sum().item())

            itl_cost = self._itl_table.lookup(avg_seqlen, batch_size, k)
            if itl_cost is None or itl_cost <= 0:
                continue

            score = total_expected / itl_cost
            if score > best_score:
                best_score = score
                best_k = k

        logger.debug(
            "DraftProbAdaptiveController: bs=%d seqlen=%.0f optimal_k=%d score=%.4f",
            batch_size,
            avg_seqlen,
            best_k,
            best_score,
        )
        return best_k

    # ------------------------------------------------------------------
    # No-op: we don't maintain a sliding-window tracker
    # ------------------------------------------------------------------

    def on_verify_complete(
        self,
        accept_lengths: list[int],
        batch_size: int = 0,
    ) -> None:
        """No-op: this controller uses draft probs, not historical accept rates."""
        pass
