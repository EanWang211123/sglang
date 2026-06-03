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

import bisect
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


def _format_draft_probs(probs: torch.Tensor, max_seqs: int = 16) -> str:
    """Format [bs, steps] greedy draft prob matrix for logging."""
    bs = probs.shape[0]
    n = min(bs, max_seqs)
    parts = []
    for s in range(n):
        pos_parts = [
            f"p{k}={probs[s, k].item():.4f}" for k in range(probs.shape[1])
        ]
        parts.append(f"seq{s}=[{', '.join(pos_parts)}]")
    if bs > n:
        parts.append(f"...+{bs - n}seq")
    return "; ".join(parts)


def _format_draft_prob_scores(rows: list[dict], best_k: int) -> str:
    parts = []
    for row in rows:
        steps = int(row["steps"])
        expected = row["expected"]
        itl_cost = row["itl_cost"]
        score = row["score"]
        marker = "*" if steps == best_k else ""
        if score is not None and expected is not None and itl_cost is not None:
            parts.append(
                f"S={steps}:E={expected:.2f}/itl={itl_cost:.2f}→{score:.4f}{marker}"
            )
        elif expected is not None:
            parts.append(f"S={steps}:E={expected:.2f}/itl=?{marker}")
        else:
            parts.append(f"S={steps}:?{marker}")
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
    # BS-routing helpers (mirror AdaptiveSpeculativeParams methods since
    # this class bypasses the parent's self.params)
    # ------------------------------------------------------------------

    @property
    def candidate_steps(self) -> list[int]:
        return sorted({s for slot in self._bs_params.values() for s in slot.candidate_steps})

    def _pad_to_cuda_graph_bs(self, batch_size: int) -> int:
        if self._cuda_graph_bs is None:
            return batch_size
        idx = bisect.bisect_left(self._cuda_graph_bs, batch_size)
        return (
            self._cuda_graph_bs[idx] if idx < len(self._cuda_graph_bs) else batch_size
        )

    def _find_closest_bs(self, target: int) -> int:
        idx = bisect.bisect_right(self._bs_list, target) - 1
        return self._bs_list[max(0, idx)]

    def _cuda_graph_bs_for_step(self, step: int) -> list[int] | None:
        if self._cuda_graph_bs is None:
            return None
        return [
            v for v in self._cuda_graph_bs
            if step in self._bs_params[self._find_closest_bs(self._pad_to_cuda_graph_bs(v))].candidate_steps
        ]

    def init_states(self, cuda_graph_bs: list[int] | None = None) -> None:
        """Build and register runtime states for all candidate steps."""
        self._cuda_graph_bs = sorted(cuda_graph_bs) if cuda_graph_bs else None
        init_max_bs = max(cuda_graph_bs) if cuda_graph_bs is not None else None

        for steps in self.candidate_steps:
            if steps in self._states:
                continue
            pruned_bs = self._cuda_graph_bs_for_step(steps)
            state = self.worker.build_adaptive_runtime_state(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=steps + 1,
                cuda_graph_bs=pruned_bs,
                init_max_bs=init_max_bs,
            )
            self._states[steps] = state

        self._activate(self._current_steps)

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
      - ``"log_draft_prob_decision": true``      — log per-position draft probs
        and per-candidate throughput scores after each draft (default true).
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
        self._log_draft_prob_decision: bool = bool(
            cfg.get("log_draft_prob_decision", True)
        )

        log_info_on_rank0(
            logger,
            f"DraftProbAdaptiveController initialized: "
            f"fixed_draft_steps={self._fixed_draft_steps}, "
            f"candidate_steps={self.candidate_steps}, "
            f"itl_cost_path={itl_cost_path!r}, "
            f"log_draft_prob_decision={self._log_draft_prob_decision}",
        )

    # ------------------------------------------------------------------
    # BS-routing helpers (mirror AdaptiveSpeculativeParams methods since
    # this class bypasses the parent's self.params)
    # ------------------------------------------------------------------

    @property
    def candidate_steps(self) -> list[int]:
        return sorted({s for slot in self._bs_params.values() for s in slot.candidate_steps})

    def candidates_for_bs(self, batch_size: int) -> list[int]:
        """Per-BS-tier candidate verify steps, clamped to ``fixed_draft_steps``.

        Routes by the actual ``batch_size`` (same as the EMA / throughput-aware
        controllers) so that per-BS restrictions like ``"1": [15]`` are honored
        instead of falling back to the global union ``self.candidate_steps``.
        """
        bs_key = self._find_closest_bs(batch_size)
        return [
            k
            for k in self._bs_params[bs_key].candidate_steps
            if 1 <= k <= self._fixed_draft_steps
        ]

    def _pad_to_cuda_graph_bs(self, batch_size: int) -> int:
        if self._cuda_graph_bs is None:
            return batch_size
        idx = bisect.bisect_left(self._cuda_graph_bs, batch_size)
        return (
            self._cuda_graph_bs[idx] if idx < len(self._cuda_graph_bs) else batch_size
        )

    def _find_closest_bs(self, target: int) -> int:
        idx = bisect.bisect_right(self._bs_list, target) - 1
        return self._bs_list[max(0, idx)]

    def _cuda_graph_bs_for_step(self, step: int) -> list[int] | None:
        if self._cuda_graph_bs is None:
            return None
        return [
            v for v in self._cuda_graph_bs
            if step in self._bs_params[self._find_closest_bs(self._pad_to_cuda_graph_bs(v))].candidate_steps
        ]

    def init_states(self, cuda_graph_bs: list[int] | None = None) -> None:
        """Build and register runtime states for all candidate steps."""
        self._cuda_graph_bs = sorted(cuda_graph_bs) if cuda_graph_bs else None
        init_max_bs = max(cuda_graph_bs) if cuda_graph_bs is not None else None

        for steps in self.candidate_steps:
            if steps in self._states:
                continue
            pruned_bs = self._cuda_graph_bs_for_step(steps)
            state = self.worker.build_adaptive_runtime_state(
                speculative_num_steps=steps,
                speculative_num_draft_tokens=steps + 1,
                cuda_graph_bs=pruned_bs,
                init_max_bs=init_max_bs,
            )
            self._states[steps] = state

        self._activate(self._fixed_draft_steps)

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

    def draft_prob_score_breakdown(
        self,
        batch_size: int,
        avg_seqlen: float,
        *,
        include_seq_detail: bool = False,
        probs_override: Optional[torch.Tensor] = None,
    ) -> tuple[list[dict], Optional[torch.Tensor], int]:
        """Score every candidate verify step from the current draft prob buffer.

        Returns ``(score_rows, probs[:batch_size], best_k)``.  Each row has
        keys ``steps``, ``expected``, ``itl_cost``, ``score``; the per-seq
        ``seq_expected`` field is only populated when ``include_seq_detail`` is
        True (it is only needed for logging and forcing it costs a device→host
        copy per candidate every decode step).

        When ``probs_override`` is given (DFlash path), it is read directly and
        the internal ``_per_step_draft_probs`` side buffer is bypassed — this
        avoids a per-step device-to-device copy and the side-buffer allocation.

        All candidate ``k`` share a single ``cumprod`` over the widest candidate
        and a single device→host sync, instead of one ``.item()`` /
        ``.cpu().tolist()`` per candidate.
        """
        if self._itl_table is None:
            return [], None, self._fixed_draft_steps

        if probs_override is not None:
            probs = probs_override  # [bs, >=max_k]
        elif self._per_step_draft_probs is not None:
            probs = self._per_step_draft_probs[:batch_size]  # [bs, fixed_draft_steps]
        else:
            return [], None, self._fixed_draft_steps

        candidates = self.candidates_for_bs(batch_size)
        if not candidates:
            return [], probs, self._fixed_draft_steps

        # Shared cumprod over the widest candidate; prefix-sum over positions then
        # over batch gives total_expected for every k from one sync.
        max_k = max(candidates)
        cum_prods = torch.cumprod(probs[:, :max_k], dim=1)  # [bs, max_k]
        # col_sums[j] = sum_seq prod(p[0:j+1]); cumsum over j → prefix sums per k.
        prefix_col_sums = torch.cumsum(cum_prods.sum(dim=0), dim=0)  # [max_k]
        prefix_cpu = prefix_col_sums.cpu().tolist()  # single device→host sync

        # Per-seq detail (1 + cumsum over positions) only when logging.
        seq_detail: Optional[list[list[float]]] = None
        if include_seq_detail:
            seq_detail = (
                (1.0 + torch.cumsum(cum_prods, dim=1)).detach().cpu().tolist()
            )

        rows: list[dict] = []
        best_k = candidates[0]
        best_score = -float("inf")

        for k in candidates:
            # total_expected(k) = sum_seq (1 + sum_{j<k} prod(p[0:j+1]))
            #                   = batch_size + prefix_col_sums[k-1]
            total_expected = float(batch_size) + prefix_cpu[k - 1]

            itl_cost = self._itl_table.lookup(avg_seqlen, batch_size, k)
            score: Optional[float] = None
            if itl_cost is not None and itl_cost > 0:
                score = total_expected / itl_cost

            row = {
                "steps": float(k),
                "expected": total_expected,
                "itl_cost": itl_cost,
                "score": score,
            }
            if seq_detail is not None:
                row["seq_expected"] = [seq_detail[s][k - 1] for s in range(batch_size)]
            rows.append(row)

            if score is not None and score > best_score:
                best_score = score
                best_k = k

        return rows, probs, best_k

    def get_optimal_verify_steps(
        self,
        batch_size: int,
        avg_seqlen: float,
        current_steps: Optional[int] = None,
        probs_override: Optional[torch.Tensor] = None,
    ) -> int:
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
            current_steps: Active STS before the verify switch (for logging only).
            probs_override: When given, read draft probs directly from this tensor
                instead of the internal side buffer (DFlash eager-prob path).
        """
        rows, probs, best_k = self.draft_prob_score_breakdown(
            batch_size,
            avg_seqlen,
            include_seq_detail=self._log_draft_prob_decision,
            probs_override=probs_override,
        )
        if probs is None:
            return self._fixed_draft_steps

        if self._log_draft_prob_decision:
            prev = (
                current_steps
                if current_steps is not None
                else self._fixed_draft_steps
            )
            switch_note = (
                f"verify {prev}->{best_k}"
                if best_k != prev
                else f"verify {best_k} (no switch)"
            )
            seq_expected_lines = []
            for row in rows:
                k = int(row["steps"])
                seq_parts = [
                    f"seq{s}={exp:.3f}"
                    for s, exp in enumerate(row["seq_expected"])
                ]
                seq_expected_lines.append(f"k={k}:[{', '.join(seq_parts)}]")
            log_info_on_rank0(
                logger,
                f"DraftProbAdaptiveController: bs={batch_size} seqlen={avg_seqlen:.0f} "
                f"{switch_note}; "
                f"draft_probs={_format_draft_probs(probs)}; "
                f"seq_expected={{{'; '.join(seq_expected_lines)}}}; "
                f"scores={_format_draft_prob_scores(rows, best_k)}",
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
