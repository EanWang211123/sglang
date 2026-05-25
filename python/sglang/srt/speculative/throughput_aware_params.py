"""Throughput-aware speculative decoding: acceptance tracking and cost lookup.

Three components:
  - PositionAcceptanceTracker  -- per-position sliding-window accept-rate tracker
                                   shared across all batch sizes.
  - ITLCostTable               -- offline profiling data (seqlen, bs, spec_size)
                                   → itl_cost multiplier.
  - ThroughputScorer           -- combines tracker + table to score each step.
"""

from __future__ import annotations

import bisect
import json
import logging
import math
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class PositionAcceptanceTracker:
    """Per-position acceptance-rate tracker using a fixed-size sliding window.

    Position k (0-indexed) is accepted when ``num_correct_drafts > k``.

    Each position maintains a circular buffer of the ``window_size`` most
    recent batch-level acceptance rates.  The estimated rate for position k
    is the arithmetic mean of its buffer.

    Lifecycle of a position's buffer:
      - **Filling** (buffer length < ``window_size``): the partial-window mean
        is used as the best available estimate.  The controller's warmup gate
        (``all_positions_warmed``) blocks step changes until every active
        position has a full window, preventing premature decisions on noisy
        early data.
      - **Steady state** (buffer full): the oldest entry is evicted on each
        new observation, giving a true ``window_size``-batch sliding average.

    When the active step count decreases, positions above the new step are
    cleared (buffer fully emptied) so they start fresh if the step is later
    raised again.

    For positions with no data at all (empty buffer after a clear, or a
    candidate step larger than the current), the tracker extrapolates from
    the geometric mean of the per-step transition probabilities computed
    from the positions that do have data.
    """

    def __init__(
        self,
        max_steps: int,
        window_size: int = 20,
    ):
        self.max_steps = max_steps
        self.window_size = window_size

        # Per-position circular buffers. ``deque(maxlen=N)`` automatically
        # evicts the oldest entry when a new value is appended beyond capacity.
        self._windows: list[deque] = [
            deque(maxlen=window_size) for _ in range(max_steps)
        ]

    def update(self, accept_lengths: list[int], current_steps: int) -> None:
        """Update per-position rates from one verify batch.

        Args:
            accept_lengths: ``num_correct_drafts_per_req`` (excludes the
                always-accepted extend token; value k means the first k draft
                positions were accepted).
            current_steps: number of draft steps actually run this batch.
                Only positions 0..current_steps-1 are updated.
        """
        n = len(accept_lengths)
        if n == 0:
            return

        for k in range(min(current_steps, self.max_steps)):
            # p[k] = fraction of requests that accepted position k
            # (i.e. had at least k+1 correct drafts)
            pos_rate = sum(1 for a in accept_lengths if a > k) / n
            self._windows[k].append(pos_rate)

    def clear_positions_above(self, steps: int) -> None:
        """Clear buffers for positions >= *steps* (called on step-count decrease)."""
        for k in range(steps, self.max_steps):
            self._windows[k].clear()

    def all_positions_warmed(self, target_steps: int) -> bool:
        """Return True if every position in ``[0, target_steps)`` has a full window.

        Used by the controller as a gate: step changes are blocked until all
        active positions have accumulated ``window_size`` observations.  This
        prevents premature decisions based on small samples, both at cold-start
        and after a step-up introduces new positions.
        """
        for k in range(min(target_steps, self.max_steps)):
            if len(self._windows[k]) < self.window_size:
                return False
        return True

    def get_expected_tokens(self, target_steps: int) -> Optional[float]:
        """Return ``E[accepted tokens | target_steps drafted]``, including the
        always-present extend token (+1).

        Uses extrapolation for positions whose buffer is empty.
        Returns ``None`` if there is no data to extrapolate from (cold start).
        """
        if target_steps <= 0:
            return None

        rates: list[float] = []
        for k in range(target_steps):
            rate = self._get_rate_or_extrapolate(k, rates)
            if rate is None:
                return None
            rates.append(rate)

        return 1.0 + sum(rates)

    def snapshot_position_rates(self, num_positions: int) -> list[Optional[float]]:
        """Return per-position rates used for scoring (window mean or extrapolated)."""
        if num_positions <= 0:
            return []
        known: list[float] = []
        out: list[Optional[float]] = []
        for k in range(num_positions):
            rate = self._get_rate_or_extrapolate(k, known)
            out.append(rate)
            if rate is not None:
                known.append(rate)
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _best_rate(self, k: int) -> Optional[float]:
        """Return the sliding-window mean for position k, or ``None`` if empty."""
        w = self._windows[k]
        if not w:
            return None
        return sum(w) / len(w)

    def _get_rate_or_extrapolate(
        self, k: int, known_rates: list[float]
    ) -> Optional[float]:
        """Return acceptance rate for position k.

        Priority:
          1. Window mean (partial or full).
          2. Geometric extrapolation from ``known_rates``.
          3. ``None`` if there is nothing to extrapolate from.
        """
        real = self._best_rate(k)
        if real is not None:
            return real

        # Extrapolate from the positions that do have data.
        if len(known_rates) == 0:
            return None
        if len(known_rates) == 1:
            # Only one data point; carry it forward (p[k] ≈ p[0]).
            return known_rates[0]

        # Geometric mean of per-step transition probabilities:
        # p[i] = p[0] * alpha^i  ⟹  alpha = (p[N-1] / p[0])^(1/(N-1))
        p0 = known_rates[0]
        pn = known_rates[-1]
        if p0 <= 0:
            return None
        alpha = (pn / p0) ** (1.0 / (len(known_rates) - 1))
        return known_rates[-1] * alpha


# ---------------------------------------------------------------------------
# ITL cost table
# ---------------------------------------------------------------------------


class ITLCostTable:
    """Load offline profiling data and answer (seqlen, bs, spec_size) queries.

    JSONL format (one JSON object per line)::

        {"seqlen": 64, "batch_size": 1, "spec_size": 3,
         "itl_baseline_ms": 6.06, "itl_spec_ms": 15.89, "itl_cost": 2.62}

    Lookup strategy:
      - ``seqlen``:     nearest entry.
      - ``batch_size``: first entry **≥** the requested size (ceiling), or the
                        largest entry when the requested size exceeds all.
      - ``spec_size``:  exact match required; ``None`` is returned on miss.
    """

    def __init__(self, jsonl_path: str):
        self._data: dict[tuple[int, int, int], float] = {}
        self._seqlens: list[int] = []
        self._batch_sizes: list[int] = []
        self._load(jsonl_path)

    def _load(self, path: str) -> None:
        with open(path) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                entry = json.loads(raw)
                key = (
                    int(entry["seqlen"]),
                    int(entry["batch_size"]),
                    int(entry["spec_size"]),
                )
                self._data[key] = float(entry["itl_cost"])

        self._seqlens = sorted({k[0] for k in self._data})
        self._batch_sizes = sorted({k[1] for k in self._data})
        logger.info(
            f"ITLCostTable loaded {len(self._data)} entries from {path}; "
            f"seqlens={self._seqlens[:5]}{'...' if len(self._seqlens) > 5 else ''}, "
            f"batch_sizes={self._batch_sizes}"
        )

    def lookup(
        self, seqlen: float, batch_size: int, spec_size: int
    ) -> Optional[float]:
        """Return itl_cost or ``None`` if the spec_size is not in the table."""
        if not self._data:
            return None

        # Nearest seqlen
        nearest_seqlen = min(self._seqlens, key=lambda s: abs(s - seqlen))

        # Ceiling batch_size (first >= requested)
        idx = bisect.bisect_left(self._batch_sizes, batch_size)
        nearest_bs = self._batch_sizes[min(idx, len(self._batch_sizes) - 1)]

        return self._data.get((nearest_seqlen, nearest_bs, spec_size))


# ---------------------------------------------------------------------------
# Throughput scorer
# ---------------------------------------------------------------------------


class ThroughputScorer:
    """Compute throughput scores and return the best step.

    Score formula::

        score(S) = E[accepted_tokens | S steps] / itl_cost(seqlen, bs, S)

    where ``E[accepted_tokens | S steps]`` comes from
    :class:`PositionAcceptanceTracker` and ``itl_cost`` comes from
    :class:`ITLCostTable`.
    """

    def __init__(self, tracker: PositionAcceptanceTracker, itl_table: ITLCostTable):
        self.tracker = tracker
        self.itl_table = itl_table

    def score_breakdown(
        self,
        candidate_steps: list[int],
        batch_size: int,
        index_seqlen: float,
    ) -> list[dict[str, Optional[float]]]:
        """Per-candidate expected tokens, ITL cost, and throughput score."""
        rows: list[dict[str, Optional[float]]] = []
        for steps in candidate_steps:
            expected = self.tracker.get_expected_tokens(steps)
            itl_cost = self.itl_table.lookup(index_seqlen, batch_size, steps)
            score: Optional[float] = None
            if (
                expected is not None
                and itl_cost is not None
                and itl_cost > 0
            ):
                score = expected / itl_cost
            rows.append(
                {
                    "steps": float(steps),
                    "expected": expected,
                    "itl_cost": itl_cost,
                    "score": score,
                }
            )
        return rows

    def pick_best_from_breakdown(
        self,
        candidate_steps: list[int],
        rows: list[dict[str, Optional[float]]],
    ) -> int:
        """Return the highest-scoring step from :meth:`score_breakdown` output."""
        best_step = candidate_steps[0]
        best_score = -math.inf
        for row in rows:
            score = row["score"]
            if score is not None and score > best_score:
                best_score = score
                best_step = int(row["steps"])
        return best_step

    def best_step(
        self,
        candidate_steps: list[int],
        batch_size: int,
        index_seqlen: float,
    ) -> int:
        """Return the candidate step with the highest throughput score.

        Falls back to the first candidate if no score can be computed
        (e.g. cold start or missing ITL data for all candidates).
        """
        return self.pick_best_from_breakdown(
            candidate_steps,
            self.score_breakdown(candidate_steps, batch_size, index_seqlen),
        )
