"""Throughput-aware speculative decoding: acceptance tracking and cost lookup.

Three components:
  - PositionAcceptanceTracker  -- per-position EMA/warmup accept-rate tracker
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
from typing import Optional

logger = logging.getLogger(__name__)


class PositionAcceptanceTracker:
    """Per-position acceptance-rate tracker, shared across all batch sizes.

    Position k (0-indexed) is accepted when ``num_correct_drafts > k``.

    Two-phase update per position:
      1. **Warmup** (first ``warmup_per_pos`` observations): accumulate a
         running mean.  No EMA is started until the warmup completes; this
         avoids the cold-start bias ``0 + alpha * p`` that EMA would produce.
      2. **EMA** (after warmup): standard exponential moving average.

    When the active step count decreases, positions above the new step are
    cleared so they don't pollute the tracker the next time those positions
    are explored.

    For positions that have no data yet (e.g. when increasing steps), the
    tracker extrapolates using the geometric mean of the per-step transition
    probabilities computed from the existing positions.
    """

    def __init__(
        self,
        max_steps: int,
        ema_alpha: float = 0.2,
        warmup_per_pos: int = 5,
    ):
        self.max_steps = max_steps
        self.ema_alpha = ema_alpha
        self.warmup_per_pos = warmup_per_pos

        # Per-position state: None means no data at all.
        self._ema_rate: list[Optional[float]] = [None] * max_steps
        self._warmup_count: list[int] = [0] * max_steps
        self._warmup_sum: list[float] = [0.0] * max_steps

    def update(self, accept_lengths: list[int], current_steps: int) -> None:
        """Update per-position rates from one verify batch.

        Args:
            accept_lengths: ``num_correct_drafts_per_req`` (excludes extend
                token; value k means the first k draft positions were accepted).
            current_steps: number of draft steps that were actually run this
                batch.  Only positions 0..current_steps-1 are updated.
        """
        n = len(accept_lengths)
        if n == 0:
            return

        for k in range(min(current_steps, self.max_steps)):
            # p[k] = fraction of requests that accepted position k
            # (i.e. had at least k+1 correct drafts)
            pos_rate = sum(1 for a in accept_lengths if a > k) / n

            if self._warmup_count[k] < self.warmup_per_pos:
                self._warmup_count[k] += 1
                self._warmup_sum[k] += pos_rate
                if self._warmup_count[k] >= self.warmup_per_pos:
                    # Graduate: set EMA to the warmup mean.
                    self._ema_rate[k] = self._warmup_sum[k] / self._warmup_count[k]
            else:
                assert self._ema_rate[k] is not None
                self._ema_rate[k] = (
                    (1.0 - self.ema_alpha) * self._ema_rate[k]
                    + self.ema_alpha * pos_rate
                )

    def clear_positions_above(self, steps: int) -> None:
        """Clear data for positions >= *steps* (called on step-count decrease)."""
        for k in range(steps, self.max_steps):
            self._ema_rate[k] = None
            self._warmup_count[k] = 0
            self._warmup_sum[k] = 0.0

    def get_expected_tokens(self, target_steps: int) -> Optional[float]:
        """Return ``E[accepted tokens | target_steps drafted]``, including the
        always-present extend token (+1).

        Uses extrapolation for positions that have not yet been warmed up.
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _best_rate(self, k: int) -> Optional[float]:
        """Return the best available rate for position k (graduated EMA first,
        then warmup mean if warmup has at least one sample)."""
        if self._ema_rate[k] is not None:
            return self._ema_rate[k]
        if self._warmup_count[k] > 0:
            return self._warmup_sum[k] / self._warmup_count[k]
        return None

    def _get_rate_or_extrapolate(
        self, k: int, known_rates: list[float]
    ) -> Optional[float]:
        """Return acceptance rate for position k.

        Priority:
          1. Real data (EMA or warmup mean).
          2. Geometric extrapolation from ``known_rates``.
          3. ``None`` if there is nothing to extrapolate from.
        """
        real = self._best_rate(k)
        if real is not None:
            return real

        # Extrapolate
        if len(known_rates) == 0:
            return None
        if len(known_rates) == 1:
            # Only one data point; use it as-is (p[k] ≈ p[0]).
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
        best_step = candidate_steps[0]
        best_score = -math.inf

        for steps in candidate_steps:
            expected = self.tracker.get_expected_tokens(steps)
            if expected is None:
                continue
            itl_cost = self.itl_table.lookup(index_seqlen, batch_size, steps)
            if itl_cost is None or itl_cost <= 0:
                continue
            score = expected / itl_cost
            if score > best_score:
                best_score = score
                best_step = steps

        return best_step
