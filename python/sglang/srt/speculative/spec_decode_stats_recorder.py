"""
Speculative decoding statistics recorder.

Writes per-batch and per-sequence metrics to disk for offline analysis of early-stopping
strategies. The recorder is backend-agnostic: any speculative decoding path can call
``record_verify_step()`` after each target-model verification round.

Activation
----------
Set the environment variable ``SGLANG_SPEC_STATS_DIR`` to a writable directory path.
Optionally set ``SGLANG_SPEC_STATS_GAMMA`` (float, default 0.2) to control the scaling
factor in the feature-entropy formula.

Output layout
-------------
<output_dir>/
    batch/
        batches.jsonl          # one JSON line per decode batch
    seqs/
        seq_<rid>.jsonl        # one JSON line per batch in which this seq participated

Batch record fields
-------------------
batch_id                 – monotonically increasing batch counter (this process)
seq_ids                  – list of request IDs present in this batch
draft_steps              – number of *actual* draft positions (= draft_token_num - 1)
global_accept_rate       – cumulative (all batches, all positions) acceptance rate
avg_accepted_steps       – cumulative mean accepted draft tokens per verify step
position_accept_rates    – {"draft_position1": 0.9, "draft_position2": 0.8, …}
max_accepted_tokens      – max accepted tokens across seqs in this batch
min_accepted_tokens      – min accepted tokens across seqs in this batch

Sequence record fields (one per batch the sequence participates in)
--------------------------------------------------------------------
seq_id                   – request ID
batch_id                 – which batch this row corresponds to
draft_steps              – number of actual draft positions in this batch
seq_global_accept_rate   – cumulative acceptance rate for this sequence
seq_avg_accepted_steps   – cumulative mean accepted steps for this sequence
seq_position_accept_rates– {"draft_position1": 0.9, …} for this sequence
seq_len                  – sequence length *before* this batch's tokens were committed
draft_max_logits         – list[float] global max raw logit per draft position.
                           Available for DFlash (all TP configs): exact global max logit
                           from _greedy_sample_from_vocab_parallel_head at zero extra cost.
                           null for EAGLE/full-logit backends (use draft_max_probs instead).
draft_max_probs          – list[float] max softmax probability per draft position (0~1).
                           E.g. [0.92, 0.85, 0.7, …].
                           DFlash TP=1: computed from base_logits at zero extra LM-head cost.
                           EAGLE/full-logit backends: computed from full logits.
                           null for DFlash TP>1 (full distribution not available).
feature_entropy          – list[float] 1 − √(γ · H_DM(x)) per draft position.
                           Available when draft_max_probs is available (TP=1 or full logits).
                           null for DFlash TP>1.
accepted_tokens          – number of draft tokens accepted for this sequence in this batch
"""

import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator > 0 else 0.0


class SpecDecodeStatsRecorder:
    """
    Records per-batch and per-sequence speculative decoding statistics to disk.

    Designed to be backend-agnostic. Call ``record_verify_step()`` after every
    speculative-decoding verify step.

    Two optional inputs control logit-level statistics:

    ``draft_max_logits`` (shape ``[bs, num_draft_slots]``, float32)
        Global max logit per draft position.  For DFlash this is produced inside
        ``_greedy_sample_from_vocab_parallel_head`` with zero extra communication
        (TP>1: gathered_max.max(dim=0) is free after the existing all-gather).
        For EAGLE or any backend that already has full global logits, the caller
        can pass ``draft_full_logits`` instead and both max log-prob and entropy
        will be derived from it automatically.

    ``draft_full_logits`` (shape ``[bs * num_draft_slots, vocab_size]``, float)
        Full global logits for all draft positions.  When provided, the recorder
        computes true max log-prob (log_softmax max) and feature entropy without
        any additional model forward pass.  Intended for EAGLE and similar backends
        where the draft model already emits complete logits via LogitsProcessor.
    """

    @classmethod
    def from_env(cls) -> Optional["SpecDecodeStatsRecorder"]:
        """
        Factory that reads configuration from environment variables.

        Returns ``None`` (recorder disabled) when ``SGLANG_SPEC_STATS_DIR`` is
        not set or is empty.
        """
        output_dir = os.environ.get("SGLANG_SPEC_STATS_DIR", "").strip()
        if not output_dir:
            return None
        gamma = float(os.environ.get("SGLANG_SPEC_STATS_GAMMA", "0.2"))
        return cls(output_dir=output_dir, gamma=gamma)

    def __init__(self, output_dir: str, gamma: float = 0.2) -> None:
        self.output_dir = output_dir
        self.gamma = gamma

        self.batch_dir = os.path.join(output_dir, "batch")
        self.seqs_dir = os.path.join(output_dir, "seqs")
        os.makedirs(self.batch_dir, exist_ok=True)
        os.makedirs(self.seqs_dir, exist_ok=True)

        # ── Global accumulators ────────────────────────────────────────────────
        self._batch_id: int = 0
        # Total draft-token slots seen (positions 1..K-1) summed over all batches×seqs.
        self._global_total_draft_slots: int = 0
        # Total draft tokens accepted (excluding the always-accepted bonus token at pos-0).
        self._global_total_accepted: int = 0
        # Total verify calls (= Σ batch_size across batches).
        self._global_total_verify_steps: int = 0
        # Per draft-position (1-indexed) running counts.
        self._global_pos_accepted: Dict[int, int] = {}
        self._global_pos_seen: Dict[int, int] = {}

        # ── Per-sequence accumulators ──────────────────────────────────────────
        # rid → {"total_draft_slots", "total_accepted", "verify_ct",
        #         "pos_accepted": {pos: int}, "pos_seen": {pos: int}}
        self._seq_stats: Dict[str, Dict[str, Any]] = {}

        # ── File handles ───────────────────────────────────────────────────────
        self._batch_fh = open(
            os.path.join(self.batch_dir, "batches.jsonl"), "a", encoding="utf-8"
        )
        self._seq_fh: Dict[str, Any] = {}  # rid → file handle

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def record_verify_step(
        self,
        *,
        batch_reqs: List[Any],
        draft_token_num: int,
        accept_length_per_req_cpu: List[int],
        seq_lens_before_verify: List[int],
        # ── Logit statistics inputs (all optional) ────────────────────────────
        draft_max_logits: Optional[torch.Tensor] = None,
        draft_max_probs: Optional[torch.Tensor] = None,
        draft_raw_entropies: Optional[torch.Tensor] = None,
        draft_full_logits: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Record one speculative-decode verify step.

        Parameters
        ----------
        batch_reqs:
            List of ``Req`` objects in this batch (must expose ``.rid``).
        draft_token_num:
            Total number of draft positions *including* position-0 (the already-committed
            current token). The number of actual draft tokens is ``draft_token_num - 1``.
        accept_length_per_req_cpu:
            Number of accepted draft tokens per request (length == batch_size). This
            excludes the bonus token at position-0.  Values in ``[0, draft_token_num-1]``.
        seq_lens_before_verify:
            Sequence length per request *before* any tokens from this batch are committed.
        draft_max_logits:
            Optional ``[bs, draft_token_num-1]`` float32 – global max raw logit per draft
            position.  Available for all TP configs from DFlash.
        draft_max_probs:
            Optional ``[bs, draft_token_num-1]`` float32 – max softmax probability per
            draft position (values in (0, 1], e.g. 0.9).  Available for TP=1 DFlash only.
        draft_raw_entropies:
            Optional ``[bs, draft_token_num-1]`` float32 – Shannon entropy H of the draft
            token distribution.  Available for TP=1 DFlash only.  The recorder applies
            the formula ``1 − √(γ · H)`` to produce ``feature_entropy``.
        draft_full_logits:
            Optional ``[bs * (draft_token_num-1), vocab_size]`` float – full global logits.
            When provided (e.g. EAGLE), both max prob and entropy are derived from it and
            the three per-computed arguments above are ignored.
        """
        if not batch_reqs:
            return

        bs = len(batch_reqs)
        num_draft_slots = draft_token_num - 1  # actual draft positions 1..K-1

        # ── Logit-based stats ──────────────────────────────────────────────────
        # Priority: full logits (EAGLE etc.) > pre-computed scalars (DFlash TP=1) > nothing.
        draft_max_logits_list: Optional[List[List[float]]] = None
        draft_max_probs_list: Optional[List[List[float]]] = None
        feature_entropies: Optional[List[List[float]]] = None

        if draft_full_logits is not None and num_draft_slots > 0:
            # Full global logits available (e.g., EAGLE): derive max prob and entropy.
            draft_max_probs_list, feature_entropies = (
                self._compute_stats_from_full_logits(
                    full_logits=draft_full_logits,
                    bs=bs,
                    num_draft_slots=num_draft_slots,
                )
            )
        elif num_draft_slots > 0:
            # Pre-computed scalar stats from DFlash _greedy_sample_from_vocab_parallel_head.
            if draft_max_logits is not None:
                try:
                    draft_max_logits_list = draft_max_logits.cpu().tolist()
                except Exception:
                    pass  # non-fatal
            if draft_max_probs is not None:
                try:
                    draft_max_probs_list = draft_max_probs.cpu().tolist()
                except Exception:
                    pass
            if draft_raw_entropies is not None:
                try:
                    gamma = self.gamma
                    feature_entropies = [
                        [1.0 - math.sqrt(max(0.0, gamma * h)) for h in row]
                        for row in draft_raw_entropies.cpu().float().tolist()
                    ]
                except Exception:
                    pass

        batch_id = self._batch_id
        self._batch_id += 1

        # ── Update global accumulators ─────────────────────────────────────────
        for acc_len in accept_length_per_req_cpu:
            for j in range(num_draft_slots):
                pos = j + 1  # 1-indexed draft position
                self._global_pos_seen[pos] = self._global_pos_seen.get(pos, 0) + 1
                if acc_len > j:
                    self._global_pos_accepted[pos] = (
                        self._global_pos_accepted.get(pos, 0) + 1
                    )

        self._global_total_draft_slots += bs * num_draft_slots
        self._global_total_accepted += sum(accept_length_per_req_cpu)
        self._global_total_verify_steps += bs

        global_accept_rate = _safe_div(
            self._global_total_accepted, self._global_total_draft_slots
        )
        avg_accepted_steps = _safe_div(
            self._global_total_accepted, self._global_total_verify_steps
        )
        global_pos_rates = {
            f"draft_position{pos}": round(
                _safe_div(self._global_pos_accepted.get(pos, 0), seen), 6
            )
            for pos, seen in self._global_pos_seen.items()
        }

        # ── Batch JSONL record ─────────────────────────────────────────────────
        batch_record = {
            "batch_id": batch_id,
            "seq_ids": [req.rid for req in batch_reqs],
            "draft_steps": num_draft_slots,
            "global_accept_rate": round(global_accept_rate, 6),
            "avg_accepted_steps": round(avg_accepted_steps, 6),
            "position_accept_rates": global_pos_rates,
            "max_accepted_tokens": max(accept_length_per_req_cpu),
            "min_accepted_tokens": min(accept_length_per_req_cpu),
        }
        self._batch_fh.write(json.dumps(batch_record, ensure_ascii=False) + "\n")
        self._batch_fh.flush()

        # ── Per-sequence JSONL records ─────────────────────────────────────────
        for i, req in enumerate(batch_reqs):
            rid: str = req.rid
            acc_len = accept_length_per_req_cpu[i]

            if rid not in self._seq_stats:
                self._seq_stats[rid] = {
                    "total_draft_slots": 0,
                    "total_accepted": 0,
                    "verify_ct": 0,
                    "pos_accepted": {},
                    "pos_seen": {},
                }
            ss = self._seq_stats[rid]

            for j in range(num_draft_slots):
                pos = j + 1
                ss["pos_seen"][pos] = ss["pos_seen"].get(pos, 0) + 1
                if acc_len > j:
                    ss["pos_accepted"][pos] = ss["pos_accepted"].get(pos, 0) + 1

            ss["total_draft_slots"] += num_draft_slots
            ss["total_accepted"] += acc_len
            ss["verify_ct"] += 1

            seq_accept_rate = _safe_div(ss["total_accepted"], ss["total_draft_slots"])
            seq_avg_steps = _safe_div(ss["total_accepted"], ss["verify_ct"])
            seq_pos_rates = {
                f"draft_position{pos}": round(
                    _safe_div(ss["pos_accepted"].get(pos, 0), seen), 6
                )
                for pos, seen in ss["pos_seen"].items()
            }

            seq_record = {
                "seq_id": rid,
                "batch_id": batch_id,
                "draft_steps": num_draft_slots,
                "seq_global_accept_rate": round(seq_accept_rate, 6),
                "seq_avg_accepted_steps": round(seq_avg_steps, 6),
                "seq_position_accept_rates": seq_pos_rates,
                "seq_len": seq_lens_before_verify[i],
                # draft_max_logits: raw logit (TP≥1), null for EAGLE/full-logit backends.
                "draft_max_logits": (
                    [round(v, 6) for v in draft_max_logits_list[i]]
                    if draft_max_logits_list is not None
                    else None
                ),
                # draft_max_probs: max softmax probability per position (0~1, e.g. 0.9).
                # Available for TP=1 DFlash, or any backend that provides full logits.
                "draft_max_probs": (
                    [round(v, 6) for v in draft_max_probs_list[i]]
                    if draft_max_probs_list is not None
                    else None
                ),
                # feature_entropy: 1 − √(γ·H), null when full distribution not available.
                "feature_entropy": (
                    [round(v, 6) for v in feature_entropies[i]]
                    if feature_entropies is not None
                    else None
                ),
                "accepted_tokens": acc_len,
            }
            fh = self._get_seq_fh(rid)
            fh.write(json.dumps(seq_record, ensure_ascii=False) + "\n")
            fh.flush()

    def close(self) -> None:
        """Flush and close all open file handles."""
        self._batch_fh.close()
        for fh in self._seq_fh.values():
            fh.close()
        self._seq_fh.clear()

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_seq_fh(self, rid: str) -> Any:
        if rid not in self._seq_fh:
            safe = (
                rid.replace("/", "_")
                .replace("\\", "_")
                .replace(":", "_")
                .replace(" ", "_")
            )
            path = os.path.join(self.seqs_dir, f"seq_{safe}.jsonl")
            self._seq_fh[rid] = open(path, "a", encoding="utf-8")
        return self._seq_fh[rid]

    def _compute_stats_from_full_logits(
        self,
        full_logits: torch.Tensor,
        bs: int,
        num_draft_slots: int,
        chunk_size: int = 512,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Compute max softmax probability and feature entropy from full global logits.

        Called for backends (e.g., EAGLE) that already materialise the complete
        vocabulary distribution.  No extra model forward pass is needed.

        Parameters
        ----------
        full_logits:
            ``[bs * num_draft_slots, vocab_size]`` – full global logits (any dtype).
        bs:
            Batch size.
        num_draft_slots:
            Number of actual draft positions per sequence (= draft_token_num - 1).

        Returns
        -------
        max_probs:         ``[bs][num_draft_slots]`` – max softmax probability (0~1).
        feature_entropies: ``[bs][num_draft_slots]`` – 1 − √(γ · H_DM(x)).
        """
        N = full_logits.shape[0]
        max_prob_parts: List[torch.Tensor] = []
        entropy_parts: List[torch.Tensor] = []

        for start in range(0, N, chunk_size):
            end = min(N, start + chunk_size)
            logits_chunk = full_logits[start:end].float()
            log_p = torch.log_softmax(logits_chunk, dim=-1)
            max_prob_parts.append(torch.exp(log_p.max(dim=-1).values).cpu())
            probs = torch.exp(log_p)
            entropy_parts.append((-(probs * log_p).sum(dim=-1)).cpu())

        max_prob_t = torch.cat(max_prob_parts).view(bs, num_draft_slots)
        entropy_t = torch.cat(entropy_parts).view(bs, num_draft_slots)

        gamma = self.gamma
        max_probs = max_prob_t.tolist()
        feature_entropies = [
            [1.0 - math.sqrt(max(0.0, gamma * h)) for h in row]
            for row in entropy_t.tolist()
        ]
        return max_probs, feature_entropies
