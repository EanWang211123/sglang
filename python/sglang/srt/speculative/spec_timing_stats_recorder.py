"""
Speculative decoding timing stats recorder.

Records per-batch timing (draft, verify, draft-extend) to batchsize_xxx.jsonl.

Activation
----------
Either ``--enable-speculative-timing-logging`` OR ``SGLANG_SPEC_TIMING_STATS_DIR``
enables sync + timing stats. Only ``--enable`` prints to terminal (rank 0).
When ``SGLANG_SPEC_TIMING_STATS_DIR`` is set, timing data is written to batchsize_xxx.jsonl.

Output layout
-------------
<output_dir>/
    querylen_16_batchsize_1.jsonl
    querylen_16_batchsize_2.jsonl
    ...

Each JSONL line contains:
seq_lens:            list of sequence lengths in this batch
avg_lens:            average sequence length
draft_times:         draft phase time (ms)
draft_extend_times:  draft-extend phase time (ms)
verify_times:        verify phase time (ms)
batch_size:          batch concurrency
query_len:           verification tokens (1 + verify_step), unified per batch
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple


class SpeculativeTimingStatsRecorder:
    """
    Records per-batch timing (draft, verify, draft-extend) to batchsize_xxx.jsonl.

    Created when ``SGLANG_SPEC_TIMING_STATS_DIR`` is set. Sync + timing are enabled
    by either this env var OR ``--enable-speculative-timing-logging``.
    """

    @classmethod
    def from_env(cls, output_dir: str) -> Optional["SpeculativeTimingStatsRecorder"]:
        """
        Factory. Returns recorder when output_dir is non-empty, else None.
        """
        if not output_dir:
            return None
        return cls(output_dir=output_dir)

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._batch_fh: Dict[Tuple[int, int], Any] = {}  # (query_len, batch_size) -> file handle

    def record_batch_timing(
        self,
        *,
        seq_lens: List[int],
        draft_times: float,
        draft_extend_times: float,
        verify_times: float,
        batch_size: int,
        query_len: int,
    ) -> None:
        """Record one batch's timing to querylen_xxx_batchsize_xxx.jsonl. Times in ms."""
        if batch_size <= 0:
            return
        avg_lens = sum(seq_lens) / len(seq_lens) if seq_lens else 0.0
        # Convert seconds to ms for storage
        record = {
            "seq_lens": seq_lens,
            "avg_lens": round(avg_lens, 6),
            "draft_times": round(draft_times * 1000, 6),
            "draft_extend_times": round(draft_extend_times * 1000, 6),
            "verify_times": round(verify_times * 1000, 6),
            "batch_size": batch_size,
            "query_len": query_len,
        }
        fh = self._get_batch_fh(query_len, batch_size)
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()

    def _get_batch_fh(self, query_len: int, batch_size: int) -> Any:
        key = (query_len, batch_size)
        if key not in self._batch_fh:
            path = os.path.join(
                self.output_dir, f"querylen_{query_len}_batchsize_{batch_size}.jsonl"
            )
            self._batch_fh[key] = open(path, "a", encoding="utf-8")
        return self._batch_fh[key]

    def close(self) -> None:
        """Flush and close all open file handles."""
        for fh in self._batch_fh.values():
            fh.close()
        self._batch_fh.clear()
