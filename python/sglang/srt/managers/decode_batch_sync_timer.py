"""Synced wall-clock timing for decode batches (baseline and DFLASH)."""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def decode_batch_sync_timing_enabled() -> bool:
    return bool(
        envs.SGLANG_DEBUG_DECODE_BATCH_SYNC.get()
        or envs.SGLANG_DFLASH_DEBUG_DECODE_BATCH_SYNC.get()
    )


def _sync_device(device: torch.device) -> None:
    if device.type == "cpu":
        return
    torch.get_device_module(device).synchronize()


class DecodeBatchSyncTimer:
    """Per decode-batch timer with optional phased breakdown (synced wall clock)."""

    __slots__ = (
        "_label",
        "_device",
        "_tp_rank",
        "_bs",
        "_enabled",
        "_batch_t0",
        "_phase_t0",
        "_phase_ms",
    )

    def __init__(
        self,
        *,
        label: str,
        device: torch.device,
        tp_rank: int,
        bs: int,
    ) -> None:
        self._label = label
        self._device = device
        self._tp_rank = int(tp_rank)
        self._bs = int(bs)
        self._enabled = decode_batch_sync_timing_enabled() and self._tp_rank == 0
        self._batch_t0: Optional[float] = None
        self._phase_t0: Optional[float] = None
        self._phase_ms: dict[str, float] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def on_batch_start(self) -> None:
        if not self._enabled:
            return
        _sync_device(self._device)
        self._batch_t0 = time.perf_counter()
        self._phase_ms = {}

    def phase_start(self) -> None:
        if not self._enabled:
            return
        _sync_device(self._device)
        self._phase_t0 = time.perf_counter()

    def phase_end(self, name: str) -> None:
        if not self._enabled:
            return
        _sync_device(self._device)
        if self._phase_t0 is None:
            return
        self._phase_ms[name] = (time.perf_counter() - self._phase_t0) * 1e3
        self._phase_t0 = None

    def on_batch_end(self) -> None:
        if not self._enabled or self._batch_t0 is None:
            return
        _sync_device(self._device)
        total_ms = (time.perf_counter() - self._batch_t0) * 1e3
        if self._label == "baseline":
            forward_ms = self._phase_ms.get("forward", float("nan"))
            sample_ms = self._phase_ms.get("sample", float("nan"))
            logger.info(
                "baseline decode-batch timing (synced wall ms): bs=%s total=%.3f "
                "forward=%.3f sample=%.3f",
                self._bs,
                total_ms,
                forward_ms,
                sample_ms,
            )
            return
        draft_ms = self._phase_ms.get("draft", float("nan"))
        verify_ms = self._phase_ms.get("verify", float("nan"))
        extend_ms = self._phase_ms.get("draft_extend", float("nan"))
        logger.info(
            "DFLASH decode-batch timing (synced wall ms): bs=%s total=%.3f "
            "draft=%.3f verify=%.3f draft_extend=%.3f",
            self._bs,
            total_ms,
            draft_ms,
            verify_ms,
            extend_ms,
        )
