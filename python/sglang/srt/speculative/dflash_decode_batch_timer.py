"""Backward-compatible re-exports; prefer decode_batch_sync_timer."""

from sglang.srt.managers.decode_batch_sync_timer import (
    DecodeBatchSyncTimer as DFlashDecodeBatchTimer,
    decode_batch_sync_timing_enabled as dflash_decode_batch_sync_timing_enabled,
)

__all__ = [
    "DFlashDecodeBatchTimer",
    "dflash_decode_batch_sync_timing_enabled",
]
