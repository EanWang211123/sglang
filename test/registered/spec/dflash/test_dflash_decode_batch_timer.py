import unittest

from sglang.srt.managers.decode_batch_sync_timer import (
    DecodeBatchSyncTimer,
    decode_batch_sync_timing_enabled,
)
from sglang.srt.speculative.dflash_decode_batch_timer import (
    dflash_decode_batch_sync_timing_enabled,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDFlashDecodeBatchTimer(CustomTestCase):
    def test_disabled_by_default(self):
        self.assertFalse(decode_batch_sync_timing_enabled())
        self.assertFalse(dflash_decode_batch_sync_timing_enabled())

    def test_disabled_timer_is_noop(self):
        timer = DecodeBatchSyncTimer(
            label="DFLASH",
            device=__import__("torch").device("cpu"),
            tp_rank=0,
            bs=4,
        )
        self.assertFalse(timer.enabled)
        timer.on_batch_start()
        timer.phase_start()
        timer.phase_end("draft")
        timer.on_batch_end()


if __name__ == "__main__":
    unittest.main()
