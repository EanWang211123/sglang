#!/usr/bin/env python3
"""
Sweep --speculative-dflash-block-size from 1 to 16.

For each block size: start engine -> wait ready -> run test -> stop engine -> next.

Uses popen_launch_server from sglang.test.test_utils (same as dflash benchmark_sglang.py).

Usage (run from sglang repo root):
    python scripts/playground/sweep_dflash_block_size.py

Override via env:
    SGLANG_SPEC_TIMING_STATS_DIR  output dir for timing jsonl
    CUDA_VISIBLE_DEVICES          GPU id (default: 6)
    ENGINE_PORT                   server port (default: 6000)
    TEST_DIR                      test script dir
"""

import os
import subprocess
import sys
import time
from pathlib import Path

# Add project root for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)

# Config (override via env)
SGLANG_SPEC_TIMING_STATS_DIR = os.environ.get(
    "SGLANG_SPEC_TIMING_STATS_DIR",
    "/workspace/workspaces/wyh/workspace-2026/spec-opt-ada-spec-stps-0316/draft-time-data/tianwen-10con-250-sample",
)
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "6")
ENGINE_PORT = int(os.environ.get("ENGINE_PORT", "6000"))
TEST_DIR = os.environ.get(
    "TEST_DIR",
    "/workspace/workspaces/wyh/test/tianwen-test",
)

BASE_URL = f"http://127.0.0.1:{ENGINE_PORT}"
MODEL_PATH = "/workspace/workspaces/gq/workspaces/SpecForge/qwen3-30b-a3b_moe_awq"
DRAFT_MODEL_PATH = "/workspace/workspaces/gq/workspaces/SpecForge/outputs/qwen3-30b-a3b_moe_awq_second_dflash/epoch_12_step_43152"
TOKENIZER_PATH = "/workspace/workspaces/gq/workspaces/SpecForge/qwen3-30b-a3b_moe_awq"


def run_engine(block_size: int):
    """Start sglang server via popen_launch_server (handles health check internally)."""
    other_args = [
        "--context-length",
        "32000",
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        DRAFT_MODEL_PATH,
        "--speculative-dflash-block-size",
        str(block_size),
        "--mem-fraction-static",
        "0.90",
        "--log-requests-level",
        "3",
        "--attention-backend",
        "fa3",
        "--max-running-requests",
        "10",
        "--skip-server-warmup",
        "--enable-speculative-timing-logging",
    ]
    env = os.environ.copy()
    env["SGLANG_SPEC_TIMING_STATS_DIR"] = SGLANG_SPEC_TIMING_STATS_DIR
    env["SGLANG_DISABLE_CUDNN_CHECK"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    print(f"[sweep] Starting engine with --speculative-dflash-block-size={block_size}")
    proc = popen_launch_server(
        model=MODEL_PATH,
        base_url=BASE_URL,
        timeout=max(DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, 600),
        other_args=other_args,
        env=env,
    )
    return proc


def run_test() -> bool:
    """Run the tianwen test script."""
    print("[sweep] Running test...")
    test_script = Path(TEST_DIR) / "test.py"
    if not test_script.exists():
        print(f"[sweep] ERROR: Test script not found: {test_script}")
        return False
    cmd = [
        sys.executable,
        str(test_script),
        "--base-url",
        BASE_URL,
        "--model",
        "qwen3-30b-a3b_moe_awq",
        "--dataset-name",
        "normal",
        "--dataset-path",
        "./benchmark.full.json",
        "--tokenizer-path",
        TOKENIZER_PATH,
        "--num-prompts",
        "250",
        "--temperature",
        "0.0",
        "--max-concurrency",
        "10",
        "--shuffle",
        "--fixed-output-len",
        "4090",
        "--disable-ignore-eos",
    ]
    ret = subprocess.run(cmd, cwd=TEST_DIR)
    return ret.returncode == 0


def main():
    for block_size in range(1, 17):
        print("")
        print(f"========== block_size={block_size} ==========")

        proc = run_engine(block_size)
        try:
            # popen_launch_server already waits for server health before returning
            if not run_test():
                print(f"[sweep] Test failed at block_size={block_size}, aborting.")
                sys.exit(1)
        finally:
            if proc.poll() is None:
                print("[sweep] Stopping engine...")
                kill_process_tree(proc.pid)
                time.sleep(5)

        print(f"[sweep] block_size={block_size} done.")

    print("")
    print("[sweep] All block sizes (1-16) completed.")


if __name__ == "__main__":
    main()
