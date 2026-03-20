#!/usr/bin/env python3
"""
Fixed DFLASH server args + repeat tianwen benchmark N times (default 8).

Same outer loop as sweep_dflash_block_size.py: each round starts a fresh engine,
runs test.py once, then kills the process (cold start + capture every time).

Usage (from sglang repo root):
    python scripts/playground/repeat_dflash_tianwen_test.py

Env overrides:
    CUDA_VISIBLE_DEVICES   default 5
    ENGINE_PORT            default 6000
    TEST_DIR               default /workspace/workspaces/wyh/test/tianwen-test
    NUM_REPEATS            default 8
    SGLANG_SPEC_TIMING_STATS_DIR  optional, forwarded to server env like sweep script
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)

# --- defaults (override via env) ---
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "5")
ENGINE_PORT = int(os.environ.get("ENGINE_PORT", "6000"))
NUM_REPEATS = int(os.environ.get("NUM_REPEATS", "8"))
TEST_DIR = os.environ.get(
    "TEST_DIR",
    "/workspace/workspaces/wyh/test/tianwen-test",
)

BASE_URL = f"http://127.0.0.1:{ENGINE_PORT}"
MODEL_PATH = "/workspace/workspaces/gq/workspaces/SpecForge/qwen3-30b-a3b_moe_awq"
DRAFT_MODEL_PATH = (
    "/workspace/workspaces/gq/workspaces/SpecForge/outputs/"
    "qwen3-30b-a3b_moe_awq_second_dflash/epoch_12_step_43152"
)
TOKENIZER_PATH = "/workspace/workspaces/gq/workspaces/SpecForge/qwen3-30b-a3b_moe_awq"


def run_engine():
    """Start sglang server (same mechanism as sweep_dflash_block_size.py)."""
    other_args = [
        "--tp",
        "1",
        "--context-length",
        "32000",
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        DRAFT_MODEL_PATH,
        "--speculative-dflash-block-size",
        "16",
        "--mem-fraction-static",
        "0.90",
        "--log-requests-level",
        "3",
        "--attention-backend",
        "fa3",
        "--max-running-requests",
        "10",
    ]
    env = os.environ.copy()
    env["SGLANG_DISABLE_CUDNN_CHECK"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    stats_dir = os.environ.get("SGLANG_SPEC_TIMING_STATS_DIR")
    if stats_dir:
        env["SGLANG_SPEC_TIMING_STATS_DIR"] = stats_dir

    print("[repeat] Starting engine (fixed DFLASH block_size=16, tp=1)...")
    proc = popen_launch_server(
        model=MODEL_PATH,
        base_url=BASE_URL,
        timeout=max(DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, 600),
        other_args=other_args,
        env=env,
    )
    return proc


def run_test() -> bool:
    """Run tianwen test.py (same argv as sweep script / your CLI)."""
    test_script = Path(TEST_DIR) / "test.py"
    if not test_script.exists():
        print(f"[repeat] ERROR: Test script not found: {test_script}")
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
    for i in range(1, NUM_REPEATS + 1):
        print("")
        print(f"========== round {i}/{NUM_REPEATS} (start engine -> test -> stop) ==========")

        proc = run_engine()
        try:
            if not run_test():
                print(f"[repeat] Test failed on round {i}, aborting.")
                sys.exit(1)
        finally:
            if proc.poll() is None:
                print("[repeat] Stopping engine...")
                kill_process_tree(proc.pid)
                time.sleep(5)

        print(f"[repeat] Round {i}/{NUM_REPEATS} done.")

    print("")
    print(f"[repeat] All {NUM_REPEATS} rounds completed.")


if __name__ == "__main__":
    main()
