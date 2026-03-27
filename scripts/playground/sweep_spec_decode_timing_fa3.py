#!/usr/bin/env python3
"""
Sweep DFLASH speculative block sizes with FA3 backend + fixed verify token count.

Same flow as sweep_spec_decode_timing.py (engine -> tianwen-test-v2 test.py -> stop),
but engine uses --attention-backend fa3, --mem-fraction-static 0.9,
no kv-cache-dtype override, and port 9002.

Usage (from sglang repo root):
    python scripts/playground/sweep_spec_decode_timing_fa3.py

Env overrides:
    SGLANG_SPEC_TIMING_STATS_DIR   output dir for timing jsonl
    CUDA_VISIBLE_DEVICES           GPU id (default: 7)
    ENGINE_PORT                    server port (default: 9002)
    TEST_DIR                       directory containing test.py
    SWEEP_DFLASH_BLOCK_SIZES       comma-separated ints (default: 1..16)
"""

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

SGLANG_SPEC_TIMING_STATS_DIR = os.environ.get(
    "SGLANG_SPEC_TIMING_STATS_DIR",
    "/workspace/workspaces/wyh/workspace-2026/spec-opt-ada-spec-stps-0316/draft-time-data/tianwen-v2-fa3-090-10con-250-sample",
)
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "7")
ENGINE_PORT = int(os.environ.get("ENGINE_PORT", "9002"))
TEST_DIR = os.environ.get(
    "TEST_DIR",
    "/workspace/workspaces/wyh/test/tianwen-test-v2",
)

BASE_URL = f"http://127.0.0.1:{ENGINE_PORT}"
MODEL_PATH = "/workspace/workspaces/gq/workspaces/SpecForge/qwen3-30b-a3b-awq-latest/"
DRAFT_MODEL_PATH = "/workspace/workspaces/gq/workspaces/SpecForge/outputs/qwen3-30b-a3b-awq-third-dflash/epoch_9_step_27000/"
TOKENIZER_PATH = MODEL_PATH
DATASET_PATH = (
    "/workspace/workspaces/gq/workspaces/SpecForge/test_use/benchmark.latest.json"
)


def _parse_block_sizes() -> list[int]:
    raw = os.environ.get("SWEEP_DFLASH_BLOCK_SIZES", "").strip()
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    return list(range(1, 17))


def run_engine(block_size: int):
    other_args = [
        "--tp",
        "1",
        "--attention-backend",
        "fa3",
        "--context-length",
        "32000",
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        DRAFT_MODEL_PATH,
        "--speculative-dflash-block-size",
        str(block_size),
        "--mem-fraction-static",
        "0.9",
        "--log-requests-level",
        "3",
        "--max-running-requests",
        "10",
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
        MODEL_PATH,
        "--dataset-name",
        "normal",
        "--dataset-path",
        DATASET_PATH,
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
    block_sizes = _parse_block_sizes()
    print(f"[sweep] SGLANG_SPEC_TIMING_STATS_DIR={SGLANG_SPEC_TIMING_STATS_DIR}")
    print(f"[sweep] ENGINE_PORT={ENGINE_PORT} BASE_URL={BASE_URL}")
    print(f"[sweep] Block sizes: {block_sizes}")

    for block_size in block_sizes:
        print("")
        print(f"========== block_size={block_size} ==========")

        proc = run_engine(block_size)
        try:
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
    print(f"[sweep] All block sizes {block_sizes} completed.")


if __name__ == "__main__":
    main()
