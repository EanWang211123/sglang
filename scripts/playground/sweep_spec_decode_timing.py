#!/usr/bin/env python3
"""
Sweep DFLASH speculative block sizes while recording spec-decode timing stats.

For each block size: start engine (timing jsonl via SGLANG_SPEC_TIMING_STATS_DIR) ->
wait ready -> run tianwen v2 benchmark -> stop engine -> next.

Mirrors scripts/playground/sweep_dflash_block_size.py; defaults match the
tianwen-v2 + qwen3-30b-a3b-awq DFLASH timing-capture setup you described.

Usage (from sglang repo root):
    python scripts/playground/sweep_spec_decode_timing.py

Env overrides:
    SGLANG_SPEC_TIMING_STATS_DIR   output dir for timing jsonl (required for files)
    CUDA_VISIBLE_DEVICES           GPU id (default: 7)
    ENGINE_PORT                    server port (default: 9001)
    TEST_DIR                       directory containing test.py
    SWEEP_DFLASH_BLOCK_SIZES       comma-separated ints, e.g. "8,16,32"
                                   (default: 1,2,...,16)
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

# --- Defaults (override via env) ---
SGLANG_SPEC_TIMING_STATS_DIR = os.environ.get(
    "SGLANG_SPEC_TIMING_STATS_DIR",
    "/workspace/workspaces/wyh/workspace-2026/spec-opt-ada-spec-stps-0316/draft-time-data/tianwen-v2-kvfp8-088-10con-250-sample",
)
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "7")
ENGINE_PORT = int(os.environ.get("ENGINE_PORT", "9001"))
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
        "--context-length",
        "32000",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        DRAFT_MODEL_PATH,
        "--speculative-dflash-block-size",
        str(block_size),
        "--mem-fraction-static",
        "0.88",
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
