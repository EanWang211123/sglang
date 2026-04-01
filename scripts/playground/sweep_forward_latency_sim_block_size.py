#!/usr/bin/env python3
"""
Sweep --speculative-dflash-block-size and run forward latency simulator for each.

For each block size: start engine (with --forward-latency-sim-batch-sizes) ->
  wait ready or fail -> stop -> next block size.
No test run; success or failure both proceed to next block size.

Uses popen_launch_server from sglang.test.test_utils.

Usage (run from sglang repo root):
    python scripts/playground/sweep_forward_latency_sim_block_size.py

Equivalent manual launch (per block size, script adds --speculative-dflash-block-size):

    SGLANG_FORWARD_LATENCY_SIM_STATS_DIR=/workspace/workspaces/wyh/workspace-2026/spec-opt-ada-spec-stps-0316/draft-time-data/sim-forward-6000-seq-len-qwen3-8b-awq \\
    CUDA_VISIBLE_DEVICES=2 \\
    python -m sglang.launch_server \\
    --model-path /nfs_models/Qwen/Qwen3-8B-AWQ/ \\
    --speculative-algorithm DFLASH \\
    --speculative-draft-model-path /workspace/workspaces/wyh/models/Qwen3-8B-DFlash-b16/ \\
    --tp-size 1 \\
    --dtype bfloat16 \\
    --attention-backend fa3 \\
    --mem-fraction-static 0.75 \\
    --max-running-requests 32 --forward-latency-sim-batch-sizes 1,2,4,8,12,16,20,24,28,32 \\
    --forward-latency-sim-fixed-seq-len 5984 \\
    --cuda-graph-max-bs 32 --context-length 6000

Defaults (Qwen3-8B-AWQ tp1 / sim-forward-6000; override via env):
    MODEL_PATH              target model path
    DRAFT_MODEL_PATH        draft model path
    SGLANG_FORWARD_LATENCY_SIM_STATS_DIR  output dir for sim jsonl
    CUDA_VISIBLE_DEVICES    GPU ids (default: 2)
    TP                      tensor parallel size / --tp-size (default: 1)
    DTYPE                   --dtype (default: bfloat16)
    ENGINE_PORT             server port (default: 6000)
    BLOCK_SIZE_START/END    block size range (default: 1..16)
    CONTEXT_LENGTH          context length (default: 6000)
    CUDA_GRAPH_MAX_BS       cuda graph max batch size (default: 32)
    MAX_RUNNING_REQUESTS    max running requests (default: 32)
    MEM_FRACTION_STATIC     mem fraction static (default: 0.75)
    FORWARD_LATENCY_SIM_BATCH_SIZES  sim batch sizes (default: 1,2,4,8,12,16,20,24,28,32)
    FORWARD_LATENCY_SIM_FIXED_SEQ_LEN  fixed seq_len for all simulated sequences (default: unset,
                            uses context_len - num_tokens_per_bs). Set this to compare latency at
                            a specific context length across different block sizes / draft token counts.
                            E.g. FORWARD_LATENCY_SIM_FIXED_SEQ_LEN=5984 with context_len=6000 and
                            num_draft_tokens=16 ensures seq_len is always 5984 regardless of
                            block_size. Without this, seq_len = context_len - num_draft_tokens,
                            which varies if num_draft_tokens changes across runs.
"""

import os
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

# Config (env overrides; defaults match Qwen3-8B-AWQ tp1 DFLASH forward-latency-sim run)
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/nfs_models/Qwen/Qwen3-8B-AWQ/",
)
DRAFT_MODEL_PATH = os.environ.get(
    "DRAFT_MODEL_PATH",
    "/workspace/workspaces/wyh/models/Qwen3-8B-DFlash-b16/",
)
SGLANG_FORWARD_LATENCY_SIM_STATS_DIR = os.environ.get(
    "SGLANG_FORWARD_LATENCY_SIM_STATS_DIR",
    "/workspace/workspaces/wyh/workspace-2026/spec-opt-ada-spec-stps-0316/draft-time-data/sim-forward-6000-seq-len-qwen3-8b-awq",
)
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "2")
ENGINE_PORT = int(os.environ.get("ENGINE_PORT", "6000"))
BLOCK_SIZE_START = int(os.environ.get("BLOCK_SIZE_START", "1"))
BLOCK_SIZE_END = int(os.environ.get("BLOCK_SIZE_END", "16"))
CONTEXT_LENGTH = int(os.environ.get("CONTEXT_LENGTH", "6000"))
CUDA_GRAPH_MAX_BS = int(os.environ.get("CUDA_GRAPH_MAX_BS", "32"))
MAX_RUNNING_REQUESTS = int(os.environ.get("MAX_RUNNING_REQUESTS", "32"))
MEM_FRACTION_STATIC = float(os.environ.get("MEM_FRACTION_STATIC", "0.75"))
FORWARD_LATENCY_SIM_BATCH_SIZES = os.environ.get(
    "FORWARD_LATENCY_SIM_BATCH_SIZES",
    "1,2,4,8,12,16,20,24,28,32",
)
FORWARD_LATENCY_SIM_FIXED_SEQ_LEN = os.environ.get("FORWARD_LATENCY_SIM_FIXED_SEQ_LEN", "")
TP = int(os.environ.get("TP", "1"))
DTYPE = os.environ.get("DTYPE", "bfloat16")

BASE_URL = f"http://127.0.0.1:{ENGINE_PORT}"


def run_engine(block_size: int, output_dir: str):
    """Start sglang server with forward-latency-sim. Returns process or None on failure."""
    other_args = [
        "--context-length", str(CONTEXT_LENGTH),
        "--dtype", DTYPE,
        "--speculative-algorithm", "DFLASH",
        "--speculative-draft-model-path", DRAFT_MODEL_PATH,
        "--speculative-dflash-block-size", str(block_size),
        "--mem-fraction-static", str(MEM_FRACTION_STATIC),
        "--attention-backend", "fa3",
        "--max-running-requests", str(MAX_RUNNING_REQUESTS),
        "--forward-latency-sim-batch-sizes", FORWARD_LATENCY_SIM_BATCH_SIZES,
        *(["--forward-latency-sim-fixed-seq-len", FORWARD_LATENCY_SIM_FIXED_SEQ_LEN]
          if FORWARD_LATENCY_SIM_FIXED_SEQ_LEN else []),
        "--cuda-graph-max-bs", str(CUDA_GRAPH_MAX_BS),
        "--tp-size", str(TP),
    ]
    env = os.environ.copy()
    env["SGLANG_FORWARD_LATENCY_SIM_STATS_DIR"] = output_dir
    env["SGLANG_DISABLE_CUDNN_CHECK"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    print(f"[sweep] block_size={block_size} | output={output_dir} | tp={TP}")
    try:
        proc = popen_launch_server(
            model=MODEL_PATH,
            base_url=BASE_URL,
            timeout=max(DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, 600),
            other_args=other_args,
            env=env,
        )
        return proc
    except Exception as e:
        print(f"[sweep] block_size={block_size} FAILED to start: {e}")
        return None


def main():
    if not MODEL_PATH or not DRAFT_MODEL_PATH:
        print("ERROR: MODEL_PATH and DRAFT_MODEL_PATH must be set.")
        sys.exit(1)
    if not SGLANG_FORWARD_LATENCY_SIM_STATS_DIR:
        print("ERROR: SGLANG_FORWARD_LATENCY_SIM_STATS_DIR must be set.")
        sys.exit(1)

    base_output = SGLANG_FORWARD_LATENCY_SIM_STATS_DIR.rstrip("/")
    block_sizes = list(range(BLOCK_SIZE_START, BLOCK_SIZE_END + 1))

    for block_size in block_sizes:
        print("")
        print(f"========== block_size={block_size} ==========")
        os.makedirs(base_output, exist_ok=True)

        proc = run_engine(block_size, base_output)
        if proc is None:
            print(f"[sweep] block_size={block_size} skipped (launch failed), continuing.")
            continue

        try:
            # Engine started (sim runs during init). Keep alive briefly then stop.
            # Or: if user wants to stop immediately after ready, we could kill right away.
            # "启动成功或者失败都就直接到下一个" - so we stop and move on.
            time.sleep(2)
        finally:
            if proc.poll() is None:
                print(f"[sweep] Stopping engine for block_size={block_size}...")
                kill_process_tree(proc.pid)
                time.sleep(5)

        print(f"[sweep] block_size={block_size} done.")

    print("")
    print(f"[sweep] All block sizes {block_sizes} completed.")


if __name__ == "__main__":
    main()
