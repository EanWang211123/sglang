#!/usr/bin/env python3
"""
两阶段重复测试：先 DFLASH + 动态 verify tokens 配置 N 轮（每轮冷启动引擎 -> test.py -> 停服），
再 DFLASH 但不带动态配置 N 轮（同样流程，作为对照 baseline）。

用法（在 sglang 仓库根目录）：
    CUDA_VISIBLE_DEVICES=3 python scripts/playground/repeat_spec_then_baseline_tianwen_test.py

服务端启动与下列命令等价（python -m sglang.launch_server，kv fp8 + DFLASH 等）：
    见 run_engine() 内拼接的 argv。

环境变量：
    CUDA_VISIBLE_DEVICES              默认 3
    ENGINE_PORT                       默认 9005
    TEST_DIR                          test.py 所在目录（cwd），默认 tianwen-test-v2
    DATASET_PATH                      可选；默认与下方手写 test 命令一致（SpecForge 数据路径）
    NUM_SPEC_ROUNDS                   带动态 verify 配置阶段轮数，默认 10
    NUM_BASE_ROUNDS                   baseline（无动态配置）阶段轮数，默认 10
    SKIP_NON_SPEC_BENCHMARK=1         若设置则跳过第二阶段（仅跑带动态配置阶段）
"""

from __future__ import annotations

import os
import shlex
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
    _launch_server_process,
    _wait_for_server_health,
)

# --- defaults (override via env) ---
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "3")
ENGINE_PORT = int(os.environ.get("ENGINE_PORT", "9005"))
NUM_SPEC_ROUNDS = int(os.environ.get("NUM_SPEC_ROUNDS", "10"))
NUM_BASE_ROUNDS = int(os.environ.get("NUM_BASE_ROUNDS", "10"))
SKIP_NON_SPEC = os.environ.get("SKIP_NON_SPEC_BENCHMARK", "").strip().lower() in (
    "1",
    "true",
    "yes",
)

DEFAULT_TEST_DIR = "/workspace/workspaces/wyh/test/tianwen-test-v2"
TEST_DIR = os.environ.get("TEST_DIR", DEFAULT_TEST_DIR)
# 与手写测试命令一致（dataset/model/tokenizer 不在 TEST_DIR 下）
DEFAULT_DATASET_PATH = (
    "/workspace/workspaces/gq/workspaces/SpecForge/test_use/benchmark.latest.json"
)
DATASET_PATH = os.environ.get("DATASET_PATH", DEFAULT_DATASET_PATH)

BASE_URL = f"http://127.0.0.1:{ENGINE_PORT}"
MODEL_PATH = (
    "/workspace/workspaces/gq/workspaces/SpecForge/qwen3-30b-a3b-awq-latest/"
)
DRAFT_MODEL_PATH = (
    "/workspace/workspaces/gq/workspaces/SpecForge/outputs/"
    "qwen3-30b-a3b-awq-third-dflash/epoch_9_step_27000/"
)
DYNAMIC_VERIFY_CONFIG = (
    "/workspace/workspaces/wyh/workspace-2026/spec-opt-ada-spec-stps-0316/"
    "acc-balance-data/results-tianwen-v2-kvfp8/cglist_entropy_fp8_top1.json"
)
TOKENIZER_PATH = MODEL_PATH


def _launch_server_argv(*, with_dynamic_verify_config: bool) -> list[str]:
    """与手写 `python3 -m sglang.launch_server ...` 一致；baseline 仅去掉动态 verify 配置项。"""
    argv = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--port",
        str(ENGINE_PORT),
        "--tp",
        "1",
        "--model",
        MODEL_PATH,
        "--context-length",
        "32000",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--speculative-algorithm",
        "DFLASH",
        "--speculative-draft-model-path",
        DRAFT_MODEL_PATH,
        "--speculative-dflash-block-size",
        "16",
        "--speculative-dflash-verify-token-num",
        "16",
        "--mem-fraction-static",
        "0.88",
        "--log-requests-level",
        "3",
        "--max-running-requests",
        "10",
    ]
    if with_dynamic_verify_config:
        argv.extend(
            [
                "--dynamic-speculative-dflash-verify-tokens-config",
                DYNAMIC_VERIFY_CONFIG,
            ]
        )
    return argv


def run_engine(*, with_dynamic_verify_config: bool):
    """Start server via `python -m sglang.launch_server` and wait for /health_generate."""
    env = os.environ.copy()
    env["SGLANG_DISABLE_CUDNN_CHECK"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    cmd = _launch_server_argv(with_dynamic_verify_config=with_dynamic_verify_config)
    label = (
        "DFLASH + dynamic verify tokens config"
        if with_dynamic_verify_config
        else "DFLASH baseline (no dynamic verify config)"
    )
    print(f"[repeat] Starting engine ({label})...")
    print(f"command={shlex.join(cmd)}")

    proc = _launch_server_process(cmd, env, None, MODEL_PATH)
    timeout = max(DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, 600)
    ok, err = _wait_for_server_health(proc, BASE_URL, None, timeout)
    if not ok:
        print(f"[repeat] Server failed to start: {err}")
        if proc.poll() is None:
            kill_process_tree(proc.pid)
        sys.exit(1)
    return proc


def run_test() -> bool:
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


def _run_phase(name: str, num_rounds: int, *, with_dynamic_verify_config: bool) -> None:
    for i in range(1, num_rounds + 1):
        print("")
        print(
            f"========== {name} round {i}/{num_rounds} "
            "(start engine -> test -> stop) =========="
        )
        proc = run_engine(with_dynamic_verify_config=with_dynamic_verify_config)
        try:
            if not run_test():
                print(f"[repeat] Test failed on {name} round {i}, aborting.")
                sys.exit(1)
        finally:
            if proc.poll() is None:
                print("[repeat] Stopping engine...")
                kill_process_tree(proc.pid)
                time.sleep(5)
        print(f"[repeat] {name} round {i}/{num_rounds} done.")


def main():
    print(
        f"[repeat] CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES} "
        f"port={ENGINE_PORT} spec_rounds={NUM_SPEC_ROUNDS} "
        f"base_rounds={NUM_BASE_ROUNDS} skip_non_spec={SKIP_NON_SPEC}"
    )

    _run_phase(
        "DFLASH + dynamic verify config",
        NUM_SPEC_ROUNDS,
        with_dynamic_verify_config=True,
    )

    if SKIP_NON_SPEC:
        print("")
        print("[repeat] SKIP_NON_SPEC_BENCHMARK set — skipping baseline phase.")
    else:
        _run_phase(
            "DFLASH baseline (no dynamic verify config)",
            NUM_BASE_ROUNDS,
            with_dynamic_verify_config=False,
        )

    print("")
    print("[repeat] All requested phases completed.")


if __name__ == "__main__":
    main()
