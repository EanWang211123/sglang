"""
aicp_speculative_size_itl_cost_warmup.py
=========================================

Sweep speculative-decoding draft step sizes against a baseline, and measure the
per-(seqlen, batch_size) ITL (inter-token latency) cost ratio.

Workflow (Linux only):

    1. Read a JSON config (`--config config.json`).
    2. Launch the sglang server WITHOUT speculative decoding (= baseline).
       For each (seqlen, batch_size), fire batch_size concurrent streaming
       requests and measure the raw inter-chunk gap (= time between consecutive
       SSE data events, NOT divided by accepted tokens). This is the baseline
       decode-step time.  Record mean_itl_ms → itl_baseline_ms.
    3. For each spec_size in `test_spec_size_list`, restart the server with
       speculative args appended, repeat the sweep.  The inter-chunk gap now
       measures one full draft+verify+draft-extend cycle → itl_spec_ms.
    4. Emit a JSONL file with one row per (input_tokens, batch_size, spec_size):
           {target_seqlen, input_tokens, seqlen, batch_size, spec_size,
            itl_baseline_ms, itl_spec_ms, itl_cost}
       ``input_tokens`` / ``seqlen`` are measured via the local tokenizer
       (chat template included, matching /v1/chat/completions).
       ``target_seqlen`` is the sweep-grid target from config.
       where itl_cost = itl_spec_ms / itl_baseline_ms.
       itl_cost < 1.0 → spec overhead fits within one baseline step (beneficial).
       itl_cost > 1.0 → spec step costs more than baseline (may hurt throughput).

Usage:

    python benchmark/speculative/aicp_speculative_size_itl_cost_warmup.py \
        --config benchmark/speculative/aicp_config.example.json

Stability notes:

    * The base command in the config MUST NOT contain any --speculative-* flag
      and MUST NOT contain --port. The script will reject it otherwise.
    * Server lifecycle uses sglang.utils.{launch_server_cmd, wait_for_server,
      terminate_process} which internally calls kill_process_tree -- safe for
      TP/EP multi-process setups.
    * Prefix-cache bleeding is minimised by always using random prompts;
      the server is started with --disable-radix-cache when possible.
    * Partial results are appended to `<output>.partial.jsonl` after every
      cell, so a crash mid-sweep does not lose data.
    * SIGINT / SIGTERM / atexit all teardown the running server.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import logging
import math
import os
import random
import re
import shlex
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import requests

try:
    from transformers import AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# sglang helpers (Linux-only assumed; these modules import fine without GPU).
# ---------------------------------------------------------------------------
from sglang.utils import (
    launch_server_cmd,
    terminate_process,
    wait_for_http_ready,
)

# ---------------------------------------------------------------------------
# These flags (and their value tokens) are stripped from `base_command` so the
# user can paste the full launch command verbatim; the script re-injects them
# with the correct per-sweep values.
# ---------------------------------------------------------------------------
_STRIP_FLAGS = frozenset(
    {
        "--speculative-algorithm",
        "--speculative-draft-model-path",
        "--speculative-num-steps",
        "--speculative-eagle-topk",
        "--speculative-num-draft-tokens",
        "--port",
    }
)


def normalize_command(cmd: str) -> str:
    """Normalize a shell command string that may contain backslash-newline
    continuations (the standard shell multiline style).

    Strips ``_STRIP_FLAGS`` (``--speculative-*``, ``--port``) and their values
    so the full launch command can be pasted verbatim as a single JSON string.
    """
    # Collapse shell-style line continuations:
    #   optional trailing spaces + backslash + optional trailing spaces + newline
    #   + optional leading indentation on the next line
    cmd = re.sub(r"[ \t]*\\[ \t]*\n[ \t]*", " ", cmd)
    # Collapse bare newlines (JSON \n without backslash).
    cmd = re.sub(r"\n[ \t]*", " ", cmd)

    try:
        tokens = shlex.split(cmd)
    except ValueError as exc:
        raise ValueError(f"Could not parse base_command with shlex: {exc}") from exc

    clean: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _STRIP_FLAGS:
            i += 2  # skip flag + value
            continue
        clean.append(tok)
        i += 1

    return " ".join(clean)


def extract_model_path_from_command(cmd: str) -> Optional[str]:
    """Read ``--model-path`` / ``--model`` from a launch command string."""
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return None
    for i, tok in enumerate(tokens):
        if tok in ("--model-path", "--model") and i + 1 < len(tokens):
            return tokens[i + 1]
    return None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    base_command: str

    seqlen_min: int = 500
    seqlen_max: int = 5000
    seqlen_step: int = 500
    output_len: int = 256

    min_batch_size: int = 1
    max_batch_size: int = 32
    batch_size_step: int = 4
    # If non-empty, use exactly this list (sorted, deduped) and ignore the
    # auto-generated sequence entirely.
    batch_size_capture_list: List[int] = field(default_factory=list)

    test_spec_size_list: List[int] = field(default_factory=lambda: [1, 3, 5, 7, 9])
    # SGLang speculative algorithm name (case-insensitive, matches
    # sglang.srt.speculative.spec_info.SpeculativeAlgorithm):
    #   "EAGLE"      – MTP self-speculation (SGLANG_ENABLE_SPEC_V2=1, no draft
    #                  model path) or classic EAGLE1 (with draft model path)
    #   "EAGLE3"     – EAGLE3 variant (requires draft model path)
    #   "DFLASH"     – DFlash (requires draft model path)
    #   "STANDALONE" – Standalone speculative worker (requires draft model path)
    #   "NGRAM"      – N-gram lookahead (no draft model path needed)
    speculative_algorithm: str = "EAGLE"
    speculative_eagle_topk: int = 1
    # Path to the draft model checkpoint.  Leave empty ("") when using MTP /
    # SGLANG_ENABLE_SPEC_V2=1 (main model reused internally).  Required for
    # EAGLE3, DFLASH, STANDALONE, and EAGLE1 with a separate draft checkpoint.
    speculative_draft_model_path: str = ""
    combo_per_batch_size: int = 3

    host: str = "127.0.0.1"
    # HuggingFace tokenizer directory.  If empty, inferred from --model-path in
    # base_command.  Used to build exact-length prompts and to measure the
    # actual input token count written to JSONL (chat template included).
    tokenizer_path: str = ""
    warmup_seqlen: int = 512
    warmup_output_len: int = 32
    seed: int = 1

    output_path: str = "itl_cost_results.json"

    @classmethod
    def from_file(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Strip comment keys (underscore-prefixed) before validation.
        raw = {k: v for k, v in raw.items() if not k.startswith("_")}
        unknown = set(raw) - {f.name for f in cls.__dataclass_fields__.values()}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        # base_command may be a plain string or a list of strings (see
        # normalize_command). Normalize before construction so validation sees
        # the cleaned, flattened form.
        raw["base_command"] = normalize_command(raw["base_command"])
        cfg = cls(**raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.seqlen_min <= 0 or self.seqlen_step <= 0:
            raise ValueError("seqlen_min and seqlen_step must be > 0")
        if self.seqlen_max < self.seqlen_min:
            raise ValueError("seqlen_max must be >= seqlen_min")
        if self.min_batch_size < 1:
            raise ValueError("min_batch_size must be >= 1")
        if self.max_batch_size < self.min_batch_size:
            raise ValueError("max_batch_size must be >= min_batch_size")
        if self.batch_size_step < 1:
            raise ValueError("batch_size_step must be >= 1")
        if self.batch_size_capture_list and any(
            bs < 1 for bs in self.batch_size_capture_list
        ):
            raise ValueError("batch_size_capture_list values must be >= 1")
        if self.combo_per_batch_size < 1:
            raise ValueError("combo_per_batch_size must be >= 1")
        if any(s < 1 for s in self.test_spec_size_list):
            raise ValueError("test_spec_size_list values must be >= 1")
        if self.output_len <= 1:
            raise ValueError("output_len must be > 1 to measure ITL")
        if not self.tokenizer_path:
            inferred = extract_model_path_from_command(self.base_command)
            if inferred:
                self.tokenizer_path = inferred
        if not self.tokenizer_path:
            raise ValueError(
                "tokenizer_path is required (set explicitly or include "
                "--model-path in base_command)"
            )

    def seqlen_list(self) -> List[int]:
        return list(range(self.seqlen_min, self.seqlen_max + 1, self.seqlen_step))

    def batch_size_list(self) -> List[int]:
        if self.batch_size_capture_list:
            return sorted(set(self.batch_size_capture_list))
        # Auto: min_batch_size as the first anchor, then step-aligned values up
        # to max_batch_size (always included).
        # The step sequence starts at ceil(min/step)*step so that:
        #   min=1,  step=4, max=32 → [1, 4, 8, 12, 16, 20, 24, 28, 32]
        #   min=12, step=4, max=24 → [12, 16, 20, 24]
        #   min=2,  step=4, max=32 → [2, 4, 8, 12, 16, 20, 24, 28, 32]
        first_step = math.ceil(self.min_batch_size / self.batch_size_step) * self.batch_size_step
        multiples = list(range(first_step, self.max_batch_size + 1, self.batch_size_step))
        bslist = sorted(set([self.min_batch_size] + multiples + [self.max_batch_size]))
        return bslist


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------
class ServerHandle:
    """Tiny wrapper around the subprocess returned by launch_server_cmd."""

    def __init__(self, process, port: int, host: str, model_name: str = ""):
        self.process = process
        self.port = port
        self.host = host
        self.model_name = model_name

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def stop(self) -> None:
        if self.process is None:
            return
        try:
            terminate_process(self.process)
        except Exception as exc:  # pragma: no cover -- best-effort teardown
            print(f"[warn] terminate_process failed: {exc}", file=sys.stderr)
        self.process = None


_active_handles: List[ServerHandle] = []


def _kill_all_active(*_args) -> None:
    while _active_handles:
        h = _active_handles.pop()
        h.stop()


atexit.register(_kill_all_active)
for _sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(_sig, lambda s, f: (_kill_all_active(), sys.exit(130)))
    except (ValueError, OSError):
        # Not on the main thread or unsupported signal -- ignore.
        pass


def build_server_command(
    cfg: Config,
    spec_size: Optional[int],
) -> str:
    """Append speculative args (or nothing) to base_command.

    base_command has already been cleaned by normalize_command (no spec flags,
    no --port).  --port is further injected by launch_server_cmd.

    For speculative mode we only inject the three core flags:
      --speculative-algorithm EAGLE
      --speculative-num-steps  <spec_size>
      --speculative-eagle-topk <topk>

    Draft model path injection:
      - speculative_draft_model_path == "" (default, MTP mode):
          --speculative-draft-model-path is NOT added.  SGLang reuses the
          main model internally when SGLANG_ENABLE_SPEC_V2=1.
      - speculative_draft_model_path != "" (EAGLE1 / EAGLE3 / DFlash / …):
          --speculative-draft-model-path <path> is appended.
    """
    cmd = cfg.base_command.strip()
    if spec_size is not None:
        cmd = (
            cmd
            + f" --speculative-algorithm {cfg.speculative_algorithm}"
            + f" --speculative-num-steps {spec_size}"
            + f" --speculative-eagle-topk {cfg.speculative_eagle_topk}"
        )
        if cfg.speculative_draft_model_path:
            cmd += f" --speculative-draft-model-path {cfg.speculative_draft_model_path}"
    return cmd


def start_server(cfg: Config, spec_size: Optional[int]) -> ServerHandle:
    cmd = build_server_command(cfg, spec_size)
    label = "baseline" if spec_size is None else f"spec_size={spec_size}"
    print(f"\n{'=' * 78}\n[server] starting ({label})\n  $ {cmd}\n{'=' * 78}")

    process, port = launch_server_cmd(cmd, host=cfg.host)
    handle = ServerHandle(process=process, port=port, host=cfg.host)
    _active_handles.append(handle)

    try:
        # timeout=None: wait forever, but _raise_if_process_exited() inside the
        # loop will immediately raise if the server process crashes.
        wait_for_http_ready(
            url=f"{handle.base_url}/v1/models",
            timeout=None,
            process=process,
            headers={"Authorization": "Bearer None"},
        )
    except Exception:
        handle.stop()
        if handle in _active_handles:
            _active_handles.remove(handle)
        raise
    # Fetch model name once so we can pass it to /v1/completions later.
    try:
        resp = requests.get(f"{handle.base_url}/v1/models", timeout=10)
        handle.model_name = resp.json()["data"][0]["id"]
    except Exception as exc:
        print(f"[warn] could not fetch model name: {exc}", file=sys.stderr)
        handle.model_name = ""
    print(f"[server] ready at {handle.base_url} ({label})  model={handle.model_name!r}")
    return handle


def stop_server(handle: ServerHandle) -> None:
    handle.stop()
    if handle in _active_handles:
        _active_handles.remove(handle)
    # Give CUDA / NCCL some breathing room; matters for TP setups.
    time.sleep(5)



# ---------------------------------------------------------------------------
# Bench: direct async SSE streaming client.
#
# Why not bench_serving?
#   bench_serving computes ITL as `chunk_gap / num_new_tokens` (time spread
#   evenly across every accepted token in a chunk). For speculative decoding
#   the number of tokens per chunk varies (1..num_steps+1), so the normalized
#   ITL is high-variance and conflates acceptance rate with step latency.
#
# What we actually want:
#   ITL = time between consecutive streaming chunks (raw chunk_gap).
#   - Baseline:  chunk_gap ≈ one decode-forward-pass time (1 token/step)
#   - Spec:      chunk_gap ≈ one draft+verify+draft-extend cycle time
#                           (independent of how many tokens were accepted)
#   This makes itl_cost = itl_spec / itl_baseline a pure measure of overhead.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SSE streaming client – copied verbatim from aicp_random_tokens_benchmark.py
# (RequestFuncInput / RequestFuncOutput / remove_prefix / async_request_openai)
# ---------------------------------------------------------------------------

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    best_of: int = 1
    ignore_eos: bool = False
    temperature: float = 0.0
    stream: bool = True


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0
    itl: List[float] = field(default_factory=list)
    prompt_len: int = 0
    completion_len: int = 0
    error: str = ""


def remove_prefix(s: str, prefix: str) -> str:
    if s.startswith(prefix):
        return s[len(prefix):]
    return s


async def async_request_openai(
    request_func_input: RequestFuncInput,
    api_key: str = "EMPTY",
    enable_thinking: bool = False,
    pbar: Optional[Any] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    connector = aiohttp.TCPConnector(ssl=False) if "https" in api_url else None
    async with aiohttp.ClientSession(
        timeout=AIOHTTP_TIMEOUT, connector=connector
    ) as session:
        payload: Dict[str, Any] = {
            "model": request_func_input.model,
            "temperature": request_func_input.temperature,
            "best_of": request_func_input.best_of,
            "max_tokens": request_func_input.output_len,
            "stream": request_func_input.stream,
            "ignore_eos": request_func_input.ignore_eos,
            "stream_options": {"include_usage": True},
            # Bug4 fix：对 Qwen3 等模型禁用 thinking 模式，否则 TTFT 会极大偏高
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
            "messages": [{"role": "user", "content": request_func_input.prompt}],
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        if (
                            not chunk_bytes
                            or chunk_bytes in {b"\r", b"\n"}
                            or b": ping - " in chunk_bytes
                        ):
                            continue
                        if b"local_rate_limited" in chunk_bytes:
                            break

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data:").strip()
                        if chunk == "[DONE]":
                            output.latency = time.perf_counter() - st
                            output.success = True
                            # Bug3 fix：若 usage chunk 未携带 completion_tokens，用 ITL 数量兜底
                            if output.completion_len == 0 and output.itl:
                                output.completion_len = len(output.itl) + 1
                        else:
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            if data.get("status", {}).get("code") == 500:
                                output.error = f"{response.reason} {chunk}".strip()
                                output.success = False
                                break

                            choices = data.get("choices", [])
                            if not choices and data.get("usage"):
                                output.prompt_len = data["usage"]["prompt_tokens"]
                                output.completion_len = data["usage"]["completion_tokens"]
                                continue

                            if not choices:
                                continue

                            delta = choices[0].get("delta", {})
                            if "content" in delta:
                                # Bug1 fix：SGLang 会发出 content="" 的空 delta，必须跳过；
                                # 否则会错误触发 TTFT 或产生接近 0ms 的虚假 ITL 条目。
                                if not delta.get("content"):
                                    continue
                                if output.ttft == 0.0:
                                    output.ttft = time.perf_counter() - st
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)
                                output.generated_text += delta["content"]
                                # Bug2 fix：most_recent_timestamp 只在有效非空 content 时更新，
                                # 与 test.py 行为一致；放在这里而非外层，避免 usage/空 chunk 干扰。
                                most_recent_timestamp = timestamp
                else:
                    error_message = "".join(
                        [c.decode("utf-8").strip() async for c in response.content]
                    )
                    output.error = f"{response.reason} {error_message}".strip()
                    output.success = False
        except Exception:
            output.success = False
            output.error = "".join(traceback.format_exception(*sys.exc_info()))
            logging.error("Request failed: %s", output.error)

    if pbar is not None:
        pbar.update(1)
    return output


# ---------------------------------------------------------------------------
# Prompt generator (tokenizer-backed)
# ---------------------------------------------------------------------------

def build_exact_prompt_tokens(
    tokenizer: Any,
    num_tokens: int,
    rng: random.Random,
) -> Tuple[str, int]:
    """Build user-message text with exactly ``num_tokens`` raw tokens."""
    if num_tokens <= 0:
        raise ValueError("num_tokens must be > 0")
    ids: List[int] = []
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    for _ in range(10000):
        while len(ids) < num_tokens:
            tid = rng.randint(0, tokenizer.vocab_size - 1)
            if tid in special:
                continue
            ids.append(tid)
        ids = ids[:num_tokens]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        enc = tokenizer.encode(text, add_special_tokens=False)
        if len(enc) == num_tokens:
            return text, num_tokens
        if len(enc) > num_tokens:
            ids = enc[:num_tokens]
        else:
            ids = enc
    raise RuntimeError(
        f"无法构造恰好 {num_tokens} 个 token 的 prompt，请换 seed 或 tokenizer。"
    )


def count_chat_prompt_tokens(tokenizer: Any, prompt_text: str) -> int:
    """Token count for /v1/chat/completions (includes chat template overhead)."""
    messages = [{"role": "user", "content": prompt_text}]
    try:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )
    except TypeError:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    # Some tokenizers return tensors / BatchEncoding instead of list[int].
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    elif hasattr(ids, "shape"):
        ids = ids.flatten().tolist()
    return len(ids)


def resolve_measured_input_tokens(
    *results: Optional[Dict[str, Any]],
) -> Optional[int]:
    """Pick authoritative input length for JSONL (never the sweep-grid target).

    Priority: server ``usage.prompt_tokens`` mean > bench ``input_tokens``.
    """
    for res in results:
        if not isinstance(res, dict):
            continue
        for key in ("mean_server_prompt_tokens", "input_tokens"):
            val = res.get(key)
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    continue
    return None


def make_prompt_batch(
    tokenizer: Any,
    target_seqlen: int,
    num_prompts: int,
    seed: int,
) -> List[str]:
    """Return exact-length user prompts for one benchmark cell."""
    rng = random.Random(seed)
    return [
        build_exact_prompt_tokens(tokenizer, target_seqlen, rng)[0]
        for _ in range(num_prompts)
    ]


# ---------------------------------------------------------------------------
# Async benchmark runner (uses async_request_openai, same as random_tokens_benchmark)
# ---------------------------------------------------------------------------

async def _async_bench(
    host: str,
    port: int,
    model: str,
    prompt_texts: List[str],
    prompt_lens: List[int],
    output_len: int,
    batch_size: int,
    api_key: str = "EMPTY",
) -> Dict[str, Any]:
    """Run all prompts concurrently (up to `batch_size` at a time) using
    async_request_openai, then aggregate ITL from successful requests.

    Returns dict with `mean_itl_ms`, `median_itl_ms`, `p95_itl_ms`, `num_chunks`,
    and optional `mean_server_prompt_tokens` from usage chunks.
    """
    url = f"http://{host}:{port}/v1/chat/completions"
    semaphore = asyncio.Semaphore(batch_size)
    all_itls: List[float] = []
    server_prompt_tokens: List[int] = []

    async def _one(prompt_text: str, prompt_len: int) -> RequestFuncOutput:
        inp = RequestFuncInput(
            prompt=prompt_text,
            api_url=url,
            prompt_len=prompt_len,
            output_len=output_len,
            model=model,
            ignore_eos=True,
            temperature=0.0,
        )
        async with semaphore:
            return await async_request_openai(inp, api_key=api_key)

    outputs = await asyncio.gather(
        *[_one(t, pl) for t, pl in zip(prompt_texts, prompt_lens)]
    )
    for out in outputs:
        if out.success:
            all_itls.extend(out.itl)
            if out.prompt_len > 0:
                server_prompt_tokens.append(out.prompt_len)

    if not all_itls:
        raise RuntimeError("No valid ITL samples collected (all requests failed?)")

    arr = np.array(all_itls) * 1000.0  # → ms
    result: Dict[str, Any] = {
        "mean_itl_ms": float(np.mean(arr)),
        "median_itl_ms": float(np.median(arr)),
        "p95_itl_ms": float(np.percentile(arr, 95)),
        "num_chunks": len(all_itls),
    }
    if server_prompt_tokens:
        result["mean_server_prompt_tokens"] = int(round(np.mean(server_prompt_tokens)))
    return result


def run_one_bench(
    cfg: Config,
    handle: ServerHandle,
    tokenizer: Any,
    target_seqlen: int,
    batch_size: int,
    output_len: int,
    num_prompts: int,
    *,
    label: str,
) -> Dict[str, Any]:
    """Generate `num_prompts` exact-length prompts, stream them in concurrent
    batches of `batch_size`, and return ITL statistics plus ``input_tokens``.
    """
    prompt_texts = make_prompt_batch(
        tokenizer,
        target_seqlen=target_seqlen,
        num_prompts=num_prompts,
        seed=cfg.seed,
    )
    prompt_lens = [count_chat_prompt_tokens(tokenizer, t) for t in prompt_texts]
    tokenizer_local_tokens = int(round(float(np.mean(prompt_lens))))
    result = asyncio.run(
        _async_bench(
            host=handle.host,
            port=handle.port,
            model=handle.model_name,
            prompt_texts=prompt_texts,
            prompt_lens=prompt_lens,
            output_len=output_len,
            batch_size=batch_size,
        )
    )
    result["tokenizer_local_tokens"] = tokenizer_local_tokens
    result["target_seqlen"] = target_seqlen
    # Prefer server-side prompt_tokens (same counter SGLang uses at runtime).
    server_tokens = result.get("mean_server_prompt_tokens")
    result["input_tokens"] = (
        int(server_tokens) if server_tokens is not None else tokenizer_local_tokens
    )
    print(
        f"  → input_tokens={result['input_tokens']} "
        f"(server={server_tokens}, local={tokenizer_local_tokens}, "
        f"target={target_seqlen})  "
        f"mean_itl={result['mean_itl_ms']:.2f}ms  "
        f"median={result['median_itl_ms']:.2f}ms  "
        f"p95={result['p95_itl_ms']:.2f}ms  "
        f"chunks={result['num_chunks']}",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# Sweep: fix one server, walk (seqlen, batch_size) grid.
# ---------------------------------------------------------------------------
def sweep_one_server(
    cfg: Config,
    handle: ServerHandle,
    tokenizer: Any,
    spec_size: Optional[int],
    partial_path: Path,
    global_step: List[int],
    global_total: int,
) -> Dict[tuple, Dict[str, Any]]:
    """Walk (seqlen, batch_size) grid for the current server.

    Returns {(seqlen, bs): bench_result_dict}.
    global_step is a single-element list used as a mutable counter shared
    across all sweep calls so progress shows overall position.
    """
    label = "baseline" if spec_size is None else f"spec_size={spec_size}"
    seqlens = cfg.seqlen_list()
    bslist = cfg.batch_size_list()
    local_total = len(seqlens) * len(bslist)

    # Server-level warmup: trigger CUDA graph capture / weight loading.
    print(
        f"\n[warmup] 1 short request to warm the server ({label}) …",
        flush=True,
    )
    try:
        run_one_bench(
            cfg=cfg,
            handle=handle,
            tokenizer=tokenizer,
            target_seqlen=cfg.warmup_seqlen,
            batch_size=1,
            output_len=cfg.warmup_output_len,
            num_prompts=1,
            label=f"{label} (warmup)",
        )
    except Exception as exc:
        print(f"[warn] warmup failed (continuing): {exc}", file=sys.stderr)

    print(f"[sweep] {label}  cells={local_total}  "
          f"overall={global_step[0]+1}–{global_step[0]+local_total}/{global_total}",
          flush=True)

    out: Dict[tuple, Dict[str, Any]] = {}
    local_step = 0
    for seqlen in seqlens:
        for bs in bslist:
            local_step += 1
            global_step[0] += 1
            num_prompts = bs * cfg.combo_per_batch_size
            print(
                f"\n{'─' * 72}\n"
                f"[progress] overall {global_step[0]}/{global_total}  "
                f"local {local_step}/{local_total}  |  "
                f"{label}  seqlen={seqlen}  bs={bs}  "
                f"num_prompts={num_prompts}",
                flush=True,
            )
            try:
                res = run_one_bench(
                    cfg=cfg,
                    handle=handle,
                    tokenizer=tokenizer,
                    target_seqlen=seqlen,
                    batch_size=bs,
                    output_len=cfg.output_len,
                    num_prompts=num_prompts,
                    label=label,
                )
            except Exception as exc:
                print(
                    f"[error] {label} seqlen={seqlen} bs={bs} failed: {exc}",
                    file=sys.stderr,
                )
                traceback.print_exc()
                res = {"error": str(exc)}

            measured = (
                resolve_measured_input_tokens(res) if isinstance(res, dict) else None
            )
            out[(seqlen, bs)] = res
            _append_partial(
                partial_path,
                {
                    "spec_size": spec_size,
                    "target_seqlen": seqlen,
                    "seqlen": measured,
                    "input_tokens": measured,
                    "batch_size": bs,
                    "result": res,
                },
            )

    return out


def _append_partial(path: Path, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Final aggregation
# ---------------------------------------------------------------------------
def _safe_get(res: Dict[str, Any], key: str) -> Optional[float]:
    if not isinstance(res, dict):
        return None
    val = res.get(key)
    if val is None:
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v


def aggregate(
    cfg: Config,
    baseline: Dict[tuple, Dict[str, Any]],
    spec_results: Dict[int, Dict[tuple, Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    # `mean_itl_ms` here is the raw mean chunk gap (NOT divided by tokens/chunk).
    # For baseline: chunk_gap ≈ one decode forward-pass → directly comparable.
    # For spec:     chunk_gap ≈ one draft+verify+extend cycle.
    # itl_cost = spec_step_time / baseline_step_time → < 1 means spec is faster.
    rows: List[Dict[str, Any]] = []
    for spec_size, spec_grid in spec_results.items():
        for (seqlen, bs), spec_res in spec_grid.items():
            base_res = baseline.get((seqlen, bs), {})
            base_itl = _safe_get(base_res, "mean_itl_ms")
            spec_itl = _safe_get(spec_res, "mean_itl_ms")
            cost = (
                spec_itl / base_itl
                if (base_itl is not None and spec_itl is not None and base_itl > 0)
                else None
            )
            measured = resolve_measured_input_tokens(base_res, spec_res)
            if measured is None:
                print(
                    f"[warn] skip row: no measured input_tokens for "
                    f"target_seqlen={seqlen} bs={bs} spec_size={spec_size}",
                    file=sys.stderr,
                )
                continue
            server_pt = None
            local_pt = None
            for res in (base_res, spec_res):
                if not isinstance(res, dict):
                    continue
                if server_pt is None and res.get("mean_server_prompt_tokens") is not None:
                    server_pt = int(res["mean_server_prompt_tokens"])
                if local_pt is None and res.get("tokenizer_local_tokens") is not None:
                    local_pt = int(res["tokenizer_local_tokens"])
            rows.append(
                {
                    # ``seqlen`` = runtime prefill prompt_tokens (server usage preferred).
                    "seqlen": measured,
                    "input_tokens": measured,
                    "target_seqlen": seqlen,
                    "server_prompt_tokens": server_pt,
                    "tokenizer_local_tokens": local_pt,
                    "batch_size": bs,
                    "spec_size": spec_size,
                    "itl_baseline_ms": base_itl,
                    "itl_spec_ms": spec_itl,
                    "itl_cost": cost,
                }
            )
    rows.sort(
        key=lambda r: (
            r["spec_size"],
            r.get("input_tokens") or r.get("target_seqlen") or 0,
            r["batch_size"],
        )
    )
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep speculative-decoding step sizes vs baseline and produce "
            "per-(seqlen, batch_size, spec_size) ITL cost ratios."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config.json (see aicp_config.example.json).",
    )
    args = parser.parse_args()

    cfg = Config.from_file(args.config)

    if not _TRANSFORMERS_AVAILABLE:
        print(
            "[error] transformers is required for tokenizer-backed prompt lengths. "
            "Install with: pip install transformers",
            file=sys.stderr,
        )
        return 1
    print(f"[plan] tokenizer_path   = {cfg.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_path, trust_remote_code=True
    )

    output_path = Path(cfg.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial.jsonl")
    if partial_path.exists():
        partial_path.unlink()

    cells_per_server = len(cfg.seqlen_list()) * len(cfg.batch_size_list())
    num_servers = 1 + len(cfg.test_spec_size_list)   # baseline + each spec_size
    global_total = cells_per_server * num_servers
    global_step: List[int] = [0]   # mutable counter shared across sweeps

    print("[plan] seqlen_list    =", cfg.seqlen_list())
    print("[plan] batch_size_list=", cfg.batch_size_list())
    print("[plan] spec_size_list =", cfg.test_spec_size_list)
    print(f"[plan] cells/server   = {cells_per_server}")
    print(f"[plan] total cells    = {global_total}  ({num_servers} servers)")
    print(f"[plan] output_path    = {output_path}")
    print(f"[plan] partial_path   = {partial_path}")

    handle: Optional[ServerHandle] = None

    # ---- Baseline ----------------------------------------------------------
    baseline: Dict[tuple, Dict[str, Any]] = {}
    try:
        handle = start_server(cfg, spec_size=None)
        baseline = sweep_one_server(
            cfg, handle, tokenizer, spec_size=None,
            partial_path=partial_path,
            global_step=global_step, global_total=global_total,
        )
    finally:
        if handle is not None:
            stop_server(handle)
            handle = None

    # ---- Spec sizes --------------------------------------------------------
    spec_results: Dict[int, Dict[tuple, Dict[str, Any]]] = {}
    for spec_size in cfg.test_spec_size_list:
        try:
            handle = start_server(cfg, spec_size=spec_size)
            spec_results[spec_size] = sweep_one_server(
                cfg, handle, tokenizer, spec_size=spec_size,
                partial_path=partial_path,
                global_step=global_step, global_total=global_total,
            )
        except Exception as exc:
            print(
                f"[error] spec_size={spec_size} sweep failed: {exc}", file=sys.stderr
            )
            traceback.print_exc()
            spec_results[spec_size] = {}
        finally:
            if handle is not None:
                stop_server(handle)
                handle = None

    # ---- Aggregate + dump (JSON Lines) -------------------------------------
    rows = aggregate(cfg, baseline, spec_results)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n[done] wrote {len(rows)} rows -> {output_path}")
    print(f"[done] partial log    -> {partial_path}")
    if rows:
        sample = rows[0]
        print(
            "[done] sample row keys:",
            sorted(sample.keys()),
            "| seqlen=",
            sample.get("seqlen"),
            "target_seqlen=",
            sample.get("target_seqlen"),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
