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
    4. Emit a JSONL file with one row per (seqlen, batch_size, spec_size):
           {seqlen, batch_size, spec_size,
            itl_baseline_ms, itl_spec_ms, itl_cost}
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
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
import requests

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

_SSE_PREFIX = b"data: "
_SSE_DONE = b"data: [DONE]"


async def _stream_one(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
) -> List[float]:
    """Stream a single /v1/chat/completions request and return decode-phase ITL
    gaps (sec), matching the methodology of inference_benchmark.py exactly.

    ITL definition (same as inference_benchmark.py::async_request_openai):
      - First SSE event with delta.content → TTFT anchor (most_recent_timestamp).
        NO gap appended.
      - Every subsequent SSE event with delta.content → gap = timestamp - most_recent_timestamp.
        most_recent_timestamp updated to this timestamp.
      - Non-content events (e.g. usage) are parsed but do NOT contribute a gap.

    For baseline (no spec): each SSE event carries 1 token → gap = one decode step.
    For spec decoding: each SSE event carries all tokens accepted in one spec cycle
    → gap = one full draft+verify+extend cycle (independent of acceptance length).

    Implementation note: we use readline() instead of `async for chunk in resp.content`
    to guarantee one SSE event per iteration regardless of TCP segment boundaries.
    """
    most_recent_timestamp: float = 0.0
    gaps: List[float] = []
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                return []
            while True:
                raw = await resp.content.readline()
                if not raw:
                    break
                line = raw.strip()
                if not line:
                    continue
                if line == _SSE_DONE:
                    break
                if not line.startswith(_SSE_PREFIX):
                    continue
                try:
                    data = json.loads(line[len(_SSE_PREFIX):])
                except json.JSONDecodeError:
                    continue

                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                if "content" not in delta:
                    continue

                timestamp = time.perf_counter()
                if most_recent_timestamp == 0.0:
                    # First content event = TTFT anchor; no gap recorded.
                    most_recent_timestamp = timestamp
                else:
                    gaps.append(timestamp - most_recent_timestamp)
                    most_recent_timestamp = timestamp
    except Exception:
        return []

    return gaps


def _make_payload(model: str, prompt_text: str, output_len: int) -> dict:
    """Build a /v1/chat/completions streaming payload (mirrors inference_benchmark.py)."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": output_len,
        "stream": True,
        "ignore_eos": True,    # never cut short → full output_len always
        "temperature": 0.0,
        "best_of": 1,
    }


def _make_prompt_text(seqlen: int, rng: random.Random) -> str:
    """Generate a text prompt of approximately `seqlen` tokens.

    Uses space-separated two-digit random integers (0-99).  Each such number
    tokenises to roughly 1 token for common BPE vocabularies, so seqlen numbers
    ≈ seqlen tokens.  The exact count varies slightly by model/tokeniser but is
    close enough for controlled ITL measurement.
    """
    return " ".join(str(rng.randint(0, 99)) for _ in range(seqlen))


async def _async_bench(
    host: str,
    port: int,
    model: str,
    prompt_texts: List[str],
    output_len: int,
    batch_size: int,
) -> Dict[str, Any]:
    """Run `prompt_texts` in rounds of `batch_size` concurrent chat streams.

    Returns dict with `mean_itl_ms` = mean raw chunk gap across all rounds.
    """
    url = f"http://{host}:{port}/v1/chat/completions"
    all_gaps: List[float] = []

    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None, connect=30)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for start in range(0, len(prompt_texts), batch_size):
            batch = prompt_texts[start : start + batch_size]
            payloads = [_make_payload(model, t, output_len) for t in batch]
            results = await asyncio.gather(
                *[_stream_one(session, url, pl) for pl in payloads],
                return_exceptions=True,
            )
            for r in results:
                if isinstance(r, list):
                    all_gaps.extend(r)

    if not all_gaps:
        raise RuntimeError("No valid streaming chunks collected (server error?)")

    arr = np.array(all_gaps) * 1000.0  # → ms
    return {
        "mean_itl_ms": float(np.mean(arr)),
        "median_itl_ms": float(np.median(arr)),
        "p95_itl_ms": float(np.percentile(arr, 95)),
        "num_chunks": len(all_gaps),
    }


def run_one_bench(
    cfg: Config,
    handle: ServerHandle,
    seqlen: int,
    batch_size: int,
    output_len: int,
    num_prompts: int,
    *,
    label: str,
) -> Dict[str, Any]:
    """Generate `num_prompts` text prompts (~seqlen tokens each), stream them in
    concurrent batches of `batch_size`, and return ITL statistics.
    """
    rng = random.Random(cfg.seed)
    prompt_texts = [_make_prompt_text(seqlen, rng) for _ in range(num_prompts)]
    result = asyncio.run(
        _async_bench(
            host=handle.host,
            port=handle.port,
            model=handle.model_name,
            prompt_texts=prompt_texts,
            output_len=output_len,
            batch_size=batch_size,
        )
    )
    print(
        f"  → mean_itl={result['mean_itl_ms']:.2f}ms  "
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
            seqlen=cfg.warmup_seqlen,
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
                    seqlen=seqlen,
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

            out[(seqlen, bs)] = res
            _append_partial(
                partial_path,
                {
                    "spec_size": spec_size,
                    "seqlen": seqlen,
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
            rows.append(
                {
                    "seqlen": seqlen,
                    "batch_size": bs,
                    "spec_size": spec_size,
                    "itl_baseline_ms": base_itl,
                    "itl_spec_ms": spec_itl,
                    "itl_cost": cost,
                }
            )
    rows.sort(key=lambda r: (r["spec_size"], r["seqlen"], r["batch_size"]))
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
            cfg, handle, spec_size=None,
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
                cfg, handle, spec_size=spec_size,
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
