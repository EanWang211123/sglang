"""
aicp_speculative_size_itl_cost_warmup.py
=========================================

Sweep speculative-decoding draft step sizes against a baseline, and measure the
per-(seqlen, batch_size) ITL (inter-token latency) cost ratio.

Workflow (Linux only):

    1. Read a JSON config (`--config config.json`).
    2. Launch the sglang server WITHOUT speculative decoding (= baseline).
       For each (seqlen, batch_size), call `sglang.bench_serving` with the
       `random` dataset, max-concurrency = batch_size, num-prompts =
       batch_size * combo_per_batch_size, ignore_eos=True, stream=True.
       Record `mean_itl_ms` -> itl_baseline.
    3. For each spec_size in `test_spec_size_list`, restart the server with
       the speculative args appended, repeat the sweep -> itl_spec.
    4. Emit a JSON file with one row per (seqlen, batch_size, spec_size):
           {seqlen, batch_size, spec_size,
            itl_baseline_ms, itl_spec_ms, itl_cost}
       where `itl_cost = itl_spec_ms / itl_baseline_ms`.

Usage:

    python benchmark/speculative/aicp_speculative_size_itl_cost_warmup.py \
        --config benchmark/speculative/aicp_config.example.json

Stability notes:

    * The base command in the config MUST NOT contain any --speculative-* flag
      and MUST NOT contain --port. The script will reject it otherwise.
    * Server lifecycle uses sglang.utils.{launch_server_cmd, wait_for_server,
      terminate_process} which internally calls kill_process_tree -- safe for
      TP/EP multi-process setups.
    * After each bench, /flush_cache is called so prefix-cache cannot bleed
      results across (seqlen, bs) cells.
    * Partial results are appended to `<output>.partial.jsonl` after every
      cell, so a crash mid-sweep does not lose data.
    * SIGINT / SIGTERM / atexit all teardown the running server.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    batch_size_step: int = 1

    test_spec_size_list: List[int] = field(default_factory=lambda: [1, 3, 5, 7, 9])
    speculative_eagle_topk: int = 1
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
        if self.min_batch_size < 1 or self.max_batch_size < self.min_batch_size:
            raise ValueError("invalid batch-size range")
        if self.batch_size_step <= 0:
            raise ValueError("batch_size_step must be > 0")
        if self.combo_per_batch_size < 1:
            raise ValueError("combo_per_batch_size must be >= 1")
        if any(s < 1 for s in self.test_spec_size_list):
            raise ValueError("test_spec_size_list values must be >= 1")
        if self.output_len <= 1:
            raise ValueError("output_len must be > 1 to measure ITL")

    def seqlen_list(self) -> List[int]:
        return list(range(self.seqlen_min, self.seqlen_max + 1, self.seqlen_step))

    def batch_size_list(self) -> List[int]:
        return list(
            range(self.min_batch_size, self.max_batch_size + 1, self.batch_size_step)
        )


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------
class ServerHandle:
    """Tiny wrapper around the subprocess returned by launch_server_cmd."""

    def __init__(self, process, port: int, host: str):
        self.process = process
        self.port = port
        self.host = host

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
    no --port). --port is further injected by launch_server_cmd.
    """
    cmd = cfg.base_command.strip()
    if spec_size is not None:
        spec_args = (
            " --speculative-algorithm EAGLE"
            f" --speculative-num-steps {spec_size}"
            f" --speculative-eagle-topk {cfg.speculative_eagle_topk}"
            f" --speculative-num-draft-tokens {spec_size + 1}"
        )
        # Best-effort: if user did not put --speculative-draft-model-path in
        # the base, derive it from --model-path. EAGLE almost always reuses
        # the same model path (matches the user's example).
        if "--speculative-draft-model-path" not in cmd:
            try:
                tokens = shlex.split(cmd)
                model_path = tokens[tokens.index("--model-path") + 1]
                spec_args += f" --speculative-draft-model-path {model_path}"
            except (ValueError, IndexError):
                raise RuntimeError(
                    "Could not infer --speculative-draft-model-path: please "
                    "ensure --model-path is in `base_command`."
                )
        cmd = cmd + spec_args
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
    print(f"[server] ready at {handle.base_url} ({label})")
    return handle


def stop_server(handle: ServerHandle) -> None:
    handle.stop()
    if handle in _active_handles:
        _active_handles.remove(handle)
    # Give CUDA / NCCL some breathing room; matters for TP setups.
    time.sleep(5)


def flush_cache(base_url: str) -> None:
    try:
        requests.post(f"{base_url}/flush_cache", timeout=30)
    except Exception as exc:
        print(f"[warn] /flush_cache failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Bench: subprocess-call into `sglang.bench_serving`. Subprocess isolation is
# deliberate: bench_serving uses module-global `args`, so calling it twice in
# the same process is fragile. A subprocess is ~free relative to the bench
# duration and gives us total clean-state reproducibility.
# ---------------------------------------------------------------------------
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
    """Run a single bench_serving call and return the parsed result dict.

    stdout/stderr of the bench_serving subprocess are NOT captured so both
    the server logs and the bench output appear live on the terminal.
    The only structured output we need is the JSONL result file written by
    bench_serving via --output-file.
    """
    with tempfile.NamedTemporaryFile(
        mode="r",
        suffix=".jsonl",
        prefix="aicp_bench_",
        delete=False,
    ) as tmp:
        out_file = tmp.name

    cmd = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend", "sglang",
        "--host", handle.host,
        "--port", str(handle.port),
        "--dataset-name", "random",
        "--random-input-len", str(seqlen),
        "--random-output-len", str(output_len),
        "--random-range-ratio", "1.0",
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(batch_size),
        "--warmup-requests", "1",
        "--seed", str(cfg.seed),
        "--output-file", out_file,
        "--disable-tqdm",
    ]

    # No capture: server logs + bench output both appear live in the terminal.
    # No timeout: bench_serving exits naturally when requests finish or the
    # server becomes unreachable (connection errors).
    proc = subprocess.run(cmd, env=os.environ.copy())
    if proc.returncode != 0:
        try:
            os.unlink(out_file)
        except OSError:
            pass
        raise RuntimeError(
            f"bench_serving exited with code {proc.returncode} "
            f"for {label} seqlen={seqlen} bs={batch_size}"
        )

    try:
        with open(out_file, "r", encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        if not lines:
            raise RuntimeError(f"bench_serving produced no JSONL output for {label}")
        result = json.loads(lines[-1])
    finally:
        try:
            os.unlink(out_file)
        except OSError:
            pass
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
            flush_cache(handle.base_url)
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
