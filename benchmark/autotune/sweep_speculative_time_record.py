import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import cycle, islice
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm.auto import tqdm


PROMPT_KEYS = ("instruction", "prompt", "input", "question", "text")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep speculative_num_steps by repeatedly launching SGLang server, "
            "running dynamic concurrency benchmarks, and collecting timing JSONs."
        )
    )
    parser.add_argument(
        "--speculative-num-steps",
        type=int,
        nargs="+",
        required=True,
        help="Candidate speculative_num_steps values to sweep.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to a JSON/JSONL file used to build prompts.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        nargs="+",
        required=True,
        help="Prompt counts for each benchmark case.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        required=True,
        help="Concurrency for each benchmark case. Must match --num-prompts length.",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=256,
        help="max_new_tokens used for each request.",
    )
    parser.add_argument(
        "--disable-ignore-eos",
        action="store_true",
        help="Disable ignore_eos. By default ignore_eos is enabled.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host if not passed through forwarded server args.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="Server port if not passed through forwarded server args.",
    )
    parser.add_argument(
        "--server-timeout",
        type=int,
        default=600,
        help="Server startup timeout in seconds.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=600,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--server-env",
        action="append",
        default=[],
        help="Extra environment variables for the server in KEY=VALUE format.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="spec_step_sweep_results",
        help="Directory for benchmark outputs.",
    )
    parser.add_argument(
        "--spec-timing-root",
        type=str,
        default=None,
        help="Root directory for SPEC_TIMING_WARMUP_SIM_RESULTS. Defaults to <output-dir>/timing.",
    )
    parser.add_argument(
        "--save-server-logs",
        action="store_true",
        help="Save server logs to <output-dir>/server_logs.",
    )
    parser.add_argument(
        "--launch-module",
        type=str,
        default="sglang.launch_server",
        help="Python module used to launch the server.",
    )
    parser.add_argument(
        "server_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the server after '--'.",
    )
    args = parser.parse_args()

    if len(args.num_prompts) != len(args.concurrency):
        parser.error("--num-prompts and --concurrency must have the same length.")
    if args.output_tokens < 1:
        parser.error("--output-tokens must be >= 1.")
    if any(x < 1 for x in args.speculative_num_steps):
        parser.error("--speculative-num-steps values must be >= 1.")

    if args.server_args and args.server_args[0] == "--":
        args.server_args = args.server_args[1:]
    if not args.server_args:
        parser.error("Missing forwarded server args after '--'.")
    if has_flag(args.server_args, "--speculative-num-steps"):
        parser.error("Do not pass --speculative-num-steps in forwarded server args.")

    return args


def has_flag(args_list: List[str], flag_name: str) -> bool:
    return flag_name in args_list


def get_flag_value(args_list: List[str], flag_name: str) -> Optional[str]:
    for index, value in enumerate(args_list):
        if value == flag_name and index + 1 < len(args_list):
            return args_list[index + 1]
    return None


def parse_env_list(env_list: List[str]) -> Dict[str, str]:
    parsed = {}
    for item in env_list:
        if "=" not in item:
            raise ValueError(f"Invalid --server-env entry: {item}")
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def load_prompt_records(dataset_path: str) -> List[dict]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            return data["data"]
        return [data]
    raise ValueError(f"Unsupported dataset format in {dataset_path}")


def extract_prompt(record: dict) -> str:
    for key in PROMPT_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    convs = record.get("conversations", record.get("conversation"))
    if isinstance(convs, list):
        for turn in convs:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role", "")).lower()
            if role in ("assistant", "gpt"):
                continue
            for key in ("content", "value"):
                value = turn.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()

    raise ValueError(f"Unable to extract prompt from record: {record}")


def build_prompt_pool(dataset_path: str, required_count: int) -> List[str]:
    prompts = [extract_prompt(record) for record in load_prompt_records(dataset_path)]
    prompts = [prompt for prompt in prompts if prompt]
    if not prompts:
        raise ValueError(f"No usable prompts found in {dataset_path}")
    return list(islice(cycle(prompts), required_count))


def make_request(
    base_url: str,
    prompt: str,
    output_tokens: int,
    ignore_eos: bool,
    timeout: int,
) -> dict:
    start = time.perf_counter()
    try:
        response = requests.post(
            f"{base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": output_tokens,
                    "ignore_eos": ignore_eos,
                },
                "return_logprob": False,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        meta_info = data.get("meta_info", {})
        return {
            "success": True,
            "latency": time.perf_counter() - start,
            "completion_tokens": int(meta_info.get("completion_tokens", 0)),
        }
    except Exception as exc:
        return {
            "success": False,
            "latency": time.perf_counter() - start,
            "error": str(exc),
            "completion_tokens": 0,
        }


def run_dynamic_benchmark(
    base_url: str,
    prompts: List[str],
    num_prompts: int,
    concurrency: int,
    output_tokens: int,
    ignore_eos: bool,
    request_timeout: int,
    progress_desc: str,
) -> dict:
    start = time.perf_counter()
    sent = 0
    results = []

    with tqdm(total=num_prompts, desc=progress_desc, leave=True) as pbar:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            inflight = set()

            def submit_one(prompt: str):
                return executor.submit(
                    make_request,
                    base_url,
                    prompt,
                    output_tokens,
                    ignore_eos,
                    request_timeout,
                )

            while sent < min(concurrency, num_prompts):
                inflight.add(submit_one(prompts[sent]))
                sent += 1

            while inflight:
                done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                for future in done:
                    results.append(future.result())
                    pbar.update(1)
                    if sent < num_prompts:
                        inflight.add(submit_one(prompts[sent]))
                        sent += 1

    elapsed_s = time.perf_counter() - start
    successes = [item for item in results if item["success"]]
    total_completion_tokens = sum(item["completion_tokens"] for item in successes)
    avg_latency_s = (
        sum(item["latency"] for item in successes) / len(successes) if successes else None
    )

    return {
        "num_prompts": num_prompts,
        "concurrency": concurrency,
        "elapsed_s": round(elapsed_s, 4),
        "success_count": len(successes),
        "error_count": len(results) - len(successes),
        "success_rate": round(len(successes) / len(results), 4) if results else 0.0,
        "total_completion_tokens": total_completion_tokens,
        "throughput_tok_s": (
            round(total_completion_tokens / elapsed_s, 4) if elapsed_s > 0 else 0.0
        ),
        "avg_latency_s": round(avg_latency_s, 4) if avg_latency_s is not None else None,
        "sample_errors": [
            item["error"] for item in results if not item["success"]
        ][:3],
    }


def wait_for_server_ready(
    proc: subprocess.Popen,
    base_url: str,
    timeout_s: int,
) -> None:
    start = time.perf_counter()
    session = requests.Session()
    try:
        while time.perf_counter() - start < timeout_s:
            if proc.poll() is not None:
                raise RuntimeError(f"Server exited early with code {proc.returncode}")

            try:
                response = session.get(f"{base_url}/health_generate", timeout=5)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
    finally:
        session.close()

    raise TimeoutError(f"Server did not become ready within {timeout_s}s")


def terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()

    output_thread = getattr(proc, "_output_thread", None)
    if output_thread is not None:
        output_thread.join(timeout=2)

    log_file = getattr(proc, "_log_file", None)
    if log_file is not None and not log_file.closed:
        log_file.close()


def stream_process_output(proc: subprocess.Popen, log_file) -> None:
    try:
        if proc.stdout is None:
            return

        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if log_file is not None:
                log_file.write(line)
                log_file.flush()
    finally:
        if proc.stdout is not None:
            proc.stdout.close()


def launch_server(
    args,
    step: int,
    timing_dir: Path,
    log_path: Optional[Path],
) -> subprocess.Popen:
    forwarded_args = list(args.server_args)
    if not has_flag(forwarded_args, "--host"):
        forwarded_args.extend(["--host", args.host])
    if not has_flag(forwarded_args, "--port"):
        forwarded_args.extend(["--port", str(args.port)])
    forwarded_args.extend(["--speculative-num-steps", str(step)])

    command = [sys.executable, "-m", args.launch_module, *forwarded_args]
    env = os.environ.copy()
    env["SPEC_TIMING_WARMUP_SIM_RESULTS"] = str(timing_dir)
    env["PYTHONUNBUFFERED"] = "1"
    env.update(parse_env_list(args.server_env))

    stdout_target = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_target = log_path.open("w", encoding="utf-8")

    kwargs = {
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
    }
    if os.name != "nt":
        kwargs["preexec_fn"] = os.setsid

    print(f"\n=== Launch step={step} ===")
    print("Command:", " ".join(command))
    proc = subprocess.Popen(command, **kwargs)
    output_thread = threading.Thread(
        target=stream_process_output,
        args=(proc, stdout_target),
        daemon=True,
    )
    output_thread.start()
    proc._output_thread = output_thread
    proc._log_file = stdout_target
    return proc


def read_timing_jsons(timing_dir: Path) -> Dict[str, list]:
    result = {}
    for stage in ("draft", "verify", "draft_extend"):
        path = timing_dir / f"{stage}_time.json"
        if path.exists():
            result[stage] = json.loads(path.read_text(encoding="utf-8"))
        else:
            result[stage] = []
    return result


def write_results(output_dir: Path, results: List[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "num_prompts",
                "concurrency",
                "elapsed_s",
                "success_count",
                "error_count",
                "success_rate",
                "total_completion_tokens",
                "throughput_tok_s",
                "avg_latency_s",
            ],
        )
        writer.writeheader()
        for step_result in results:
            for case_result in step_result["cases"]:
                writer.writerow(
                    {
                        "step": step_result["step"],
                        **{
                            key: case_result.get(key)
                            for key in writer.fieldnames
                            if key != "step"
                        },
                    }
                )


def format_markdown_table(title: str, row_name: str, col_names: List[str], rows: List[List[str]]):
    lines = [f"## {title}"]
    header = [row_name, *col_names]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def print_markdown_table(title: str, row_name: str, col_names: List[str], rows: List[List[str]]):
    print()
    print(format_markdown_table(title, row_name, col_names, rows))


def format_float(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def get_expected_bucket_value(stage: str, step: int) -> int:
    if stage == "draft":
        return step
    return step + 1


def build_throughput_table(results: List[dict]):
    step_values = [result["step"] for result in results]
    batch_sizes = sorted(
        {
            case["concurrency"]
            for result in results
            for case in result["cases"]
        }
    )
    step_to_case = {
        result["step"]: {
            case["concurrency"]: case for case in result["cases"]
        }
        for result in results
    }
    rows = []
    for batch_size in batch_sizes:
        row = [str(batch_size)]
        for step in step_values:
            case = step_to_case.get(step, {}).get(batch_size)
            row.append(
                format_float(case.get("throughput_tok_s"), digits=2) if case else "-"
            )
        rows.append(row)
    return [str(step) for step in step_values], rows


def build_stage_time_table(results: List[dict], stage: str):
    step_values = [result["step"] for result in results]
    batch_sizes = sorted(
        {
            int(entry["batch-size"])
            for result in results
            for entry in result.get("timing_json", {}).get(stage, [])
        }
    )
    stage_maps = {}
    for result in results:
        entry_map = {}
        expected_bucket_value = get_expected_bucket_value(stage, result["step"])
        for entry in result.get("timing_json", {}).get(stage, []):
            bucket_key = "steps" if stage == "draft" else "query-len"
            if int(entry.get(bucket_key, -1)) != expected_bucket_value:
                continue
            entry_map[int(entry["batch-size"])] = float(entry["time"])
        stage_maps[result["step"]] = entry_map

    rows = []
    for batch_size in batch_sizes:
        row = [str(batch_size)]
        for step in step_values:
            row.append(format_float(stage_maps.get(step, {}).get(batch_size), digits=3))
        rows.append(row)
    return [str(step) for step in step_values], rows


def emit_summary_tables(output_dir: Path, results: List[dict]) -> None:
    if not results:
        return

    sections = []

    throughput_cols, throughput_rows = build_throughput_table(results)
    print_markdown_table("表1 吞吐变化表", "batch-size", throughput_cols, throughput_rows)
    sections.append(
        format_markdown_table(
            "表1 吞吐变化表", "batch-size", throughput_cols, throughput_rows
        )
    )

    stage_titles = {
        "draft": "表2 draft 时间变化表",
        "verify": "表2 verify 时间变化表",
        "draft_extend": "表2 draft-extend 时间变化表",
    }
    for stage in ("draft", "verify", "draft_extend"):
        cols, rows = build_stage_time_table(results, stage)
        print_markdown_table(stage_titles[stage], "batch-size", cols, rows)
        sections.append(format_markdown_table(stage_titles[stage], "batch-size", cols, rows))

    (output_dir / "summary_tables.md").write_text(
        "\n\n".join(sections) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    timing_root = (
        Path(args.spec_timing_root).resolve()
        if args.spec_timing_root
        else output_dir / "timing"
    )
    log_root = output_dir / "server_logs"

    forwarded_port = get_flag_value(args.server_args, "--port")
    forwarded_host = get_flag_value(args.server_args, "--host")
    base_url = f"http://{forwarded_host or args.host}:{forwarded_port or args.port}"

    prompt_pool = build_prompt_pool(args.dataset_path, required_count=max(args.num_prompts))
    all_results = []

    for step in args.speculative_num_steps:
        timing_dir = timing_root / f"step_{step}"
        if timing_dir.exists():
            shutil.rmtree(timing_dir)
        timing_dir.mkdir(parents=True, exist_ok=True)

        log_path = log_root / f"step_{step}.log" if args.save_server_logs else None
        proc = launch_server(args, step, timing_dir, log_path)

        try:
            wait_for_server_ready(proc, base_url, args.server_timeout)
            step_result = {"step": step, "cases": []}

            for num_prompts, concurrency in zip(args.num_prompts, args.concurrency):
                case_prompts = prompt_pool[:num_prompts]
                print(
                    f"Running step={step}, num_prompts={num_prompts}, concurrency={concurrency}"
                )
                case_result = run_dynamic_benchmark(
                    base_url=base_url,
                    prompts=case_prompts,
                    num_prompts=num_prompts,
                    concurrency=concurrency,
                    output_tokens=args.output_tokens,
                    ignore_eos=not args.disable_ignore_eos,
                    request_timeout=args.request_timeout,
                    progress_desc=(
                        f"step={step} concurrency={concurrency} prompts={num_prompts}"
                    ),
                )
                print(
                    "  "
                    f"throughput={case_result['throughput_tok_s']} tok/s, "
                    f"avg_latency={case_result['avg_latency_s']} s, "
                    f"success={case_result['success_count']}/{case_result['num_prompts']}"
                )
                step_result["cases"].append(case_result)

            step_result["timing_json"] = read_timing_jsons(timing_dir)
            all_results.append(step_result)
            write_results(output_dir, all_results)
        finally:
            terminate_process(proc)

    emit_summary_tables(output_dir, all_results)
    print("\nFinished. Results written to:", output_dir)


if __name__ == "__main__":
    main()
