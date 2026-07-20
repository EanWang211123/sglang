#!/usr/bin/env python3
"""Replay captured NewAPI request logs against an Anthropic-compatible endpoint.

The two JSON files in the repo root are Elasticsearch/NewAPI log exports. This
script extracts `request_body` and POSTs it to `/v1/messages` so you can try to
reproduce garbled or incorrect model responses locally or on a staging server.

Examples:
  # Replay both logs to a local SGLang server
  python replay_request_logs.py

  # Replay one file to a custom endpoint (matches the original log path)
  python replay_request_logs.py \\
    --file 000031_20260718021118414349793Bq8xEpGG.json \\
    --base-url http://127.0.0.1:30000 \\
    --beta

  # Non-streaming replay (easier to inspect full JSON response)
  python replay_request_logs.py --no-stream --file 20260717085444934755845QBsJUv9X.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

import requests

DEFAULT_FILES = [
    "000031_20260718021118414349793Bq8xEpGG.json",
    "20260717085444934755845QBsJUv9X.json",
]


def load_log(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_request_body(log: dict[str, Any]) -> dict[str, Any]:
    if "request_body" not in log:
        raise ValueError(f"{log!r} has no request_body field")
    body = log["request_body"]
    if not isinstance(body, dict):
        raise ValueError("request_body must be a JSON object")
    return body


def build_url(base_url: str, beta: bool) -> str:
    base = base_url.rstrip("/")
    if not base.endswith("/v1/messages"):
        base = f"{base}/v1/messages"

    if not beta:
        return base

    parsed = urlparse(base)
    query = dict(
        x.split("=", 1) for x in parsed.query.split("&") if x
    ) if parsed.query else {}
    query["beta"] = "true"
    return urlunparse(parsed._replace(query=urlencode(query)))


def build_headers(api_key: str | None, extra_headers: list[str]) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream, application/json",
        "anthropic-version": "2023-06-01",
    }
    if api_key:
        headers["x-api-key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"

    for item in extra_headers:
        if ":" not in item:
            raise ValueError(f"Invalid header {item!r}, expected Name: Value")
        name, value = item.split(":", 1)
        headers[name.strip()] = value.strip()
    return headers


def apply_overrides(
    body: dict[str, Any],
    *,
    stream: bool | None,
    model: str | None,
) -> dict[str, Any]:
    if stream is None and model is None:
        return body
    patched = dict(body)
    if stream is not None:
        patched["stream"] = stream
        if not stream:
            patched.pop("stream_options", None)
    if model is not None:
        patched["model"] = model
    return patched


def summarize_request_body(body: dict[str, Any]) -> str:
    messages = body.get("messages") or []
    tools = body.get("tools") or []
    parts = [
        f"model={body.get('model')!r}",
        f"max_tokens={body.get('max_tokens')}",
        f"stream={body.get('stream')}",
        f"messages={len(messages)}",
    ]
    if tools:
        parts.append(f"tools={len(tools)}")
    if "system" in body:
        parts.append("system=present")
    if "thinking" in body:
        parts.append(f"thinking={body.get('thinking')}")
    return ", ".join(parts)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def replay_streaming(response: requests.Response, output_path: Path) -> str:
    chunks: list[str] = []
    print("--- streaming response ---")
    for line in response.iter_lines(decode_unicode=True):
        if line is None:
            continue
        print(line)
        chunks.append(line)
    text = "\n".join(chunks)
    save_text(output_path, text)
    return text


def replay_non_streaming(response: requests.Response, output_path: Path) -> str:
    try:
        payload = response.json()
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    except ValueError:
        text = response.text
    print(text)
    save_text(output_path, text)
    return text


def replay_one(
    *,
    log_path: Path,
    base_url: str,
    api_key: str | None,
    beta: bool,
    stream_override: bool | None,
    model_override: str | None,
    timeout: float,
    output_dir: Path,
    extra_headers: list[str],
    dry_run: bool,
) -> int:
    log = load_log(log_path)
    body = extract_request_body(log)
    body = apply_overrides(body, stream=stream_override, model=model_override)
    stream = bool(body.get("stream", False))

    url = build_url(base_url, beta)
    headers = build_headers(api_key, extra_headers)

    print("=" * 72)
    print(f"file: {log_path.name}")
    if "llm_endpoint" in log:
        print(f"original endpoint: {log.get('llm_endpoint')}")
    if "prompt_tokens" in log:
        print(f"original prompt_tokens: {log.get('prompt_tokens')}")
    print(f"target url: {url}")
    print(f"request: {summarize_request_body(body)}")

    suffix = "sse.txt" if stream else "json"
    output_path = output_dir / f"{log_path.stem}.response.{suffix}"

    if dry_run:
        print(f"[dry-run] would POST to {url}, save to {output_path}")
        return 0

    started = time.perf_counter()
    with requests.post(
        url,
        headers=headers,
        json=body,
        stream=stream,
        timeout=timeout,
    ) as response:
        elapsed = time.perf_counter() - started
        print(f"status: {response.status_code}, elapsed: {elapsed:.2f}s")
        if response.status_code >= 400:
            error_text = response.text
            print(error_text)
            save_text(output_dir / f"{log_path.stem}.response.error.txt", error_text)
            return response.status_code

        if stream:
            replay_streaming(response, output_path)
        else:
            replay_non_streaming(response, output_path)

    print(f"saved response to: {output_path}")
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay captured NewAPI request logs to /v1/messages."
    )
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        help="Log JSON file to replay. Repeatable. Defaults to both root log files.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:30000",
        help="Server base URL or full /v1/messages URL. Default: http://127.0.0.1:30000",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key sent as x-api-key and Authorization: Bearer.",
    )
    parser.add_argument(
        "--beta",
        action="store_true",
        help="Append ?beta=true to /v1/messages (matches one of the original logs).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Force stream=true in request_body.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Force stream=false in request_body.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override request_body.model (useful when local server uses a different model id).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600.0,
        help="Request timeout in seconds. Default: 3600.",
    )
    parser.add_argument(
        "--output-dir",
        default="replay_outputs",
        help="Directory for saved responses. Default: replay_outputs",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="NAME: VALUE",
        help="Extra HTTP header. Repeatable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be sent without making HTTP requests.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    repo_root = Path(__file__).resolve().parent

    if args.stream and args.no_stream:
        print("Use only one of --stream or --no-stream.", file=sys.stderr)
        return 2

    stream_override: bool | None
    if args.stream:
        stream_override = True
    elif args.no_stream:
        stream_override = False
    else:
        stream_override = None

    file_names = args.files or DEFAULT_FILES
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir

    exit_code = 0
    for name in file_names:
        path = Path(name)
        if not path.is_absolute():
            path = repo_root / path
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            exit_code = 1
            continue

        code = replay_one(
            log_path=path,
            base_url=args.base_url,
            api_key=args.api_key,
            beta=args.beta,
            stream_override=stream_override,
            model_override=args.model,
            timeout=args.timeout,
            output_dir=output_dir,
            extra_headers=args.header,
            dry_run=args.dry_run,
        )
        exit_code = max(exit_code, code)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
