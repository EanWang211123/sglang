# -*- coding: utf-8 -*-
"""
aicp_random_tokens_benchmark.py
================================

基于 ``inference_benchmark.py`` 同类方法的压测脚本：使用 **OpenAI 兼容**
``/v1/chat/completions`` 流式接口，对 **随机 prompt** 测 TTFT / TPOT / ITL / 吞吐等指标。

典型场景：投机解码（speculative）下，用随机 token 分布的输入避免 radix/prefix 命中，
在 **固定并发** 下复现稳定负载。

用法示例：

    python benchmark/speculative/aicp_random_tokens_benchmark.py \\
        --base-url http://127.0.0.1:30000 \\
        --tokenizer-path /path/to/model \\
        --input-len 512 \\
        --output-len 256 \\
        --num-prompts 64 \\
        --max-concurrency 8 \\
        --api-key EMPTY

可选 ``--prompt-mode approx``：不加载 tokenizer，用空格分隔随机数字近似 ``input-len``
个 token（与 ``aicp_speculative_size_itl_cost_warmup._make_prompt_text`` 思路一致）。

默认在 **当前工作目录** 写出 ``{模型名}_{时间戳}.xlsx`` 汇总表；可用 ``--disable-excel`` 关闭。

**Goodput（``--goodput``）**：对每个 **成功且** ``completion_len > 1`` **的单条请求** 单独判断；
仅当该请求的 ttft / tpot / e2el（未在 CLI 中指定则不对该项设限）与该请求的 **decode 吞吐**
同时满足阈值时，该请求计入 good。``goodput_percentage`` = 满足条件的请求数 / 成功请求数；
**不是**先看“整轮平均 TTFT / 平均吞吐”再判一次。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import pandas as pd
import requests
from tqdm.asyncio import tqdm as async_tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

###############################################################################
# 与 inference_benchmark.py 对齐的常量与数据结构
###############################################################################

MILLISECONDS_TO_SECONDS_CONVERSION = 1000
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class BenchmarkMetrics:
    completed: int
    mean_input: int
    mean_output: int
    total_input: int
    total_output: int
    request_throughput: float
    request_goodput: float
    goodput_percentage: float
    output_throughput: float
    mean_output_throughput: float
    total_token_throughput: float
    mean_goodput_ttft: float
    mean_goodput_tpot: float
    mean_goodput_e2el: float
    mean_goodput_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: List[Tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: List[Tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: List[Tuple[float, float]]
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: List[Tuple[float, float]]
    mean_prefilling_throughput: float
    median_prefilling_throughput: float
    std_prefilling_throughput: float
    percentiles_prefilling_throughput: List[Tuple[float, float]]
    mean_decoding_throughput: float
    median_decoding_throughput: float
    std_decoding_throughput: float
    percentiles_decoding_throughput: List[Tuple[float, float]]


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    best_of: int = 1
    logprobs: Optional[int] = None
    multi_modal_content: Optional[dict] = None
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
        return s[len(prefix) :]
    return s


def parse_goodput(slo_pairs: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for slo_pair in slo_pairs:
        slo_name, slo_val = slo_pair.split(":", 1)
        out[slo_name.strip()] = float(slo_val.strip())
    return out


def safe_model_stem_for_filename(model_id: str) -> str:
    """模型名中文件系统非法字符替换为下划线，并限制长度。"""
    name = Path(str(model_id)).name.strip() or "model"
    name = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", name)
    name = name.strip(" ._") or "model"
    return name[:180]


def write_summary_excel(path: Path, row: Dict[str, Any]) -> None:
    """单行指标表写入 ``.xlsx``（需已安装 ``openpyxl``）。"""
    try:
        df = pd.DataFrame([row])
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(path, index=False, engine="openpyxl")
    except ImportError as exc:
        raise RuntimeError(
            "写入 Excel 需要安装 openpyxl：pip install openpyxl"
        ) from exc


def check_goodput_args(slo_args: Optional[List[str]]) -> Dict[str, float]:
    if not slo_args:
        return {}
    VALID_NAMES = ["ttft", "tpot", "e2el", "throughput", "ttft2", "throughput2"]
    cfg = parse_goodput(slo_args)
    for slo_name, slo_val in cfg.items():
        if slo_name not in VALID_NAMES:
            raise ValueError(f"Invalid goodput metric: {slo_name}")
        if slo_val < 0:
            raise ValueError(f"Invalid goodput value: {slo_name}:{slo_val}")
    return cfg


def calculate_metrics(
    outputs: Tuple[RequestFuncOutput, ...],
    dur_s: float,
    selected_percentiles: List[float],
    goodput_config_dict: Dict[str, float],
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    good_completed = 0
    itls: List[float] = []
    tpots: List[float] = []
    all_tpots: List[float] = []
    ttfts: List[float] = []
    e2els: List[float] = []
    out_throughputs: List[float] = []
    prefill_throughputs: List[float] = []
    decode_throughputs: List[float] = []

    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].completion_len
            actual_output_lens.append(output_len)
            total_input += outputs[i].prompt_len

            tpot, throughput = 0, 0
            if output_len > 1:
                tpot = (outputs[i].latency - outputs[i].ttft) / (output_len - 1)
                tpots.append(tpot)
                throughput = output_len / outputs[i].latency
                out_throughputs.append(throughput)
                prefill_throughput = outputs[i].prompt_len / outputs[i].ttft
                prefill_throughputs.append(prefill_throughput)
                decode_throughput = (output_len - 1) / (
                    outputs[i].latency - outputs[i].ttft
                )
                decode_throughputs.append(decode_throughput)

            all_tpots.append(tpot)
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            e2els.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    mean_goodput_ttft = -1.0
    mean_goodput_tpot = -1.0
    mean_goodput_e2el = -1.0
    mean_goodput_throughput = -1.0

    if goodput_config_dict:
        valid_metrics: Dict[str, List[float]] = {}
        slo_values: Dict[str, float] = {}
        valid_metrics["ttft"] = ttfts
        valid_metrics["tpot"] = all_tpots
        valid_metrics["e2el"] = e2els
        valid_metrics["throughput"] = decode_throughputs

        slo_values["ttft"] = (
            goodput_config_dict["ttft"] / MILLISECONDS_TO_SECONDS_CONVERSION
            if "ttft" in goodput_config_dict
            else float("inf")
        )
        slo_values["tpot"] = (
            goodput_config_dict["tpot"] / MILLISECONDS_TO_SECONDS_CONVERSION
            if "tpot" in goodput_config_dict
            else float("inf")
        )
        slo_values["e2el"] = (
            goodput_config_dict["e2el"] / MILLISECONDS_TO_SECONDS_CONVERSION
            if "e2el" in goodput_config_dict
            else float("inf")
        )
        slo_values["throughput"] = (
            goodput_config_dict["throughput"]
            if "throughput" in goodput_config_dict
            else float("-inf")
        )

        good_ttfts = np.array([])
        good_tpots = np.array([])
        good_e2els = np.array([])
        good_throughputs = np.array([])
        # 注意：原版 inference_benchmark 此处 zip 的第三列误写为 tpot；此处改为 e2el
        for _ttft, _tpot, _e2el, _throughput in zip(
            valid_metrics["ttft"],
            valid_metrics["tpot"],
            valid_metrics["e2el"],
            valid_metrics["throughput"],
        ):
            if (
                _ttft <= slo_values["ttft"]
                and _tpot <= slo_values["tpot"]
                and _e2el <= slo_values["e2el"]
                and _throughput >= slo_values["throughput"]
            ):
                good_ttfts = np.append(good_ttfts, _ttft)
                good_tpots = np.append(good_tpots, _tpot)
                good_e2els = np.append(good_e2els, _e2el)
                good_throughputs = np.append(good_throughputs, _throughput)

        good_completed = len(good_ttfts)
        if good_completed > 0:
            mean_goodput_ttft = float(np.mean(good_ttfts))
            mean_goodput_tpot = float(np.mean(good_tpots))
            mean_goodput_e2el = float(np.mean(good_e2els))
            mean_goodput_throughput = float(np.mean(good_throughputs))
        else:
            mean_goodput_ttft = np.nan
            mean_goodput_tpot = np.nan
            mean_goodput_e2el = np.nan
            mean_goodput_throughput = np.nan

    if completed == 0:
        logging.warning(
            "All requests failed. Check server URL, model name, and GPU health."
        )
        completed = int(float("-inf"))

    metrics = BenchmarkMetrics(
        completed=completed,
        mean_input=int(total_input / completed) if completed > 0 else 0,
        mean_output=int(np.mean(actual_output_lens) if actual_output_lens else 0),
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s if completed > 0 else 0.0,
        request_goodput=good_completed / dur_s,
        goodput_percentage=(good_completed / completed) * 100 if completed > 0 else 0.0,
        output_throughput=sum(actual_output_lens) / dur_s,
        mean_output_throughput=float(np.mean(out_throughputs or 0)),
        total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
        mean_goodput_ttft=float(mean_goodput_ttft),
        mean_goodput_tpot=float(mean_goodput_tpot),
        mean_goodput_e2el=float(mean_goodput_e2el),
        mean_goodput_throughput=float(mean_goodput_throughput),
        mean_ttft_ms=np.mean(ttfts or 0) * 1000,
        std_ttft_ms=np.std(ttfts or 0) * 1000,
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        percentiles_ttft_ms=[
            (p, np.percentile(ttfts or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        std_tpot_ms=np.std(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        percentiles_tpot_ms=[
            (p, np.percentile(tpots or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_itl_ms=np.mean(itls or 0) * 1000,
        std_itl_ms=np.std(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        percentiles_itl_ms=[
            (p, np.percentile(itls or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_e2el_ms=np.mean(e2els or 0) * 1000,
        std_e2el_ms=np.std(e2els or 0) * 1000,
        median_e2el_ms=np.median(e2els or 0) * 1000,
        percentiles_e2el_ms=[
            (p, np.percentile(e2els or 0, p) * 1000) for p in selected_percentiles
        ],
        mean_prefilling_throughput=float(np.mean(prefill_throughputs or 0)),
        std_prefilling_throughput=float(np.std(prefill_throughputs or 0)),
        median_prefilling_throughput=float(np.median(prefill_throughputs or 0)),
        percentiles_prefilling_throughput=[
            (p, float(np.percentile(prefill_throughputs or 0, p)))
            for p in selected_percentiles
        ],
        mean_decoding_throughput=float(np.mean(decode_throughputs or 0)),
        std_decoding_throughput=float(np.std(decode_throughputs or 0)),
        median_decoding_throughput=float(np.median(decode_throughputs or 0)),
        percentiles_decoding_throughput=[
            (p, float(np.percentile(decode_throughputs or 0, p)))
            for p in selected_percentiles
        ],
    )

    return metrics, actual_output_lens


async def async_request_openai(
    request_func_input: RequestFuncInput,
    api_key: str,
    pbar: Optional[async_tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    connector = aiohttp.TCPConnector(ssl=False) if "https" in api_url else None
    async with aiohttp.ClientSession(
        timeout=AIOHTTP_TIMEOUT, connector=connector
    ) as session:
        payload = {
            "model": request_func_input.model,
            "temperature": request_func_input.temperature,
            "best_of": request_func_input.best_of,
            "max_tokens": request_func_input.output_len,
            "stream": request_func_input.stream,
            "ignore_eos": request_func_input.ignore_eos,
            "stream_options": {"include_usage": True},
            "messages": [
                {
                    "role": "user",
                    "content": request_func_input.prompt,
                }
            ],
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
                            logging.info("%s — rate limited, break", chunk_bytes)
                            break

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data:").strip()
                        if chunk == "[DONE]":
                            output.latency = time.perf_counter() - st
                            output.success = True
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

                            delta = choices[0].get("delta", {})
                            if "content" in delta:
                                if output.ttft == 0.0:
                                    output.ttft = time.perf_counter() - st
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)
                                output.generated_text += delta["content"] or ""
                            most_recent_timestamp = timestamp
                else:
                    error_message = "".join(
                        [chunk.decode("utf-8").strip() async for chunk in response.content]
                    )
                    output.error = f"{response.reason} {error_message}".strip()
                    output.success = False
        except Exception:
            output.success = False
            output.error = "".join(traceback.format_exception(*sys.exc_info()))
            logging.error("Request failed: %s", output.error)

    if pbar:
        pbar.update(1)
    return output


async def limited_request_func(
    semaphore: Optional[asyncio.Semaphore],
    request_func_input: RequestFuncInput,
    api_key: str,
    pbar: Optional[async_tqdm],
) -> RequestFuncOutput:
    if semaphore is None:
        return await async_request_openai(request_func_input, api_key=api_key, pbar=pbar)
    async with semaphore:
        return await async_request_openai(request_func_input, api_key=api_key, pbar=pbar)


async def get_request(
    input_requests: List[Tuple[str, int, int, Optional[dict]]],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[Tuple[str, int, int, Optional[dict]], None]:
    input_requests = iter(input_requests)
    assert burstiness > 0
    theta = 1.0 / (request_rate * burstiness)

    for request in input_requests:
        yield request
        if request_rate == float("inf"):
            continue
        interval = float(np.random.gamma(shape=burstiness, scale=theta))
        await asyncio.sleep(interval)


def build_approx_prompt_tokens(seqlen: int, rng: random.Random) -> str:
    """与 ``aicp_speculative_size_itl_cost_warmup._make_prompt_text`` 相同思路。"""
    return " ".join(str(rng.randint(0, 99)) for _ in range(seqlen))


def build_exact_prompt_tokens(
    tokenizer: PreTrainedTokenizerBase,
    num_tokens: int,
    rng: random.Random,
) -> Tuple[str, int]:
    """生成长度恰好为 ``num_tokens``（在本地 tokenizer 下）的文本。"""
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
            while len(ids) < num_tokens:
                tid = rng.randint(0, tokenizer.vocab_size - 1)
                if tid in special:
                    continue
                ids.append(tid)
            ids = ids[:num_tokens]

    raise RuntimeError(
        f"无法构造恰好 {num_tokens} 个 token 的 prompt，请换一种随机种子或 tokenizer。"
    )


def make_input_requests(
    *,
    prompt_mode: str,
    tokenizer: Optional[PreTrainedTokenizerBase],
    num_prompts: int,
    input_len: int,
    output_len: int,
    seed: int,
) -> List[Tuple[str, int, int, None]]:
    rng = random.Random(seed)
    out: List[Tuple[str, int, int, None]] = []
    for _ in range(num_prompts):
        if prompt_mode == "exact":
            assert tokenizer is not None
            prompt, plen = build_exact_prompt_tokens(tokenizer, input_len, rng)
            out.append((prompt, plen, output_len, None))
        else:
            prompt = build_approx_prompt_tokens(input_len, rng)
            out.append((prompt, input_len, output_len, None))
    return out


def fetch_model_name(base_url: str, api_key: str) -> Optional[str]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    r = requests.get(base_url.rstrip("/") + "/v1/models", headers=headers, timeout=30)
    if r.status_code != 200:
        logging.warning("GET /v1/models failed: %s %s", r.status_code, r.text[:200])
        return None
    data = r.json()
    if data.get("data"):
        return data["data"][0]["id"]
    return None


async def warmup_one(
    api_url: str,
    base_url: str,
    model_id: str,
    api_key: str,
    input_len: int,
    output_len: int,
    prompt_mode: str,
    tokenizer: Optional[PreTrainedTokenizerBase],
    seed: int,
) -> None:
    rng = random.Random(seed ^ 0x9E3779B9)
    if prompt_mode == "exact":
        assert tokenizer is not None
        prompt, plen = build_exact_prompt_tokens(
            tokenizer, max(1, min(32, input_len)), rng
        )
    else:
        prompt = build_approx_prompt_tokens(max(1, min(32, input_len)), rng)
        plen = max(1, min(32, input_len))

    inp = RequestFuncInput(
        model=model_id,
        prompt=prompt,
        api_url=api_url,
        prompt_len=plen,
        output_len=max(8, min(64, output_len)),
        ignore_eos=True,
    )
    res = await async_request_openai(inp, api_key=api_key, pbar=None)
    if not res.success:
        raise RuntimeError(f"Warmup failed: {res.error}")


def metrics_to_result_dict(
    metrics: BenchmarkMetrics,
    *,
    model_id: str,
    input_len: int,
    output_len: int,
    num_prompts: int,
    max_concurrency: int,
    goodput_config_dict: Dict[str, float],
) -> "OrderedDict[str, Any]":
    pct_ttft = metrics.percentiles_ttft_ms
    pct_tpot = metrics.percentiles_tpot_ms
    pct_itl = metrics.percentiles_itl_ms
    pct_e2el = metrics.percentiles_e2el_ms
    pct_prefill = metrics.percentiles_prefilling_throughput
    pct_decode = metrics.percentiles_decoding_throughput
    return OrderedDict(
        {
            "model": Path(model_id).name,
            "question_label": f"{input_len}:{output_len}",
            "batch": max_concurrency,
            "num_prompts": num_prompts,
            "completed": (metrics.completed / num_prompts * 100)
            if num_prompts and metrics.completed > 0
            else 0.0,
            "mean_input_tokens": int(metrics.mean_input),
            "mean_output_tokens": int(metrics.mean_output),
            "total_prompt_tokens": metrics.total_input,
            "total_completion_tokens": metrics.total_output,
            "mean_TTFT": metrics.mean_ttft_ms,
            "median_TTFT": metrics.median_ttft_ms,
            "std_TTFT": metrics.std_ttft_ms,
            "TTFT_P90": pct_ttft[0][1] if len(pct_ttft) > 0 else float("nan"),
            "TTFT_P95": pct_ttft[1][1] if len(pct_ttft) > 1 else float("nan"),
            "TTFT_P99": pct_ttft[2][1] if len(pct_ttft) > 2 else float("nan"),
            "mean_output_throughput": metrics.mean_output_throughput,
            "output_throughput": metrics.output_throughput,
            "total_token_throughput": metrics.total_token_throughput,
            "request_throughput": metrics.request_throughput,
            "request_goodput": metrics.request_goodput if goodput_config_dict else -1.0,
            "goodput_percentage": metrics.goodput_percentage
            if goodput_config_dict
            else -1.0,
            "mean_goodput_ttft": metrics.mean_goodput_ttft,
            "mean_goodput_tpot": metrics.mean_goodput_tpot,
            "mean_goodput_e2el": metrics.mean_goodput_e2el,
            "mean_goodput_throughput": metrics.mean_goodput_throughput,
            "mean_TPOT": metrics.mean_tpot_ms,
            "median_TPOT": metrics.median_tpot_ms,
            "std_TPOT": metrics.std_tpot_ms,
            "TPOT_P90": pct_tpot[0][1] if len(pct_tpot) > 0 else float("nan"),
            "TPOT_P95": pct_tpot[1][1] if len(pct_tpot) > 1 else float("nan"),
            "TPOT_P99": pct_tpot[2][1] if len(pct_tpot) > 2 else float("nan"),
            "mean_ITL": metrics.mean_itl_ms,
            "median_ITL": metrics.median_itl_ms,
            "std_ITL": metrics.std_itl_ms,
            "ITL_P90": pct_itl[0][1] if len(pct_itl) > 0 else float("nan"),
            "ITL_P95": pct_itl[1][1] if len(pct_itl) > 1 else float("nan"),
            "ITL_P99": pct_itl[2][1] if len(pct_itl) > 2 else float("nan"),
            "mean_E2EL": metrics.mean_e2el_ms,
            "median_E2EL": metrics.median_e2el_ms,
            "std_E2EL": metrics.std_e2el_ms,
            "E2EL_P90": pct_e2el[0][1] if len(pct_e2el) > 0 else float("nan"),
            "E2EL_P95": pct_e2el[1][1] if len(pct_e2el) > 1 else float("nan"),
            "E2EL_P99": pct_e2el[2][1] if len(pct_e2el) > 2 else float("nan"),
            "mean_prefill_throughput": metrics.mean_prefilling_throughput,
            "median_prefill_throughput": metrics.median_prefilling_throughput,
            "std_prefill_throughput": metrics.std_prefilling_throughput,
            "P90_prefill_throughput": pct_prefill[0][1] if len(pct_prefill) > 0 else float("nan"),
            "P95_prefill_throughput": pct_prefill[1][1] if len(pct_prefill) > 1 else float("nan"),
            "P99_prefill_throughput": pct_prefill[2][1] if len(pct_prefill) > 2 else float("nan"),
            "mean_decode_throughput": metrics.mean_decoding_throughput,
            "median_decode_throughput": metrics.median_decoding_throughput,
            "std_decode_throughput": metrics.std_decoding_throughput,
            "P90_decode_throughput": pct_decode[0][1] if len(pct_decode) > 0 else float("nan"),
            "P95_decode_throughput": pct_decode[1][1] if len(pct_decode) > 1 else float("nan"),
            "P99_decode_throughput": pct_decode[2][1] if len(pct_decode) > 2 else float("nan"),
        }
    )


async def run_benchmark_async(args: argparse.Namespace) -> Dict[str, Any]:
    base_url = args.base_url.rstrip("/")
    api_url = base_url + args.endpoint

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")

    model_id = args.model or ""
    if not model_id:
        model_id = fetch_model_name(base_url, api_key) or ""
    if not model_id:
        raise SystemExit(
            "未指定 --model 且无法从 /v1/models 自动探测模型名，请显式传入 --model。"
        )

    tokenizer: Optional[PreTrainedTokenizerBase] = None
    if args.prompt_mode == "exact":
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )

    input_requests = make_input_requests(
        prompt_mode=args.prompt_mode,
        tokenizer=tokenizer,
        num_prompts=args.num_prompts,
        input_len=args.input_len,
        output_len=args.output_len,
        seed=args.seed,
    )

    goodput_config_dict = check_goodput_args(args.goodput)
    selected_percentiles = [float(p) for p in args.metric_percentiles.split(",")]

    if not args.disable_warmup:
        logging.info("Warming up (1 request)...")
        await warmup_one(
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            api_key=api_key,
            input_len=args.input_len,
            output_len=args.output_len,
            prompt_mode=args.prompt_mode,
            tokenizer=tokenizer,
            seed=args.seed,
        )

    semaphore = (
        asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None
    )
    pbar = async_tqdm(total=len(input_requests)) if not args.disable_pbar else None
    t0 = time.perf_counter()
    tasks: List[asyncio.Task] = []
    async for request in get_request(
        input_requests, request_rate=args.request_rate, burstiness=args.burstiness
    ):
        prompt, prompt_len, output_len, _mm = request
        inp = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            ignore_eos=True,
            temperature=0.0,
        )
        tasks.append(
            asyncio.create_task(
                limited_request_func(
                    semaphore=semaphore,
                    request_func_input=inp,
                    api_key=api_key,
                    pbar=pbar,
                )
            )
        )
    outputs = await asyncio.gather(*tasks)
    if pbar is not None:
        pbar.close()

    dur_s = time.perf_counter() - t0
    metrics, _ = calculate_metrics(
        outputs=tuple(outputs),
        dur_s=dur_s,
        selected_percentiles=selected_percentiles,
        goodput_config_dict=goodput_config_dict,
    )

    result = metrics_to_result_dict(
        metrics,
        model_id=model_id,
        input_len=args.input_len,
        output_len=args.output_len,
        num_prompts=args.num_prompts,
        max_concurrency=args.max_concurrency,
        goodput_config_dict=goodput_config_dict,
    )

    # 日志摘要（风格贴近 inference_benchmark.benchmark）
    logging.info("%s", "=" * 50 + " Serving Benchmark Result " + "=" * 12)
    logging.info(
        "%-40s %s",
        "Successful requests:",
        f"{metrics.completed}/{len(input_requests)}",
    )
    logging.info("%-40s %.2f", "Benchmark duration (s):", dur_s)
    logging.info("%-40s %s", "Mean input tokens:", metrics.mean_input)
    logging.info("%-40s %s", "Mean output tokens:", metrics.mean_output)
    logging.info("%-40s %.2f", "Request throughput (req/s):", metrics.request_throughput)
    logging.info(
        "%-40s %.2f",
        "Mean decoding throughput (tok/s):",
        metrics.mean_decoding_throughput,
    )
    logging.info("%-40s %.2f", "Total E2E output throughput (tok/s):", metrics.output_throughput)
    logging.info("%-40s %.2f", "Mean TTFT (ms):", metrics.mean_ttft_ms)
    logging.info("%-40s %.2f", "Mean TPOT (ms):", metrics.mean_tpot_ms)
    logging.info("%-40s %.2f", "Mean ITL (ms):", metrics.mean_itl_ms)
    logging.info("%-40s %.2f", "Mean E2EL (ms):", metrics.mean_e2el_ms)
    logging.info("%s", "=" * 72)

    return {
        "result": result,
        "duration_s": dur_s,
        "base_url": base_url,
        "model": model_id,
        "prompt_mode": args.prompt_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random-token fixed-concurrency benchmark (inference_benchmark 同款指标)."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="例如 http://127.0.0.1:30000",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/chat/completions",
        help="聊天补全 endpoint，默认 /v1/chat/completions",
    )
    parser.add_argument("--model", type=str, default=None, help="请求体 model；可省略以自动探测")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Bearer token；默认同环境变量 OPENAI_API_KEY，可写 EMPTY",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default=None,
        help="``--prompt-mode exact`` 时必需：HuggingFace tokenizer / 模型目录",
    )
    parser.add_argument(
        "--prompt-mode",
        type=str,
        choices=["exact", "approx"],
        default="exact",
        help="exact：本地 tokenizer 下严格指定输入 token 数；approx：两位随机整数近似",
    )
    parser.add_argument("--input-len", type=int, required=True, help="输入长度（exact 为 token 数）")
    parser.add_argument("--output-len", type=int, required=True, help="max_tokens / 输出 token 数")
    parser.add_argument("--num-prompts", type=int, required=True, help="总请求数")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        required=True,
        help="固定并发（asyncio.Semaphore），与 inference_benchmark 中含义一致",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="请求到达率 (req/s)；默认 inf 表示尽快填满并发",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Gamma 到达过程形状参数；仅 request-rate < inf 时有效",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--disable-warmup", action="store_true")
    parser.add_argument("--disable-pbar", action="store_true")
    parser.add_argument(
        "--metric-percentiles",
        type=str,
        default="90,95,99",
        help="与 inference_benchmark 一致，逗号分隔",
    )
    parser.add_argument(
        "--goodput",
        nargs="*",
        default=[],
        help='可选，格式 "ttft:5000" "throughput:100"（数值含义与 inference_benchmark 相同）',
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="将 result 字典写入该 JSON 文件（UTF-8）",
    )
    parser.add_argument(
        "--disable-excel",
        action="store_true",
        help="不写入默认 Excel（默认：当前目录 {模型名}_{时间戳}.xlsx）",
    )
    parser.add_argument(
        "--excel-path",
        type=str,
        default=None,
        help="指定 Excel 路径；默认使用当前工作目录下 模型名_时间戳.xlsx",
    )
    parser.add_argument("--log-level", type=str, default="INFO")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s: %(message)s",
    )

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.prompt_mode == "exact" and not args.tokenizer_path:
        raise SystemExit("--prompt-mode exact 需要 --tokenizer-path")

    if args.output_len <= 1:
        raise SystemExit("--output-len 必须 > 1（以便计算 TPOT/ITL）")

    out = asyncio.run(run_benchmark_async(args))
    run_id = str(uuid.uuid4())

    excel_row: Dict[str, Any] = dict(out["result"])
    excel_row.update(
        {
            "duration_s": out["duration_s"],
            "base_url": out["base_url"],
            "prompt_mode": out["prompt_mode"],
            "run_id": run_id,
            "goodput_slo": " ".join(args.goodput) if args.goodput else "",
        }
    )

    if not args.disable_excel:
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        if args.excel_path:
            xlsx_path = Path(args.excel_path)
        else:
            stem = safe_model_stem_for_filename(out["model"])
            xlsx_path = Path.cwd() / f"{stem}_{ts}.xlsx"
        write_summary_excel(xlsx_path, excel_row)
        logging.info("已写入 Excel %s", xlsx_path.resolve())

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            "result": dict(out["result"]),
            "duration_s": out["duration_s"],
            "base_url": out["base_url"],
            "model": out["model"],
            "prompt_mode": out["prompt_mode"],
            "run_id": run_id,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        logging.info("已写入 %s", p.resolve())


if __name__ == "__main__":
    main()
