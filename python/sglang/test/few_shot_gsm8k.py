"""
Run few-shot GSM-8K evaluation.

.. deprecated::
    This module is deprecated. Use ``sglang.test.run_eval`` with
    ``eval_name="gsm8k"`` instead, which routes through the unified
    Chat API evaluation framework with dump_metric support.

``--data-path`` accepts:

- Original GSM-8K JSONL with ``question`` / ``answer`` per line.
- A JSON file that is either one object, or an array of objects, each with
  ``instruction`` and ``output`` (same semantics as ``question`` / ``answer``).
  If whole-file JSON parsing fails, falls back to JSONL line-by-line parsing.

Inference talks to a running SGLang Runtime via ``RuntimeEndpoint`` (not OpenAI-style URL flags).
Defaults: ``--host 127.0.0.1``, ``--port 30000``. Override when the server is remote or uses another port.

Usage:
python3 -m sglang.test.few_shot_gsm8k --num-questions 200
python3 -m sglang.test.few_shot_gsm8k --data-path /path/to/gsm8k_main_test.json
python3 -m sglang.test.few_shot_gsm8k --parallel 32  # concurrent requests (same as --concurrency)
python3 -m sglang.test.few_shot_gsm8k --host 192.168.0.5 --port 30001
"""

import argparse
import ast
import json
import re
import time
import warnings

import numpy as np

from sglang.lang.api import set_default_backend
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.utils import (
    download_and_cache_file,
    dump_state_text,
    normalize_base_url,
    read_jsonl,
)

INVALID = -9999999


def _normalize_gsm8k_record(rec: dict) -> dict:
    """Map instruction/output records to question/answer format."""
    if "question" in rec and "answer" in rec:
        return rec
    if "instruction" in rec and "output" in rec:
        return {"question": rec["instruction"], "answer": rec["output"]}
    raise ValueError(
        "Each record must have either ('question', 'answer') or "
        f"('instruction', 'output'); got keys: {sorted(rec.keys())}"
    )


def load_gsm8k_records(path: str) -> list:
    """Load GSM-8K-style data from JSON (object/array) or JSONL."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    text = text.strip()
    if not text:
        return []

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [_normalize_gsm8k_record(r) for r in read_jsonl(path)]

    if isinstance(data, dict):
        records = [data]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(
            f"Top-level JSON must be an object or array, got {type(data).__name__}"
        )

    return [_normalize_gsm8k_record(r) for r in records]


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def run_eval(args):
    warnings.warn(
        "sglang.test.few_shot_gsm8k is deprecated. "
        "Use sglang.test.run_eval with eval_name='gsm8k' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Select backend
    set_default_backend(RuntimeEndpoint(normalize_base_url(args.host, args.port)))

    if args.data_path is None:
        # Read data
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        filename = download_and_cache_file(url)
    else:
        filename = args.data_path

    lines = load_gsm8k_records(filename)

    # Construct prompts
    num_questions = args.num_questions
    num_shots = args.num_shots
    few_shot_examples = get_few_shot_examples(lines, num_shots)

    questions = []
    labels = []
    for i in range(len(lines[:num_questions])):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels)
    arguments = [{"question": q} for q in questions]

    #####################################
    ######### SGL Program Begin #########
    #####################################

    import sglang as sgl

    @sgl.function
    def few_shot_gsm8k(s, question):
        s += few_shot_examples + question
        s += sgl.gen(
            "answer",
            max_tokens=args.max_new_tokens,
            stop=["Question", "Assistant:", "<|separator|>"],
        )

    #####################################
    ########## SGL Program End ##########
    #####################################

    # Run requests
    tic = time.perf_counter()
    states = few_shot_gsm8k.run_batch(
        arguments,
        temperature=args.temperature if hasattr(args, "temperature") else 0,
        num_threads=args.parallel,
        progress_bar=True,
        return_logprob=getattr(args, "return_logprob", None),
        logprob_start_len=getattr(args, "logprob_start_len", None),
    )
    latency = time.perf_counter() - tic

    preds = []
    for i in range(len(states)):
        preds.append(get_answer_value(states[i]["answer"]))

    # print(f"{preds=}")
    # print(f"{labels=}")

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    # Compute speed
    num_output_tokens = sum(
        s.get_meta_info("answer")["completion_tokens"] for s in states
    )
    output_throughput = num_output_tokens / latency

    # Print results
    print(f"Accuracy: {acc:.3f}")
    print(f"Invalid: {invalid:.3f}")
    print(f"Latency: {latency:.3f} s")
    print(f"Output throughput: {output_throughput:.3f} token/s")

    # Dump results
    dump_state_text("tmp_output_gsm8k.txt", states)

    return {
        "accuracy": acc,
        "invalid": invalid,
        "latency": latency,
        "output_throughput": output_throughput,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--parallel",
        "--concurrency",
        type=int,
        default=128,
        dest="parallel",
        metavar="N",
        help="Concurrent requests sent to the runtime (run_batch num_threads). Default: 128.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="SGLang Runtime hostname (no http://). Default: 127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="SGLang Runtime HTTP port. Default: 30000",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    run_eval(args)
