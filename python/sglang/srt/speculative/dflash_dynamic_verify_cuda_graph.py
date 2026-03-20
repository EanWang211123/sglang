# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""DFLASH dynamic TARGET_VERIFY CUDA graph: per-batch-size verify lengths.

Builds a merged map ``batch_size -> verify_len`` from ``--cuda-graph-bs`` (seed list)
and a JSON config, then groups batch sizes by ``verify_len`` so each
:class:`~sglang.srt.model_executor.cuda_graph_runner.CudaGraphRunner` captures only
one query length (tokens per sequence) — matching the invariant that a single runner
uses a fixed ``num_tokens_per_bs`` for all captured batch sizes.

JSON format (object; keys are batch sizes, values are verify lengths or lists of lengths;
only the **first** length per key is used today — lists reserve future selection)::

    { "1": [4], "2": 5, "4": 8 }

Merge rule:

* Key universe = union of ``cuda_graph_bs`` (after server defaults) and JSON keys.
* For each batch size ``bs`` in that universe: use the first verify length from JSON if
  ``bs`` is present in JSON, otherwise ``default_verify_len`` (from
  ``--speculative-dflash-verify-token-num`` if set, else block_size).
* Drop ``bs`` that fail divisibility / pool-size filters (see
  :func:`filter_batch_sizes_for_verify_capture`).
"""

from __future__ import annotations

import json
import logging
from typing import Dict, Iterable, List, Mapping, Tuple

logger = logging.getLogger(__name__)


def load_dflash_dynamic_verify_tokens_json(path: str) -> Dict[int, List[int]]:
    """Load and normalize JSON to ``Dict[batch_size, list of verify lengths]``."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"Dynamic DFLASH verify JSON must be a JSON object, got {type(raw).__name__}"
        )
    out: Dict[int, List[int]] = {}
    for k, v in raw.items():
        bs = int(k)
        if isinstance(v, list):
            lens = [int(x) for x in v]
        else:
            lens = [int(v)]
        if not lens:
            raise ValueError(f"Empty verify length list for batch_size={bs}")
        out[bs] = lens
    return out


def _mul_base_for_capture(server_args) -> int:
    from sglang.srt.layers.dp_attention import get_attention_cp_size, get_attention_tp_size
    from sglang.srt.utils.common import require_gathered_buffer

    mul_base = 1
    if server_args.enable_two_batch_overlap:
        mul_base *= 2
    if require_gathered_buffer(server_args):
        mul_base *= get_attention_tp_size()
    if mul_base % get_attention_cp_size() != 0:
        mul_base *= get_attention_cp_size()
    return mul_base


def filter_batch_sizes_for_verify_capture(
    *,
    batch_sizes: Iterable[int],
    verify_len: int,
    model_runner,
    server_args,
) -> List[int]:
    """Keep batch sizes where ``bs * verify_len`` passes the same divisibility rules as cuda graph capture."""
    mul_base = _mul_base_for_capture(server_args)
    pool = model_runner.req_to_token_pool.size
    out: List[int] = []
    for bs in sorted(set(batch_sizes)):
        if bs <= 0 or bs > pool:
            continue
        if bs * verify_len % mul_base != 0:
            logger.warning(
                "DFLASH dynamic verify: dropping bs=%s verify_len=%s: "
                "bs * verify_len=%s not divisible by mul_base=%s",
                bs,
                verify_len,
                bs * verify_len,
                mul_base,
            )
            continue
        out.append(bs)
    return sorted(set(out))


def merge_dflash_verify_lens_by_batch_size(
    *,
    seed_cuda_graph_bs: List[int],
    json_map: Mapping[int, List[int]],
    default_verify_len: int,
    block_size: int,
) -> Dict[int, int]:
    """Merge JSON overrides with defaults for all batch sizes in the union of keys."""
    if default_verify_len <= 0 or default_verify_len > block_size:
        raise ValueError(
            f"default_verify_len must be in (0, block_size], got {default_verify_len}, block_size={block_size}"
        )
    universe = sorted(set(int(x) for x in seed_cuda_graph_bs) | set(json_map.keys()))
    merged: Dict[int, int] = {}
    for bs in universe:
        if bs in json_map:
            vlen = int(json_map[bs][0])
        else:
            vlen = int(default_verify_len)
        if vlen <= 0 or vlen > block_size:
            raise ValueError(
                f"Invalid verify_len={vlen} for batch_size={bs} (must be in (0, block_size], block_size={block_size})"
            )
        merged[bs] = vlen
    return merged


def group_batch_sizes_by_verify_len(bs_to_verify_len: Mapping[int, int]) -> Dict[int, List[int]]:
    """Invert to ``verify_len -> [batch sizes]`` (sorted)."""
    groups: Dict[int, List[int]] = {}
    for bs, vlen in sorted(bs_to_verify_len.items()):
        groups.setdefault(vlen, []).append(bs)
    for vlen in groups:
        groups[vlen] = sorted(set(groups[vlen]))
    return dict(sorted(groups.items()))


def cuda_graph_envelope_for_groups(
    verify_len_to_batch_sizes: Mapping[int, List[int]],
) -> Tuple[int, int]:
    """(global_max_bs, global_max_num_tokens) for a single attn ``init_cuda_graph_state``."""
    global_max_bs = 0
    global_max_num_tokens = 0
    for vlen, bs_list in verify_len_to_batch_sizes.items():
        if not bs_list:
            continue
        mb = max(bs_list)
        global_max_bs = max(global_max_bs, mb)
        global_max_num_tokens = max(global_max_num_tokens, mb * vlen)
    return global_max_bs, global_max_num_tokens


def build_filtered_merge_and_groups(
    *,
    server_args,
    model_runner,
    json_map: Mapping[int, List[int]],
) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    """Full pipeline: default verify len, merge, per-(bs,vlen) filter, drop empty groups."""
    block_size = int(server_args.speculative_num_draft_tokens)
    if server_args.speculative_dflash_verify_token_num is not None:
        default_verify_len = int(server_args.speculative_dflash_verify_token_num)
    else:
        default_verify_len = block_size

    seed = list(server_args.cuda_graph_bs)
    if seed and max(seed) > model_runner.req_to_token_pool.size:
        seed = seed + [model_runner.req_to_token_pool.size]
    merged_raw = merge_dflash_verify_lens_by_batch_size(
        seed_cuda_graph_bs=seed,
        json_map=json_map,
        default_verify_len=default_verify_len,
        block_size=block_size,
    )

    filtered_bs_to_vlen: Dict[int, int] = {}
    for bs, vlen in sorted(merged_raw.items()):
        kept = filter_batch_sizes_for_verify_capture(
            batch_sizes=[bs],
            verify_len=vlen,
            model_runner=model_runner,
            server_args=server_args,
        )
        if kept:
            filtered_bs_to_vlen[bs] = vlen
        else:
            logger.warning(
                "DFLASH dynamic verify: batch_size=%s verify_len=%s removed after capture filter",
                bs,
                vlen,
            )

    if not filtered_bs_to_vlen:
        raise RuntimeError(
            "DFLASH dynamic verify: no valid (batch_size, verify_len) pairs after filtering; "
            "check --cuda-graph-bs, JSON, and divisibility constraints."
        )

    groups = group_batch_sizes_by_verify_len(filtered_bs_to_vlen)
    groups = {v: bss for v, bss in groups.items() if bss}
    logger.info(
        "DFLASH dynamic verify config: bs->verify_len=%s, verify_len->batch_sizes=%s",
        filtered_bs_to_vlen,
        groups,
    )
    return filtered_bs_to_vlen, groups
