"""
Helper utilities for DFLASH dynamic per-batch-size verify token num.
Used by --dynamic-speculative-dflash-verify-tokens-config.

JSON format:
  {
    "<batch_size>": [<verify_len>, ...],   # list for future multi-qlen support
    ...
  }

Currently only the first element of each list is used.
"""

from __future__ import annotations

import bisect
import json
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def load_dflash_dynamic_verify_tokens_json(path: str) -> Dict[int, List[int]]:
    """Load and parse the dynamic verify tokens config JSON.

    Returns a dict mapping int(batch_size) -> List[int(qlen)].
    Raises ValueError on invalid format.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"DFLASH dynamic verify config must be a JSON object, got {type(raw)}"
        )

    result: Dict[int, List[int]] = {}
    for k, v in raw.items():
        try:
            bs = int(k)
        except ValueError:
            raise ValueError(
                f"DFLASH dynamic verify config: key {k!r} is not an integer"
            )
        if bs <= 0:
            raise ValueError(
                f"DFLASH dynamic verify config: batch_size must be positive, got {bs}"
            )
        if isinstance(v, int):
            v = [v]
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError(
                f"DFLASH dynamic verify config: value for bs={bs} must be a non-empty list"
            )
        qlens = []
        for q in v:
            try:
                qlen = int(q)
            except (TypeError, ValueError):
                raise ValueError(
                    f"DFLASH dynamic verify config: qlen {q!r} for bs={bs} is not an integer"
                )
            qlens.append(qlen)
        result[bs] = qlens

    return result


def build_dflash_bs_to_qlen(
    server_args: "ServerArgs",
    capture_bs: List[int],
) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    """Build the merged {bs -> qlen} map for CUDA graph capture.

    Rules:
    - Default qlen = speculative_dflash_verify_token_num or speculative_num_draft_tokens.
    - JSON entries whose batch_size IS in capture_bs override the default.
    - JSON entries whose batch_size is NOT in capture_bs are silently ignored
      (no extra graphs are captured – satisfies requirement 1).
    - JSON values are List[int]; only [0] is used now (requirement 2 prep).

    Returns:
        merged  : Dict[bs -> effective_qlen]   (one entry per captured bs)
        groups  : Dict[qlen -> List[bs]]        (for logging / future use)
    """
    default_qlen = int(
        server_args.speculative_dflash_verify_token_num
        or server_args.speculative_num_draft_tokens
    )
    block_size = int(server_args.speculative_num_draft_tokens)

    json_map = load_dflash_dynamic_verify_tokens_json(
        server_args.dynamic_speculative_dflash_verify_tokens_config
    )

    capture_bs_set = set(capture_bs)
    merged: Dict[int, int] = {bs: default_qlen for bs in capture_bs}

    for bs, qlens in json_map.items():
        if bs not in capture_bs_set:
            logger.info(
                "DFLASH dynamic verify: batch_size=%d from JSON not in capture_bs %s, "
                "ignored (no extra graph)",
                bs,
                sorted(capture_bs_set),
            )
            continue
        qlen = qlens[0]  # only first element used now
        if qlen <= 0 or qlen > block_size:
            raise ValueError(
                f"DFLASH dynamic verify config: qlen={qlen} for bs={bs} must be in "
                f"(0, block_size={block_size}]"
            )
        merged[bs] = qlen

    # Build inverse map for logging
    groups: Dict[int, List[int]] = {}
    for bs, qlen in merged.items():
        groups.setdefault(qlen, []).append(bs)

    logger.info(
        "DFLASH dynamic verify config: bs->verify_len=%s, verify_len->batch_sizes=%s",
        merged,
        {q: sorted(bs_list) for q, bs_list in sorted(groups.items())},
    )

    return merged, groups


def resolve_verify_len_for_batch_size(
    raw_bs: int,
    sorted_capture_bs: List[int],
    bs_to_qlen: Dict[int, int],
) -> int:
    """Return the effective verify_len for raw_bs.

    Finds the smallest captured_bs >= raw_bs (ceiling padding) and returns its
    assigned qlen.  Mirrors the padding logic in CudaGraphRunner.replay_prepare.
    """
    idx = bisect.bisect_left(sorted_capture_bs, raw_bs)
    if idx >= len(sorted_capture_bs):
        idx = len(sorted_capture_bs) - 1
    padded_bs = sorted_capture_bs[idx]
    return bs_to_qlen[padded_bs]
