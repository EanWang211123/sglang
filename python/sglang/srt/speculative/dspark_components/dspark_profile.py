"""Startup profiling for DSpark adaptive verify costs.

Config example::

    {
      "batch_sizes": [1, 4, 8, 16],
      "seq_len": 128,
      "query_lens_per_req": [2, 4, 6, 8],
      "n_warmup": 1,
      "n_measure": 3
    }
"""

from __future__ import annotations

import dataclasses
import json
import logging
import statistics
from array import array
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.distributed import get_world_group
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.runtime_context import get_parallel, get_schedule
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.dspark_components.dspark_planner import (
    ragged_capture_max_slots,
    ragged_capture_num_tokens,
)
from sglang.srt.speculative.dspark_components.dspark_sps_fit import (
    fit_additive_sps_table,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdaptiveVerifyProfileConfig:
    batch_sizes: Optional[list[int]] = None
    seq_len: int = 128
    query_lens_per_req: Optional[list[int]] = None
    n_warmup: int = 1
    n_measure: int = 3


def load_adaptive_verify_profile_config(value: str) -> AdaptiveVerifyProfileConfig:
    raw = json.loads(value)
    if not isinstance(raw, dict):
        raise ValueError("adaptive verify profile config must be a JSON object")
    known = {field.name for field in dataclasses.fields(AdaptiveVerifyProfileConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"Unknown adaptive verify profile config keys: {sorted(unknown)}"
        )
    cfg = AdaptiveVerifyProfileConfig(**raw)
    _validate_positive_list("batch_sizes", cfg.batch_sizes)
    _validate_positive_list("query_lens_per_req", cfg.query_lens_per_req)
    if (
        not isinstance(cfg.seq_len, int)
        or isinstance(cfg.seq_len, bool)
        or cfg.seq_len < 1
    ):
        raise ValueError(f"seq_len must be a positive int, got {cfg.seq_len!r}")
    if (
        not isinstance(cfg.n_warmup, int)
        or isinstance(cfg.n_warmup, bool)
        or cfg.n_warmup < 0
    ):
        raise ValueError(f"n_warmup must be a non-negative int, got {cfg.n_warmup!r}")
    if (
        not isinstance(cfg.n_measure, int)
        or isinstance(cfg.n_measure, bool)
        or cfg.n_measure < 1
    ):
        raise ValueError(f"n_measure must be a positive int, got {cfg.n_measure!r}")
    return cfg


def _validate_positive_list(name: str, values: Optional[list[int]]) -> None:
    if values is None:
        return
    if (
        not isinstance(values, list)
        or not values
        or not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in values
        )
    ):
        raise ValueError(f"{name} must be a non-empty list of positive ints or null")


def resolve_profile_grid(
    cfg: AdaptiveVerifyProfileConfig,
    *,
    max_batch_size_per_rank: int,
    max_query_len_per_req: int,
) -> tuple[list[int], list[int]]:
    if max_batch_size_per_rank < 1 or max_query_len_per_req < 1:
        raise ValueError("resolved profile batch/query limits must be positive")
    if cfg.batch_sizes is None:
        batch_sizes = [1, 4, 8]
        batch_sizes.extend(range(16, max_batch_size_per_rank + 1, 8))
        batch_sizes.append(max_batch_size_per_rank)
        batch_sizes = [
            batch_size
            for batch_size in batch_sizes
            if batch_size <= max_batch_size_per_rank
        ]
    else:
        batch_sizes = cfg.batch_sizes
    batch_sizes = sorted(set(batch_sizes))
    if batch_sizes[-1] > max_batch_size_per_rank:
        raise ValueError(
            f"profile batch size {batch_sizes[-1]} exceeds resolved per-rank "
            f"max_running_requests={max_batch_size_per_rank}"
        )
    default_query_lens = list(range(2, max_query_len_per_req + 1, 2))
    default_query_lens.append(max_query_len_per_req)
    query_lens = sorted(set(cfg.query_lens_per_req or default_query_lens))
    if query_lens[-1] > max_query_len_per_req:
        raise ValueError(
            f"profile query length {query_lens[-1]} exceeds "
            f"max_query_len_per_req={max_query_len_per_req}"
        )
    return batch_sizes, query_lens


def run_adaptive_verify_profile(
    *, worker, tree_cache, max_running_requests: int
) -> None:
    config_value = worker.server_args.adaptive_verify_profile_config
    cfg = (
        load_adaptive_verify_profile_config(config_value)
        if config_value
        else AdaptiveVerifyProfileConfig()
    )
    batch_sizes, query_lens = resolve_profile_grid(
        cfg,
        max_batch_size_per_rank=max_running_requests,
        max_query_len_per_req=worker.verify_num_draft_tokens,
    )
    batch_sizes = _validate_profile_grid(worker, batch_sizes, query_lens)
    num_cells = len(batch_sizes) * len(query_lens)
    if num_cells < 4:
        raise ValueError(
            f"adaptive verify profiling needs at least 4 cells, got {num_cells}"
        )
    required_context = (
        cfg.seq_len + cfg.n_warmup + cfg.n_measure + worker.verify_num_draft_tokens
    )
    if required_context > int(worker.model_config.context_len):
        raise ValueError(
            "adaptive verify profile needs seq_len + warmup + measure + verify "
            f"window <= context length, got {required_context} > "
            f"{worker.model_config.context_len}"
        )

    logger.info(
        "Starting DSpark adaptive verify profiling: batch_sizes=%s, "
        "query_lens_per_req=%s, seq_len=%d, warmup=%d, measure=%d",
        batch_sizes,
        query_lens,
        cfg.seq_len,
        cfg.n_warmup,
        cfg.n_measure,
    )
    cells = []
    with _profile_acceptance_override(worker):
        for batch_size in batch_sizes:
            for query_len in query_lens:
                budget = batch_size * (query_len - 1)
                worker._verify_planner.set_profile_verify_token_budget(budget)
                worker._verify_planner.set_profile_verify_len_per_req(query_len)
                try:
                    step_seconds = DSparkProfileSession(
                        worker=worker,
                        tree_cache=tree_cache,
                        batch_size=batch_size,
                        query_len_per_req=query_len,
                        seq_len=cfg.seq_len,
                        n_warmup=cfg.n_warmup,
                        n_measure=cfg.n_measure,
                    ).measure()
                finally:
                    worker._verify_planner.set_profile_verify_len_per_req(None)
                    worker._verify_planner.set_profile_verify_token_budget(None)
                batch_tokens = batch_size * query_len
                cells.append({"bs": batch_size, "M": batch_tokens, "T": step_seconds})
                logger.info(
                    "DSpark adaptive verify profile: bs=%d, query_len_per_req=%d, "
                    "batch_tokens=%d, median=%.3fms",
                    batch_size,
                    query_len,
                    batch_tokens,
                    step_seconds * 1000.0,
                )

    table = fit_additive_sps_table(cells=cells)
    worker._verify_planner.install_sps_table(table)
    logger.info(
        "DSpark adaptive verify profiling complete: %d cells, bs_probes=%s, "
        "batch_token_probes=%s",
        len(cells),
        table.bs_probes,
        table.m_probes,
    )


def _validate_profile_grid(worker, batch_sizes, query_lens) -> list[int]:
    capture_tokens = ragged_capture_num_tokens(model_runner=worker.model_runner)
    max_slots = ragged_capture_max_slots(model_runner=worker.model_runner)
    if capture_tokens is None or max_slots is None:
        raise ValueError("adaptive verify profiling requires compact CUDA graphs")
    if batch_sizes[-1] > max_slots:
        original_max = batch_sizes[-1]
        batch_sizes = sorted({min(batch_size, max_slots) for batch_size in batch_sizes})
        logger.info(
            "Clamped adaptive verify profile batch sizes to compact graph max "
            "batch size %d (configured max was %d)",
            max_slots,
            original_max,
        )
    max_profile_tokens = max(
        batch_size * query_len for batch_size in batch_sizes for query_len in query_lens
    )
    if max_profile_tokens > capture_tokens[-1]:
        raise ValueError(
            f"profile batch tokens {max_profile_tokens} exceed compact graph "
            f"maximum {capture_tokens[-1]}"
        )
    return batch_sizes


@contextmanager
def _profile_acceptance_override(worker):
    worker_value = worker._simulate_acc_len
    executor_value = worker._verify_executor._simulate_acc_len
    worker._simulate_acc_len = 1.0
    worker._verify_executor._simulate_acc_len = 1.0
    try:
        yield
    finally:
        worker._simulate_acc_len = worker_value
        worker._verify_executor._simulate_acc_len = executor_value


class DSparkProfileSession:
    """Measure real DSpark decode steps for one ``(bs, query_len)`` cell."""

    def __init__(
        self,
        *,
        worker,
        tree_cache,
        batch_size: int,
        query_len_per_req: int,
        seq_len: int,
        n_warmup: int,
        n_measure: int,
    ) -> None:
        self.worker = worker
        self.tree_cache = tree_cache
        self.batch_size = batch_size
        self.query_len_per_req = query_len_per_req
        self.seq_len = seq_len
        self.n_warmup = n_warmup
        self.n_measure = n_measure
        self.device_mod = torch.get_device_module(worker.device)
        self.forward_iter = 0

    def measure(self) -> float:
        reqs = self._build_reqs()
        primary_error = None
        try:
            batch = self._run_prefill(reqs)

            for _ in range(self.n_warmup):
                self._run_decode(batch)
            self.device_mod.synchronize()

            samples = self._measure_decode_steps(batch)
            return statistics.median(self._reduce_mean_across_ranks(samples))
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                self._teardown(reqs)
            except Exception:
                if primary_error is None:
                    raise
                logger.exception(
                    "Failed to clean up DSpark profiling requests while handling "
                    "an earlier profiling error"
                )

    def _build_reqs(self) -> list[Req]:
        model_runner = self.worker.model_runner
        model_config = model_runner.model_config
        vocab_size = getattr(model_config, "vocab_size", 32000)
        sampling_params = SamplingParams(
            temperature=0.0,
            max_new_tokens=self.n_warmup + self.n_measure + 8,
            ignore_eos=True,
        )
        sampling_params.normalize(None)

        reqs = []
        rng = np.random.default_rng(0)
        for index in range(self.batch_size):
            token_ids = rng.integers(
                1, max(2, vocab_size), size=self.seq_len, dtype=np.int64
            )
            req = Req(
                rid=(
                    f"dspark_profile_q{self.query_len_per_req}_"
                    f"b{self.batch_size}_{index}"
                ),
                origin_input_text="",
                origin_input_ids=array("q", token_ids.tolist()),
                sampling_params=sampling_params,
            )
            req.full_untruncated_fill_ids = req.origin_input_ids
            req.logprob_start_len = -1
            req.init_next_round_input(self.tree_cache)
            req.set_extend_range(
                len(req.prefix_indices), len(req.full_untruncated_fill_ids)
            )
            reqs.append(req)
        return reqs

    def _build_batch(self, reqs: list[Req]) -> ScheduleBatch:
        model_runner = self.worker.model_runner
        req_to_token_pool = model_runner.req_to_token_pool
        token_to_kv_pool_allocator = model_runner.token_to_kv_pool_allocator
        if req_to_token_pool is None or token_to_kv_pool_allocator is None:
            raise RuntimeError(
                "DSpark startup profiling requires initialized target memory pools"
            )
        return ScheduleBatch.init_new(
            reqs,
            req_to_token_pool,
            token_to_kv_pool_allocator,
            self.tree_cache,
            model_runner.model_config,
            False,
            model_runner.spec_algorithm,
        )

    def _run_prefill(self, reqs: list[Req]) -> ScheduleBatch:
        # DeepSeek-V4's compressor prefill plan stores num_q_tokens in uint16.
        # Also honor the scheduler's resolved prefill admission/memory ceiling.
        max_prefill_tokens = min(
            int(get_schedule().max_prefill_tokens or (2**16 - 1)), 2**16 - 1
        )
        reqs_per_batch = max_prefill_tokens // self.seq_len
        if reqs_per_batch < 1:
            raise ValueError(
                "adaptive verify profile seq_len exceeds the per-forward prefill "
                f"limit: {self.seq_len} > {max_prefill_tokens}"
            )

        merged_batch = None
        for start in range(0, len(reqs), reqs_per_batch):
            prefill_reqs = reqs[start : start + reqs_per_batch]
            batch = self._build_batch(prefill_reqs)
            self._set_dp_counts(batch, len(prefill_reqs) * self.seq_len)
            batch.prepare_for_extend()
            if (
                batch.input_ids is None
                and getattr(batch, "prefill_input_ids_cpu", None) is not None
            ):
                batch.input_ids = batch.prefill_input_ids_cpu.to(
                    self.worker.device, non_blocking=True
                )
                batch.prefill_input_ids_cpu = None
            self._run_forward_isolated(batch)
            if merged_batch is None:
                merged_batch = batch
            else:
                merged_batch.merge_batch(batch)

        assert merged_batch is not None
        return merged_batch

    def _run_decode(self, batch: ScheduleBatch) -> None:
        # Profiling runs the same synthetic decode batch on every rank, so this
        # mirrors the scheduler's successful decode-graph eligibility vote.
        batch.can_run_decode_cuda_graph = True
        batch.prepare_for_decode()
        self._set_dp_counts(batch, self.batch_size)
        verify_tokens = self.batch_size * self.query_len_per_req
        batch.spec_verify_tier_num_tokens = verify_tokens
        if get_parallel().enable_dp_attention:
            batch.global_spec_verify_tier_num_tokens = [
                verify_tokens
            ] * get_parallel().dp_size
        result = self._run_forward_isolated(batch)
        if not result.can_run_cuda_graph:
            raise RuntimeError(
                "DSpark adaptive verify profiling did not run on a CUDA graph: "
                f"batch_size={self.batch_size}, "
                f"query_len_per_req={self.query_len_per_req}"
            )

    def _measure_decode_steps(self, batch: ScheduleBatch) -> list[float]:
        events = []
        for _ in range(self.n_measure):
            start = self.device_mod.Event(enable_timing=True)
            end = self.device_mod.Event(enable_timing=True)
            start.record()
            self._run_decode(batch)
            end.record()
            events.append((start, end))
        self.device_mod.synchronize()
        return [start.elapsed_time(end) / 1000.0 for start, end in events]

    def _reduce_mean_across_ranks(self, samples: list[float]) -> list[float]:
        if not dist.is_initialized():
            return samples
        group = get_world_group().device_group
        world_size = dist.get_world_size(group=group)
        if world_size == 1:
            return samples
        values = torch.tensor(samples, dtype=torch.float64, device=self.worker.device)
        dist.all_reduce(values, op=dist.ReduceOp.SUM, group=group)
        values /= world_size
        return values.cpu().tolist()

    def _set_dp_counts(self, batch: ScheduleBatch, per_rank_tokens: int) -> None:
        if not get_parallel().enable_dp_attention:
            return
        counts = [per_rank_tokens] * get_parallel().dp_size
        batch.global_num_tokens = counts
        batch.global_num_tokens_for_logprob = counts

    def _run_forward_isolated(self, batch: ScheduleBatch):
        self.forward_iter += 1
        batch.forward_iter = self.forward_iter
        snapshot = {f.name: getattr(batch, f.name) for f in dataclasses.fields(batch)}
        sampling_info = batch.sampling_info
        if sampling_info is not None:
            batch.sampling_info = sampling_info.copy_for_forward()
        try:
            result = self.worker.forward_batch_generation(batch)
        finally:
            for name, value in snapshot.items():
                setattr(batch, name, value)

        batch.spec_info = result.next_draft_input
        if result.new_seq_lens is not None:
            batch.seq_lens = result.new_seq_lens
            batch.seq_lens_cpu = result.new_seq_lens.to("cpu")
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
            for req, seq_len in zip(batch.reqs, batch.seq_lens_cpu.tolist()):
                req.kv.kv_committed_len = int(seq_len)
        batch.input_ids = None
        return result

    def _teardown(self, reqs: list[Req]) -> None:
        errors = []
        for req in reqs:
            try:
                release_kv_cache(req, self.tree_cache, is_insert=False)
            except Exception as exc:
                errors.append(exc)
                logger.exception("Failed to free profiling request %s", req.rid)
        if errors:
            raise RuntimeError(
                f"Failed to free {len(errors)} profiling request(s)"
            ) from errors[0]
