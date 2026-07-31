"""Unit tests for the DFlash spec-v2 host-sync removal: compact-rebuild
kernel bit-exactness, vocab-parallel draft sampler select, host seq-lens
upper bound, hybrid needs_cpu_seq_lens delegation, filter_batch host
keep-list."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=13, stage="base-b", runner_config="1-gpu-small")

_HAS_CUDA = torch.cuda.is_available()


def _compact_lens_exact(seq_lens, window, page):
    fake_self = SimpleNamespace(
        device=seq_lens.device, draft_window_size=window, page_size=page
    )
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    return DFlashWorkerV2._compute_compact_draft_seq_lens(fake_self, seq_lens)


def _compact_lens_host(seq_lens, window, page):
    fake_self = SimpleNamespace(draft_window_size=window, page_size=page)
    out = torch.empty(seq_lens.numel(), dtype=torch.int32)
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    DFlashWorkerV2._compute_compact_draft_seq_lens_host(fake_self, seq_lens, out)
    return out


class TestCompactSeqLensHostBound(CustomTestCase):
    def test_upper_bound_of_exact(self):
        g = torch.Generator().manual_seed(0)
        for window, page in [(4096, 64), (4096, 1), (128, 32), (64, 1)]:
            seq = torch.randint(1, 3 * window, (512,), generator=g)
            exact = _compact_lens_exact(seq, window, page).to(torch.int64)
            bound = _compact_lens_host(seq, window, page).to(torch.int64)
            self.assertTrue(
                bool((bound >= exact).all()),
                f"host bound under-shoots exact at window={window} page={page}",
            )

    def test_sawtooth_counterexample(self):
        # exact(4160) = 4096 < exact(4100) = 4100 at window=4096 page=64:
        # a host mirror of the exact math fed the reserved over-estimate
        # (4160 >= true 4100) would under-shoot; the envelope must not.
        window, page = 4096, 64
        true_len = torch.tensor([4100])
        reserved = torch.tensor([4160])
        exact_true = _compact_lens_exact(true_len, window, page).to(torch.int64)
        exact_reserved = _compact_lens_exact(reserved, window, page).to(torch.int64)
        self.assertLess(int(exact_reserved), int(exact_true))
        bound = _compact_lens_host(reserved, window, page).to(torch.int64)
        self.assertGreaterEqual(int(bound), int(exact_true))


class _FakeTpGroup:
    """Single-process stand-in for the TP GroupCoordinator: replays the
    concatenation of all ranks' recorded all-gather inputs."""

    def __init__(self, world_size):
        self.world_size = world_size
        self.phase = 0
        self.recorded = {}  # (rank, call_idx) -> tensor
        self.rank = 0
        self.call_idx = 0

    def all_gather_into_tensor(self, output, input_):
        if self.phase == 0:
            self.recorded[(self.rank, self.call_idx)] = input_.clone()
        else:
            output.copy_(
                torch.cat(
                    [self.recorded[(r, self.call_idx)] for r in range(self.world_size)]
                )
            )
        self.call_idx += 1

    def all_reduce(self, input_):
        reduce_key = ("reduce", self.rank, self.call_idx)
        if self.phase == 1:
            self.recorded[reduce_key] = input_.clone()
            output = input_
        elif self.phase == 2:
            output = sum(
                self.recorded[("reduce", r, self.call_idx)]
                for r in range(self.world_size)
            )
        else:
            output = input_
        self.call_idx += 1
        # Deliberately out-of-place, matching supported GroupCoordinator paths.
        return output


class TestDflashDraftSamplerVocabParallel(CustomTestCase):
    def _run(
        self, vocab, hidden, bs, block_size, world, dtype, weight=None, emit_probs=False
    ):
        from sglang.srt.speculative.dflash_worker_v2 import _DflashDraftSampler

        device = torch.device("cuda" if _HAS_CUDA else "cpu")
        g = torch.Generator(device=device).manual_seed(0)
        if weight is None:
            weight = torch.randn(vocab, hidden, generator=g, device=device, dtype=dtype)
        hs = torch.randn(
            bs * block_size, hidden, generator=g, device=device, dtype=dtype
        )
        shard = vocab // world
        group = _FakeTpGroup(world)
        samplers = [
            _DflashDraftSampler(
                weight=weight[r * shard : (r + 1) * shard].contiguous(),
                block_size=block_size,
                num_org=shard,
                org_vocab_start=r * shard,
                max_bs=bs,
                tp_group=group,
                emit_probs=emit_probs,
            )
            for r in range(world)
        ]
        # Phase 0 records all-gather inputs; phase 1 records partitions after
        # those gathers resolve; phase 2 replays the out-of-place all-reduce.
        for phase in range(3):
            group.phase = phase
            for r, s in enumerate(samplers):
                group.rank, group.call_idx = r, 0
                s(hs)

        n = bs * (block_size - 1)
        ref_hs = hs.view(bs, block_size, -1)[:, 1:, :].reshape(-1, hidden)
        ref = torch.argmax(torch.matmul(ref_hs.to(weight.dtype), weight.T), dim=-1).to(
            torch.long
        )
        for r, s in enumerate(samplers):
            torch.testing.assert_close(
                s.out[:n], ref, rtol=0, atol=0, msg=f"rank {r} mismatch"
            )
            if emit_probs:
                logits = torch.matmul(ref_hs.to(weight.dtype), weight.T).float()
                expected_probs = torch.softmax(logits, dim=-1).amax(dim=-1)
                torch.testing.assert_close(
                    s.out_probs[:n],
                    expected_probs,
                    rtol=1e-5,
                    atol=1e-6,
                    msg=f"rank {r} probability mismatch",
                )

    def test_matches_full_vocab_argmax(self):
        self._run(
            vocab=512, hidden=64, bs=3, block_size=8, world=4, dtype=torch.float32
        )

    def test_emits_selected_token_probability_for_adaptive_verify(self):
        from sglang.srt.speculative.dflash_worker_v2 import _DflashDraftSampler

        device = torch.device("cuda" if _HAS_CUDA else "cpu")
        generator = torch.Generator(device=device).manual_seed(7)
        bs, block_size, vocab, hidden = 2, 5, 64, 16
        weight = torch.randn(
            vocab,
            hidden,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        hidden_states = torch.randn(
            bs * block_size,
            hidden,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        sampler = _DflashDraftSampler(
            weight=weight,
            block_size=block_size,
            num_org=vocab,
            org_vocab_start=0,
            max_bs=bs,
            emit_probs=True,
        )
        sampler(hidden_states)

        draft_hidden = hidden_states.view(bs, block_size, hidden)[:, 1:].reshape(
            -1, hidden
        )
        logits = draft_hidden @ weight.T
        expected = torch.softmax(logits, dim=-1).amax(dim=-1)
        actual = sampler.out_probs[: bs * (block_size - 1)]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_tp_emits_globally_normalized_probability(self):
        self._run(
            vocab=64,
            hidden=16,
            bs=2,
            block_size=5,
            world=4,
            dtype=torch.float32,
            emit_probs=True,
        )

    def test_shard_boundary_tie_resolves_to_first_global_index(self):
        # Duplicate row 10 (shard 0) at row 200 (shard 1): identical logits, so
        # a correct fold must pick 10 (torch.argmax first-max semantics).
        vocab, hidden = 256, 32
        device = torch.device("cuda" if _HAS_CUDA else "cpu")
        weight = torch.zeros(vocab, hidden, device=device)
        weight[10] = 1.0
        weight[200] = 1.0
        self._run(
            vocab=vocab,
            hidden=hidden,
            bs=1,
            block_size=4,
            world=2,
            dtype=torch.float32,
            weight=weight,
        )


class TestSelectorAdaptiveProbability(CustomTestCase):
    def test_follows_realized_transition_rows(self):
        from sglang.srt.speculative.dflash_worker_v2 import _selector_path_probs

        candidate_ids = torch.tensor([[[10, 11], [20, 21], [30, 31]]])
        scores = torch.tensor(
            [
                [
                    [[0.0, 2.0], [9.0, 9.0]],
                    [[1.0, 3.0], [4.0, 0.0]],
                    [[0.0, 5.0], [2.0, 1.0]],
                ]
            ]
        )
        tokens = torch.tensor([[11, 20, 31]])

        actual = _selector_path_probs(
            candidate_ids=candidate_ids, scores=scores, tokens=tokens
        )
        expected = torch.stack(
            (
                torch.softmax(scores[0, 0, 0], dim=-1)[1],
                torch.softmax(scores[0, 1, 1], dim=-1)[0],
                torch.softmax(scores[0, 2, 0], dim=-1)[1],
            )
        )[None]
        torch.testing.assert_close(actual, expected)


class TestQuantizedHeadAdaptiveProbability(CustomTestCase):
    def test_emits_selected_token_probability(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        weight = torch.randn(13, 7, generator=torch.Generator().manual_seed(3))
        hidden = torch.randn(5, 7, generator=torch.Generator().manual_seed(4))

        class _FakeQuantMethod:
            @staticmethod
            def apply(layer, x, bias):
                del bias
                return x @ layer.qweight.T

        lm_head = SimpleNamespace(
            qweight=weight,
            quant_method=_FakeQuantMethod(),
            org_vocab_size=weight.shape[0],
        )
        with patch(
            "sglang.srt.speculative.dflash_worker_v2.get_tp_group",
            return_value=SimpleNamespace(world_size=1),
        ):
            tokens, probs = DFlashWorkerV2._greedy_sample_from_quantized_head(
                SimpleNamespace(),
                hidden_states=hidden,
                lm_head=lm_head,
                chunk_size=2,
                return_probs=True,
            )

        logits = hidden @ weight.T
        torch.testing.assert_close(tokens, logits.argmax(dim=-1))
        torch.testing.assert_close(
            probs, torch.softmax(logits, dim=-1).amax(dim=-1)
        )

@unittest.skipUnless(_HAS_CUDA, "triton kernel requires CUDA")
class TestRebuildCompactDraftReqToToken(CustomTestCase):
    def _legacy(self, draft, target, req_idx, start, lens, verify_2d, bs, block):
        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        lens64 = lens.to(torch.int64)
        max_len = int(lens64.max().item())
        offs = torch.arange(max_len, device=lens.device).unsqueeze(0)
        pos2d = start.to(torch.int64).unsqueeze(1) + offs
        mask = offs < lens64.unsqueeze(1)
        packed = target[req_idx.to(torch.int64)[:, None], pos2d.masked_fill(~mask, 0)][
            mask
        ].to(torch.int64)
        assign_req_to_token_pool_func(
            req_idx, draft, torch.zeros_like(lens), lens, packed, bs
        )
        assign_req_to_token_pool_func(
            req_idx, draft, lens, lens + block, verify_2d.reshape(-1), bs
        )

    def test_bitexact_vs_legacy(self):
        from sglang.kernels.ops.speculative.cache_locs import (
            rebuild_compact_draft_req_to_token_func,
        )

        device = torch.device("cuda")
        for bs, window, page, block, seed in [
            (1, 64, 1, 8, 0),
            (16, 64, 32, 8, 1),
            (13, 128, 64, 8, 2),
            (7, 512, 64, 16, 3),
        ]:
            g = torch.Generator(device=device).manual_seed(seed)
            pool_rows, width = 4 * bs, 4 * window
            seq = torch.randint(
                1, width - block - 1, (bs,), generator=g, device=device
            ).to(torch.int64)
            lens = _compact_lens_exact(seq, window, page).to(device)
            start = seq - lens.to(torch.int64)
            req_idx = torch.randperm(pool_rows, generator=g, device=device)[:bs]
            target = torch.randint(
                0, 2**30, (pool_rows, width), generator=g, device=device
            ).to(torch.int32)
            verify_2d = torch.randint(
                0, 2**30, (bs, block), generator=g, device=device
            ).to(torch.int64)
            draft_width = window + page + block + 8
            draft_a = torch.full(
                (pool_rows, draft_width), -1, dtype=torch.int32, device=device
            )
            draft_b = draft_a.clone()

            self._legacy(draft_a, target, req_idx, start, lens, verify_2d, bs, block)
            rebuild_compact_draft_req_to_token_func(
                draft_req_to_token=draft_b,
                target_req_to_token=target,
                req_pool_indices=req_idx,
                suffix_start=start,
                draft_prefix_lens=lens,
                verify_out_cache_loc_2d=verify_2d,
                batch_size=bs,
                block_size=block,
            )
            torch.testing.assert_close(draft_b, draft_a, rtol=0, atol=0)
            for i in range(bs):
                total = int(lens[i].item()) + block
                self.assertTrue(
                    bool((draft_b[req_idx[i], total:] == -1).all()),
                    "kernel wrote past the verify block",
                )


class TestHybridNeedsCpuSeqLens(CustomTestCase):
    def _make(self, prefill_flag, decode_flag, spec_mode="decode"):
        from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend

        def backend(flag):
            return SimpleNamespace(
                needs_cpu_seq_lens=flag,
                extend_dummy_seqs_capped_by_req_pool=False,
            )

        runner = SimpleNamespace(
            server_args=SimpleNamespace(speculative_attention_mode=spec_mode),
            kv_cache_dtype=torch.bfloat16,
            token_to_kv_pool=None,
            req_to_token_pool=None,
            kv_index_translator=None,
            model_config=SimpleNamespace(context_len=2048),
        )
        # The backend takes the mode from the published configuration, not from
        # the runner it is handed.
        override = get_context().override_server_args(
            speculative_attention_mode=spec_mode
        )
        override.install()
        self.addCleanup(override.restore)
        return HybridAttnBackend(runner, backend(prefill_flag), backend(decode_flag))

    def test_delegation(self):
        # Only backends serving the spec decode loop count: decode always,
        # prefill only when speculative_attention_mode routes verify to it.
        self.assertFalse(self._make(False, False).needs_cpu_seq_lens)
        self.assertFalse(self._make(True, False).needs_cpu_seq_lens)
        self.assertTrue(self._make(False, True).needs_cpu_seq_lens)
        self.assertTrue(self._make(True, False, spec_mode="prefill").needs_cpu_seq_lens)
        self.assertFalse(
            self._make(False, False, spec_mode="prefill").needs_cpu_seq_lens
        )


class TestFilterBatchHostIndices(CustomTestCase):
    def test_host_keep_list_matches_gpu_indices(self):
        from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2

        def make():
            info = DFlashDraftInputV2.create_idle_input(device=torch.device("cpu"))
            info.nxt_kv_lens_cpu = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
            info.nxt_kv_lens_sum = 100
            info.future_indices = torch.tensor([5, 6, 7, 8])
            return info

        keep = [0, 2]
        a, b = make(), make()
        a.filter_batch(new_indices=torch.tensor(keep))
        b.filter_batch(
            new_indices=torch.tensor(keep),
            new_indices_cpu=keep,
        )
        torch.testing.assert_close(a.nxt_kv_lens_cpu, b.nxt_kv_lens_cpu)
        self.assertEqual(a.nxt_kv_lens_sum, b.nxt_kv_lens_sum)
        torch.testing.assert_close(a.future_indices, b.future_indices)


if __name__ == "__main__":
    unittest.main()
