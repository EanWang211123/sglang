# DSpark Compact Verify Performance Optimization

## Background

Compact verify packs each request according to its selected verify length and
keys CUDA Graphs only by the resulting batched-token tier. This is the intended
design and must remain unchanged: it avoids a Cartesian product of batch-size
and token-count graph variants.

Hybrid GDN and KDA target verification historically assumed a uniform
`[batch_size, draft_token_num]` layout. Compact verify instead supplies
`sum(verify_lens)` packed rows. Applying the uniform assumption to that input
can cause invalid reshapes, token-to-request misalignment, and illegal memory
accesses.

The initial correctness fix converts packed rows to a dense per-request window
before causal convolution and gathers the valid rows afterwards. This makes
ragged verification correct, but introduces work that static verification does
not perform. The cost is most visible at low concurrency because a token-tier
graph may contain more captured request slots than live requests.

This document records the follow-up optimizations needed to recover that cost
without changing the batched-token CUDA Graph indexing model.

## Constraints

- Keep compact CUDA Graphs keyed only by batched-token tier.
- Preserve variable per-request verify lengths and graph padding semantics.
- Preserve `intermediate_conv_window` for acceptance-time Mamba state commit.
- Keep GDN and KDA behavior equivalent to their uniform dense reference paths.
- Keep each optimization independently reviewable and revertible.
- Do not trade correctness for a static fallback on an adaptive request.

## Optimization 1: Native packed varlen causal convolution

### Why it is needed

The current compatibility path performs the following operations in every GDN
or KDA layer:

```text
packed QKV
  -> scatter into [captured_slots, draft_window, dim]
  -> fixed-width causal_conv1d_update
  -> gather valid rows back into packed QKV
```

Even after removing redundant buffer initialization, the scatter and gather
remain two additional memory kernels per linear-attention layer. The dense
temporary also reflects captured slots rather than live requests.

### Proposed change

Add a graph-capturable varlen causal-convolution kernel that:

- consumes packed `[num_tokens, dim]` input directly;
- reads request boundaries from `query_start_loc`;
- processes at most `draft_token_num` sequential positions per request;
- reads the request's initial convolution state through `cache_indices`;
- writes packed output directly;
- writes the per-step `intermediate_conv_window` used by state commit;
- skips padded request slots whose cache index is `-1`.

The launch remains compatible with the captured token tier. No new graph key or
graph variant is introduced.

### Expected result

- Remove packed-to-dense scatter and dense-to-packed gather from every GDN/KDA
  layer.
- Remove the dense causal-convolution scratch tensor.
- Preserve the existing recurrent varlen verify kernels and commit path.

## Optimization 2: Compact epilogue identity fast path

### Why it is needed

The compact epilogue scatters logits and hidden states back to a fixed-stride
layout. When every request verifies the full window, packed and strided layouts
are identical for live requests, but the generic scatter still runs.

### Proposed change

Teach the graph-captured epilogue to recognize a full uniform layout from its
device-side verify lengths and use an identity copy/alias-compatible path. The
decision must remain inside the existing graph and must not create a separate
batch-size graph family.

### Expected result

Reduce full-budget overhead in profiler runs and in production steps where the
adaptive cost model selects the complete verify window.

## Optimization 3: Fuse ragged verify-window construction

### Why it is needed

Building compact verify input currently performs separate indexing work for
token IDs, positions, and cache locations. These small kernels and temporary
tensors are proportionally expensive at low concurrency.

### Proposed change

Fuse construction of packed token IDs, positions, request rows, and cache
locations into one graph-safe kernel where their ownership and dtype contracts
allow it. Reuse cached uniform layout objects when the selected budget covers
the complete window.

### Expected result

Reduce per-step launch count and allocator pressure before target verification.

## Optimization 4: Reduce cost-model control-plane overhead

### Why it is needed

Adaptive serving must compute and relay confidence and select a verify budget.
This work is required for partial budgets, but some operations are unnecessary
when a forced or resolved budget already covers the full window.

### Proposed change

- Keep full-budget detection ahead of device top-k scheduling.
- Cache uniform layouts by local request count and DP-global tier.
- Avoid redundant host/device conversions in forced-budget profiling.
- Preserve confidence history while skipping calculations whose result is
  already fixed by the forced budget.

### Expected result

Reduce fixed per-step CPU overhead without changing adaptive budget decisions.

## Commit plan

Each item is submitted as a separate source commit:

1. `perf(linear-attn): add packed varlen verify convolution`
2. `perf(dspark): optimize full-layout compact epilogue`
3. `perf(dspark): fuse compact verify window construction`
4. `perf(dspark): reduce adaptive budget control overhead`

Correctness fixes and performance changes must not be squashed together. A
later optimization may depend on an earlier one, but every commit must retain a
working compact verify path.

## Validation

For each optimization:

- compare kernel outputs and intermediate convolution windows with the dense
  reference for mixed verify lengths;
- cover graph padding, ghost tokens, and non-divisible packed token counts;
- replay one captured token tier with more than one live layout;
- compare static and compact forced-full throughput at batch sizes
  `1 2 4 8 16 24 32 48`;
- compare adaptive end-to-end throughput and acceptance metrics;
- run a multimodal mRoPE correctness case in addition to text-only workloads.

The principal reference platform is 2x H200 with the Qwen3.8-27B target and its
DSpark draft checkpoint.
