# SGLang Fork Branch Registry

This orphan branch records how long-lived branches in `EanWang211123/sglang`
are maintained. It is metadata only and must never be merged into a source
branch.

## Branch types

- `community_pr`: a focused branch submitted to `sgl-project/sglang`.
- `maintained_feature`: a deployable integration branch maintained in the fork.

## Current registry

| Branch | Type | Purpose |
| --- | --- | --- |
| `fix/dspark-sps-profiler-verify-all` | Community PR | Fix profiler budget pins, ragged speculative mRoPE positions, and compact GDN target verification. |
| `feat/dflash/dp-attn` | Community PR | Enable DFlash speculative decoding under DP attention with DP-local draft execution and synchronization. |
| `fix/fix-dspark-verlen-confict-tp-ranks` | Community PR | Keep DSpark verify budgets and compact graph tiers consistent across TP ranks. |
| `fix/fix-dspark-compact-confidence-head-with-vargamma` | Community PR | Use runtime gamma in the DSpark confidence head. |
| `fix/gptq-marlin-moe-bf16-w2-scales` | Community PR | Fix BF16 and TP act-order w2 scale handling in GPTQ Marlin MoE. |
| `feat/ssm-ring-replay/main` | Community PR | Support ReplaySSM verification for DFlash and DSpark on hybrid GDN models. |
| `feat/adaptive_spec_dspark/main` | Maintained feature | Integrate DSpark runtime gamma, TP consistency, batch-aware adaptive graph tiers, and online startup cost profiling. |
| `feat/pack/main` | Maintained feature | Combine adaptive DSpark, ReplaySSM, and GPTQ Marlin MoE BF16 fixes in one development branch. |
| `xkernel/slo-aware-prefill-controller` | Maintained feature | Rebase xkernel's SLO-aware prefill controller onto `feat/pack/main`. |

Machine-readable details live in [`branches.yaml`](branches.yaml). Composition
details for the maintained features live in
[`stacks/dspark-adaptive.yaml`](stacks/dspark-adaptive.yaml) and
[`stacks/adaptive-dspark-replayssm-gptq-marlin-pack.yaml`](stacks/adaptive-dspark-replayssm-gptq-marlin-pack.yaml), and
[`stacks/xkernel-slo-aware-prefill-controller.yaml`](stacks/xkernel-slo-aware-prefill-controller.yaml).

The generated branch and stack diagrams live in
[`BRANCH_GRAPH.md`](BRANCH_GRAPH.md).

## Generate the branch graph

Install the only dependency and render the graph from the YAML registry:

```bash
python -m pip install -r requirements.txt
python scripts/render_branch_graph.py
```

Check that the committed graph is current without modifying files:

```bash
python scripts/render_branch_graph.py --check
```

Refresh every registered `remote_head` with `git ls-remote`, update
`last_updated`, and regenerate the graph:

```bash
python scripts/render_branch_graph.py --refresh-heads
```

## Update rules

1. Update `remote_head` after rebasing or force-pushing a registered branch.
2. Keep PR status and links current.
3. Record logical dependencies even when a patch was rebased and therefore has
   a different commit SHA in the integration branch.
4. Before rebuilding a maintained feature, create a backup tag and use
   `--force-with-lease` when publishing it.
5. Record conflict decisions and validation results in the corresponding stack
   manifest.
6. Regenerate `BRANCH_GRAPH.md` after changing registry or stack YAML.
