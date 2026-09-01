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
| `fix/fix-dspark-verlen-confict-tp-ranks` | Community PR | Keep DSpark verify budgets and compact graph tiers consistent across TP ranks. |
| `fix/fix-dspark-compact-confidence-head-with-vargamma` | Community PR | Use runtime gamma in the DSpark confidence head. |
| `feat/adaptive_spec_dspark/main` | Maintained feature | Integrate DSpark runtime gamma, TP consistency, and batch-aware adaptive graph tiers. |

Machine-readable details live in [`branches.yaml`](branches.yaml). Composition
details for the maintained feature live in
[`stacks/dspark-adaptive.yaml`](stacks/dspark-adaptive.yaml).

## Update rules

1. Update `remote_head` after rebasing or force-pushing a registered branch.
2. Keep PR status and links current.
3. Record logical dependencies even when a patch was rebased and therefore has
   a different commit SHA in the integration branch.
4. Before rebuilding a maintained feature, create a backup tag and use
   `--force-with-lease` when publishing it.
5. Record conflict decisions and validation results in the corresponding stack
   manifest.

