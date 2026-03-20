# DFLASH：解耦验证长度（`--speculative-dflash-verify-token-num`）

本文说明 SGLang 中 **DFLASH** 如何将 **draft 块大小（block_size）** 与 **target 验证 token 数** 分开配置，以及实现上为何对 `ServerArgs` 做临时覆盖、各模块如何对齐形状。

## 动机

- **block_size**（通常由 `--speculative-num-draft-tokens` / DFLASH 的 `--speculative-dflash-block-size` 表示）决定 draft 一次产出多少候选 token。
- 默认情况下，target **TARGET_VERIFY** 路径会按同一长度构图、跑 attention 与 CUDA Graph。
- 若希望 **draft 仍跑满 block**，但 **target 只验证前 N 个 token**（降低 verify 侧算力 / 图规模），需要单独参数 **`--speculative-dflash-verify-token-num`**。

约束：**N 必须在 `(0, block_size]`**。校验见 `ServerArgs` 中 DFLASH 相关逻辑。

## 用户可见行为

| 项目 | 未设置 verify-token-num | 已设置 `= N` |
|------|-------------------------|--------------|
| Draft | 产出 `block_size` 个 token | 不变，仍为 `block_size` |
| Target verify | 验证 `block_size` 个 | 只验证前 **N** 个；`DFlashVerifyInput.draft_token_num = N` |
| 共享 `ServerArgs.speculative_num_draft_tokens`（运行时） | 即 block_size | **仍为 block_size**（draft 与其它读者不受影响） |

> 实现上，仅在 **target 进程** 初始化 attention / 捕获 CUDA Graph 的短窗口内，会把 `speculative_num_draft_tokens` **临时改成 N**；结束后恢复为 block_size。

## CLI 与配置字段

- **命令行**：`--speculative-dflash-verify-token-num <int>`
- **`ServerArgs` 字段**：`speculative_dflash_verify_token_num: Optional[int]`
- **帮助文案**（摘要）：仅 DFLASH；验证用的 draft token 个数，必须 `<= block_size`；draft 仍产出完整 block。

无效配置会在启动校验阶段抛出 `ValueError`（例如 `N > block_size` 或 `N <= 0`）。

## 设计思路：为何临时覆盖 `speculative_num_draft_tokens`

Target 侧大量代码（各 **attention backend**、`CudaGraphRunner` 等）在 **构造 / capture** 时直接读 **`ServerArgs.speculative_num_draft_tokens`** 作为「每序列 verify 长度」或相关形状来源。

若要为 DFLASH 单独增加「verify 长度」并去改每一个 backend，改动面很大。当前做法与 **AutoSpec（PR #17749）** 类似：

1. **进入**（`ModelRunner._dflash_target_verify_token_hack_enter`）：若 DFLASH + **非 draft worker** + 已设置 `speculative_dflash_verify_token_num`，则  
   - 保存当前的 `speculative_num_draft_tokens`（一般为 block_size）；  
   - 将其设为 **N**；  
   - 把 N 记入 `ModelRunner.dflash_target_verify_num_tokens_per_bs`（供后续 `_dummy_run` 等与图长度一致）。
2. **退出**（`_dflash_target_verify_token_hack_exit`）：在 `finally` 里把 `speculative_num_draft_tokens` **恢复**为保存值。

**Draft worker** 不会进入上述分支，始终按 block_size 使用共享 `ServerArgs`。

## 代码路径一览

### 1. `server_args.py`

- 定义 `speculative_dflash_verify_token_num` 与 CLI。
- DFLASH 校验：`N in (0, block_size]`。

### 2. `model_executor/model_runner.py`

- **`_dflash_target_verify_token_hack_enter` / `_dflash_target_verify_token_hack_exit`**  
  在以下阶段用 `try/finally` 包裹：  
  - CUDA/MUSA：`init_attention_backend`、`kernel_warmup`、`init_device_graphs`  
  - NPU/CPU：`init_attention_backend`、`init_device_graphs`  
  - 其它 device：仅 `init_attention_backend`  
  - **`update_weights` 触发重捕 CUDA Graph**：仅 `init_device_graphs` 段  

  保证捕获图时读到的「draft/verify 长度」为 **N**，而全局 `ServerArgs` 在窗口外回到 **block_size**。

- **`dflash_target_verify_num_tokens_per_bs`**  
  在 hack enter 时设为 N；**exit 之后仍保留**，因为 `_dummy_run` 等路径在 `ServerArgs` 已恢复后仍需用 **N** 与已捕获图对齐。

- **`_dummy_run`（TARGET_VERIFY + DFlash）**  
  `num_tokens_per_bs` / `DFlashVerifyInput.draft_token_num`：若 `dflash_target_verify_num_tokens_per_bs` 已设置则用之，否则回退 `server_args.speculative_num_draft_tokens`。

### 3. `model_executor/cuda_graph_runner.py`

- `CudaGraphRunner` 的 `num_tokens_per_bs` 在初始化时来自当时的 `ServerArgs`；DFLASH target 在 capture 窗口内已被 hack 为 **N**。
- **`get_spec_info`** 中 DFLASH 分支：`DFlashVerifyInput(..., draft_token_num=self.num_tokens_per_bs)`，与 Eagle/NGRAM 一样，**长度与 runner 的 tokens_per_bs 一致**，无需再 unwrap 其它 backend 形状。

### 4. `speculative/dflash_worker.py`

- 计算 `verify_n`：若设置了 `speculative_dflash_verify_token_num` 则为该值，否则为 `block_size`。
- 对 `draft_tokens`、`positions_2d` **切片**到前 `verify_n`，构造 `DFlashVerifyInput(draft_token_num=verify_n, ...)`，再 `prepare_for_verify` 与 `ForwardMode.TARGET_VERIFY`。

此处使用的是 **恢复后的** `ServerArgs`（block_size 仍在字段里用于 draft）；`verify_n` 单独来自 `speculative_dflash_verify_token_num`，与 target 构图长度一致。

## 不变量与排查要点

1. **图捕获长度 == 运行时 verify 长度**：target 的 CUDA Graph / attention 按 **N** 捕获；worker 侧必须只送 **N** 个 token 进 verify，否则形状不匹配。
2. **Draft 始终按 block_size**：共享 `ServerArgs.speculative_num_draft_tokens` 在 hack 外应为 block_size；draft worker 不执行 hack。
3. **权重更新重捕图**：若开启重捕，需同样包一层 enter/exit，否则新图可能按错误的 `speculative_num_draft_tokens` 捕获。

## 相关符号速查

| 符号 | 含义 |
|------|------|
| `block_size` | DFLASH draft 块长，通常 `speculative_num_draft_tokens` |
| `verify_n` / `N` | `speculative_dflash_verify_token_num`，验证前 N 个 draft token |
| `dflash_target_verify_num_tokens_per_bs` | Target 侧记录的 verify 长度，供 graph warmup 等与 hack 后长度一致 |

---

*文档对应实现位于上述 Python 模块；若行为变更，请同步更新本节。*
