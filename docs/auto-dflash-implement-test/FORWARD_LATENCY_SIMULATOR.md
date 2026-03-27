# 前向延迟模拟（`ForwardLatencySimulator`）

本文说明在 **CUDA Graph 捕获完成之后、真实请求到达之前**，如何通过 **可配置 batch size 与每条序列的 `seq_lens`** 做一次前向传播延迟测量；与 `_dummy_run` 的用途区分开。

## 动机

- `_dummy_run` 为 warmup / autotune 设计：`seq_lens` 使用 attention backend 的 **统一填充值**（如 FlashAttention 常为 `1`），**无法表达「每条序列不同上下文长度」**。
- 本特性在独立模块中构造带 **逐序列 `seq_lens`** 的 `ForwardBatch`，走与运行时一致的 **`replay_prepare` → `init_forward_metadata_replay_cuda_graph`**（或 eager 回退），用于 **注意力算量与延迟随上下文分布变化** 的离线分析。

## `seq_lens` 默认值

| 场景 | 行为 |
|------|------|
| **未传** `--forward-latency-sim-seq-lens` | 该 batch 内 **全部 `bs` 条序列** 的 `seq_lens` 均为 **`ModelConfig.context_len`**（模型最大上下文长度，与 `model_config.context_len` 一致）。 |
| **传了 JSON**，但某 `batch_size` 未出现在映射中 | 同上，该 batch 仍全部为 **`context_len`**。 |
| **传了 JSON**，且某 key 对应列表 **长度 &lt; bs** | 已给出的前缀按列表取值，**剩余位置** 用 **`context_len`** 补齐。 |
| **列表长度 &gt; bs** | 截断为 **恰好 `bs` 个**。 |

实现见 `ForwardLatencySimulator._resolve_seq_lens`（`python/sglang/srt/model_executor/forward_latency_simulator.py`）。

> 注意：后续 `_build_forward_batch` 会把每条 `seq_len` **clamp 到 `[1, context_len]`**，避免越界。

## 与 DFLASH / 投机解码的关系

| 配置 | 模拟的前向模式 |
|------|----------------|
| **未启用** `speculative_algorithm`（或为 none） | `ForwardMode.DECODE`，每序列 **1 个** decode token（`num_tokens_per_bs = 1`）。 |
| **启用** Eagle / Standalone / NGRAM / **DFLASH** 等 | `ForwardMode.TARGET_VERIFY`，`num_tokens_per_bs` 与 target 侧 verify 长度对齐（DFLASH 见 `dflash_target_verify_num_tokens_per_bs` 或 `speculative_num_draft_tokens`）。 |

即：**有投机解码则模拟 verify，无则模拟 decode**，与需求一致。

## CLI 与 `ServerArgs` 字段

| CLI | `ServerArgs` 字段 | 说明 |
|-----|-------------------|------|
| `--forward-latency-sim-batch-sizes` | `forward_latency_sim_batch_sizes` | 逗号分隔 batch size，如 `1,4,8,16`。**未设置则不运行**模拟。 |
| `--forward-latency-sim-seq-lens` | `forward_latency_sim_seq_lens` | JSON：`{ "<bs>": [len0, len1, ...] }`，key 可为字符串或数字（解析后统一为 int）。 |
| `--forward-latency-sim-warmup` | `forward_latency_sim_warmup` | 计时前 warmup 次数，默认 **3**。 |
| `--forward-latency-sim-repeat` | `forward_latency_sim_repeat` | 计时重复次数，默认 **10**。 |

### 环境变量：结果写入（模拟投机解码时序格式）

```bash
export SGLANG_FORWARD_LATENCY_SIM_STATS_DIR=/path/to/output
```

设置后，每个 `(batch_size, seq_lens)` 配置测完会向 **`<output_dir>/querylen_<N>_batchsize_<M>.jsonl`** 追加一行 JSON，格式与 `SGLANG_SPEC_TIMING_STATS_DIR`（见 `SPEC_DECODE_STATS_RECORDER.md` 9.2/9.3）一致，便于与运行时 verify 耗时对比或合并分析。

| 字段 | 模拟器写入值 |
|------|--------------|
| `seq_lens` | 该 batch 的序列长度列表 |
| `avg_lens` | 平均长度 |
| `draft_times` | **0**（不模拟 draft） |
| `draft_extend_times` | **0**（不模拟 draft_extend） |
| `verify_times` | 模拟的 verify 前向耗时（ms） |
| `batch_size` | 并发数 |
| `query_len` | `num_tokens_per_bs` |

### 日志字段

- 启动行会打印 **`query_len`**：即 `num_tokens_per_bs`。在 **TARGET_VERIFY** 下为每条序列在一次 verify 前向里的 **query token 数**（与 `speculative_num_draft_tokens` / DFLASH verify 长度一致）；在 **DECODE** 下为 **1**。
- 每条配置会打印 **`total_tokens = bs * query_len`**：与 `ForwardBatch.input_ids` 长度一致。
- **`seq_lens`** 列表超过 8 条时，只打印前 8 个元素并附加 **`(+N more)`**（省略号写法，**不是问号**），避免日志过长。

### 示例

```bash
# 仅指定 batch sizes：每个 batch 内所有序列 seq_len = 安全默认（见上文「seq_lens 默认值」）
--forward-latency-sim-batch-sizes "1,4,8"

# 指定 bs=4 时四条序列为 1024/2048/4096/512，其余 batch 仍全为 context_len
--forward-latency-sim-batch-sizes "4,8" \
--forward-latency-sim-seq-lens '{"4": [1024, 2048, 4096, 512]}'
```

## 触发时机与调用链

1. `ModelRunner` 在完成 `init_piecewise_cuda_graphs`、`prealloc_symmetric_memory_pool` 之后调用 **`run_forward_latency_simulation()`**。
2. **`is_draft_worker == True` 时直接返回**（不跑模拟）。投机解码的 **draft** 进程（如 DFLASH 的 `DFlashDraftModel`）`forward` 需要 **`input_embeds`** 等来自 target 的输入，与 target 的 decode/verify 路径不同；延迟模拟只在 **target `ModelRunner`** 上执行一次即可。
3. 若 `forward_latency_sim_batch_sizes` 为空，直接返回。
4. 否则构造 `ForwardLatencySimulator(self)` 并调用 `run(...)`。
5. **TP rank 0** 打印汇总表（mean/std/min/max、`cuda_graph` 是否命中）。

核心逻辑 **不修改** `_dummy_run`；数据构造与 `model.forward` / graph replay **仅在** `forward_latency_simulator.py` 内完成。

## 数据构造要点（与运行时差异）

- **与运行时一致**：`ForwardBatch.seq_lens` / `seq_lens_cpu` 写入后，经 `populate_from_forward_batch` → `init_forward_metadata_replay_cuda_graph`，`cache_seqlens_int32`、`cu_seqlens_k` 等与真实 decode/verify 路径一致。
- **KV 访存对齐（v2 修复）**：`req_pool_indices = arange(bs)`，且在模拟前调用 `_prepare_scattered_kv_mapping` 向 `req_to_token_pool` 第 0..bs-1 行写入**非重叠的散列页索引**（KV pool size 取自 `mr.token_to_kv_pool.size`），使 FlashAttention 的 `page_table` 对每条序列指向不同物理 HBM 地址——与运行时「每请求独立散落页」更接近。`out_cache_loc` 仍全 `0`（写入侧不影响 KV 读取延迟）。
- **MoE 路由多样性（v3 修复）**：`input_ids` 改为 `torch.randint(0, vocab_size, ...)` 随机 token ID。`input_ids = zeros` 时所有 token 得到相同 embedding，MoE 路由集中于少数相同 expert，其权重被 L2 缓存复用，weight-loading 带宽压力远低于真实推理；随机 ID 使不同 token 路由到不同 expert，需从 HBM 加载不同 expert 权重矩阵，更贴近真实的 MoE 带宽开销。
- **CUDA Graph**：若 `graph_runner.can_run(forward_batch)` 为真则 **replay**；否则 **eager** `init_forward_metadata` + `model.forward`。

## 相关文件

| 文件 | 职责 |
|------|------|
| `python/sglang/srt/model_executor/forward_latency_simulator.py` | `ForwardLatencySimulator`、`parse_sim_batch_sizes` / `parse_sim_seq_lens`。 |
| `python/sglang/srt/model_executor/model_runner.py` | `run_forward_latency_simulation()` 及初始化末尾调用。 |
| `python/sglang/srt/server_args.py` | 上述 4 个字段与 CLI 注册。 |

## `seq_lens` 上限与非法显存访问（`cudaErrorIllegalAddress`）

注意力后端在 TARGET_VERIFY 模式下计算的是 **`max_seq_len_k = max(seq_lens) + num_tokens_per_bs`**，然后对 `strided_indices[:max_seq_pages]` 和 `page_table[:, :max_seq_pages]` 做索引，而这两个张量在 `init_cuda_graph_state` 时只分配了 `max_num_pages = ceil(context_len / page_size)` 个元素。

若 `seq_len == context_len`，则 `max_seq_len_k > context_len`，`max_seq_pages > max_num_pages`，导致 GPU 侧越界读写 → `cudaErrorIllegalAddress`。

在生产场景里不会出现此问题，因为 TARGET_VERIFY 时每条序列的上下文长度最多为 `context_len - draft_token_num`（草稿 token 还要占位）。模拟器中默认 seq_len 必须遵循相同的约束：

- `_resolve_seq_lens` 的填充默认值已改为 `context_len - num_tokens_per_bs`（而非 `context_len`）。
- `_build_forward_batch` 内也对用户显式传入的 seq_lens 做同样 clamp，确保不越界。

## 与运行时 verify 耗时的差异

可与 **`--enable-speculative-timing-logging`** / `SGLANG_SPEC_TIMING_STATS_DIR` 打出来的 **`verify_times`**（见 `spec_timing_stats_recorder.py`、`dflash_worker.py`）对照阅读。

### 运行时 `verify_times` 量了什么

在 `dflash_worker.py` 里，计时区间为：

1. `torch.cuda.synchronize()`
2. `verify_start = time.perf_counter()`
3. **`target_worker.forward_batch_generation(model_worker_batch, is_verify=True)`**
4. `torch.cuda.synchronize()`
5. `verify_time = perf_counter() - verify_start`

而 `forward_batch_generation`（`tp_worker.py`）在每次 verify 里会先做 **`ForwardBatch.init_new(model_worker_batch, model_runner)`**（从真实 `ScheduleBatch` 组 batch、LoRA、`clamp_position` / spec positions、部分 `.to(device)` 等），再调 **`model_runner.forward`**。

**不包含**：`verify_input.verify(...)`（接受/拒绝草稿、改 batch 等）——在计时结束之后执行。

### 模拟器量了什么

`_measure` 里每次循环是：`synchronize → perf_counter → _run_once → synchronize`。

`_run_once` 在走 CUDA Graph 时主要是 **`graph_runner.replay(forward_batch)`**，内部含 `replay_prepare` + `graph.replay()`，**不包含** 每次请求路径上的 `ForwardBatch.init_new`。

`ForwardBatch` 在 **进入 `_measure` 之前只构造一次**，warmup/repeat 共用同一份张量；而真实服务里每次 verify 都会 `init_new`（通常仍远小于整段 verify 的 GPU 时间，但非零）。

### 系统性偏差的常见原因

| 因素 | 模拟器 | 真实运行 |
|------|--------|----------|
| **KV / page_table**（**v2 已修复**） | 每序列使用独立 `req_pool_indices[i]=i`，`req_to_token_pool[i]` 写入非重叠散列页索引 → FlashAttention 读取 **不同 HBM 位置**，带宽压力接近线上；KV *值* 仍为 0（未真实计算），对算量分布有少量影响 | 每请求独立页表，**跨页、跨槽** 随机性大，DRAM 带宽与 TLB 压力高 |
| **输入内容 / MoE 路由**（**v3 已修复**） | v3 中 `input_ids = randint(0, vocab_size)` → 不同 token 路由到不同 MoE expert → 不同 expert 权重从 HBM 加载，带宽压力接近线上。AWQ dequant 分布仍与真实 token 略有差异，影响相对次要。 | 真实 token，MoE 路由多样 → 多种 expert 权重从 HBM 加载 |
| **DFLASH 辅助 hidden** | `spec_info` 为最小占位（如 `draft_token=None`） | 线上 `prepare_for_verify` 等会带 **真实 draft / mask / positions**，若模型侧有 **aux hidden 捕获** 等，算量更大 |
| **计时环境** | 启动后 **连续、紧耦合** 多次 replay，GPU 频率与 cache 处于「稳态」 | verify 与 **draft / draft_extend / 调度** 交替，间隔与竞争不同 |
| **其它** | 无真实 `init_new` 每轮成本（相对次要） | 含 `init_new` + `set_hicache_consumer` 等 |

v2+v3 修复后 KV 访存模式与 MoE 路由多样性均显著改善，剩余差距主要来自 **计时环境差异**（draft/调度交替 vs 连续 replay）与**启动时 log**，预计缩小到 1–3 ms 量级。启动时会打印 `[ForwardLatencySimulator] scattered KV mapping applied: ...` 日志，可确认分散映射是否生效。

### 若希望更接近线上

1. 以 **`verify_times`（及同批 `seq_lens`、`query_len`）** 为「真值」做回归，模拟器作 **相对** 对比（同机、同 commit、同配置）。
2. 接受模拟器偏 **乐观**（MoE 内容差异），在报告中 **显式注明**「下界 / shape-only 估计」。
3. 深度对齐需构造更接近真实的 `input_ids` / `spec_info`（工作量大，且仍难完全复现访存模式）。

## 与本文档目录下其它文档的关系

- **`dflash_verify_token_num.md`**：DFLASH target verify 长度 N 与 block_size 解耦、hack `ServerArgs` 与构图对齐。
- **`SPEC_DECODE_STATS_RECORDER.md`**：线上/实验运行时投机各阶段耗时统计。

本 **`FORWARD_LATENCY_SIMULATOR.md`** 描述的是 **启动后、接流量前** 的 **离线形状/延迟探针**，与上述运行时统计互补。
