# DFLASH 动态验证长度（`--dynamic-speculative-dflash-verify-tokens-config`）

本文说明 **DFLASH 目标模型（target）侧** 在启用 **FlashAttention** 时，如何按 **capture 列表中的 batch size** 为每个 `bs` 指定不同的 **verify query 长度（qlen）**，并完成 **CUDA Graph 捕获与运行时回放** 的对接；以及与 **draft-extend** 图、**固定** `--speculative-dflash-verify-token-num` 的关系。

> **范围**：仅 **target worker** + **TARGET_VERIFY** + **FA backend（topk≤1）**。Draft 侧 `DRAFT_EXTEND` 的 CUDA Graph **不参与**本特性。

---

## 动机与约束

- **解耦**：起草仍按 `speculative_num_draft_tokens`（block size）生成；验证前可只把前 `N` 个 draft token 送入 target（固定 `N` 见 `--speculative-dflash-verify-token-num`）。
- **按 bs 变 N**：不同并发 `bs` 可使用不同 `verify_len`（qlen），以在 **图数量不变** 的前提下，用 **更小的 `(bs × qlen)`** 捕获图，从而 **降低单图显存**（总 buffer 仍按 max qlen 分配，与原先一致）。
- **不额外捕获**：JSON 里出现的 `batch_size` 若 **不在** `--cuda-graph-bs` 推导出的 `capture_bs` 中，该条目 **静默忽略**（打 info 日志），**不会**为其实例化新图。
- **默认 qlen**：`capture_bs` 里 **未在 JSON 出现** 的 `bs`，使用默认 qlen：`speculative_dflash_verify_token_num`（若设置）否则 `speculative_num_draft_tokens`（block size）。

---

## 使用方法

### CLI 与 `ServerArgs`

| CLI | `ServerArgs` 字段 | 说明 |
|-----|-------------------|------|
| `--dynamic-speculative-dflash-verify-tokens-config` | `dynamic_speculative_dflash_verify_tokens_config` | JSON 文件路径；**仅 DFLASH target + CUDA/MUSA** 生效。 |
| `--speculative-dflash-verify-token-num` | `speculative_dflash_verify_token_num` | **可选**。作为 JSON 未覆盖时的 **默认 verify qlen**（仍须 ≤ block size）。 |
| `--speculative-num-draft-tokens` | `speculative_num_draft_tokens` | Block size；JSON 中每个 qlen 必须 **≤** 该值。 |
| `--cuda-graph-bs` | `cuda_graph_bs` | 显式指定要捕获的 batch size 列表；若缺省则走 `ServerArgs._generate_cuda_graph_batch_sizes()`（投机模式默认 **不含 9** 等部分整数，见下文）。 |

### JSON 格式

根对象：`"<batch_size>": [verify_len, ...]`。键为 **字符串或数字**（解析为 int）；值为 **非空列表**；当前实现 **只使用每个列表的第一个元素**，其余为 **将来多 qlen / 策略** 预留。

```json
{
  "1": [9],
  "2": [9],
  "3": [9],
  "4": [10],
  "5": [16],
  "6": [16],
  "7": [11],
  "8": [16],
  "10": [13]
}
```

### 启动校验（节选）

- 文件必须存在且为合法 JSON 对象。
- 设备须为 `cuda` 或 `musa`（与 `server_args` 中 DFLASH 动态 verify 校验一致）。
- 每个 `(bs, qlen)`：`qlen ∈ (0, block_size]`。

### 典型日志含义

1. **`batch_size=X from JSON not in capture_bs [...], ignored`**  
   JSON 写了 `X`，但当前 `capture_bs` 里没有 `X`（例如投机默认列表从 8 跳到 10，没有 9）。**不为其捕获图**。

2. **`bs->verify_len={...}` / `verify_len->batch_sizes={...}`**  
   合并后的 **每个 capture_bs → 实际用于捕获的 qlen**，以及按 qlen 分组的 bs 列表（便于核对）。

3. **`Capture cuda graph bs [...] (num_tokens_per_bs=16)`**  
   这里的 `num_tokens_per_bs` 是 **buffer 分配用的上限**（max qlen），**不是**每张图的真实 qlen；真实形状在捕获循环里为 `bs * dflash_bs_to_qlen[bs]`。

---

## 实现改动总览（按模块）

### 1. `python/sglang/srt/speculative/dflash_dynamic_verify_cuda_graph.py`（新建）

- `load_dflash_dynamic_verify_tokens_json(path)`：读 JSON，校验类型与取值。
- `build_dflash_bs_to_qlen(server_args, capture_bs)`：对 **每个** `capture_bs` 中的 `bs` 填默认 qlen，再用 JSON 中 **且属于 capture_bs** 的项覆盖；返回 `(merged_dict, groups_for_log)`。
- `resolve_verify_len_for_batch_size(raw_bs, sorted_capture_bs, bs_to_qlen)`：对 `raw_bs` 做 **与 CUDA graph padding 一致** 的 bisect（最小 `padded_bs ≥ raw_bs`），返回该 `padded_bs` 对应的 qlen。

### 2. `python/sglang/srt/server_args.py`

- 新增字段与 CLI：`dynamic_speculative_dflash_verify_tokens_config`。
- DFLASH 分支内：文件存在性、设备、JSON 根类型等校验。

### 3. `python/sglang/srt/model_executor/cuda_graph_runner.py`

- **仅当** `is_dflash() and not is_draft_worker and config 非空` 时，在已有 `capture_bs` 上调用 `build_dflash_bs_to_qlen`，得到 `self.dflash_bs_to_qlen`。
- **捕获**：对每个 `bs ∈ capture_bs`，`_current_capture_qlen = dflash_bs_to_qlen[bs]`，`num_tokens = bs * qlen`，图 key 为 `(bs, qlen)`（pdmux 下为带 stream 与 qlen 的字符串）。
- **can_run / replay_prepare / replay**：在 `dflash_bs_to_qlen is not None` 时，用 **padded bs + 表内 qlen** 组成与捕获相同的 key；`replay_prepare` 中 `raw_num_token = raw_bs * qlen`、`num_tokens_per_bs_for_batch = qlen`。
- **`require_mlp_tp_gather` + DFLASH dynamic**：`global_num_tokens = raw_bs * qlen`，不能再用 `// num_tokens_per_bs`（max qlen）直接除；当前代码用 **估计 bs → bisect 得 padded_bs → 查 qlen → 再反算 bs**，与 `replay_prepare` 对齐。

### 4. `python/sglang/srt/model_executor/model_runner.py`

- `init_device_graphs` 结束后，若 `graph_runner.dflash_bs_to_qlen` 存在，拷贝到  
  `dflash_dynamic_verify_bs_to_qlen` 与 `dflash_dynamic_verify_sorted_bs_keys`（供 worker 查表）。
- `resolve_dflash_verify_len_for_batch_size(raw_bs)`：有动态表则调用 `resolve_verify_len_for_batch_size`；否则返回固定默认（`speculative_dflash_verify_token_num` 或 block size）。  
  **注意**：对 `dflash_dynamic_verify_cuda_graph` 的 import 放在该方法内部，避免与 `server_args` / `model_runner` 的循环 import。

### 5. `python/sglang/srt/layers/attention/flashattention_backend.py`

- **TARGET_VERIFY 且 topk≤1**：`init_forward_metadata_capture_cuda_graph` / `init_forward_metadata_replay_cuda_graph` 中，`_qlen` 取自 `spec_info.draft_token_num`（缺省回退 `speculative_num_draft_tokens`），**metadata 字典 key 从 `bs` 改为 `(bs, _qlen)`**，与每张图的形状一致。

### 6. `python/sglang/srt/speculative/dflash_worker.py`

- 验证前截断：`verify_n = model_runner.resolve_dflash_verify_len_for_batch_size(bs)`，构造 `DFlashVerifyInput(..., draft_token_num=verify_n)`，保证 **tensor 长度、spec_info、FA replay key、CUDA graph key** 一致。

---

## 数据流（捕获 → 运行）

```text
build_dflash_bs_to_qlen(capture_bs, JSON)
    → dflash_bs_to_qlen[bs] = qlen

捕获（每个 bs）:
  _current_capture_qlen = qlen
  num_tokens = bs * qlen
  get_spec_info → DFlashVerifyInput.draft_token_num = qlen
  FA capture → target_verify_metadata[(bs, qlen)] = ...
  graphs[(bs, qlen)] = graph

运行（target verify）:
  dflash_worker: verify_n = resolve_dflash_verify_len_for_batch_size(raw_bs)
  forward_batch.spec_info.draft_token_num = verify_n
  replay_prepare: padded bs + qlen = dflash_bs_to_qlen[padded_bs]
  FA replay: target_verify_metadata[(bs, verify_n)]
  replay: graphs[(bs, _current_replay_qlen)]
```

**一致性条件**：`resolve_dflash_verify_len_for_batch_size(raw_bs)` 与 `replay_prepare` 里对 `raw_bs` / `global_num_tokens` 的 padding（bisect 到同一 `padded_bs`）必须一致；当前实现均基于 **同一 `capture_bs` 有序列表**。

---

## 与 draft-extend CUDA Graph 的关系

- **Draft `ModelRunner`**：`CudaGraphRunner.__init__` 中 **`not is_draft_worker`** 不满足，`dflash_bs_to_qlen` 恒为 `None`。
- Draft 侧仍按 **整 block** 的 `num_tokens_per_bs = speculative_num_draft_tokens` 捕获，图 key 仍为 **整数 `bs`**。
- **互不影响**。

---

## 默认 `capture_bs` 为何可能没有 9

投机解码默认生成（`ServerArgs._generate_cuda_graph_batch_sizes`，`speculative_algorithm is not None`）：

- `range(1, 9)` → 1..8  
- `range(10, 33, 2)` → 10, 12, …  

因此 **9 不在默认列表**。若 JSON 或策略需要 `bs=9`，需 **`--cuda-graph-bs`** 显式包含 `9`，或 **`--disable-cuda-graph-padding`**（连续 1..max）。

---

## 将来：捕获完成后「运行时策略」选择 qlen

当前：**每个 `bs` 在 JSON 里最多生效一个 qlen**（列表只取 `[0]`），运行时 **无分支**，直接查表。

若要 **同一 `bs` 多张图、运行时选一**：

1. **扩展 `build_dflash_bs_to_qlen`**（或新函数）  
   输出 `Dict[int, List[int]]`：`bs -> [qlen1, qlen2, ...]`，且 **仅对 `capture_bs` 内的 bs** 展开，避免偷偷增加「未在 capture 列表中的 bs」。

2. **扩展 `CudaGraphRunner.capture()`**  
   对每个 `bs` **内层再循环** `for qlen in qlens_for_bs:`，每次设置 `_current_capture_qlen = qlen`，以 **同一 `(bs, qlen)` key** 多捕获几张图（图数量 = Σ 每 bs 的候选数，需接受显存与启动时间）。

3. **运行时**  
   - 在 **`dflash_worker`**（或 scheduler）实现 `policy.select(raw_bs, candidate_qlens, context) -> qlen`。  
   - 将返回值写入 **`DFlashVerifyInput.draft_token_num`** 与截断长度 **`verify_n`**。  
   - **`replay_prepare` / `can_run` / `replay`**：在已有 `(padded_bs, qlen)` key 设计下，需根据 **当前选的 qlen** 组 key（而不是仅从 `dflash_bs_to_qlen[padded_bs]` 取单值）。可行改法：  
     - 把 `dflash_bs_to_qlen: Dict[int, int]` 改为 `Dict[int, List[int]]`，replay 时 **`qlen = spec_info.draft_token_num`**（与 worker 策略一致），**`can_run` 检查 `(padded_bs, qlen) in graphs`**；若不在则回退 eager 或报错，由产品策略决定。

4. **FA backend**  
   已使用 `(bs, qlen)` 作为 `target_verify_metadata` key，**一般无需再改**，只要 capture 阶段对 **每个用到的 `(bs, qlen)`** 都跑过一次即可。

---

## 已知限制

- **仅 FA + topk≤1**；`topk > 1` 的 target verify 路径仍按原 `bs` key，未接动态 qlen。
- **多 qlen 列表**：当前仅第一个元素有效；多候选需按上一节扩展捕获与 replay。
- **`ForwardLatencySimulator`** 等离线工具若模拟 TARGET_VERIFY，需自行与 `num_tokens_per_bs` / 动态配置对齐（见同目录 `FORWARD_LATENCY_SIMULATOR.md`）。

---

## 相关文件速查

| 路径 | 作用 |
|------|------|
| `speculative/dflash_dynamic_verify_cuda_graph.py` | JSON、合并表、bisect 解析 |
| `server_args.py` | CLI、校验 |
| `model_executor/cuda_graph_runner.py` | 捕获 key、replay、MLP gather 估计 |
| `model_executor/model_runner.py` | 暴露 `resolve_dflash_verify_len_for_batch_size` |
| `layers/attention/flashattention_backend.py` | `(bs, qlen)` metadata |
| `speculative/dflash_worker.py` | 运行时截断与 `draft_token_num` |
