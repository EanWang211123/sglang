# Speculative Decode Statistics Recorder

投机解码统计记录器，用于记录每个 verify 批次后的数据，供后续早停策略、动态步长等算法离线分析与调参。

---

## 1. 激活方式

```bash
export SGLANG_SPEC_STATS_DIR=/path/to/output
export SGLANG_SPEC_STATS_GAMMA=0.2   # 可选，feature_entropy 公式中的 γ，默认 0.2
```

不设置 `SGLANG_SPEC_STATS_DIR` 时，记录器不激活，**零运行时开销**。

---

## 2. 输出目录结构

```
<output_dir>/
├── batch/
│   └── batches.jsonl      # 每个 decode 批次一行
└── seqs/
    ├── seq_req-001.jsonl   # 每个序列一个文件，每参与一个批次追加一行
    ├── seq_req-002.jsonl
    └── ...
```

---

## 3. 索引约定（重要）

### 3.1 DFlash 的 verify 输入结构

DFlash 送入目标模型验证的 token 序列为：

| 位置 | 含义 | 是否起草产生 |
|------|------|--------------|
| **位置 0** | `verified_id` / 当前已提交 token | ❌ 否，上一步已确定 |
| **位置 1** | 第 1 个起草 token | ✅ 是 |
| **位置 2** | 第 2 个起草 token | ✅ 是 |
| **位置 3** | 第 3 个起草 token | ✅ 是 |
| **位置 4** | 第 4 个起草 token | ✅ 是 |

若 `block_size=5`，则 `draft_token_num=5`，**实际起草的 token 数 = 4**（`draft_steps = block_size - 1`）。

### 3.2 统计量中的索引规则

| 变量/字段 | 索引方式 | 含义 | 示例 |
|-----------|----------|------|------|
| `draft_position1` | **1-indexed** | 第 1 个起草 token（即 verify 输入的位置 1） | 第一个被起草的 token |
| `draft_position2` | **1-indexed** | 第 2 个起草 token | 第二个被起草的 token |
| `draft_positionK` | **1-indexed** | 第 K 个起草 token | K = 1..draft_steps |
| `draft_max_logits[i]` | **0-indexed 数组** | `draft_max_logits[i]` = 第 (i+1) 个起草位置的 max logit | `[0]`→position1, `[1]`→position2 |
| `accepted_tokens` | 整数 | 本批次该序列**被接受的起草 token 数** | 3 表示 position1,2,3 被接受 |
| `seq_len` | 整数 | verify **之前**的序列总长度（含 prompt + 已生成） | 128 |

### 3.3 索引对应关系速查

```
起草位置（1-indexed）  →  数组下标（0-indexed）  →  含义
─────────────────────────────────────────────────────────
draft_position1       →  j=0, draft_max_logits[0]  →  第 1 个起草 token
draft_position2       →  j=1, draft_max_logits[1]  →  第 2 个起草 token
draft_position3       →  j=2, draft_max_logits[2]  →  第 3 个起草 token
draft_position4       →  j=3, draft_max_logits[3]  →  第 4 个起草 token
```

若 `accepted_tokens=3`，表示 draft_position1、2、3 被接受，draft_position4 被拒绝。

---

## 4. 统计量说明

### 4.1 batch/batches.jsonl（每行一个批次）

| 字段 | 类型 | 含义 |
|------|------|------|
| `batch_id` | int | 单调递增的批次 ID |
| `seq_ids` | list[str] | 本批次包含的请求 ID |
| `draft_steps` | int | 本批次每个序列的起草 token 数（= block_size - 1） |
| `global_accept_rate` | float | 截至本批次的**全局**接受率 = 总接受数 / 总起草槽位数 |
| `avg_accepted_steps` | float | 截至本批次的**全局**平均每次 verify 接受的 token 数 |
| `position_accept_rates` | dict | `{"draft_position1": 0.9, "draft_position2": 0.8, ...}` 各起草位置的历史接受率 |
| `max_accepted_tokens` | int | 本批次中，被接受最多的序列接受了几个 token |
| `min_accepted_tokens` | int | 本批次中，被接受最少的序列接受了几个 token |

### 4.2 seqs/seq_&lt;rid&gt;.jsonl（每行 = 该序列在某批次的一次参与）

| 字段 | 类型 | 含义 |
|------|------|------|
| `seq_id` | str | 请求 ID |
| `batch_id` | int | 对应的批次 ID |
| `draft_steps` | int | 本批次的起草 token 数 |
| `seq_global_accept_rate` | float | 该序列**历史累计**接受率 |
| `seq_avg_accepted_steps` | float | 该序列**历史累计**平均每次 verify 接受的 token 数 |
| `seq_position_accept_rates` | dict | 该序列各起草位置的**历史**接受率 |
| `seq_len` | int | verify **之前**的序列长度 |
| `draft_max_logits` | list[float] \| null | 本批次各起草位置的 max logit，`[0]`=position1，`[1]`=position2，… |
| `feature_entropy` | list[float] \| null | 本批次各起草位置的 1−√(γ·H)，仅 EAGLE 等有完整 logits 时非 null |
| `accepted_tokens` | int | 本批次该序列被接受的起草 token 数 |

---

## 5. 使用示例

### 5.1 读取并解析

```python
import json

# 读取某个批次的统计
with open("batch/batches.jsonl") as f:
    for line in f:
        batch = json.loads(line)
        print(f"batch_id={batch['batch_id']}, "
              f"draft_steps={batch['draft_steps']}, "
              f"accept_rate={batch['global_accept_rate']}")

# 读取某序列的逐批次记录
with open("seqs/seq_req-001.jsonl") as f:
    for line in f:
        row = json.loads(line)
        # draft_max_logits[0] = 第 1 个起草位置的 max logit
        # draft_max_logits[1] = 第 2 个起草位置的 max logit
        if row["draft_max_logits"]:
            pos1_max = row["draft_max_logits"][0]  # draft_position1
            pos2_max = row["draft_max_logits"][1]  # draft_position2
        accepted = row["accepted_tokens"]  # 被接受了几个
```

### 5.2 索引对应示例

假设 `draft_steps=4`，某序列 `accepted_tokens=3`，`draft_max_logits=[3.2, 2.1, 1.8, 0.9]`：

| 起草位置 | 数组下标 | max logit | 是否被接受 |
|----------|----------|-----------|------------|
| draft_position1 | 0 | 3.2 | ✅ 是 |
| draft_position2 | 1 | 2.1 | ✅ 是 |
| draft_position3 | 2 | 1.8 | ✅ 是 |
| draft_position4 | 3 | 0.9 | ❌ 否（第一个被拒绝） |

`accepted_tokens=3` 表示前 3 个起草 token 被接受，第 4 个被拒绝。

---

## 6. 代码改动摘要

### 6.1 dflash_worker.py

| 改动点 | 说明 |
|--------|------|
| `_greedy_sample_from_vocab_parallel_head` | 新增 `return_max_logit` 参数，返回 `(token_ids, max_logits|None)`。TP=1 时用 `max` 替代 `argmax` 顺手取 max；TP>1 时对已有 `gathered_max` 做 `max(dim=0)`，**无额外通信**。 |
| `_prepare_for_speculative_decoding` | 调用时传入 `return_max_logit=(recorder is not None)`，将返回的 `max_logits` 存到 `self._stats_draft_max_logits`。 |
| `forward_batch_generation` | verify 后调用 `recorder.record_verify_step(..., draft_max_logits=self._stats_draft_max_logits)`，不再传 `draft_hidden` / `lm_head`。 |
| `__init__` | 移除 `_stats_draft_hidden`、`_stats_lm_head`，改为 `_stats_draft_max_logits`。 |

### 6.2 spec_decode_stats_recorder.py

| 改动点 | 说明 |
|--------|------|
| `record_verify_step` | 参数改为 `draft_max_logits`、`draft_full_logits`，移除 `draft_hidden`、`lm_head`。 |
| 移除 `_compute_logit_stats` | 不再在 recorder 内做 matmul，避免重复计算。 |
| 新增 `_compute_stats_from_full_logits` | 仅当 EAGLE 等传入 `draft_full_logits` 时调用，用于计算 max log-prob 和 feature_entropy。 |
| DFlash 路径 | 直接使用 `draft_max_logits.cpu().tolist()`，无 GPU 计算。 |

---

## 7. 边界情况检查

### 7.1 索引一致性

- `position_accept_rates` 的 key 为 `draft_position{pos}`，`pos = j + 1`（1-indexed）。
- `acc_len > j` 表示第 `j+1` 个起草位置被接受；`j` 从 0 到 `num_draft_slots-1`。
- `draft_max_logits` 的 list 下标 `j` 对应 `draft_position{j+1}`，与上述一致。

### 7.2 accept_length_per_req_cpu 的含义

DFlash verify 中：`accept_length_per_req_cpu[i] = max(0, appended - 1)`，其中 `appended` 为本批次实际提交到 `output_ids` 的 token 数（含 1 个 bonus token）。因此：

- `accepted_tokens = appended - 1` = 本批次**被接受的起草 token 数**
- 若因 stop、max_new_tokens 等提前截断，`appended` 可能小于 `acc_len + 1`，此时以实际提交数为准
- `appended = 0` 时，`accepted_tokens = 0`

### 7.3 空批次

- `batch_reqs` 为空时直接 return，不写入。
- `num_draft_slots=0` 时，`draft_max_logits_list` 和 `feature_entropies` 保持 None。

### 7.4 DFlash 与 EAGLE 的差异

| 后端 | draft_max_logits 来源 | feature_entropy |
|------|------------------------|-----------------|
| DFlash | `_greedy_sample_from_vocab_parallel_head` 返回的全局 max logit（未归一化） | null（无完整分布） |
| EAGLE | 从 `draft_full_logits` 算出的 max log-prob | 从 `draft_full_logits` 算出 |

---

## 8. 扩展其他投机解码路径

在 EAGLE 等路径的 verify 之后调用：

```python
if self._stats_recorder is not None:
    # EAGLE 有完整 logits，传入 draft_full_logits
    self._stats_recorder.record_verify_step(
        batch_reqs=batch.reqs,
        draft_token_num=...,
        accept_length_per_req_cpu=...,
        seq_lens_before_verify=...,
        draft_full_logits=draft_logits_output.next_token_logits,  # [N, vocab_size]
    )
```

若某后端只有 max logit、没有完整 logits，则传 `draft_max_logits`，不传 `draft_full_logits`。

---

## 9. 投机解码时序统计（Speculative Timing Stats）

用于统计每个 run-batch 中 draft、verify、draft-extend 三阶段的耗时，写入 `batchsize_xxx.jsonl`。

### 9.1 激活方式

`--enable-speculative-timing-logging` 或 `SGLANG_SPEC_TIMING_STATS_DIR` 满足其一即可启用同步与耗时统计。仅 `--enable` 会在终端打印日志（rank 0）；设置环境变量时写入 `batchsize_xxx.jsonl`。

```bash
# 方式一：仅终端日志（rank 0）
--enable-speculative-timing-logging

# 方式二：仅写入文件，不打印终端
export SGLANG_SPEC_TIMING_STATS_DIR=/path/to/output

# 方式三：终端 + 文件
--enable-speculative-timing-logging
export SGLANG_SPEC_TIMING_STATS_DIR=/path/to/output
```

### 9.2 输出目录结构

```
<output_dir>/
├── querylen_16_batchsize_1.jsonl
├── querylen_16_batchsize_2.jsonl
├── querylen_16_batchsize_4.jsonl
└── ...
```

### 9.3 每行 JSON 字段

| 字段 | 类型 | 含义 |
|------|------|------|
| `seq_lens` | list[int] | 该批次内每个序列的长度 |
| `avg_lens` | float | 该批次所有序列的平均长度 |
| `draft_times` | float | 起草时间（ms） |
| `draft_extend_times` | float | draft-extend 时间（ms） |
| `verify_times` | float | 大模型验证时间（ms） |
| `batch_size` | int | 该批次并发数 |
| `query_len` | int | 验证 tokens（1+验证步长），单批次内统一 |
