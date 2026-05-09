# Speculative Decoding Step-Size ITL Cost Sweep

`benchmark/speculative/aicp_speculative_size_itl_cost_warmup.py` 是一个自动化 benchmark 脚本，用于在不同 **序列长度（seqlen）** 和 **并发数（batch_size）** 条件下，扫描多个 **投机解码步数（spec_size）**，量化每个组合相对于无投机解码 baseline 的 ITL（Inter-Token Latency）开销比，输出结构化的 JSON Lines 文件，供自适应策略选择最优 spec_size 使用。

---

## 背景

本脚本的目标是离线量化：在不同 **并发数（batch_size）**、**输入长度（seqlen）**、**起草步数（spec_size）** 组合下，**开启投机解码本身带来的时间开销**，即 `itl_cost = itl_spec / itl_baseline`（一次 spec cycle 耗时 / 一次 baseline decode 耗时）。

得到这张 `(seqlen, batch_size, spec_size) → itl_cost` 映射表后，结合各场景下数据集的实测**平均接受步长（mean accepted tokens per step）**，即可计算每个 `(seqlen, batch_size)` 下各 `spec_size` 的实际收益，从而选出最优起草步数：

```
effective_gain(spec_size) = mean_accepted_tokens / itl_cost(seqlen, batch_size, spec_size)
best_spec_size = argmax over spec_size of effective_gain
```

---

## ITL 定义

本脚本的 ITL 测量方式与 `inference_benchmark.py` 完全对齐：

- 使用 `/v1/chat/completions` 流式接口，以 `text/event-stream` SSE 格式接收响应。
- **第一个** 含 `delta.content` 的 SSE 事件作为 TTFT 锚点，**不计入** ITL。
- 每个后续含 `delta.content` 的 SSE 事件记录一个 gap：`gap = timestamp - most_recent_timestamp`。
- `most_recent_timestamp` 在每个 content 事件后更新。

| 模式 | 每个 SSE 事件的含义 | gap 代表 |
|------|-------------------|---------|
| Baseline（无投机） | 1 个 token | 1 次 decode forward 时间 |
| 投机解码 | 本轮接受的全部 token | 1 次完整 spec cycle（draft + verify + extend） |

所有 gap 取均值得到 `mean_itl_ms`，与接受率无关，可直接比较两种模式的"每步时间"。

---

## 工作流程

```
读取 config.json
    │
    ▼
启动 baseline 服务（无任何 --speculative-* 参数）
    │  poll /v1/models 直至就绪（进程崩溃立即报错）
    │  执行 1 次 warmup 请求
    │
    ├─ for seqlen in seqlen_list:
    │    for bs in batch_size_list:
    │      发送 bs 个并发 /v1/chat/completions 流式请求
    │      （随机文本 prompt，ignore_eos=True，stream=True）
    │      统计各请求的 SSE inter-chunk gap → mean_itl_ms
    │      记录 itl_baseline_ms → partial.jsonl
    │
    ▼  关闭 baseline 服务
    │
for spec_size in test_spec_size_list:
    │
    ▼
    启动 spec 服务（注入 --speculative-algorithm <algo>
    │              --speculative-num-steps <spec_size>
    │              --speculative-eagle-topk <topk>
    │              [--speculative-draft-model-path <path>]  ← 仅非 MTP 模式）
    │  warmup → sweep (seqlen × bs)
    │  记录 itl_spec_ms → partial.jsonl
    ▼  关闭 spec 服务

聚合：itl_cost = itl_spec_ms / itl_baseline_ms
写入 output_path (.jsonl，每行一条记录)
```

---

## 快速开始

### 1. 准备配置文件

```bash
cp benchmark/speculative/aicp_config.example.json my_sweep_config.json
# 按需修改 base_command、seqlen_*、batch_size_* 等参数
```

### 2. 运行扫描

```bash
python benchmark/speculative/aicp_speculative_size_itl_cost_warmup.py \
    --config my_sweep_config.json
```

脚本会依次启动 `1 + len(test_spec_size_list)` 个服务，全程自动管理服务生命周期，无需手动干预。

---

## 配置文件说明

```json
{
  "base_command": "<完整的 sglang 启动命令，见下方说明>",

  "seqlen_min": 64,
  "seqlen_max": 512,
  "seqlen_step": 64,
  "output_len": 256,

  "min_batch_size": 1,
  "max_batch_size": 32,
  "batch_size_step": 4,
  "batch_size_capture_list": [],

  "test_spec_size_list": [1, 3, 5, 7, 9],
  "speculative_algorithm": "EAGLE",
  "speculative_eagle_topk": 1,
  "speculative_draft_model_path": "",

  "combo_per_batch_size": 5,

  "host": "127.0.0.1",
  "warmup_seqlen": 512,
  "warmup_output_len": 32,
  "seed": 1,

  "output_path": "itl_cost_results.jsonl"
}
```

### `base_command` 填写规范

将你实际使用的 `sglang.launch_server` 启动命令**原样粘贴**为一个 JSON 字符串。脚本会自动：

1. 解析 `\` + 换行的 shell 续行符（也支持纯换行）。
2. **剥除** `--speculative-algorithm`、`--speculative-draft-model-path`、`--speculative-num-steps`、`--speculative-eagle-topk`、`--speculative-num-draft-tokens`、`--port` 这六个 flag 及其值。
3. 按 sweep 当前的参数重新注入（baseline 轮不注入）。

因此，带或不带 spec 参数、有没有 `--port` 都不影响正确性，直接粘贴即可：

```json
"base_command": "SGLANG_ENABLE_SPEC_V2=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server --model-path /models/Qwen/Qwen3.5-27B/ --tp-size 4 --dtype bfloat16 --mem-fraction-static 0.75 --context-length 40960 --max-running-requests 32 --disable-radix-cache"
```

### 参数说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `base_command` | `str` | — | sglang 启动命令（必填） |
| `seqlen_min` | `int` | `500` | 扫描的最小输入序列长度（token 数） |
| `seqlen_max` | `int` | `5000` | 扫描的最大输入序列长度 |
| `seqlen_step` | `int` | `500` | seqlen 步长，生成等差列表 |
| `output_len` | `int` | `256` | 每个请求的输出 token 数（需 > 1 才有 ITL） |
| `min_batch_size` | `int` | `1` | 最小并发数（始终作为列表第一项） |
| `max_batch_size` | `int` | `32` | 最大并发数（始终作为列表最后一项） |
| `batch_size_step` | `int` | `4` | 并发步长；自动生成序列 `[min, step, 2×step, …, max]` |
| `batch_size_capture_list` | `list[int]` | `[]` | 非空时忽略上面三个字段，直接使用此列表 |
| `test_spec_size_list` | `list[int]` | `[1,3,5,7,9]` | 待测的投机解码步数列表 |
| `speculative_algorithm` | `str` | `"EAGLE"` | 投机解码算法名（见下表） |
| `speculative_eagle_topk` | `int` | `1` | EAGLE topk，注入至所有 spec 服务 |
| `speculative_draft_model_path` | `str` | `""` | draft 模型路径；MTP 模式留空（见下表） |
| `combo_per_batch_size` | `int` | `3` | 每个并发点的请求倍数（`num_prompts = bs × combo`） |
| `host` | `str` | `"127.0.0.1"` | 服务监听地址 |
| `warmup_seqlen` | `int` | `512` | warmup 请求的输入长度 |
| `warmup_output_len` | `int` | `32` | warmup 请求的输出长度 |
| `seed` | `int` | `1` | 随机种子（保证可复现） |
| `output_path` | `str` | `"itl_cost_results.jsonl"` | 结果输出路径 |

### 投机解码算法配置

`speculative_algorithm` 对应 SGLang 的 `SpeculativeAlgorithm` enum（大小写不敏感）：

| 场景 | `speculative_algorithm` | `speculative_draft_model_path` |
|------|------------------------|-------------------------------|
| MTP / `SGLANG_ENABLE_SPEC_V2=1`（主模型自复用） | `"EAGLE"` | `""` |
| 经典 EAGLE1（独立 draft checkpoint） | `"EAGLE"` | `/path/to/draft-eagle1/` |
| EAGLE3 | `"EAGLE3"` | `/path/to/draft-eagle3/` |
| DFlash | `"DFLASH"` | `/path/to/draft-dflash/` |
| STANDALONE | `"STANDALONE"` | `/path/to/draft-standalone/` |
| NGRAM | `"NGRAM"` | `""` |

### `batch_size_list` 生成规则

```
min=1,  step=4, max=32  →  [1, 4, 8, 12, 16, 20, 24, 28, 32]
min=12, step=4, max=24  →  [12, 16, 20, 24]
min=2,  step=4, max=32  →  [2, 4, 8, 12, 16, 20, 24, 28, 32]
```

若 `batch_size_capture_list` 非空，则直接使用该列表（排序去重后），忽略以上三个参数。

---

## 输出格式

结果文件为 **JSON Lines**，每行一条完整记录：

```jsonl
{"seqlen": 64,  "batch_size": 1,  "spec_size": 3, "itl_baseline_ms": 6.06,  "itl_spec_ms": 15.89, "itl_cost": 2.62}
{"seqlen": 64,  "batch_size": 24, "spec_size": 3, "itl_baseline_ms": 10.43, "itl_spec_ms": 28.88, "itl_cost": 2.77}
{"seqlen": 128, "batch_size": 1,  "spec_size": 5, "itl_baseline_ms": 7.10,  "itl_spec_ms": 14.30, "itl_cost": 2.01}
...
```

| 字段 | 说明 |
|------|------|
| `seqlen` | 输入序列长度（token 数） |
| `batch_size` | 并发请求数 |
| `spec_size` | 本轮投机解码步数（`--speculative-num-steps`） |
| `itl_baseline_ms` | 无投机解码时的均值 ITL（毫秒）= 均值 decode-step 时间 |
| `itl_spec_ms` | 启用投机解码时的均值 ITL（毫秒）= 均值 spec-cycle 时间 |
| `itl_cost` | `itl_spec_ms / itl_baseline_ms`，**< 1.0 有收益，> 1.0 有损耗** |

中途崩溃时，已完成的 cell 保存在 `<output>.partial.jsonl`，重启后可手动合并。

读取示例（pandas）：

```python
import pandas as pd
df = pd.read_json("itl_cost_results.jsonl", lines=True)
# 找到每个 (seqlen, batch_size) 下 itl_cost 最低的 spec_size
best = df.loc[df.groupby(["seqlen", "batch_size"])["itl_cost"].idxmin()]
```

---

## 稳定性设计

| 风险 | 处理方式 |
|------|---------|
| TP / EP 多进程服务残留 | `terminate_process` → `kill_process_tree`，`atexit` + `SIGINT` / `SIGTERM` 三重兜底 |
| 服务启动失败（crash） | `wait_for_http_ready` 内每秒 `poll()` 进程，进程退出立即抛异常，不会挂等 |
| 服务加载缓慢 | 无硬超时，等到就绪为止；Ctrl+C 可随时中断 |
| Prefix cache 污染 | 每次 bench 使用随机文本 prompt（空格分隔的随机整数），天然无复用；建议 `base_command` 加 `--disable-radix-cache` |
| 中途崩溃丢数据 | 每个 cell 完成后追加写 `<output>.partial.jsonl` |
| 单 spec_size 失败 | 仅跳过该 spec_size，不影响其他 spec_size 和 baseline 的结果 |

---

## 扫描规模估算

全格扫描的测试点数为：

```
total_cells = len(seqlen_list) × len(batch_size_list) × (1 + len(test_spec_size_list))
```

以示例配置（`seqlen_step=64, max=512` → 8 个 seqlen；`step=4, max=32` → 9 个 bs；5 个 spec_size）为例：

```
8 × 9 × 6 = 432 个 bench 点，6 次服务启停
```

如需快速验证，建议粗粒度参数：

```json
{
  "seqlen_min": 64, "seqlen_max": 256, "seqlen_step": 64,
  "batch_size_capture_list": [1, 8, 24],
  "test_spec_size_list": [3, 5],
  "combo_per_batch_size": 3
}
```

此配置共 `4 × 3 × 3 = 36` 个 bench 点，扫描时间大幅缩短。

---

## 关键 seqlen 选择建议

`itl_cost` 的趋势（随 `batch_size` 增大是升还是降）**强烈依赖 seqlen**，原因是 FlashAttention 的 KV 共享效率随序列长度变化：

- **短 seqlen（< ~100 token）**：FlashAttention KV 共享收益有限，spec 模式的 KV 读取开销随 bs 增长更快 → `itl_cost` 随 bs **增大**。
- **较长 seqlen（> ~200 token）**：FlashAttention 摊薄了 KV 读取 → `itl_cost` 随 bs **降低**或趋于平稳。

> **建议：`seqlen_min` / `seqlen_max` 务必覆盖你实际生产工作负载的输入 token 分布，否则策略表在实际场景下会失效。**

---

## 终端输出示意

脚本运行时，终端同时显示两路输出：

- **sglang 服务日志**（服务进程继承终端 fd，直接打印）
- **脚本进度行**（前缀 `[progress]`，显示全局和局部计数及实时 ITL 统计）

```
════════════════════════════════════════════════════
[server] starting (spec_size=3)
  $ SGLANG_ENABLE_SPEC_V2=1 ... --speculative-algorithm EAGLE --speculative-num-steps 3 ...
════════════════════════════════════════════════════

────────────────────────────────────────────────────────────────────────
[progress] overall 15/432  local 6/72  |  spec_size=3  seqlen=128  bs=8  num_prompts=40
  → mean_itl=18.34ms  median=17.91ms  p95=22.10ms  chunks=9840
```
