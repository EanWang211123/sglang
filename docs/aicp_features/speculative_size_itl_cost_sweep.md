# Speculative Decoding Step-Size ITL Cost Sweep

`benchmark/speculative/aicp_speculative_size_itl_cost_warmup.py` 是一个自动化 benchmark 脚本，用于在不同 **序列长度（seqlen）** 和 **并发数（batch_size）** 条件下，扫描多个 **投机解码步数（spec_size）**，量化每个组合相对于无投机解码 baseline 的 ITL（Inter-Token Latency）开销比，输出结构化的 JSON Lines 文件，供自适应策略选择最优 spec_size 使用。

---

## 背景

投机解码（Speculative Decoding，EAGLE 等）在低并发、短上下文时可以显著降低 ITL，但在高并发或长序列下草稿步数越多反而可能拖慢 decode，产生 `itl_cost > 1`。要在运行时动态选择最优 spec_size，首先需要一张离线测得的 `(seqlen, batch_size, spec_size) → itl_cost` 映射表，本脚本就是生成这张表的工具。

---

## 工作流程

```
读取 config.json
    │
    ▼
启动 baseline 服务（无投机解码）
    │  poll /v1/models 直至就绪（进程崩溃立即报错）
    │  执行 1 次 warmup 请求
    │
    ├─ for seqlen in seqlen_list:
    │    for bs in batch_size_list:
    │      POST /flush_cache
    │      bench_serving (random dataset, ignore_eos, stream)
    │      记录 itl_baseline_ms → partial.jsonl
    │
    ▼  关闭 baseline 服务
    │
for spec_size in test_spec_size_list:
    │
    ▼
    启动 spec 服务（注入 --speculative-algorithm EAGLE
    │              --speculative-num-steps <spec_size>
    │              --speculative-eagle-topk <topk>
    │              --speculative-num-draft-tokens <spec_size+1>）
    │  warmup → sweep (seqlen × bs)
    │  记录 itl_spec_ms → partial.jsonl
    ▼  关闭 spec 服务

聚合：itl_cost = itl_spec_ms / itl_baseline_ms
写入 output_path (.jsonl，每行一条记录)
```

---

## 快速开始

### 1. 准备配置文件

复制示例配置并按需修改：

```bash
cp benchmark/speculative/aicp_config.example.json my_sweep_config.json
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

  "seqlen_min": 500,
  "seqlen_max": 5000,
  "seqlen_step": 500,
  "output_len": 256,

  "min_batch_size": 1,
  "max_batch_size": 32,
  "batch_size_step": 1,

  "test_spec_size_list": [1, 3, 5, 7, 9],
  "speculative_eagle_topk": 1,
  "combo_per_batch_size": 3,

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
3. 按 sweep 当前的 `spec_size` 重新注入上述参数（baseline 轮不注入）。

因此，带或不带 spec 参数、有没有 `--port` 都不影响正确性，直接粘贴即可：

```json
"base_command": "SGLANG_DISABLE_CUDNN_CHECK=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server --model-path /models/Qwen/Qwen3___5-27B/ --tp-size 4 --dtype bfloat16 --mem-fraction-static 0.75 --context-length 40960 --max-running-requests 32 --speculative-algorithm EAGLE --speculative-draft-model-path /models/Qwen/Qwen3___5-27B/ --speculative-num-steps 16 --speculative-eagle-topk 1 --speculative-num-draft-tokens 17 --port 10000"
```

### 参数说明

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `base_command` | `str` | — | sglang 启动命令（必填） |
| `seqlen_min` | `int` | `500` | 扫描的最小输入序列长度（token 数） |
| `seqlen_max` | `int` | `5000` | 扫描的最大输入序列长度 |
| `seqlen_step` | `int` | `500` | seqlen 步长，生成等差列表 |
| `output_len` | `int` | `256` | 每个请求的输出 token 数（需 > 1 才有 ITL） |
| `min_batch_size` | `int` | `1` | 最小并发数 |
| `max_batch_size` | `int` | `32` | 最大并发数 |
| `batch_size_step` | `int` | `1` | 并发步长，建议粗扫时设为 `4` 或 `8` |
| `test_spec_size_list` | `list[int]` | `[1,3,5,7,9]` | 待测的投机解码步数列表 |
| `speculative_eagle_topk` | `int` | `1` | EAGLE topk，注入至所有 spec 服务 |
| `combo_per_batch_size` | `int` | `3` | 每个并发点的测试请求倍数（`num_prompts = bs × combo`） |
| `host` | `str` | `"127.0.0.1"` | 服务监听地址 |
| `warmup_seqlen` | `int` | `512` | warmup 请求的输入长度 |
| `warmup_output_len` | `int` | `32` | warmup 请求的输出长度 |
| `seed` | `int` | `1` | 随机种子（保证可复现） |
| `output_path` | `str` | `"itl_cost_results.jsonl"` | 结果输出路径 |

---

## 输出格式

结果文件为 **JSON Lines**，每行一条完整记录：

```jsonl
{"seqlen": 500, "batch_size": 1, "spec_size": 1, "itl_baseline_ms": 14.23, "itl_spec_ms": 9.81, "itl_cost": 0.689}
{"seqlen": 500, "batch_size": 1, "spec_size": 3, "itl_baseline_ms": 14.23, "itl_spec_ms": 8.95, "itl_cost": 0.629}
{"seqlen": 500, "batch_size": 4, "spec_size": 5, "itl_baseline_ms": 16.40, "itl_spec_ms": 18.12, "itl_cost": 1.105}
...
```

| 字段 | 说明 |
|---|---|
| `seqlen` | 输入序列长度（token 数） |
| `batch_size` | 并发请求数 |
| `spec_size` | 本轮投机解码步数（`--speculative-num-steps`） |
| `itl_baseline_ms` | 无投机解码时的均值 ITL（毫秒） |
| `itl_spec_ms` | 启用投机解码时的均值 ITL（毫秒） |
| `itl_cost` | `itl_spec_ms / itl_baseline_ms`，**< 1.0 表示有收益，> 1.0 表示有损耗** |

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
|---|---|
| TP / EP 多进程服务残留 | `terminate_process` → `kill_process_tree`，`atexit` + `SIGINT` / `SIGTERM` 三重兜底 |
| 服务启动失败（crash） | `wait_for_http_ready` 内每秒 `poll()` 进程，进程退出立即抛异常，不会挂等 |
| 服务加载缓慢 | 无硬超时，等到就绪为止；Ctrl+C 可随时中断 |
| prefix cache 污染 | 每次 bench 前 `POST /flush_cache` |
| 中途崩溃丢数据 | 每个 cell 完成后追加写 `<output>.partial.jsonl` |
| 单 spec_size 失败 | 仅跳过该 spec_size，不影响其他 spec_size 和 baseline 的结果 |

---

## 扫描规模估算

全格扫描的测试点数为：

```
total = len(seqlen_list) × len(batch_size_list) × (1 + len(test_spec_size_list))
```

以默认配置为例：`10 × 32 × 6 = 1920` 个 bench 点，加上 6 次服务启停。如需快速验证，建议先用粗粒度参数：

```json
{
  "seqlen_step": 1000,
  "batch_size_step": 4,
  "test_spec_size_list": [3, 7]
}
```

此配置共 `5 × 8 × 3 = 120` 个 bench 点，单次扫描时间大幅缩短。

---

## 输出屏幕说明

脚本运行时，终端同时显示三路输出：

- **sglang 服务日志**（服务进程继承终端 fd，直接打印）
- **bench_serving 输出**（每个 bench 点的指标摘要）
- **脚本进度行**（前缀为 `[progress]`，显示全局和局部计数）

```
────────────────────────────────────────────────────────────────────────
[progress] overall 47/1920  local 47/320  |  spec_size=3  seqlen=2000  bs=15  num_prompts=45
```
