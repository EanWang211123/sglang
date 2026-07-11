# DFlash 自适应验证集成方案

## 目标与关键语义

- 复用 DSpark 的“历史分数决定总预算 + 当前分数做前缀 top-k + cost table 选择 CUDA graph bucket”框架。
- DFlash 的 `block_size=K` 中，第 0 列是 anchor/上一轮 bonus，真正的 draft 是 `K-1` 个。调度输入为 `[bs, K-1]`，每请求至少验证 1 个 anchor，最多验证 `K` 个 token。
- 实验已确认 DFlash draft top-1 probability 与逐位置条件接受率正相关，因此可直接作为当前步 top-k 排序信号。
- raw draft probability 不严格等于接受概率。若要提高 cost-table 总预算的绝对收益估计，可增加逐位置温度缩放或单调校准，但校准不是 correctness 或 compact 启用的硬依赖。
- 首版限定 greedy DFlash，最终执行形态直接采用 compact ragged verify；未启用时保持 fixed-block 行为不变。

## 1. 产出 DFlash draft probability

- 扩展 `python/sglang/srt/speculative/dflash_worker_v2.py` 的 `_DflashDraftSampler` 与 eager TP-safe head 路径，在生成 greedy token 时同时返回该 token 的 softmax probability。
- TP>1 使用全局 max 与分布式 log-sum-exp，不能只归一化本地 vocab shard。
- 增加观测组件，用未裁剪 `num_correct_drafts` 构造前缀标签，持续验证 raw probability 的 rank correlation。
- top-k 排序直接使用 raw probability；cost-table budget 可选择使用逐位置校准后的条件概率。
- 首版只支持 all-greedy batch；sampling verify 回退 fixed block。sampling 后续需单独估计接受事件，不能直接套用 greedy probability 语义。

## 2. 接入异步预算与当前步前缀分配

- 新建 DFlash planner 薄适配层，首版复用：
  - `dspark_scheduler.py` 的 cost-table argmax、generation freshness 与 lag 逻辑。
  - `schedule_verify_lens_topk.py` 的 GPU 前缀 top-k kernel。
- 调度参数使用 `gamma=K-1`、`min_verify_len=1`、`max_verify_len=K`。
- 扩展 `overlap_utils.py` 和 `scheduler.py`：启用 DFlash adaptive compact 时发布当前分数，通过 pinned ring 异步 D2H；调度线程使用滞后且 generation 匹配的历史 survival 计算下一步总预算。
- worker 使用“历史预算 + 当前块分数”运行 top-k：历史值决定 bucket 大小，当前值决定每请求获得的连续前缀长度。
- 增加 DFlash 专属 SPS table 参数。不能直接复用 DSpark profile；需按目标模型、GPU、TP、attention backend、CUDA graph grid 和 block size 测量。

## 3. 直接接入 compact ragged verify

- 最终实现直接采用 `compact`，减少 target verify token 数与 FLOPs。`cap-accept` mode 不作为阶段性交付或性能方案。
- 复用 `dflash_info.py` 已有的 `ragged_verify_layout`、`ragged_verify.py`、compact IDs/positions/cache locations 构造，以及 compact-to-strided scatter 数据流。
- compact target forward 后，将 logits/hidden scatter 回 `[bs, K, ...]`，保持现有 greedy accept 与输出布局。
- 接受阶段必须执行：

```python
num_correct_drafts = min(num_correct_drafts, verify_lens - 1)
```

- bonus 从裁剪后索引选取；KV commit、`commit_lens`、`new_seq_lens`、`block_accept_lens` 和 `cap_lens` 均按裁剪结果更新。
- `decode_cuda_graph_runner.py` 的 ragged capture/replay 机制已基本算法无关，只需将 `ragged_verify_full_mode_enabled()` 从 `is_dspark()` 放宽到 `is_dflash_or_dspark()`，并让 DFlash worker 传入 layout。
- Attention backend 只读取 `ragged_verify_layout`，与 DSpark/DFlash 无关。实际能力保持现状：
  - FlashAttention、TRTLLM MHA 支持 ragged verify graph。
  - 普通 FlashInfer 当前明确拒绝 ragged verify graph。
- 已支付的 CUDA graph bucket padding 可按当前 survival 排序回填真实前缀；超出 capture grid、backend 不支持或出现 mixed sampling/extend/idle 时回退 uniform verify。

## 4. 验证与上线

- 单元测试：softmax/TP 归一化、rank correlation、可选校准、survival 前缀语义、预算 argmax、连续前缀 top-k、generation 失配回退、`verify_lens-1` 上限和 bonus 索引。
- GPU 集成测试：adaptive 与 fixed DFlash 在 greedy 下逐 token 一致；覆盖 overlap 开关、CUDA graph/eager、TP=1/2、batch churn、短请求结束、不同 bucket 和长上下文。
- 性能测试：draft probability 计算开销、D2H/CPU budget 是否隐藏、target verify token 数、bucket padding、accept length 和端到端 tokens/s。
- 首先启用观测模式验证建议 budget 与真实收益，再开放 compact 执行；观测模式不是 `cap-accept`，不会作为最终路径。
- 稳定后将 `HostConfidenceBudgetPlanner`、SPS table、top-k 和 compact target executor 从 `dspark_components` 提取到通用 `adaptive_verify` 模块。

## 实施清单

- [ ] 实现 greedy DFlash draft probability 输出与观测。
- [ ] 接入滞后异步预算、DFlash cost table 和当前步 GPU 前缀 top-k。
- [ ] 接入 compact ragged target verify、接受长度裁剪与安全回退。
- [ ] 补齐 correctness、CUDA graph、TP 和端到端性能验证。
- [ ] 稳定后提取算法无关 adaptive verify 模块。
