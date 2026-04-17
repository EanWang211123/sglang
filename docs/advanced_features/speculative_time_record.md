# Speculative Time Recording

This page documents the lightweight speculative timing recorder and the companion sweep script:

- `python/sglang/srt/speculative/spec_timing_warmup_sim_recorder.py`
- `benchmark/autotune/sweep_speculative_time_record.py`

The feature is intended for EAGLE speculative decode tuning when you want to:

- record per-stage timing for `draft`, `verify`, and `draft_extend`
- aggregate timing by `batch-size` and `steps` / `query-len`
- sweep multiple `--speculative-num-steps` values automatically
- compare output throughput and stage timing across steps

## Server-Side Timing Recording

Set the environment variable `SPEC_TIMING_WARMUP_SIM_RESULTS` to enable timing aggregation.

When this variable is set, SGLang will:

- synchronize CPU/GPU around `draft`, `verify`, and `draft_extend`
- record decode-path timings into JSON files
- avoid printing timing logs unless `--enable-speculative-time-logging` is also enabled

Example:

```bash
SPEC_TIMING_WARMUP_SIM_RESULTS=/tmp/spec-timing \
python -m sglang.launch_server \
  --model-path /path/to/model \
  --speculative-algorithm EAGLE \
  --speculative-eagle-topk 1 \
  --speculative-num-steps 3
```

### Output Files

The recorder writes three files under the target directory:

- `draft_time.json`
- `verify_time.json`
- `draft_extend_time.json`

`draft_time.json` uses `steps` as the bucket key:

```json
[
  {
    "batch-size": 8,
    "steps": 3,
    "num": 42,
    "time": 4.318
  }
]
```

`verify_time.json` and `draft_extend_time.json` use `query-len`:

```json
[
  {
    "batch-size": 8,
    "query-len": 4,
    "num": 42,
    "time": 12.771
  }
]
```

The `time` field is the running average in milliseconds for that bucket, and `num` is the number of writes merged into that average.

### Bucket Semantics

- `draft`: bucketed by `steps = speculative_num_steps`
- `verify`: bucketed by `query-len = speculative_num_steps + 1`
- `draft_extend`: bucketed by `query-len = speculative_num_steps + 1`

For example, when `step=1`, the expected verify `query-len` is `2`.

## Sweep Script

Use `benchmark/autotune/sweep_speculative_time_record.py` to launch the server repeatedly with different `--speculative-num-steps` values and benchmark each configuration.

The script:

- launches one server per step value
- injects `SPEC_TIMING_WARMUP_SIM_RESULTS` automatically
- reuses a prompt dataset
- runs dynamic-concurrency load generation
- shows one progress bar per benchmark case
- prints server startup logs directly to the terminal
- emits summary tables after all sweeps finish

### Dynamic Concurrency Behavior

The load generator is producer-style dynamic filling:

- it starts up to `concurrency` requests
- when one request finishes, it immediately submits the next prompt
- it continues until `num_prompts` requests have completed

This is different from launching all requests in fixed waves.

## Script Example

```bash
python -m benchmark.autotune.sweep_speculative_time_record \
  --speculative-num-steps 3 5 7 9 \
  --dataset-path /path/to/data.json \
  --num-prompts 10 100 100 \
  --concurrency 1 10 100 \
  --output-tokens 256 \
  --spec-timing-root /tmp/spec-timing \
  --server-env CUDA_VISIBLE_DEVICES=1 \
  --save-server-logs \
  -- \
  --model-path /nfs_models/Qwen/Qwen3.5-4B \
  --speculative-algorithm EAGLE \
  --speculative-eagle-topk 1 \
  --mem-fraction-static 0.7 \
  --context-length 8192 \
  --max-running-requests 16 \
  --mamba-scheduler-strategy extra_buffer \
  --port 30000
```

## Important Arguments

### Sweep Arguments

- `--speculative-num-steps`: list of step values to test
- `--dataset-path`: JSON or JSONL prompt file
- `--num-prompts`: number of requests per benchmark case
- `--concurrency`: concurrency per benchmark case
- `--output-tokens`: `max_new_tokens` for each request
- `--disable-ignore-eos`: disable `ignore_eos`; by default `ignore_eos=True`

### Server Forwarding

All arguments after `--` are forwarded to `python -m sglang.launch_server`.

Do not pass `--speculative-num-steps` in the forwarded server args, because the script controls it during the sweep.

## Generated Outputs

The script writes:

- `results.json`: full structured results
- `results.csv`: flat benchmark summary
- `summary_tables.md`: markdown tables for throughput and timing
- `timing/step_<n>/...`: per-step timing JSONs, unless overridden by `--spec-timing-root`
- `server_logs/step_<n>.log`: per-step server logs when `--save-server-logs` is enabled

## Summary Tables

After the run, the script prints and saves:

### Table 1: Throughput Change Table

- columns: `step`
- rows: `batch-size`
- values: output throughput = `total completion tokens / total elapsed time`

In the current script, `batch-size` corresponds to the benchmark `concurrency`.

### Table 2: Time Change Tables

Three timing tables are produced:

- `draft`
- `verify`
- `draft-extend`

Each table uses:

- columns: `step`
- rows: `batch-size`
- values: average stage time in milliseconds

The verify and draft-extend tables are filtered using the expected `query-len = step + 1`.

## Notes

- The timing recorder only updates decode batches, not prefill batches.
- `draft_extend` is only written when that stage actually runs.
- Timing recording is rank-filtered to avoid duplicate writes from multiple workers.
- If startup fails, server output is streamed directly to the terminal to simplify debugging.
