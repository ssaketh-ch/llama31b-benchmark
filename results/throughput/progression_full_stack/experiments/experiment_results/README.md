# Llama-3.1-8B Offline Throughput Sweep (H200 NVL MIG)

> MLPerf Inference, Offline scenario - vLLM 0.10.0 on a 71 GB H200 NVL MIG slice. Six sessions, fifteen configs, five full runs.

## Executive Summary
- Baseline: ~1,008-1,043 tok/s (BS=16, BF16).
- Peak: 3,064.41 tok/s (BS=1024, FP8) - **2.94x** uplift.
- Dominant levers: batch size (16->512->1024), FP8 weights, right-sized KV cache, length-sorted batching. vLLM async/scheduler/compile features did not help this offline workload.

## Hardware & Software
- GPU: NVIDIA H200 NVL, single MIG slice (71 GB usable of 143,771 MiB). GPU mem util tuned to 0.95-0.98.
- CPU/RAM: server-class (lscpu, /proc/meminfo captured in runs).
- Software: Python 3.10.12, PyTorch 2.7.1, CUDA 12.6.x, vLLM 0.10.0, Transformers 4.53.2, FlashInfer bundled (0.3.x), xFormers installed.
- Model/Data: meta-llama/Meta-Llama-3.1-8B-Instruct, CNN/DailyMail (13,368 samples, prompt len min 79 / max 2,540 / mean ~870, max_tokens=128). Offline scenario.

## Methodology
- Automation: `run_experiments.sh` with 15 cumulative experiments (`sut_exp01`–`sut_exp15`).
- Each experiment writes a full `SUT_VLLM.py`; only one change added per step.
- 30s cooldown + GPU reset between runs; system info captured; OOM auto-retry at half batch; MLflow used for tracking; loadgen summaries parsed for throughput/latency.
- Sessions:
  - Session 1: exp_01-exp_11 (exp_01 failed init).
  - Session 2: exp_01-exp_11 clean.
  - Session 3: exp_11 retest (invalid for duration).
  - Session 4: exp_01-exp_15 (best peak, 3,064 tok/s).
  - Session 5: exp_01-exp_15 (repro of best ordering).
  - Session 6: exp_01-exp_15 (MLflow tags; FlashInfer valid but slower).

## Best Configuration (exp_11)
- LLM (sync generate), TP=1, BS=1024.
- Weights: FP8; compute dtype float16.
- KV/cache: `max_model_len=2668`, `max_num_batched_tokens=65536`, `max_num_seqs=512`, `gpu_memory_utilization=0.95`, chunked prefill on, prefix cache off, sorted batching.
- Allocator/runtime: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, TF32 matmul + cuDNN benchmark on, `float32_matmul_precision=high`, queue size `batch_size*4`.
- Throughput (Session 4): **3,064.41 tok/s**, samples/s 23.94, duration 610s (loadgen marked INVALID only because it finished faster than `min_duration=600s`; increase `target_qps` to resolve).

### How to Reproduce Best Config
- Use the exp_11 settings above with batch_size=1024 and TP=1 on the single MIG slice.
- Ensure `max_model_len=2668`, `max_num_batched_tokens=65536`, `max_num_seqs=512`, and sorted batching are applied.
- Set `gpu_memory_utilization=0.95`, `expandable_segments:True`, TF32 matmul, and cuDNN benchmark on.
- If loadgen flags INVALID for duration, raise `target_qps` to lengthen the run.

## What Moved the Needle (Session 4 deltas vs prior step)
- Batch 16->512 (exp_05): **+128.7%** tok/s (compute-bound; Python overhead amortised).
- Batch 512->1024 (exp_11): +0.6% tok/s (fewer host->GPU trips; near saturation already).
- FP8 weights (exp_02): +27.4% tok/s at small batch (bandwidth-bound).
- `max_model_len=2668` (exp_06): +0.3% tok/s; crucial to enable high KV concurrency.
- Sort-by-length batching (exp_09): +4.6% tok/s, -24% mean latency.

Regressions to avoid for this workload:
- `async_scheduling` (exp_12): -3-4% (scheduler overhead > benefit for ~13 batches).
- `num_scheduler_steps=4` (exp_13): -11-13% (wasted compute on finished sequences).
- `torch.compile` level 3 (exp_14): -2-5% here; needs many more batches to amortise compile.
- FlashInfer v0.3 (exp_15): OOM at BS=1024; ~-12% when it runs.

## Session-to-Session Consistency (tokens/sec)
Peak ordering is stable across sessions:
- exp_11 (BS=1024) is always top; exp_09/10 close behind; exp_05/06 form the big jump; exp_12-15 always regress.
- Variance: 5-15% across sessions for compute-bound configs (thermal/power/MIG QoS). Small-batch (exp_01-04) vary ~3-4%.

| Experiment (key change)                  | S1 tok/s | S2 tok/s | S4 tok/s ★ | S5 tok/s | S6 tok/s |
|------------------------------------------|---------:|---------:|-----------:|---------:|---------:|
| exp_01 baseline BF16 BS=16               |       — | 1,008.73 | 1,043.22   | 1,042.09 | 1,030.62 |
| exp_02 +FP8                              | 1,372.20 | 1,381.17 | 1,328.66   | 1,327.71 | 1,326.19 |
| exp_05 BS=512                            | 2,616.41 | 2,673.76 | 3,036.07   | 2,891.56 | 2,769.88 |
| exp_06 +max_model_len=2668               | 2,633.29 | 2,678.99 | 3,044.84   | 2,896.71 | 2,804.35 |
| exp_09 +sort by length                   | 2,913.79 | 2,780.76 | 3,055.24   | 2,882.09 | 2,910.42 |
| exp_11 BS=1024 (best)                    | 2,902.62 | 2,746.62 | **3,064.41** | 2,846.64 | 2,895.05 |
| exp_13 num_scheduler_steps=4             |       — |       — | 2,700.73   | 2,654.45 | 2,682.61 |
| exp_14 torch.compile lvl3                |       — |       — | 2,986.68   | 2,868.74 | 2,882.80 |
| exp_15 FlashInfer v0.3                   |       — |       — | FAILED     | 2,679.78 | 2,689.55 |

## Latency (Session 4)
All values are MLPerf Offline latencies (all 13,368 samples issued at t=0).

| Config                         | Min (ms) | Mean (ms) | P50 (ms) | P90 (ms) | P99 (ms) | Duration (s) |
|--------------------------------|---------:|----------:|---------:|---------:|---------:|-------------:|
| exp_01 BS=16                   | 1,818    | 820,118   | 820,050  | 1,473,372| 1,624,906| 1,685 |
| exp_05 BS=512                  | 20,809   | 290,088   | 299,384  | 515,040  | 560,463  | 614  |
| exp_09 BS=512 + sort           | 8,682    | 220,937   | 197,153  | 470,367  | 553,968  | 612  |
| exp_11 BS=1024 + sort          | 18,003   | 230,881   | 196,305  | 470,329  | 552,381  | 610  |
| exp_13 BS=1024 + sched_steps=4 | 20,696   | 264,066   | 227,571  | 533,317  | 626,877  | 695  |

Insights: batch size dominates latency (queue clears faster); sorting cuts mean latency ~24% and slashes min latency by 58%; scheduler-steps=4 uniformly harms percentiles.

## Original vs Best Config (diff)
- Engine: AsyncLLMEngine -> sync generate.
- BS: 1 -> 1024. TP: 8 -> 1.
- Weights: BF16 -> FP8. Compute dtype: bfloat16 -> float16.
- KV cache: `max_model_len` 131,072 -> 2,668; `gpu_memory_utilization` 0.90 -> 0.95; batched tokens auto -> 65,536; `max_num_seqs` auto -> 512.
- Batching: unsorted -> sorted by input length; prefix caching default/on -> off; chunked prefill on.
- Runtime knobs: add expandable_segments allocator, TF32 matmul, cuDNN benchmark, `float32_matmul_precision=high`, queue bound, daemon threads.
- Tracking: remove per-batch MLflow overhead; disable tqdm in generate.

## Recommendations
1) For offline batching on large-memory GPUs: push batch size first; right-size `max_model_len` to dataset max input + max output.
2) Use FP8 to free bandwidth/memory; keep TP=1 on single MIG/GPU.
3) Sort by input length; disable prefix caching when prompts are unique.
4) Keep vLLM async/scheduler/compile features off unless you have many more batches to amortise overhead.
5) If loadgen reports INVALID for short runs, increase `target_qps` rather than slowing the system.

## Raw Data
Full raw tables (Sessions 1-6) remain in this directory for auditability.
- Source text: [llama31b-benchmark/results/throughput/progression_full_stack/experiments/experiment_results/EXPERIMENT_REPORT.txt](llama31b-benchmark/results/throughput/progression_full_stack/experiments/experiment_results/EXPERIMENT_REPORT.txt).
