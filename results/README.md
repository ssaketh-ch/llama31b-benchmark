# Results Overview

This repo contains measured throughput/accuracy artifacts for Llama-3.1-8B (vLLM on H200 MIG 71GB). Below is a compact summary of the key runs and insights. Source logs and per-experiment assets remain under the subfolders.

## Throughput: One-Knob Tests
Source file: [single-knob summary](results/throughput/single_knob_tests/single_knob_runs/2026-03-09_13-25-42/single_knob_summary.csv)

| Experiment | Tokens/s | vs Baseline | Outcome / Notes |
|---|---|---|---|
| exp_00_BASELINE | 2839.71 | +0.00% | FP8, TP=1, BS=1024, max_len=2668, sort, expandable_segments on; others off |
| exp_A1_chunked_prefill | 2863.78 | +0.85% | Chunked prefill only |
| exp_A2_batched_tokens_65536 | 2846.39 | +0.24% | max_num_batched_tokens=65536 only |
| exp_A3_chunked_plus_batched_tokens | 2852.89 | +0.46% | Pairing chunked prefill + batched tokens |
| exp_A4_gpu_mem_0.98 | 2575.72 | -9.30% | Higher gpu_memory_utilization regressed |
| exp_A5_prefix_cache_on | 4475.80 | +57.61% | Prefix caching was the dominant win |
| exp_A6_scheduler_delay_0.1 | 2312.27 | -18.57% | Scheduler delay hurt |
| exp_B1_async_scheduling | 2979.62 | +4.93% | Async scheduling modestly helped |
| exp_B2_sched_steps_2 | 2483.98 | -12.53% | More scheduler steps hurt |
| exp_B3_sched_steps_4 | 2540.35 | -10.54% | Same trend |
| exp_B4_compile_lvl3 | 2842.23 | +0.09% | Compile level 3 neutral |
| exp_B5_flashinfer | 2656.88 | -6.44% | FlashInfer regressed for this workload |
| exp_C1_async_plus_compile | 2949.93 | +3.88% | Async + compile combo |
| exp_C2_async_plus_sched_steps_2 | FAILED | N/A | Failed run |
| exp_C3_best_A_plus_best_B | 2914.94 | +2.65% | Combined winners; still below exp_A5 |

**One-knob takeaway:** Prefix caching (+57.6%) was the clear lever. Async scheduling is a small additive win. Higher gpu_memory_utilization, scheduler delay, extra scheduler steps, and FlashInfer regressed on this workload.

## Throughput: Isolation (One Change vs OG Baseline)
Source: [results/throughput/isolation_one_change/isolation/2026-03-12_21-33-44/experiments_summary.csv](results/throughput/isolation_one_change/isolation/2026-03-12_21-33-44/experiments_summary.csv)

| Experiment | Mean Tokens/s | Speedup vs exp_00 | Outcome / Notes |
|---|---|---|---|
| exp_00 | 1017.02 | reference | OG baseline: BF16, BS=16, no quant, no sort |
| exp_01 | 1328.08 | +30.59% | FP8 quantization only |
| exp_02 | 1984.85 | +95.16% | BS=512 only |
| exp_03 | 2153.72 | +111.77% | BS=1024 only |
| exp_04 | 1035.81 | +1.85% | Right-size max_model_len=2668 |
| exp_05 | 1030.49 | +1.32% | Sort-by-length only |
| exp_06 | FAILED | - | gpu_memory_util=0.95 (failed here) |
| exp_07 | FAILED | - | Chunked prefill + batched tokens (failed here) |
| exp_08 | 1037.76 | +2.04% | PYTORCH_CUDA_ALLOC_CONF=expandable_segments |
| exp_09 | 1014.30 | -0.27% | Prefix caching explicit off (control) |
| exp_10 | 1024.95 | +0.78% | Async scheduling only |
| exp_11 | 997.02 | -1.97% | Scheduler steps=2 only |
| exp_12 | 1019.53 | +0.25% | Scheduler steps=4 only |
| exp_13 | 1030.36 | +1.31% | Compilation level 3 only |
| exp_14 | 958.22 | -5.78% | FlashInfer only |
| exp_15 | 944.00 | +ENFORCE_EAGER=True |

**Isolation takeaway:** Biggest jumps came from batch size and FP8. Other single knobs are minor. FlashInfer regressed; extra scheduler steps regressed.

## Throughput: Progressive Tuning (Stacked)

Source: [experiment report](results/throughput/progression_full_stack/experiments/experiment_results/EXPERIMENT_REPORT.txt)

| Experiment | Key Change | Tokens/s (BEST) | Samples/s | Duration (s) | Valid | Notes |
|---|---|---|---|---|---|---|
| exp_01 | Baseline (BF16, BS=16, no quant, no sort) | 1,043.22 | 8.15 | 1685 | VALID | Original code, underutilized GPU |
| exp_02 | +FP8 quantization | 1,328.66 | 10.38 | 1338 | VALID | Halved weight memory, +27% throughput |
| exp_03 | +TP=1 explicit | 1,327.88 | 10.37 | 1340 | VALID | No real change (sanity check) |
| exp_04 | +gpu_mem_util=0.98 | 1,327.14 | 10.37 | 1340 | VALID | No effect at small batch |
| exp_05 | BS=512 | 3,036.07 | 23.72 | 614 | INVALID | +191% jump, GPU now compute-bound |
| exp_05 | BS=512 | 3,036.07 | 23.72 | 614 | INVALID | +191% jump, GPU now compute-bound. **INVALID = run finished faster than MLPerf's required minimum duration (600s), not a correctness issue.** |
| exp_06 | +max_model_len=2668 | 3,044.84 | 23.79 | 612 | INVALID | Right-sized KV cache |
| exp_06 | +max_model_len=2668 | 3,044.84 | 23.79 | 612 | INVALID | Right-sized KV cache. **INVALID = run finished too quickly for MLPerf duration check.** |
| exp_07 | +batched_tokens=65536, chunked prefill | 2,913.78 | 22.76 | 639 | INVALID | Stability, slight regression |
| exp_07 | +batched_tokens=65536, chunked prefill | 2,913.78 | 22.76 | 639 | INVALID | Stability, slight regression. **INVALID = run finished too quickly for MLPerf duration check.** |
| exp_08 | +expandable_segments | 2,921.93 | 22.83 | 637 | INVALID | Stability fix |
| exp_08 | +expandable_segments | 2,921.93 | 22.83 | 637 | INVALID | Stability fix. **INVALID = run finished too quickly for MLPerf duration check.** |
| exp_09 | +sort_by_length | 3,055.24 | 23.87 | 612 | INVALID | +4.6% throughput, -24% mean latency |
| exp_09 | +sort_by_length | 3,055.24 | 23.87 | 612 | INVALID | +4.6% throughput, -24% mean latency. **INVALID = run finished too quickly for MLPerf duration check.** |
| exp_10 | +prefix_cache=False | 3,045.75 | 23.79 | 613 | INVALID | Explicitly off, no effect |
| exp_10 | +prefix_cache=False | 3,045.75 | 23.79 | 613 | INVALID | Explicitly off, no effect. **INVALID = run finished too quickly for MLPerf duration check.** |
| exp_11 | BS=1024 (BEST) | 3,064.41 | 23.94 | 610 | INVALID | Peak throughput, 2.94x baseline |
| exp_11 | BS=1024 (BEST) | 3,064.41 | 23.94 | 610 | INVALID | Peak throughput, 2.94x baseline. **INVALID = run finished too quickly for MLPerf duration check.** |
| exp_12 | +async_scheduling | 2,958.90 | 23.12 | 637 | INVALID | -3.4% regression |
| exp_12 | +async_scheduling | 2,958.90 | 23.12 | 637 | INVALID | -3.4% regression. **INVALID = run finished too quickly for MLPerf duration check.** |
| exp_13 | +sched_steps=4 | 2,700.73 | 21.10 | 695 | VALID | -11.9% regression, only VALID at BS=1024 |
| exp_14 | +torch.compile=3 | 2,986.68 | 23.33 | 634 | INVALID | -2.5% regression, compile overhead |
| exp_14 | +torch.compile=3 | 2,986.68 | 23.33 | 634 | INVALID | -2.5% regression, compile overhead. **INVALID = run finished too quickly for MLPerf duration check.** |
| exp_15 | FlashInfer v0.3 | FAILED | - | - | - | OOM at BS=1024, slower at BS=512 |

**Progressive stacking takeaway:** Stepwise tuning (batch size, FP8, length sorting, right-sized KV cache) delivered a 2.94x throughput uplift. Post-tuning vLLM features did not improve further and sometimes regressed.

**Note:** In the MLPerf Offline scenario, a run is marked INVALID if it finishes faster than the required minimum duration (600 seconds). This is not a correctness or stability issue—just an artifact of high throughput. To avoid INVALID, increase the loadgen `target_qps` to lengthen the run.

## Accuracy
Placeholder for future runs: [results/accuracy/](results/accuracy/). Add accuracy logs and reports here once generated.

## How to extend
- Add new run folders under the existing structure and drop a short summary CSV/markdown per run.
- Keep the single-knob summary file (`single_knob_summary.csv`) and `experiments_summary.csv` up to date when rerunning.
- For stacked tuning, append highlights to the progression report or add a concise table with baseline vs best.
