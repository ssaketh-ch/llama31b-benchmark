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

**Isolation takeaway:** Biggest jumps came from batch size and FP8. Other single knobs are minor. FlashInfer regressed; extra scheduler steps regressed.

## Throughput: Progressive Tuning (Stacked)
Source: [results/throughput/progression_full_stack/experiments/experiment_results/EXPERIMENT_REPORT.txt](results/throughput/progression_full_stack/experiments/experiment_results/EXPERIMENT_REPORT.txt)

- Baseline starting point: ~1,008–1,043 tok/s (BF16, BS=16, no quant, no sort).
- Peak observed: 3,064.41 tok/s (2.94x) in the BEST session (2026-03-07_09-54-10).
- Dominant contributors (stacked): batch size increases (16→512→1024), FP8 quantization, length sorting, right-sized max_model_len (2668), stable allocator settings. vLLM 0.10.0 extras (async scheduling, scheduler steps, compile, FlashInfer) did not beat the tuned stack.

## Accuracy
Placeholder for future runs: [results/accuracy/](results/accuracy/). Add accuracy logs and reports here once generated.

## How to extend
- Add new run folders under the existing structure and drop a short summary CSV/markdown per run.
- Keep the single-knob summary file (`single_knob_summary.csv`) and `experiments_summary.csv` up to date when rerunning.
- For stacked tuning, append highlights to the progression report or add a concise table with baseline vs best.
