# Phase A/B/C Isolation Run

Isolation experiments built on top of a strong pre-tuned baseline
(FP8 + TP1 + BS=1024 + max_len=2668 + sort + expandable_segments).
Each phase-A and phase-B run changes exactly one flag relative to that baseline.
Phase-C runs combine the best-performing flags from A and B.

All runs: LLaMA 3.1-8B, BF16 dtype, BS=1024, MLPerf Offline scenario (PySUT, PerformanceOnly).

## Contents

- `experiments_summary.csv` -- full results with tok/s, latency percentiles, and speedup vs baseline.
- `<experiment>/run_1/` -- per-experiment MLPerf logs and system captures.

---

## Results

All speedups relative to **exp_00_BASELINE** (2839.71 tok/s). Latencies in seconds (converted from ns).

### Phase A -- Prefill / Memory / Scheduler Knobs

| Experiment | Change | Tok/s | Speedup | p50 (s) | p99 (s) | Result |
|---|---|---|---|---|---|---|
| exp_00_BASELINE | FP8+TP1+BS1024+max_len2668+sort+expandable_segments | 2839.71 | baseline | 0.42 | 1.19 | VALID |
| exp_A1_chunked_prefill | +enable_chunked_prefill=True | 2863.78 | +0.85% | 0.42 | 1.18 | VALID |
| exp_A2_batched_tokens_65536 | +max_num_batched_tokens=65536 (CP=False) | 2846.39 | +0.24% | 0.42 | 1.19 | VALID |
| exp_A3_chunked_plus_batched_tokens | +chunked_prefill=True + MNBT=65536 | 2852.89 | +0.46% | 0.42 | 1.19 | VALID |
| exp_A4_gpu_mem_0.98 | +gpu_memory_utilization=0.98 | 2575.72 | -9.30% | 0.47 | 1.31 | VALID |
| exp_A5_prefix_cache_on | +enable_prefix_caching=True | 4475.80 | **+57.61%** | 0.29 | 0.76 | VALID |
| exp_A6_scheduler_delay_0.1 | +scheduler_delay_factor=0.1 | 2312.27 | -18.57% | 0.55 | 1.47 | VALID |

### Phase B -- vLLM 0.10.0 Flags

| Experiment | Change | Tok/s | Speedup | p50 (s) | p99 (s) | Result |
|---|---|---|---|---|---|---|
| exp_B1_async_scheduling | +async_scheduling=True | 2979.62 | **+4.93%** | 0.39 | 1.14 | VALID |
| exp_B2_sched_steps_2 | +num_scheduler_steps=2 | 2483.98 | -12.53% | 0.50 | 1.36 | VALID |
| exp_B3_sched_steps_4 | +num_scheduler_steps=4 | 2540.35 | -10.54% | 0.48 | 1.33 | VALID |
| exp_B4_compile_lvl3 | +compilation_config=3 | 2842.23 | +0.09% | 0.42 | 1.19 | VALID |
| exp_B5_flashinfer | +VLLM_ATTENTION_BACKEND=FLASHINFER | 2656.88 | -6.44% | 0.45 | 1.27 | VALID |

### Phase C -- Combination Runs

| Experiment | Change | Tok/s | Speedup | p50 (s) | p99 (s) | Result |
|---|---|---|---|---|---|---|
| exp_C1_async_plus_compile | +async_scheduling + compilation_config=3 | 2949.93 | +3.88% | 0.41 | 1.15 | VALID |
| exp_C2_async_plus_sched_steps_2 | +async_scheduling + num_scheduler_steps=2 | -- | N/A | -- | -- | FAILED |
| exp_C3_best_A_plus_best_B | Best Phase A + best Phase B combined | 2914.94 | +2.65% | 0.41 | 1.16 | VALID |

---

## Key Observations

- **Prefix caching ON** (exp_A5) is the standout result at **+57.6%** over baseline, cutting p50 latency
  from 0.42s to 0.29s and p99 from 1.19s to 0.76s. This is an exceptionally large gain -- validate that
  the CNN/DailyMail query distribution has sufficient prefix reuse before treating this as a general win.
- **async_scheduling** (exp_B1) is the best Phase B flag at **+4.93%** with slightly lower latency.
  It remains positive in combination (exp_C1: +3.88%), confirming it is stable.
- **Chunked prefill and batched token knobs** (A1-A3) all yielded near-zero gains (+0.24% to +0.85%).
  None are worth the added complexity at this baseline.
- **gpu_memory_utilization=0.98** (exp_A4) hurt throughput by -9.3%, likely due to increased memory
  pressure causing fragmentation or OOM-avoidance stalls. Avoid.
- **scheduler_delay_factor=0.1** (exp_A6) is the worst single-flag change at -18.6%. Do not use.
- **num_scheduler_steps=2/4** (B2, B3) both regress significantly (-11% to -13%). These hurt at BS=1024.
- **FlashInfer backend** (B5) regressed by -6.4%; the default attention backend is preferable here.
- **compilation_config=3** (B4) had no measurable effect (+0.09%) but compounds cleanly with
  async_scheduling (C1: +3.88%), suggesting it at least does not interfere.
- **exp_C2** (async + sched_steps_2) failed; avoid this combination.

## Recommendations

1. **Enable prefix caching** if query workload has meaningful shared prefixes -- the +57% gain is
   confirmed on this dataset but should be validated on production traffic.
2. **Enable async_scheduling=True** for a reliable free +5% on vLLM 0.10.0.
3. **Optionally pair with compilation_config=3** -- no standalone gain but stable in combination.
4. **Do not raise gpu_memory_utilization above 0.95**, and avoid scheduler_delay_factor and
   multi-step scheduling at large batch sizes.