# Isolation Run (vLLM v0.17.1)

This folder contains the results of isolation experiments, where each run changes a single parameter relative to the production baseline (BF16, BS=16, stock vLLM V1 defaults).

## Baseline

| Experiment | Description | Tok/s |
|---|---|---|
| exp_00 | PRODUCTION BASELINE: stock vLLM V1 defaults, BS=16 | 1082.82 |
| exp_00b | RIGHT-SIZED KV: max_model_len=2668 on production baseline | 1165.24 (+7.61%) |

> All speedups are computed relative to **exp_00** (1082.82 tok/s).

---

## Contents

- `experiments_summary.csv` -- summary table with tokens/s, latency percentiles, and speedup vs baseline for each experiment.
- `exp_*` folders -- individual experiment runs, logs, and system captures.

---

## Experiment Results Table

| Experiment | Description | Tok/s | Speedup vs exp_00 |
|---|---|---|---|
| exp_00 | PRODUCTION BASELINE: stock vLLM V1 defaults, BS=16 | 1082.82 | baseline |
| exp_00b | RIGHT-SIZED KV: max_model_len=2668 on production baseline | 1165.24 | +7.61% |
| exp_01 | -prefix_caching=False (disable V1 default APC) | 1048.73 | -3.15% |
| exp_02 | -async_output_proc OFF (disable V1 default async) | 1006.82 | -7.02% |
| exp_03 | compilation_config=O1 (below V1 default O2) | 1062.58 | -1.87% |
| exp_04 | compilation_config=O0 (no compile vs V1 default O2) | 857.43 | -20.82% |
| exp_05 | compilation_config=O3 (above V1 default O2) | 1098.13 | +1.41% |
| exp_06 | max_num_batched_tokens=2668 (neutralise chunked prefill) | 1063.47 | -1.79% |
| exp_07 | async_scheduling=True (experimental) | 1115.30 | +3.00% |
| exp_08 | max_num_batched_tokens=8192 | 1120.72 | +3.50% |
| exp_09 | max_num_batched_tokens=65536 | 1089.07 | +0.58% |
| exp_10 | quantization=fp8 weight only | 1456.36 | **+34.50%** |
| exp_11 | kv_cache_dtype=fp8 KV only | 1131.84 | +4.53% |
| exp_12 | fp8 weights + fp8 KV | 1277.73 | +18.00% |
| exp_13 | +batch_size=1024 | 2302.79 | **+112.67%** |
| exp_14 | +batch_size=2048 | 2379.46 | **+119.75%** |
| exp_15 | sort_by_input_length | 1115.30 | +3.00% |
| exp_16 | gpu_memory_utilization=0.95 | 1104.48 | +2.00% |
| exp_17 | attention_backend=FLASHINFER | 1081.93 | -0.08% |
| exp_18 | skip_tokenizer_init=True | 1083.77 | +0.09% |

---

## Summary of Key Results

- **Batch size increases** (exp_13, exp_14) provided the largest throughput gains: **+113% at BS=1024** and **+120% at BS=2048**.
- **FP8 weight quantization** (exp_10) gave a strong **+34.5% boost** over baseline at BS=16. Combined fp8 weights + fp8 KV cache (exp_12) yields **+18%**, lower than weights-only, suggesting KV fp8 introduces some overhead.
- **FP8 KV cache alone** (exp_11) gave a modest **+4.5%** improvement.
- **Right-sizing KV** (exp_00b, max_model_len=2668) is a free **+7.6%** win on the production baseline at no accuracy cost.
- **async_scheduling, max_num_batched_tokens=8192, sort_by_length, compilation O3, gpu_mem_util=0.95** all gave small positive effects in the **+1% to +4%** range.
- **Disabling V1 defaults** (prefix_caching off, async_output_proc off) confirmed these defaults are net-positive; disabling them regresses throughput by 3-7%.
- **compilation_config=O0** (no compile) is the worst single-knob change at **-20.8%**, a strong signal that torch compilation is essential.
- **FLASHINFER backend** and **skip_tokenizer_init** had virtually zero impact (within +-0.1%).

---

## Recommendations

1. **Apply BS=1024 or BS=2048** for offline/batch workloads -- the single biggest lever.
2. **Enable FP8 weight quantization** for a free ~34% gain at standard batch sizes.
3. **Right-size KV cache** with max_model_len=2668 -- zero-cost gain.
4. **Do not disable prefix_caching or async_output_proc** -- V1 defaults are beneficial. Or is it.....
5. **Keep compilation at O2 (default) or try O3** -- never run O0 in production.

See `experiments_summary.csv` for full details and per-experiment statistics.
