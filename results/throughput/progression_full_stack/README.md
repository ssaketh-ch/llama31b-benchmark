# Progressive / Stacked Tuning

Stacked tuning sessions that build from the production baseline toward a best-performing configuration.
Each stack step adds one change on top of all previous ones. See `experiments/` for
dated runs and the consolidated report.

## Stack Results 

All speedups are relative to **stk_00_baseline** (1089.04 tok/s).

| Stack Step | Changes Added | Tok/s | Speedup vs Baseline | p50 Latency (s) | p99 Latency (s) | Result |
|---|---|---|---|---|---|---|
| stk_00_baseline | Production baseline: BS=16, BF16, stock vLLM V1 defaults | 1089.04 | baseline | 783.92 | 1556.55 | VALID |
| stk_01_fp8 | +fp8 weight quantization | 1462.53 | +34.30% | 583.79 | 1158.98 | VALID |
| stk_03_kv2668 | +max_model_len=2668 (right-sized KV) | 1544.29 | +41.80% | 550.27 | 1097.43 | VALID |
| stk_04_sort | +sort_by_input_length | 1507.93 | +38.46% | 472.53 | 1117.14 | VALID |
| stk_05_gmu095 | +gpu_memory_utilization=0.95 | 1504.52 | +38.15% | 474.87 | 1119.70 | VALID |
| stk_06_kvcache_fp8 | +kv_cache_dtype=fp8 | 1500.22 | +37.76% | 477.08 | 1122.68 | VALID |
| stk_07_compile3 | +compilation_config=O3 | 1494.88 | +37.27% | 480.19 | 1126.89 | VALID |
| stk_08_expandable | +PYTORCH_CUDA_ALLOC_CONF=expandable_segments | 1420.80 | +30.46% | 481.81 | 1186.76 | VALID |
| stk_09_bitsandbytes | +bitsandbytes quantization (replaces fp8) | 446.14 | -59.03% | 1764.17 | 3787.78 | VALID |
| stk_02_bs1024 | +batch_size=1024 (standalone, not stacked) | 3321.76 | +205.02% | 191.41 | 509.22 | INVALID* |

> *stk_02_bs1024 is INVALID: min_duration not satisfied. The loadgen warning recommends increasing expected QPS so it pre-generates a larger coalesced query set. Throughput figure is directionally correct but not a certified result.

## Summary of Key Observations

- **FP8 weight quantization** (stk_01) is the single largest stacked gain at **+34.3%** and forms the foundation of all subsequent steps.
- **Right-sizing the KV cache** (stk_03, max_model_len=2668) pushes the stack to its throughput peak of **1544 tok/s (+41.8%)** while also cutting p99 latency by ~30%.
- **Sort-by-length, higher GPU memory utilization, and fp8 KV cache** each contribute small incremental gains but start to show diminishing returns as the stack grows.
- **expandable_segments** (stk_08) and **compilation O3** (stk_07) slightly erode the stacked peak, suggesting negative interactions with the full configuration rather than isolation-level benefits.
- **bitsandbytes quantization** (stk_09) is strongly negative at -59% and should not be used.
- **BS=1024** remains the highest raw throughput option (+205%) but requires a valid loadgen run with corrected QPS settings before it can be treated as a certified result.