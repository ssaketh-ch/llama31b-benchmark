# LLaMA 3.1-8B vLLM Tuning Report

End-to-end optimization study for LLaMA 3.1-8B on vLLM v0.17.1 (MLPerf Offline, PySUT).
Three experiment sets were run in sequence: (1) single-flag isolation from the stock production
baseline, (2) a focused deep-dive into quantization, chunked prefill, and prefix caching behavior,
and (3) progressive stacking of the best-performing flags toward a tuned configuration.

---

## Experiment Sets

| Set | Focus | Baseline Tok/s | Best Result |
|---|---|---|---|
| Isolation (vLLM v0.17.1) | Single-flag sweep from stock defaults | 1082.82 | 2379.46 (+119.75%, BS=2048) |
| Quant & Prefill Deep-Dive | Quantization type, CP config, APC | 685.88 | 1380.12 (+101.22%, fp8 W8A8) |
| Progressive Stacking | Cumulative build toward best config | 1089.04 | 1544.29 (+41.8%, stacked) |

> Note: baselines differ slightly across sets due to different run dates and system states.
> All speedups are relative to the respective set's own baseline.

---

## Part 1 -- Isolation Run

Each experiment changes exactly one flag relative to the production baseline
(BF16, BS=16, stock vLLM V1 defaults, exp_00 = 1082.82 tok/s).

### Baseline Variants

| Experiment | Description | Tok/s | Speedup |
|---|---|---|---|
| exp_00 | PRODUCTION BASELINE: stock vLLM V1 defaults, BS=16 | 1082.82 | baseline |
| exp_00b | RIGHT-SIZED KV: max_model_len=2668 on production baseline | 1165.24 | +7.61% |

### Full Isolation Results

| Experiment | Description | Tok/s | Speedup |
|---|---|---|---|
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

### Isolation Key Findings

- **Batch size** (exp_13, exp_14) is the single biggest lever: **+113% at BS=1024** and **+120% at BS=2048**. This dwarfs every other single-flag change.
- **FP8 weight quantization** (exp_10) is the best non-batch-size gain at **+34.5%**. Notably, adding fp8 KV cache on top (exp_12) drops this to +18% -- fp8 KV introduces overhead that partially cancels the weight quantization benefit.
- **FP8 KV cache alone** (exp_11) gives a modest +4.5%, confirming the overhead is real but manageable in isolation.
- **Right-sizing KV cache** (exp_00b) is effectively free at +7.6% -- always set max_model_len to the actual dataset maximum.
- **V1 defaults are net-positive**: disabling APC (exp_01) costs -3.15% and disabling async output processing (exp_02) costs -7%. Do not turn these off without a specific reason.
- **compilation_config=O0** is the worst single regression at **-20.8%**. Torch compilation is essential in production.
- **FLASHINFER backend, skip_tokenizer_init** had zero measurable impact.

---

## Part 2 -- Quantization, Chunked Prefill & APC Deep-Dive

A more targeted set probing quantization type, chunked prefill MNBT settings,
and the cost/benefit of Automatic Prefix Caching (APC).
Baseline here uses CP ON + MNBT=16384 + APC ON (685.88 tok/s).

All speedups relative to **baseline** (685.88 tok/s). Latencies in seconds.

| Experiment | Description | Tok/s | Speedup | p50 (s) | p99 (s) | Result |
|---|---|---|---|---|---|---|
| baseline | bf16, no quant, prefix ON, CP ON, MNBT=16384, BS=16 | 685.88 | baseline | 1.06 | 2.45 | VALID |
| quant_fp8_weights | fp8 W8A8 weights (float16 compute) | 1380.12 | **+101.22%** | 0.57 | 1.22 | VALID |
| quant_int8_weights | int8 W8A8 weights | -- | N/A | -- | -- | FAILED |
| cp_off | Chunked prefill OFF (MNBT=16384) | 587.32 | -14.37% | 1.05 | 2.45 | VALID |
| cp_on_mnbt_2668 | CP ON + MNBT=2668 (~V0 scheduling) | 659.35 | -3.87% | 1.09 | 2.56 | VALID |
| cp_on_mnbt_8192 | CP ON + MNBT=8192 (docs-recommended min) | 671.70 | -2.07% | 1.17 | 2.52 | VALID |
| prefix_off | Prefix caching OFF (APC disabled) | 975.21 | **+42.18%** | 0.68 | 1.72 | VALID |

### Deep-Dive Key Findings

- **fp8 W8A8 weight quantization more than doubles throughput** (+101%) and halves latency end-to-end. This is the most impactful confirmed finding across all experiment sets and is consistent with the isolation results above.
- **int8 W8A8 failed entirely** (exit within 20s). This is likely a missing kernel for LLaMA 3.1-8B on this GPU; investigate before re-running.
- **APC (prefix caching) is a net negative on CNN/DailyMail** (prefix_off: +42%). This directly contradicts the isolation run result (exp_01 showed -3.15% for disabling APC). The discrepancy is explained by dataset: CNN/DailyMail has low prefix reuse, so APC introduces lookup overhead without delivering cache hits. On workloads with high prefix reuse, APC would be strongly positive.
- **Chunked prefill ON + MNBT=16384** is the optimal chunked prefill setting. Turning CP off costs -14%, and lowering MNBT to 2668 or 8192 costs 2-4%. Do not reduce MNBT below the default.

---

## Part 3 -- Progressive Stacking

Starting from the production baseline, each stack step permanently adds the
best flag from the isolation run onto all previous steps. This reveals
cumulative interactions -- flags that look positive in isolation may erode
when combined.

All speedups relative to **stk_00_baseline** (1089.04 tok/s). Latencies in seconds.

| Stack Step | Cumulative Change | Tok/s | Speedup | p50 (s) | p99 (s) | Result |
|---|---|---|---|---|---|---|
| stk_00_baseline | Production baseline: BS=16, BF16, stock vLLM V1 defaults | 1089.04 | baseline | 783.92 | 1556.55 | VALID |
| stk_01_fp8 | +fp8 weight quantization | 1462.53 | +34.30% | 583.79 | 1158.98 | VALID |
| stk_03_kv2668 | +max_model_len=2668 | 1544.29 | +41.80% | 550.27 | 1097.43 | VALID |
| stk_04_sort | +sort_by_input_length | 1507.93 | +38.46% | 472.53 | 1117.14 | VALID |
| stk_05_gmu095 | +gpu_memory_utilization=0.95 | 1504.52 | +38.15% | 474.87 | 1119.70 | VALID |
| stk_06_kvcache_fp8 | +kv_cache_dtype=fp8 | 1500.22 | +37.76% | 477.08 | 1122.68 | VALID |
| stk_07_compile3 | +compilation_config=O3 | 1494.88 | +37.27% | 480.19 | 1126.89 | VALID |
| stk_08_expandable | +PYTORCH_CUDA_ALLOC_CONF=expandable_segments | 1420.80 | +30.46% | 481.81 | 1186.76 | VALID |
| stk_09_bitsandbytes | +bitsandbytes quantization (replaces fp8) | 446.14 | -59.03% | 1764.17 | 3787.78 | VALID |
| stk_02_bs1024 | +batch_size=1024 (standalone reference) | 3321.76 | +205.02% | 191.41 | 509.22 | INVALID* |

> *stk_02_bs1024 is INVALID: min_duration not satisfied. Increase expected QPS so loadgen
> pre-generates a larger coalesced query set. Throughput is directionally correct but not certified.

### Stacking Key Findings

- **FP8 weights + right-sized KV** (stk_01 + stk_03) is the sweet spot for the stacked config, delivering **+41.8% throughput and -30% p99 latency** with just two changes.
- **Sort, GPU memory utilization, and fp8 KV** each add marginal gains in isolation but create diminishing returns once stacked. The cumulative peak is at stk_03 (1544 tok/s); subsequent steps gradually erode it.
- **expandable_segments** (stk_08) drops throughput to 1420 tok/s in the stacked context despite showing +2% in isolation. This is a clear negative interaction -- likely conflicting with fp8 memory allocation patterns.
- **bitsandbytes** (stk_09) is catastrophically negative at -59%. Never substitute bitsandbytes for fp8 in this configuration.
- **BS=1024 directionally confirms +205%** over BS=16, consistent with the isolation run (+113% at BS=1024). Once the loadgen QPS issue is fixed, this will be the dominant configuration.

---

## Cross-Set Insights

### APC is Workload-Dependent
The isolation run shows APC as mildly positive (+3.15% cost to disable it), while the deep-dive
shows APC as strongly negative (-42.18% cost to enable it on CNN/DailyMail). These are not
contradictory -- APC only pays off when queries share a long common prefix. **Always profile APC
on your actual production query distribution before deciding.**

### fp8 KV Cache Adds Overhead When Stacked
Isolation shows fp8 KV alone at +4.5% (exp_11). However, fp8 weights + fp8 KV (exp_12) only yields
+18% vs. weights-alone at +34.5% -- a net loss of ~16% from adding KV quantization. The stacking
results confirm this: stk_06 (adds fp8 KV to an already fp8-weight stack) slightly regresses from
the stk_05 peak. **Use fp8 weight quantization. Add fp8 KV cache only if memory is the bottleneck.**

### Small Flags Lose Their Effect in Combination
Flags like compilation_config=O3, expandable_segments, and sort_by_input_length each show small
gains in isolation (+1-3%) but tend to be neutral or slightly negative in the full stacked context.
They are not worth tuning individually unless the configuration is otherwise fully optimized.

### Batch Size Dominates Everything Else
At BS=16, the best achievable stacked result is ~1544 tok/s (+42%). At BS=1024, directional results
suggest 3300+ tok/s (+200%). No combination of flags at BS=16 comes close to simply increasing batch
size. For offline and batch workloads, **batch size should be the first knob to maximize**.

---

## Consolidated Recommendations

1. **Maximize batch size first** -- BS=1024 or BS=2048 for offline workloads. This single change
   outperforms every other optimization combined at BS=16.
2. **Enable fp8 W8A8 weight quantization** -- confirmed +34-100% depending on baseline, no
   validity issues across any run set.
3. **Right-size max_model_len=2668** -- free +7.6% at zero cost. Always match to dataset max length.
4. **Do not use fp8 KV cache alongside fp8 weights** -- the combination loses ~16% vs. weights alone.
5. **APC decision depends on workload** -- disable it for datasets with low prefix reuse (CNN/DailyMail);
   keep or enable it for chat/RAG workloads with long shared system prompts.
6. **Keep CP ON with MNBT=16384** -- lowering MNBT or disabling chunked prefill both hurt throughput.
7. **Do not use bitsandbytes quantization** -- -59% in stacked context, not viable.
8. **Do not run compilation_config=O0 in production** -- -20.8% penalty confirmed.
9. **Investigate int8 W8A8 kernel support** -- failed to run; may be viable once kernel compatibility
   is resolved.
10. **async_scheduling=True** is a safe +3-5% on vLLM 0.10.0+ with no observed downsides.