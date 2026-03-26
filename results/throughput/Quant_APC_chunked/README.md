# Quantization & Prefill & APC experiments

This folder contains isolation experiments probing the effect of quantization strategy, chunked prefill
configuration, and Automatic prefix caching on LLaMA 3.1-8B offline throughput. All runs use BS=16, BF16 dtype,
and the MLPerf Offline scenario (PySUT) unless otherwise noted.

## Contents

- `experiments_summary.csv` -- summary table with tok/s, latency percentiles, and speedup vs baseline.
- `<experiment>/run_1/` -- per-experiment MLPerf logs, detail logs, and system captures.

---

## Experiment Results

All speedups are relative to **baseline** (685.88 tok/s). Latencies converted from nanoseconds to seconds.

| Experiment | Description | Tok/s | Speedup vs Baseline | p50 Latency (s) | p99 Latency (s) | Result |
|---|---|---|---|---|---|---|
| baseline | bf16, no quant, prefix ON, CP ON, MNBT=16384, BS=16 | 685.88 | baseline | 1.06 | 2.45 | VALID |
| quant_fp8_weights | fp8 W8A8 weights (float16 compute), BS=16 | 1380.12 | **+101.22%** | 0.57 | 1.22 | VALID |
| quant_int8_weights | int8 W8A8 weights, BS=16 | -- | N/A | -- | -- | FAILED |
| cp_off | Chunked prefill OFF (MNBT=16384), BS=16 | 587.32 | -14.37% | 1.05 | 2.45 | VALID |
| cp_on_mnbt_2668 | CP ON + MNBT=2668 (~V0 scheduling), BS=16 | 659.35 | -3.87% | 1.09 | 2.56 | VALID |
| cp_on_mnbt_8192 | CP ON + MNBT=8192 (docs-recommended min), BS=16 | 671.70 | -2.07% | 1.17 | 2.52 | VALID |
| prefix_off | Prefix caching OFF (APC disabled), BS=16 | 975.21 | **+42.18%** | 0.68 | 1.72 | VALID |

---

## Key Observations

- **FP8 weight quantization** is the single largest lever in this set, more than doubling baseline throughput
  (+101%) and halving p50/p99 latency. This is fp8 W8A8 with float16 compute -- a strong candidate for
  production use.
- **int8 W8A8** failed entirely (exit within 20s, no MLPerf output). Likely a missing kernel or
  compatibility issue; investigate before retrying.
- **Prefix caching OFF** (prefix_off) yields a surprising **+42%** gain over the baseline. The baseline
  runs with APC enabled by default; this result suggests APC introduces meaningful overhead for this
  dataset's query distribution, where cache hit rate may be low.
- **Chunked prefill OFF** (cp_off) regresses throughput by -14%. The V1 default CP ON + MNBT=16384
  is net-positive and should be kept.
- **Lowering MNBT** (cp_on_mnbt_2668 and cp_on_mnbt_8192) degrades throughput by 2-4% compared to the
  baseline MNBT=16384. The docs-recommended minimum of 8192 is still slightly worse than 16384 for this
  workload.

## Recommendations

1. **Enable fp8 W8A8 weight quantization** -- confirmed +100% throughput, no validity issues.
2. **Evaluate prefix caching need** -- APC appears to hurt throughput here; disable it if the workload
   has low prefix reuse.
3. **Keep CP ON with MNBT=16384** -- both CP OFF and lower MNBT values regress performance.
4. **Investigate int8 failure** -- check vLLM int8 kernel support for LLaMA 3.1-8B on the target GPU
   before re-running.