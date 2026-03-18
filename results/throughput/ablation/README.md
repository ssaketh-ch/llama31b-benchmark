## Main Variant Experiment Results

| Experiment | Tokens/s | Samples/s | vs Baseline (%) | Valid | Exit Code | Duration (s) | Description |
|---|---|---|---|---|---|---|---|
| exp_00_BASELINE | 2839.71 | 22.19 | +0.00 | VALID | 0 | 1293 | BASELINE: FP8+TP1+BS1024+max_len2668+sort+expandable_segments. All other flags OFF. |
| exp_A1_chunked_prefill | 2863.78 | 22.37 | +0.85 | VALID | 0 | 1404 | +enable_chunked_prefill=True only. All other flags=baseline. |
| exp_A2_batched_tokens_65536 | 2846.39 | 22.24 | +0.24 | VALID | 0 | 1291 | +max_num_batched_tokens=65536 only. chunked_prefill=False. All other flags=baseline. |
| exp_A3_chunked_plus_batched_tokens | 2852.89 | 22.29 | +0.46 | VALID | 0 | 1289 | +chunked_prefill=True + max_num_batched_tokens=65536. Logical pair. All other flags=baseline. |
| exp_A4_gpu_mem_0.98 | 2575.72 | 20.12 | -9.30 | VALID | 0 | 1533 | +gpu_memory_utilization=0.98 only. All other flags=baseline. |
| exp_A5_prefix_cache_on | 4475.80 | 34.97 | +57.61 | VALID | 0 | 849 | +enable_prefix_caching=True only. 
| exp_A6_scheduler_delay_0.1 | 2312.27 | 18.06 | -18.57 | VALID | 0 | 1574 | +scheduler_delay_factor=0.1 only. All other flags=baseline. |
| exp_B1_async_scheduling | 2979.62 | 23.28 | +4.93 | VALID | 0 | 1364 | +async_scheduling=True only [vLLM 0.10.0]. All other flags=baseline. |
| exp_B2_sched_steps_2 | 2483.98 | 19.41 | -12.53 | VALID | 0 | 1589 | +num_scheduler_steps=2 only [vLLM 0.10.0]. All other flags=baseline. |
| exp_B3_sched_steps_4 | 2540.35 | 19.85 | -10.54 | VALID | 0 | 1439 | +num_scheduler_steps=4 only [vLLM 0.10.0]. All other flags=baseline. |
| exp_B4_compile_lvl3 | 2842.23 | 22.20 | +0.09 | VALID | 0 | 1288 | +compilation_config=3 only [vLLM 0.10.0]. First batch ~2-5min compile. All other flags=baseline. |
| exp_B5_flashinfer | 2656.88 | 20.76 | -6.44 | VALID | 0 | 1544 | +VLLM_ATTENTION_BACKEND=FLASHINFER only [vLLM 0.10.0]. All other flags=baseline. |
| exp_C1_async_plus_compile | 2949.93 | 23.05 | +3.88 | VALID | 0 | 1369 | +async_scheduling + compilation_config=3. Run only if B1 AND B4 both positive. |
| exp_C2_async_plus_sched_steps_2 | FAILED | FAILED | N/A | FAILED | 0 | 92 | +async_scheduling + num_scheduler_steps=2. Run only if B1 OR B2 positive. |
| exp_C3_best_A_plus_best_B | 2914.94 | 22.77 | +2.65 | VALID | 0 | 1384 | Best Phase A + best Phase B. EDIT SUT BEFORE RUNNING based on A+B results. |

## Reproducibility Check: SPECIAL1 and SPECIAL2

The following runs were performed to check if the results of exp_A5 (prefix caching) are reproducible:

| Experiment | Tokens/s | Samples/s | vs Baseline (%) | Valid | Exit Code | Duration (s) | Description |
|---|---|---|
| exp_A5_prefix_cache_on | 4475.80 | 34.97 | +57.61 | VALID | 0 | 849 | +enable_prefix_caching=True only. 
| exp_SPECIAL1 | 4458.3 | 34.83 | +57.06 | VALID | 0 | 852 | Reproducibility check for exp_A5. Same config as exp_A5. |
| exp_SPECIAL2 | 4493.4 | 35.10 | +58.29 | VALID | 0 | 847 | Reproducibility check for exp_A5. Same config as exp_A5. |
# Variant Run Set (2026-03-09_13-25-42)

This folder contains results from single-factor variant experiments, where each run changes one parameter relative to the tuned baseline (FP8, TP=1, BS=1024, max_len=2668).

Contents:
- `variant_summary.csv` — summary of all variant results.
- `variant_summary_with_latency.csv` — variant results with detailed latency stats.
- `exp_*` folders — individual variant experiments and logs.

## Highlights
- **Prefix caching** (+57.6%) was the most impactful single change.
- **Async scheduling** provided a small additive gain.
- Higher GPU memory utilization, scheduler delay, extra scheduler steps, and FlashInfer regressed throughput.

See the summary CSVs for full results and latency breakdowns.
