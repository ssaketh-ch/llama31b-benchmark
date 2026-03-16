# Single-Knob Runs (2026-03-09)

Summary of one-at-a-time toggles measured on the fixed baseline (FP8, TP=1, BS=1024, max_len=2668, sorted batching, expandable_segments). Source data: [summary_with_latency.csv](summary_with_latency.csv).

## Quick Takeaways
- Baseline throughput: 2,839.71 tok/s (VALID).
- Biggest gain: enabling prefix caching here (exp_A5) reached 4,475.80 tok/s (+57.61%). 
- Minor gains: chunked prefill alone (+0.85%), async scheduling alone (+4.93%), compile level 3 alone (+0.09%).
- Regressions: gpu_memory_utilization=0.98 (-9.30%), scheduler_delay_factor=0.1 (-18.57%), num_scheduler_steps=2 (-12.53%), num_scheduler_steps=4 (-10.54%), FlashInfer backend (-6.44%).
- Fail: async + scheduler_steps=2 (exp_C2) failed early.

## Best/Worst (tokens/sec)
| Run | tok/s | Delta vs baseline | Notes |
| --- | ----: | ----------------: | ----- |
| exp_A5_prefix_cache_on | 4,475.80 | +57.61% | Surprising win; validate accuracy and cache hit behavior. |
| exp_A1_chunked_prefill | 2,863.78 | +0.85% | Small gain; stability knob. |
| exp_B1_async_scheduling | 2,979.62 | +4.93% | Helped slightly for this run. |
| exp_B4_compile_lvl3 | 2,842.23 | +0.09% | Nearly neutral; compile overhead amortized. |
| exp_A4_gpu_mem_0.98 | 2,575.72 | -9.30% | Lower throughput. |
| exp_A6_scheduler_delay_0.1 | 2,312.27 | -18.57% | Worst valid regression. |
| exp_B2_sched_steps_2 | 2,483.98 | -12.53% | Multi-step scheduling hurts. |
| exp_B3_sched_steps_4 | 2,540.35 | -10.54% | Same trend. |
| exp_B5_flashinfer | 2,656.88 | -6.44% | Slower than default backend. |
| exp_C2_async_plus_sched_steps_2 | FAILED | N/A | Combination unstable here. |

## Latency Snapshots (ns, Session 4 style metrics)
- Baseline mean/p50/p99: 473,394,786,056 / 421,444,226,431 / 1,192,970,803,884.
- Fastest latency profile: exp_A5 mean/p50/p99: 314,407,292,283 / 292,436,948,982 / 757,265,193,011.
- Worst latency profile among valids: exp_A6 mean/p50/p99: 602,213,785,291 / 552,532,486,812 / 1,466,374,797,311.

## Recommendations
- Re-verify exp_A5 correctness (prefix cache ON) with accuracy/ROUGE; if valid, it is the best single knob here.
- Avoid scheduler delay and multi-step scheduling for offline batches; they consistently hurt throughput and latency.
- FlashInfer remains slower than default; stick to the default attention backend for this setup.
- Keep chunked prefill as a stability knob; benefit is small but non-negative.
