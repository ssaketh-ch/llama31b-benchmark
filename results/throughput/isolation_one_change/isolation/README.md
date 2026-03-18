## Experiment Results Table

| Experiment | Description | Mean tok/s | Stdev tok/s | CV% | Min tok/s | Max tok/s | Speedup vs exp_00 |
|---|---|---|---|---|---|---|---|
| exp_00 | OG baseline: BF16 BS=16 no-quant no-sort. Reference for all deltas. | 1017.02 | 4.38 | 0.4% | 1011.81 | 1021.69 | - |
| exp_01 | +quantization=fp8 only. All else=baseline. | 1328.08 | 2.55 | 0.2% | 1324.84 | 1330.22 | +30.59% |
| exp_02 | +batch_size=512 only. No quant, no sort, all else=baseline. | 1984.85 | 4.37 | 0.2% | 1981.81 | 1991.28 | +95.16% |
| exp_03 | +batch_size=1024 only. No quant, no sort, all else=baseline. | 2153.72 | 58.26 | 2.7% | 2073.42 | 2197.75 | +111.77% |
| exp_04 | +max_model_len=2668 only. No quant, no sort, all else=baseline. | 1035.81 | 8.85 | 0.9% | 1023.95 | 1042.92 | +1.85% |
| exp_05 | +sort_by_input_length only. No quant, no KV resize, all else=baseline. | 1030.49 | 1.32 | 0.1% | 1028.75 | 1031.64 | +1.32% |
| exp_06 | +gpu_mem_util=0.95 only. All else=baseline. | FAILED | - | - | - | - | - |
| exp_07 | +chunked_prefill=True + max_num_batched_tokens=65536. All else=baseline. | FAILED | - | - | - | - | - |
| exp_08 | +PYTORCH_CUDA_ALLOC_CONF=expandable_segments only. All else=baseline. | 1037.76 | 3.46 | 0.3% | 1033.13 | 1040.74 | +2.04% |
| exp_09 | prefix_caching=False explicit (control). Expected: ~=baseline. | 1014.30 | 5.26 | 0.5% | 1011.03 | 1022.15 | -0.27% |
| exp_10 | +async_scheduling=True only [vLLM 0.10.0]. All else=baseline. | 1024.95 | 8.79 | 0.9% | 1015.70 | 1036.52 | +0.78% |
| exp_11 | +num_scheduler_steps=2 only [vLLM 0.10.0]. All else=baseline. | 997.02 | 5.80 | 0.6% | 990.34 | 1004.20 | -1.97% |
| exp_12 | +num_scheduler_steps=4 only [vLLM 0.10.0]. All else=baseline. | 1019.53 | 6.05 | 0.6% | 1012.25 | 1025.97 | +0.25% |
| exp_13 | +compilation_config=3 only [vLLM 0.10.0]. All else=baseline. | 1030.36 | 5.47 | 0.5% | 1026.01 | 1038.22 | +1.31% |
| exp_14 | +VLLM_ATTENTION_BACKEND=FLASHINFER only [vLLM 0.10.0]. All else=baseline. | 958.22 | 10.18 | 1.1% | 943.07 | 964.62 | -5.78% |
| exp_15 | +ENFORCE_EAGER=True only [vLLM 0.10.0]. All else=baseline. | 944.00 | 4.70 | 0.5% | 938.52 | 949.44 | -7.19% |
# Isolation Run Set (2026-03-12_21-33-44)

This folder contains the results of isolation experiments, where each run changes a single parameter relative to the original baseline (BF16, BS=16, no quantization, no sorting).

Contents:
- `experiments_summary.csv` — summary table with mean tokens/s, coefficient of variation, and speedup vs baseline for each experiment.
- `exp_*` folders — individual experiment runs (4 per experiment), logs, and system captures.

## Summary of Key Results

- **Batch size increases** (exp_02, exp_03) provided the largest throughput gains, with up to +112% speedup.
- **FP8 quantization** (exp_01) gave a +30% boost over baseline.
- **Expandable segments, sort-by-length, and compile level 3** gave small positive effects.
- **Prefix caching off** (exp_09) and extra scheduler steps (exp_11, exp_12) had little or negative impact.
- **FlashInfer backend** and **enforce eager** (exp_14, exp_15) regressed throughput.
- Some configurations (e.g., chunked prefill + batched tokens, high GPU memory utilization) failed to run successfully.

See `experiments_summary.csv` for full details and per-experiment statistics.
