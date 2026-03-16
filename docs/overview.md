# Llama-3.1-8B vLLM Benchmark Harness

Scope:
- Reproducible benchmarking and tuning harness for Llama-3.1-8B on NVIDIA H200 (MIG 71GB) using vLLM.
- One-knob ablations and isolation runs; MLflow logging; summarized findings.

Audience:
- Practitioners/infra engineers benchmarking Hopper-class GPUs.
- MLOps teams needing MLflow-logged benchmarking templates.
- Engineers comparing vLLM knobs (FP8, chunked prefill, async scheduling, prefix caching, etc.).

Hardware/Software (tested):
- GPU: NVIDIA H200 NVL MIG 71GB
- CUDA: 12.6.x
- vLLM: (fill actual)
- PyTorch: (fill actual)
- Transformers: (fill actual)
- Python: (fill actual)

Baselines:
- OG baseline (main_run.sh): BF16, no quant, BS=16, TP=1, max_len=131072, no sorting, knobs off.
- Tuned baseline (baseline.sh exp_00): FP8, TP=1, BS=1024, max_len=2668, sorting on, expandable_segments on; other knobs off.

Key findings (from ablation):
- Prefix caching: +57% tok/s (validate accuracy/hit-rate).
- Async scheduling: +5%.
- Regressions: high gpu_mem_util (0.98), scheduler delay, scheduler steps>1, FlashInfer (for this workload).
