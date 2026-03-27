# Llama‑3.1‑8B vLLM Benchmark Harness

**Reproducible benchmarking and tuning harness for Llama‑3.1‑8B on NVIDIA H200 (MIG 71GB) using vLLM.**

This repository provides a one‑knob harness for benchmarking and tuning Llama‑3.1‑8B on Hopper‑class GPUs. It includes MLflow‑logged experiments, isolated ablations, and summarized findings, making it easy to evaluate vLLM configuration knobs (FP8, chunked prefill, async scheduling, prefix caching, etc.) against well‑defined baselines.

---

## Scope

- Reproducible benchmarking and tuning for **Llama‑3.1‑8B** on **NVIDIA H200 NVL MIG 71GB** using **vLLM**.
- One‑knob configuration for controlled ablations and isolation runs.
- MLflow‑based logging of parameters, metrics, and artifacts.
- Clear, summarized results per configuration knob (throughput, latency, GPU util, scheduler behavior).

---

## Target Audience

- **Practitioners and infra engineers** benchmarking Hopper‑class GPUs and vLLM for large‑context Llama‑3.1‑8B workloads.  
- **MLOps teams** looking for reusable, MLflow‑integrated benchmark templates.  
- **ML engineers** comparing vLLM tuning knobs:
  - FP8 vs BF16  
  - Chunked prefill vs full prefill  
  - Async scheduling  
  - Prefix caching  
  - Expandable segments, sorting, FlashInfer, etc.

---

## Hardware & Software

| Category  |  Version / Configuration    |
|-----------|-----------------------------|
| GPU       | NVIDIA H200 NVL, MIG 71GB   |
| CUDA      | 12.8                        |
| vLLM      | v0.17.1                     |
| PyTorch   | 2.10.0+cu129             |
| Transformers | 4.57.6         |
| Python    | 3.12.13             |

All experiments are tested on this configuration; results may vary under different vLLM, CUDA, or driver versions.

---

## Baselines

1. **OG baseline** (`main_run.sh`)  
   - BF16, no quantization  
   - BS=16, TP=1, max_len=131072  
   - No sorting, no expansion, all advanced knobs off  

2. **Tuned baseline** (`baseline.sh exp_00`)  
   - FP8, TP=1, BS=1024, max_len=2668  
   - Sorting on, expandable_segments on  
   - All other knobs (prefix caching, async scheduling, FlashInfer, etc.) off  

All ablations are measured relative to these baselines.

---

## Key Findings (from Ablation)

- **Prefix caching**: **+57% tokens/sec** (throughput), subject to accuracy and hit‑rate validation.  
- **Async scheduling**: **+5% tokens/sec** improvement at acceptable latency cost.  
- **Regressions**:  
  - High **GPU memory utilization** (up to **0.98**) under some configs.  
  - Increased **scheduler delay** and **scheduler steps > 1** with certain knob combinations.  
  - **FlashInfer** introduces regressions for this workload and is not beneficial here.

These results are MLflow‑logged and summarized in the `results/` and `profiling/` directories.