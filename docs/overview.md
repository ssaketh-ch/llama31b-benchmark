# Llama-3.1-8B vLLM Benchmark Harness

Concise orientation guide for the repository: what it measures, which baselines matter, and where
to find the corresponding scripts and result bundles.

---

## Scope

- Reproducible benchmarking for **Llama-3.1-8B** on **NVIDIA H200 NVL MIG 71GB**
- vLLM-focused tuning across isolation, stacked, and accuracy workflows
- MLPerf Offline scenario as the primary benchmark path
- Logged artifacts, summarized reports, and profiling captures kept in-repo

---

## Main Workflows

| Workflow | Script | Purpose |
|---|---|---|
| Isolation sweep | `scripts/run_isolation.sh` | One change at a time vs the stock BF16/BS=16 production baseline |
| Stacked / combination sweep | `scripts/run_stacked.sh` | Tuned FP8/BS=1024 baseline plus Phase A-D follow-up studies |
| Legacy progression sweep | `scripts/run_experiments.sh` | Older progression experiments used for the original stacked report |
| Accuracy validation | `scripts/accuracy_test.sh` | ROUGE checks for throughput candidates and negative controls |
| Quant / APC / chunked prefill deep-dive | `scripts/quant_prefill.sh` | Focused study on quantization, APC, and chunked-prefill interactions |

---

## Baselines

### Stock production baseline

- BF16
- batch size `16`
- stock vLLM V1 defaults
- used by the isolation study and as the main reference point in the repo-level report

### Tuned serving baseline

- FP8 weights
- batch size `1024`
- `max_model_len=2668`
- used by the combination study to test follow-up backend and scheduling changes

---

## Current Headline Findings

- **FP8 weight quantization** is the highest-value single improvement from the stock baseline.
- **`max_model_len=2668`** is a free throughput win and is accuracy-safe for CNN/DailyMail.
- **Large batch sizes** unlock the largest throughput gains but are blocked by a major accuracy regression.
- **APC is workload-dependent**: harmful on low-prefix-reuse CNN/DailyMail, potentially useful on chat/RAG-style prompts.
- **FLASHINFER is neutral in the stock-baseline isolation run but positive in the tuned-baseline combination sweep**, so backend conclusions depend on the baseline context.

---

## Where To Read Results

| Location | What it contains |
|---|---|
| `README.md` | Repo-level summary, setup, and consolidated findings |
| `results/README.md` | Full report across throughput and accuracy studies |
| `results/throughput/isolation_one_change/` | Stock-baseline one-knob sweep |
| `results/throughput/progression_full_stack/` | Progressive stacking study |
| `results/throughput/Quant_APC_chunked/` | Quantization, APC, and chunked-prefill deep-dive |
| `results/throughput/Combination results/` | Tuned-baseline combination sweep and raw captures |
| `profiling/README.md` | Nsight Systems analysis for `batch=16` vs `batch=1024` |

---

## Notes

- Raw benchmark logs and profiling captures are intentionally retained in the repository.
- Some result directories preserve older naming for historical continuity.
- For the latest interpretation of the data, prefer the root [README](../README.md) and
  [results report](../results/README.md).
