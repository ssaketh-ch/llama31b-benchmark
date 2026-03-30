# Scripts Overview

Reference guide for the automation scripts in `scripts/`.

---

## Common Environment Variables

- `INFERENCE_DIR`: path to the local `llama3.1-8b/` harness directory
- `CHECKPOINT_PATH`: model directory, usually `${INFERENCE_DIR}/model`
- `DATASET_PATH`: dataset directory, usually `${INFERENCE_DIR}/dataset`
- `GPU_COUNT`: tensor-parallel size / GPU count used by the benchmark
- `MLFLOW_TRACKING_URI`: MLflow server URI or local SQLite-backed URI
- `RUNS_PER_EXP`, `TARGET`, `RUN_SCOPE`: experiment selection controls used by several scripts

---

## Scripts

| Script | Purpose | Typical usage |
|---|---|---|
| `accuracy_test.sh` | Accuracy / ROUGE validation runs | `./scripts/accuracy_test.sh` |
| `download_model.sh` | Download the HF model into a local directory | `./scripts/download_model.sh <repo> <path>` |
| `quant_prefill.sh` | Quantization, APC, and chunked-prefill study | `./scripts/quant_prefill.sh` |
| `run_experiments.sh` | Legacy progression / stacked experiment workflow | `./scripts/run_experiments.sh` |
| `run_isolation.sh` | Stock-baseline one-knob isolation sweep | `./scripts/run_isolation.sh` |
| `run_stacked.sh` | Tuned-baseline stacked / combination sweep | `./scripts/run_stacked.sh` |
| `server_run.sh` | Server-scenario benchmark workflow | `./scripts/server_run.sh` |

---

## Notes

- Run scripts from the repository root unless you have a specific reason not to.
- Most scripts emit a generated `SUT_VLLM.py` per experiment and then execute `llama3.1-8b/main.py`.
- Several scripts assume the MLPerf inference tree, dataset, and model paths are already available.
- MLflow defaults vary by script: some use a local SQLite backend, while others assume a server at
  `http://localhost:5000` unless overridden.
- If you are deciding where to start:
  - use `run_isolation.sh` for stock-baseline ablations
  - use `run_stacked.sh` for tuned-baseline follow-up studies
  - use `accuracy_test.sh` to validate any throughput candidate before treating it as production-safe
