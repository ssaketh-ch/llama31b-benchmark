**Scripts Overview**

- **Purpose:** Concise descriptions and usage notes for automation scripts used to run MLPerf-style benchmarks and ablations with vLLM and the Llama-3.1-8B model.
- **Location:** `scripts/` — run these from the repository root or the `scripts/` directory.

**Common environment variables**
- `INFERENCE_DIR`: path to inference code and `main.py` (default: scripts dir or project-specific path).
- `CHECKPOINT_PATH`: model directory (default: `${INFERENCE_DIR}/model`).
- `DATASET_PATH`: dataset root (default: `${INFERENCE_DIR}/data`).
- `GPU_COUNT`: number of GPUs / tensor parallel size used by benchmarks.
- `RESULTS_BASE` / `RESULTS_DIR`: where runs write outputs (scripts set sensible defaults).
- `MLFLOW_TRACKING_URI`: MLflow server or `sqlite:////path/to/mlflow.db` (many scripts default to a local SQLite file).
- `RUNS_PER_EXP`, `TARGET`, `RUN_SCOPE`: control which experiments run and how many repetitions.

**Scripts**

- `accuracy_test.sh` — Run accuracy ablations (generates per-experiment `SUT_VLLM.py`, runs MLPerf Offline accuracy, computes ROUGE, aggregates results).
  - Usage examples:
    - `./accuracy_test.sh` — run all experiments.
    - `RUNS_PER_EXP=2 ./accuracy_test.sh` — two runs per experiment.
    - `RUN_SCOPE=only ./accuracy_test.sh exp_12` — run only `exp_12`.

- `download_model.sh` — Helper to download a Hugging Face repo to a local directory using `huggingface-cli`.
  - Usage: `./download_model.sh <huggingface_repo> <local_dir>`.
  - Note: requires `huggingface_hub` / `huggingface-cli` installed.

- `quant_prefill.sh` — Run quantisation, prefix-cache, and chunked-prefill ablations; produces throughput results and CSV summary.
  - Usage examples:
    - `./quant_prefill.sh` — run baseline + quant/prefill experiments.
    - `TARGET=quant_fp8_kv RUN_SCOPE=only ./quant_prefill.sh` — run a single target.

- `run_experiments.sh` — Throughput/stacked experiments progression; integrates with MLflow (defaults to `http://localhost:5000`).
  - Usage: `./run_experiments.sh` or `./run_experiments.sh exp_12`.

- `run_isolation.sh` — Isolation ablation suite (one change per experiment) for vLLM v0.17.1; writes `isolation_results_v17/<timestamp>`.
  - Usage: `./run_isolation.sh`, `TARGET=exp_04 ./run_isolation.sh`, `RUNS_PER_EXP=2 ./run_isolation.sh`.

- `run_stacked.sh` — Stacked study (phases A–D) using vLLM V1 engine features and combinations.
  - Usage: `./run_stacked.sh`, `./run_stacked.sh exp_A1`, `TARGET=exp_C ./run_stacked.sh`.

- `server_run.sh` — Server-scenario ablation suite producing Server-specific MLPerf metrics (TTFT, TPOT, E2E latencies). Includes `#SBATCH` headers for SLURM environments.
  - Usage: `chmod +x server_run.sh && ./server_run.sh` or submit via `sbatch server_run.sh`.

**Quick tips & gotchas**
- Scripts expect `main.py`, `evaluation.py`, and a `dataset` module to be present under the `INFERENCE_DIR` path (set via `INFERENCE_DIR` env var if needed).
- Many scripts emit a `SUT_VLLM.py` for each experiment and copy it to `INFERENCE_DIR` before running `main.py`.
- If you run on a machine without an MLflow server, the scripts default to a local SQLite tracking DB inside the script folder; set `MLFLOW_TRACKING_URI` to a remote server to centralize runs.
- Some experiments require `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` or other environment flags — check the emitted SUTs and `README` notes in each results folder.
- For SLURM, adjust `#SBATCH` settings (`--chdir`, `--gres`, partition) to match your cluster.

**Next steps**
- I created `scripts/README.md`. Tell me if you want this committed and pushed, or if you want expanded details per script (examples, required packages, or recommended SLURM settings).
