# Llama-3.1-8B vLLM Benchmark Harness (H200 MIG)

## What this is
Reproducible benchmarking and tuning harness for Llama-3.1-8B on NVIDIA H200 (MIG 71GB) using vLLM. One-knob tests, isolation runs, MLflow logging, and summarized results.

## Quickstart
```bash
# From repo root (this repo is self-contained)
export INFERENCE_DIR=$(pwd)/llama3.1-8b
export CHECKPOINT_PATH=$INFERENCE_DIR/model
export DATASET_PATH=$INFERENCE_DIR/dataset
export MLFLOW_TRACKING_URI=http://localhost:5000

# (Optional) Download model locally; requires a Hugging Face token with access
# Usage: ./scripts/download_model.sh <model_repo> <local_dir>
./scripts/download_model.sh meta-llama/Meta-Llama-3.1-8B-Instruct "$CHECKPOINT_PATH"

# Tuned baseline (FP8, TP=1, BS=1024, max_len=2668)
./scripts/baseline.sh exp_00

# One single-knob test (e.g., prefix caching)
./scripts/baseline.sh exp_A5

# Isolation (OG baseline, one change vs OG, 4 runs)
./scripts/main_run.sh exp_00
./scripts/main_run.sh exp_A5
```

Repo notes:
- You still need the upstream MLPerf Inference repo for loadgen and reference assets (clone from https://github.com/mlcommons/inference.git as shown below).
- If you want to reuse the processed datasets/logs stored in this repo (instead of downloading fresh), also clone this repo and keep its `results/` and `llama3.1-8b` paths intact. Otherwise, follow the download steps below.

Note: The Quickstart scripts use the tuned FP8 / BS=1024 baseline. The plain Python examples below show the reference BF16 / BS=16 commands from the main repo.

## MLflow logging and UI
- What we log: run params (experiment id, batch size, flags), metrics (tokens/s, samples/s, duration, validity), and latency summaries when available. Artifacts include mlperf logs and the SUT snapshot for that run.
- Environment: set `MLFLOW_TRACKING_URI` (Quickstart above). If unset, MLflow defaults to a local `mlruns` folder.
- Start a local server (from repo root):
```bash
export MLFLOW_BACKEND_URI=sqlite:///$(pwd)/mlflow.db
export MLFLOW_ARTIFACT_ROOT=file:$(pwd)/mlartifacts

mlflow server \
	--backend-store-uri $MLFLOW_BACKEND_URI \
	--default-artifact-root $MLFLOW_ARTIFACT_ROOT \
	--host 0.0.0.0 --port 5000

# Then point runs at it (already in Quickstart)
export MLFLOW_TRACKING_URI=http://localhost:5000
```
- UI: open http://localhost:5000 to browse runs. Stop the server with Ctrl+C.
 - Artifacts land in `mlruns/` by default (local), or in `mlartifacts/` when using the server env vars above.

## Install (submission-agnostic)
```bash
# Clone MLPerf Inference (needed for loadgen and reference files)
git clone --recurse-submodules https://github.com/mlcommons/inference.git

# Python + CUDA runtime
conda create -y -n llama3.1-8b python=3.10
conda activate llama3.1-8b
conda install -y -c conda-forge libstdcxx-ng=12

# Project deps
pip install -r env/requirements.txt

# LoadGen (for MLPerf harness integration, still useful for local runs)
cd $(pwd)/../loadgen
pip install -e .
cd -
```

## Get the model 
Requires Hugging Face access to meta-llama/Meta-Llama-3.1-8B-Instruct.
```bash
huggingface-cli login  # ensure your token has model access
./scripts/download_model.sh meta-llama/Meta-Llama-3.1-8B-Instruct "$CHECKPOINT_PATH"
```

Alternative with git-xet (needs a Hugging Face write token):
```bash
curl -sSfL https://hf.co/git-xet/install.sh | sh
# When prompted for a password, use a Hugging Face access token with write access (create at https://huggingface.co/settings/tokens)
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

## Get the dataset (CNN/DailyMail eval + calibration, no mlcr)
```bash
DL_SCRIPT="https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh"

# Full datacenter eval
bash <(curl -s "$DL_SCRIPT") https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-eval.uri -d "$DATASET_PATH"

# 5k eval subset
bash <(curl -s "$DL_SCRIPT") https://inference.mlcommons-storage.org/metadata/llama3-1-8b-sample-cnn-eval-5000.uri -d "$DATASET_PATH"

# Calibration set
bash <(curl -s "$DL_SCRIPT") https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-dailymail-calibration.uri -d "$DATASET_PATH"
```

## Run performance (plain python, no submission tooling)
```bash
python -u llama3.1-8b/main.py --scenario Offline \
	--model-path "$CHECKPOINT_PATH" \
	--batch-size 16 \
	--dtype bfloat16 \
	--user-conf llama3.1-8b/user.conf \
	--total-sample-count 13368 \
	--dataset-path "$DATASET_PATH" \
	--output-log-dir output \
	--tensor-parallel-size 1 \
	--vllm
```

### Server scenario (not tested in this harness)
Reference command from upstream; adjust batch size and TP for your hardware.
```bash
python -u llama3.1-8b/main.py --scenario Server \
	--model-path "$CHECKPOINT_PATH" \
	--batch-size 16 \
	--dtype bfloat16 \
	--user-conf llama3.1-8b/user.conf \
	--total-sample-count 13368 \
	--dataset-path "$DATASET_PATH" \
	--output-log-dir output \
	--tensor-parallel-size 1 \
	--vllm
```
Note: The upstream reference notes that Server SUT was not tested for GPU runs; same here.

## Run accuracy (plain python)
```bash
python -u llama3.1-8b/main.py --scenario Offline \
	--model-path "$CHECKPOINT_PATH" \
	--batch-size 16 \
	--accuracy \
	--dtype bfloat16 \
	--user-conf llama3.1-8b/user.conf \
	--total-sample-count 13368 \
	--dataset-path "$DATASET_PATH" \
	--output-log-dir output \
	--tensor-parallel-size 1 \
	--vllm

# Evaluate the accuracy log
python llama3.1-8b/evaluation.py --mlperf-accuracy-file output/mlperf_log_accuracy.json \
	--dataset-file "$DATASET_PATH" --dtype int32
```

### Upstream accuracy baselines (BF16, reference GPU run)
- Datacenter (full CNN/DailyMail): rouge1 38.7792, rouge2 15.9075, rougeL 24.4957, rougeLsum 35.793, gen_len 8167644, gen_num 13368. Target is 99% of these for rouge metrics and 90% for gen_len.
 - Edge (5k) baseline values are available in the upstream reference doc if needed.
For edge (5k) baselines, see the upstream reference doc if needed.

## Repository layout (key paths)
- Code + configs: [llama3.1-8b/](llama3.1-8b/)
- Scripts: [scripts/](scripts/)
- Throughput results: [results/throughput/](results/throughput/)
	- One-knob tests: [results/throughput/single-knob-tests/](results/throughput/single-knob-tests/)
	- Isolation (one change vs OG): [results/throughput/isolation_one_change/](results/throughput/isolation_one_change/)
	- Progression (stacked tuning): [results/throughput/progression_full_stack/](results/throughput/progression_full_stack/)
- Accuracy optimization (placeholder): [results/accuracy/](results/accuracy/)

## Scenarios tested in this harness
- Offline: tested (tuned FP8 baseline and sweeps).
- Server: reference command shown above, but not tested here (the main repo also notes GPU Server SUT untested).
- Accuracy: Offline scripts provided; Server accuracy not tested here.

## Optional references
- Official MLPerf reference instructions and Docker/mlcr automation: https://github.com/mlcommons/inference/tree/master/language/llama3.1-8b

Add accuracy runs and reports under [results/accuracy/](results/accuracy/) when you start accuracy-focused tuning.

# llama31b-benchmark
Reference implementations of MLPerf® inference benchmarks
