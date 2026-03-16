# Llama-3.1-8B vLLM Benchmark Harness on H200

## What this is repo for?
Reproducible benchmarking and tuning harness for Llama-3.1-8B on NVIDIA H200 (MIG 71GB) using vLLM. One-knob tests, isolation runs, MLflow logging, and summarized results.

## System info (runs captured here)
- GPU: NVIDIA H200 NVL, MIG enabled, driver 570.133.20, CUDA 12.8
- CPU: Dual Intel Xeon Platinum 8480+ (224 vCPUs total)
- RAM: 256 GB
- Key libraries: vLLM 0.10.0, PyTorch 2.7.1, Transformers 4.53.2

## Quickstart (step-by-step)
1) Install prerequisites
- GPU: NVIDIA driver + CUDA 12.1
- Conda: e.g.
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init
```

2) Clone the upstream MLPerf Inference repo (needed for loadgen/reference)
```bash
git clone --recurse-submodules https://github.com/mlcommons/inference.git --depth 1
cd inference
```

3) Set helper paths
```bash
export ROOT=$PWD/inference
export LLAMA_FOLDER=$PWD/language/llama3.1-8b
export LOADGEN_FOLDER=$PWD/loadgen
export DATASET_FOLDER=$LLAMA_FOLDER/dataset
```

4) Create and activate the conda env
```bash
conda create -y -n llama3.1-8b python=3.10
conda activate llama3.1-8b
conda install -y -c conda-forge libstdcxx-ng=12
```

5) Install requirements and loadgen (from $LLAMA_FOLDER)
```bash
cd $LLAMA_FOLDER
pip install -r requirements.txt
cd $LOADGEN_FOLDER && pip install -e . && cd -
```

6) Download the dataset (datacenter eval) with the MLCommons downloader
```bash
mkdir -p "$DATASET_FOLDER"
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
	https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-eval.uri \
	-d "$DATASET_FOLDER"
```

7) Download the model (choose one)
- Scripted HF download (needs a Hugging Face token with access):
```bash
huggingface-cli login
./scripts/download_model.sh meta-llama/Meta-Llama-3.1-8B-Instruct "$LLAMA_FOLDER/model"
```
- Simplest git-xet clone (needs a HF token with write access):
```bash
curl -sSfL https://hf.co/git-xet/install.sh | sh
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct "$LLAMA_FOLDER/model"
```

8) Clone this repo inside the LLAMA folder and use the scripts
```bash
cd $LLAMA_FOLDER
git clone https://github.com/ssaketh-ch/llama31b-benchmark.git
cd llama31b-benchmark

# Optional: set local helper vars for this repo
export INFERENCE_DIR=$(pwd)/llama3.1-8b
export CHECKPOINT_PATH=$INFERENCE_DIR/model
export DATASET_PATH=$INFERENCE_DIR/dataset
export MLFLOW_TRACKING_URI=http://localhost:5000

# Tuned baseline (FP8, TP=1, BS=1024, max_len=2668)
./scripts/baseline.sh exp_00

# One single-knob test (e.g., prefix caching)
./scripts/baseline.sh exp_A5

# Isolation (OG baseline vs one change)
./scripts/main_run.sh exp_00
./scripts/main_run.sh exp_A5
```

Notes:
- Quickstart scripts use the tuned FP8 / BS=1024 baseline. The plain Python examples below show the upstream BF16 / BS=16 reference commands.
- Keep `results/` and `llama3.1-8b/` intact after cloning if you want to reuse processed datasets/logs.

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

## Setup (detailed)
- Clone the upstream MLPerf repo: `git clone --recurse-submodules https://github.com/mlcommons/inference.git`
- Clone this repo and `cd` into it.
- Set helper vars (same as Quickstart): `INFERENCE_DIR`, `CHECKPOINT_PATH`, `DATASET_PATH`, `MLFLOW_TRACKING_URI`.
- Create env: `conda create -y -n llama3.1-8b python=3.10 && conda activate llama3.1-8b && conda install -y -c conda-forge libstdcxx-ng=12`.
- Install deps: `pip install -r env/requirements.txt`.
- Install loadgen: `cd ../inference/loadgen && pip install -e . && cd -`.

## Get the model 
Requires Hugging Face access to meta-llama/Meta-Llama-3.1-8B-Instruct.
```bash
huggingface-cli login  # ensure your token has model access
./scripts/download_model.sh meta-llama/Meta-Llama-3.1-8B-Instruct "$CHECKPOINT_PATH"
```

Alternative with git-xet (needs a Hugging Face write token):
```bash
# Install git-xet: https://hf.co/docs/hub/git-xet
curl -sSfL https://hf.co/git-xet/install.sh | sh

# When prompted for a password, use an access token with write permissions
# (create at https://huggingface.co/settings/tokens)
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

## Get the dataset (CNN/DailyMail)
Datacenter eval set (required):
```bash
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
	https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-eval.uri \
	-d "$DATASET_PATH"
```
Optional:
```bash
# 5k eval subset
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
	https://inference.mlcommons-storage.org/metadata/llama3-1-8b-sample-cnn-eval-5000.uri \
	-d "$DATASET_PATH"

# Calibration set
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
	https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-dailymail-calibration.uri \
	-d "$DATASET_PATH"
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
