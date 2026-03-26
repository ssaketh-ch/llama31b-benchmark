# Llama-3.1-8B vLLM Benchmark Harness on H200

## What is this repo for?
Reproducible benchmarking and tuning harness for Llama-3.1-8B on NVIDIA H200 (MIG 71GB) using vLLM.
One-knob tests, isolation runs, stacked tuning, accuracy validation, MLflow logging, and summarized results.

---

## System Info
- **GPU:** NVIDIA H200 NVL, MIG enabled, driver 570.133.20, CUDA 12.8
- **CPU:** Dual Intel Xeon Platinum 8480+ (224 vCPUs total)
- **RAM:** 256 GB
- **Key libraries:**
  - vLLM 0.17.1a
  - PyTorch 2.10.0+cu129
  - Transformers 4.57.6
  - CUDA 12.9 (nvidia-cuda-runtime-cu12 12.9.79)
  - FlashInfer 0.6.4
  - bitsandbytes 0.49.2
  - accelerate 1.13.0
  - ray 2.54.0
  - triton 3.6.0
  - huggingface_hub 0.36.2
  - safetensors 0.7.0
  - tokenizers 0.22.2

---

## Key Results at a Glance

> Full tables and per-experiment logs live in the result subfolders.
> Reference baseline: BF16, BS=16, stock vLLM V1 defaults (~1082-1089 tok/s depending on run date).
> Best certified stacked config: FP8 weights + max_model_len=2668 (1544.29 tok/s, +41.8% vs baseline).
> BS=1024 directional result: ~2303-3322 tok/s (+112 to +205%) -- not yet certified due to loadgen QPS issue.

### Top Throughput Wins (Isolation, from OG baseline)

| Change | Tok/s | Speedup | Safe to Ship? |
|---|---|---|---|
| batch_size=1024 | ~2154-3322 | +112 to +205% | PENDING (accuracy regression -- see below) |
| FP8 weight quantization | ~1328-1456 | +34% | PENDING (accuracy run failed -- re-run needed) |
| max_model_len=2668 (right-sized KV) | ~1165 | +7.6% | YES -- confirmed accuracy-neutral |
| async_scheduling=True | ~1115 | +3-5% | YES -- low risk, no accuracy concern |
| sort_by_input_length | ~1115 | +3% | PENDING (accuracy run failed -- re-run needed) |
| prefix_caching=True | ~4476 | +57.6% | CONDITIONAL -- workload-dependent (see below) |

### Critical Accuracy Findings

| Finding | Impact | Status |
|---|---|---|
| BS=1024 causes ~13 ROUGE-1 regression | Hard blocker for certified runs | Open -- root cause under investigation |
| max_model_len < 2668 truncates inputs | -2.24 R1 at len=1500; fails at len=1000 | Resolved -- lock to 2668 |
| FP8 accuracy run failed to produce output | Cannot certify FP8 without this | Blocker -- re-run needed |
| APC neutral on unique queries, negative on CNN/DM | +42% from disabling on low-reuse dataset | Workload-dependent |
| compilation_config=3, chunked_prefill, dtype=fp16 | All confirmed ROUGE-neutral | Safe |

### Things That Hurt -- Do Not Use

| Change | Throughput Impact | Notes |
|---|---|---|
| compilation_config=O0 | -20.82% | Torch compilation is mandatory |
| scheduler_delay_factor=0.1 | -18.57% | Worst single flag in tuned-baseline set |
| bitsandbytes quantization | -59% (stacked) | Never substitute for fp8 |
| num_scheduler_steps=2/4 | -11 to -13% | Harmful at BS=1024 |
| FLASHINFER backend | -6 to -7% | Default backend is better for this workload |
| gpu_memory_utilization=0.98 | -9.30% | Memory pressure causes stalls |
| fp8 KV cache alongside fp8 weights | -16% vs weights-alone | Use weights-only fp8 |

---

## Quickstart (step-by-step)

1) Install prerequisites
- GPU: NVIDIA driver + CUDA 12.1
- Conda:
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init
```

2) Clone the upstream MLPerf Inference repo
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

5) Install requirements and loadgen
```bash
cd $LLAMA_FOLDER
pip install -r requirements.txt
cd $LOADGEN_FOLDER && pip install -e . && cd -
```

6) Download the dataset
```bash
mkdir -p "$DATASET_FOLDER"
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-eval.uri \
    -d "$DATASET_FOLDER"
```

7) Download the model (choose one)
```bash
# Scripted HF download (needs a HF token with access)
huggingface-cli login
./scripts/download_model.sh meta-llama/Meta-Llama-3.1-8B-Instruct "$LLAMA_FOLDER/model"

# Or git-xet clone (needs a HF write token)
curl -sSfL https://hf.co/git-xet/install.sh | sh
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct "$LLAMA_FOLDER/model"
```

8) Clone this repo and run
```bash
cd $LLAMA_FOLDER
git clone https://github.com/ssaketh-ch/llama31b-benchmark.git
cd llama31b-benchmark

export INFERENCE_DIR=$(pwd)/llama3.1-8b
export CHECKPOINT_PATH=$INFERENCE_DIR/model
export DATASET_PATH=$INFERENCE_DIR/dataset
export MLFLOW_TRACKING_URI=http://localhost:5000

# Tuned baseline (FP8, TP=1, BS=1024, max_len=2668)
./scripts/baseline.sh exp_00

# One single-knob test
./scripts/baseline.sh exp_A5

# Isolation run
./scripts/main_run.sh exp_00
./scripts/main_run.sh exp_A5
```

> Quickstart scripts use the tuned FP8/BS=1024 baseline. Plain Python examples below use the
> upstream BF16/BS=16 reference commands.

---

## MLflow Logging

- **What is logged:** run params (experiment id, batch size, flags), metrics (tokens/s, samples/s,
  duration, validity), latency summaries, MLPerf logs, and the SUT snapshot for each run.
- **Start a local server:**
```bash
export MLFLOW_BACKEND_URI=sqlite:///$(pwd)/mlflow.db
export MLFLOW_ARTIFACT_ROOT=file:$(pwd)/mlartifacts

mlflow server \
    --backend-store-uri $MLFLOW_BACKEND_URI \
    --default-artifact-root $MLFLOW_ARTIFACT_ROOT \
    --host 0.0.0.0 --port 5000

export MLFLOW_TRACKING_URI=http://localhost:5000
```
- **UI:** open http://localhost:5000. Artifacts land in `mlruns/` (local) or `mlartifacts/` (server).

---

## Setup (detailed)

- Clone upstream MLPerf repo: `git clone --recurse-submodules https://github.com/mlcommons/inference.git`
- Clone this repo and `cd` into it.
- Set helper vars: `INFERENCE_DIR`, `CHECKPOINT_PATH`, `DATASET_PATH`, `MLFLOW_TRACKING_URI`.
- Create env: `conda create -y -n llama3.1-8b python=3.10 && conda activate llama3.1-8b && conda install -y -c conda-forge libstdcxx-ng=12`
- Install deps: `pip install -r env/requirements.txt`
- Install loadgen: `cd ../inference/loadgen && pip install -e . && cd -`

---

## Get the Model

Requires Hugging Face access to `meta-llama/Meta-Llama-3.1-8B-Instruct`.
```bash
huggingface-cli login
./scripts/download_model.sh meta-llama/Meta-Llama-3.1-8B-Instruct "$CHECKPOINT_PATH"
```

Alternative with git-xet (needs a HF write token):
```bash
curl -sSfL https://hf.co/git-xet/install.sh | sh
git clone https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
```

---

## Get the Dataset (CNN/DailyMail)

```bash
# Datacenter eval set (required)
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-eval.uri \
    -d "$DATASET_PATH"

# 5k eval subset (optional)
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    https://inference.mlcommons-storage.org/metadata/llama3-1-8b-sample-cnn-eval-5000.uri \
    -d "$DATASET_PATH"

# Calibration set (optional)
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
    https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-dailymail-calibration.uri \
    -d "$DATASET_PATH"
```

---

## Run Performance (plain Python)

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

### Server Scenario (not tested in this harness)

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

> The upstream reference notes Server SUT was not tested for GPU runs; same applies here.

---

## Run Accuracy (plain Python)

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
python llama3.1-8b/evaluation.py \
    --mlperf-accuracy-file output/mlperf_log_accuracy.json \
    --dataset-file "$DATASET_PATH" \
    --dtype int32
```

### Upstream Accuracy Baselines (BF16, reference GPU run)

| Metric | Reference Value | MLPerf Target (99%) |
|---|---|---|
| ROUGE-1 | 38.7792 | 38.3914 |
| ROUGE-2 | 15.9075 | 15.7484 |
| ROUGE-L | 24.4957 | 24.2507 |
| RougeLsum | 35.7930 | 35.4551 |
| gen_len | 8,167,644 | 7,350,880 (90%) |

Our measured BF16 baseline: R1=38.80, R2=15.95, RL=24.52 -- within spec.

---

## Throughput Results

### Isolation Run (vLLM v0.17.1)
Source: `results/throughput/isolation_one_change/`

Each run changes a single parameter relative to the production baseline (BF16, BS=16, stock vLLM V1 defaults).
All speedups relative to **exp_00** (1082.82 tok/s).

| Experiment | Description | Tok/s | Speedup |
|---|---|---|---|
| exp_00 | PRODUCTION BASELINE: stock vLLM V1 defaults, BS=16 | 1082.82 | baseline |
| exp_00b | RIGHT-SIZED KV: max_model_len=2668 on production baseline | 1165.24 | +7.61% |
| exp_01 | -prefix_caching=False (disable V1 default APC) | 1048.73 | -3.15% |
| exp_02 | -async_output_proc OFF (disable V1 default async) | 1006.82 | -7.02% |
| exp_03 | compilation_config=O1 (below V1 default O2) | 1062.58 | -1.87% |
| exp_04 | compilation_config=O0 (no compile vs V1 default O2) | 857.43 | -20.82% |
| exp_05 | compilation_config=O3 (above V1 default O2) | 1098.13 | +1.41% |
| exp_06 | max_num_batched_tokens=2668 (neutralise chunked prefill) | 1063.47 | -1.79% |
| exp_07 | async_scheduling=True (experimental) | 1115.30 | +3.00% |
| exp_08 | max_num_batched_tokens=8192 | 1120.72 | +3.50% |
| exp_09 | max_num_batched_tokens=65536 | 1089.07 | +0.58% |
| exp_10 | quantization=fp8 weight only | 1456.36 | **+34.50%** |
| exp_11 | kv_cache_dtype=fp8 KV only | 1131.84 | +4.53% |
| exp_12 | fp8 weights + fp8 KV | 1277.73 | +18.00% |
| exp_13 | +batch_size=1024 | 2302.79 | **+112.67%** |
| exp_14 | +batch_size=2048 | 2379.46 | **+119.75%** |
| exp_15 | sort_by_input_length | 1115.30 | +3.00% |
| exp_16 | gpu_memory_utilization=0.95 | 1104.48 | +2.00% |
| exp_17 | attention_backend=FLASHINFER | 1081.93 | -0.08% |
| exp_18 | skip_tokenizer_init=True | 1083.77 | +0.09% |

- **Batch size** (exp_13, exp_14) is the single biggest lever: **+113% at BS=1024** and **+120% at BS=2048**.
- **FP8 weight quantization** (exp_10) gives **+34.5%** at BS=16. Adding fp8 KV on top (exp_12) drops this to +18% -- fp8 KV introduces overhead that partially cancels the weight quantization benefit.
- **Right-sizing KV** (exp_00b) is a free **+7.6%** -- always set max_model_len to the dataset maximum.
- **V1 defaults are net-positive**: disabling APC costs -3.15%, disabling async output costs -7%. Do not turn these off without a specific reason.
- **compilation_config=O0** is the worst single regression at **-20.8%**. Torch compilation is mandatory.
- **FLASHINFER and skip_tokenizer_init** had zero measurable impact.

---

### Progressive / Stacked Tuning
Source: `results/throughput/progression_full_stack/`

Stacked tuning sessions that build from the production baseline toward a best-performing configuration.
Each stack step adds one change on top of all previous ones.

All speedups relative to **stk_00_baseline** (1089.04 tok/s). Latencies in seconds.

| Stack Step | Changes Added | Tok/s | Speedup | p50 (s) | p99 (s) | Result |
|---|---|---|---|---|---|---|
| stk_00_baseline | Production baseline: BS=16, BF16, stock vLLM V1 defaults | 1089.04 | baseline | 783.92 | 1556.55 | VALID |
| stk_01_fp8 | +fp8 weight quantization | 1462.53 | +34.30% | 583.79 | 1158.98 | VALID |
| stk_03_kv2668 | +max_model_len=2668 (right-sized KV) | 1544.29 | +41.80% | 550.27 | 1097.43 | VALID |
| stk_04_sort | +sort_by_input_length | 1507.93 | +38.46% | 472.53 | 1117.14 | VALID |
| stk_05_gmu095 | +gpu_memory_utilization=0.95 | 1504.52 | +38.15% | 474.87 | 1119.70 | VALID |
| stk_06_kvcache_fp8 | +kv_cache_dtype=fp8 | 1500.22 | +37.76% | 477.08 | 1122.68 | VALID |
| stk_07_compile3 | +compilation_config=O3 | 1494.88 | +37.27% | 480.19 | 1126.89 | VALID |
| stk_08_expandable | +PYTORCH_CUDA_ALLOC_CONF=expandable_segments | 1420.80 | +30.46% | 481.81 | 1186.76 | VALID |
| stk_09_bitsandbytes | +bitsandbytes quantization (replaces fp8) | 446.14 | -59.03% | 1764.17 | 3787.78 | VALID |
| stk_02_bs1024 | +batch_size=1024 (standalone, not stacked) | 3321.76 | +205.02% | 191.41 | 509.22 | INVALID* |

> *stk_02_bs1024 is INVALID: min_duration not satisfied. Increase expected QPS so loadgen
> pre-generates a larger coalesced query set. Throughput is directionally correct but not certified.

- **FP8 weights + right-sized KV** (stk_01 + stk_03) is the sweet spot at **+41.8% throughput and -30% p99 latency** with just two changes.
- **Sort, GPU memory utilization, and fp8 KV** add marginal gains but show diminishing returns once stacked. Peak is at stk_03 (1544 tok/s); subsequent steps gradually erode it.
- **expandable_segments** (stk_08) drops to 1420 tok/s in the stacked context despite being positive in isolation -- likely conflicting with fp8 memory allocation patterns.
- **bitsandbytes** (stk_09) is strongly negative at -59%. Never substitute for fp8.
- **BS=1024** directionally confirms +205% but requires a corrected loadgen QPS run before certification.

---

### Quantization, APC & Chunked Prefill Deep-Dive
Source: `results/throughput/Quant_APC_chunked/`

Targeted isolation probing fp8 vs int8, chunked prefill MNBT settings, and APC cost/benefit.
All speedups relative to **baseline** (685.88 tok/s). Latencies in seconds.

| Experiment | Description | Tok/s | Speedup | p50 (s) | p99 (s) | Result |
|---|---|---|---|---|---|---|
| baseline | BF16, no quant, prefix ON, CP ON, MNBT=16384, BS=16 | 685.88 | baseline | 1.06 | 2.45 | VALID |
| quant_fp8_weights | fp8 W8A8 weights (float16 compute) | 1380.12 | **+101.22%** | 0.57 | 1.22 | VALID |
| quant_int8_weights | int8 W8A8 weights | -- | N/A | -- | -- | FAILED |
| cp_off | Chunked prefill OFF (MNBT=16384) | 587.32 | -14.37% | 1.05 | 2.45 | VALID |
| cp_on_mnbt_2668 | CP ON + MNBT=2668 (~V0 scheduling) | 659.35 | -3.87% | 1.09 | 2.56 | VALID |
| cp_on_mnbt_8192 | CP ON + MNBT=8192 (docs-recommended min) | 671.70 | -2.07% | 1.17 | 2.52 | VALID |
| prefix_off | Prefix caching OFF (APC disabled) | 975.21 | **+42.18%** | 0.68 | 1.72 | VALID |

- **fp8 W8A8** more than doubles throughput (+101%) and halves latency. Consistent with isolation results.
- **int8 W8A8** failed entirely (exit within 20s). Likely a missing kernel -- investigate before retrying.
- **APC is strongly negative on CNN/DailyMail** (+42% from disabling it) due to low prefix reuse. Workload-dependent -- beneficial for chat/RAG with long shared system prompts.
- **CP ON + MNBT=16384** is optimal. CP OFF costs -14%; lower MNBT values cost 2-4%.

---

## Accuracy Results

Reference baseline (acc_baseline, BF16 BS=16): **R1=38.80 | R2=15.95 | RL=24.52**
Source: `results/accuracy/`

| Experiment | ROUGE-1 | ROUGE-L | vs Baseline | Verdict |
|---|---|---|---|---|
| acc_baseline | 38.8001 | 24.5197 | baseline | Reference |
| acc_chunked_65536 | 38.8000 | 24.5189 | ~= baseline | SAFE |
| acc_compile_cfg_3 | 38.7993 | 24.5194 | ~= baseline | SAFE |
| acc_dtype_fp16 | 38.8381 | 24.5365 | +0.04 R1 | SAFE |
| acc_fp8_quant | N/A | N/A | -- | FAILED -- re-run needed |
| acc_prefix_cache_on | 38.8006 | 24.5196 | ~= baseline | SAFE |
| acc_sched_steps_4 | 38.8007 | 24.5227 | ~= baseline | SAFE |
| acc_sort_by_len | N/A | N/A | -- | FAILED -- re-run needed |
| acc_kv_len_1500 | 36.5577 | 23.2105 | **-2.24 R1** | UNSAFE |
| acc_kv_len_2668 | 38.7976 | 24.5183 | ~= baseline | SAFE |
| acc_batch_512 | 26.2134 | 16.5851 | **-12.59 R1** | UNSAFE |
| acc_batch_1024 | 25.5702 | 16.2307 | **-13.23 R1** | UNSAFE |
| combo_acc_fp8_kv2668 | 38.8376 | 24.5358 | ~= baseline | SAFE |
| combo_acc_full_optimal | N/A | N/A | -- | FAILED -- re-run needed |

---

## Open Issues & Blockers

| Issue | Priority | Action |
|---|---|---|
| BS=1024 ROUGE-1 regression (~13 pts) | CRITICAL | Investigate non-determinism at large batch sizes |
| acc_fp8_quant failed -- no output | CRITICAL | Re-run; fp8 is the primary throughput optimization |
| combo_acc_full_optimal failed | CRITICAL | Re-run; final accuracy answer for the production config |
| acc_sort_by_len failed -- no output | HIGH | Re-run; expected neutral but unconfirmed |
| int8 W8A8 kernel failure | MEDIUM | Check vLLM int8 support for LLaMA 3.1-8B on this GPU |
| stk_02_bs1024 INVALID (loadgen) | MEDIUM | Rerun with corrected expected QPS |


---

## Scenarios Tested

- **Offline:** tested -- tuned FP8 baseline and all sweeps.
- **Server:** reference command shown above; not tested in this harness.
- **Accuracy:** Offline scripts provided and run; Server accuracy not tested.

---

## References

- Official MLPerf reference + Docker/mlcr automation: https://github.com/mlcommons/inference/tree/master/language/llama3.1-8b
- See result subfolder READMEs for full tables, latency percentiles, and per-experiment details.
