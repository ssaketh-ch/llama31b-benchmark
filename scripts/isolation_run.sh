#!/bin/bash
# =============================================================================
# vLLM MLPerf Isolation Ablation — Llama-3.1-8B-Instruct on H200 MIG 71GB
# Offline Scenario — ONE change vs OG baseline, 4 runs per experiment
#
# DESIGN PRINCIPLE: every experiment changes EXACTLY ONE thing from the OG
# baseline. Nothing is stacked. Results show the isolated contribution of
# each knob in raw tok/s relative to the unchanged baseline.
#
# BASELINE (exp_00): exact reproduction of SUT_VLLM_OG.py behaviour
#   Engine:                  LLM (sync generate) — functionally equiv to OG
#   dtype:                   bfloat16
#   quantization:            None            ← NO quantization
#   tensor_parallel_size:    1               ← MIG slice has 1 GPU
#   batch_size:              16              ← OG default (--batch-size 16)
#   max_model_len:           131072          ← Llama 3.1 default (OG default)
#   gpu_memory_utilization:  0.90            ← vLLM default (OG never sets this)
#   enable_prefix_caching:   False           ← explicitly off
#   enable_chunked_prefill:  False           ← explicitly off
#   enforce_eager:           False           ← CUDA graphs (OG never disables)
#   cpu_offload_gb:          0
#   scheduler_delay_factor:  0.0
#   max_num_batched_tokens:  NOT SET         ← vLLM auto
#   async_scheduling:        NOT SET         ← vLLM default
#   num_scheduler_steps:     NOT SET         ← vLLM default (=1)
#   compilation_config:      NOT SET         ← no compile
#   VLLM_ATTENTION_BACKEND:  NOT SET         ← default FA3
#   PYTORCH_CUDA_ALLOC_CONF: NOT SET         ← OG never sets this
#   Sort by length:          NO              ← OG processes in loadgen order
#
# EXPERIMENTS (each changes exactly ONE thing from baseline):
#   exp_01:  +quantization="fp8"                   (FP8 weight quant)
#   exp_02:  +batch_size=512                        (larger batch)
#   exp_03:  +batch_size=1024                       (largest batch)
#   exp_04:  +max_model_len=2668                    (right-sized KV cache)
#   exp_05:  +sort_by_input_length                  (shortest-first batching)
#   exp_06:  +gpu_memory_utilization=0.95           (more HBM for KV cache)
#   exp_07:  +chunked_prefill=True                  (requires batched_tokens)
#            +max_num_batched_tokens=65536           (inseparable pair)
#   exp_08:  +PYTORCH_CUDA_ALLOC_CONF=expandable    (allocator stability)
#   exp_09:  enable_prefix_caching=False explicit   (confirm OG implicit OFF)
#   exp_10:  +async_scheduling=True                 [vLLM 0.10.0]
#   exp_11:  +num_scheduler_steps=2                 [vLLM 0.10.0]
#   exp_12:  +num_scheduler_steps=4                 [vLLM 0.10.0]
#   exp_13:  +compilation_config=3                  [vLLM 0.10.0]
#   exp_14:  +VLLM_ATTENTION_BACKEND=FLASHINFER     [vLLM 0.10.0]
#
# Each experiment is run RUNS_PER_EXP times (default 4) for reproducibility.
# A 30-second cooldown + GPU reset separates every individual run.
# The per-experiment mean, stdev, min, max are reported in the summary table.
#
# MLflow: each run is logged as a separate MLflow run within one experiment.
#
# Usage:
#   chmod +x run_isolation_ablation.sh
#   ./run_isolation_ablation.sh                  # all experiments
#   ./run_isolation_ablation.sh exp_00           # baseline only
#   ./run_isolation_ablation.sh exp_01           # one experiment (4 runs)
#   ./run_isolation_ablation.sh exp_07           # chunked prefill pair
#   RUNS_PER_EXP=2 ./run_isolation_ablation.sh   # quick pass (2 runs each)
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INFERENCE_DIR="${INFERENCE_DIR:-${REPO_ROOT}/llama3.1-8b}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${INFERENCE_DIR}/model}"
DATASET_PATH="${DATASET_PATH:-${INFERENCE_DIR}/dataset}"
GPU_COUNT="${GPU_COUNT:-1}"
TOTAL_SAMPLE_COUNT=13368
SCENARIO="Offline"
USER_CONF="${INFERENCE_DIR}/user.conf"
RUNS_PER_EXP="${RUNS_PER_EXP:-4}"

RESULTS_BASE="${REPO_ROOT}/results/throughput/isolation_one_change"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RESULTS_DIR="${RESULTS_BASE}/${TIMESTAMP}"

MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
MLFLOW_EXPERIMENT_NAME="llama3.1-8b_isolation-${TIMESTAMP}"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
log()     { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"; }
success() { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓${NC} $*"; }
warn()    { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠${NC} $*"; }
error()   { echo -e "${RED}[$(date '+%H:%M:%S')] ✗${NC} $*"; }
info()    { echo -e "${CYAN}[$(date '+%H:%M:%S')] ○${NC} $*"; }

# ---------------------------------------------------------------------------
# GPU utilities
# ---------------------------------------------------------------------------
cleanup_gpu() {
    log "Resetting GPU state (30s cooldown)..."
    set +e
    pkill -f "vllm"      2>/dev/null; sleep 2
    pkill -f "SUT_VLLM"  2>/dev/null; sleep 2
    pkill -f "main.py"   2>/dev/null; sleep 2
    nvidia-smi --gpu-reset 2>/dev/null
    sleep 10
    set -e
}

capture_system_info() {
    local run_dir="$1"
    mkdir -p "${run_dir}/system"
    set +e
    nvidia-smi > "${run_dir}/system/nvidia-smi.txt" 2>&1
    nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,\
utilization.gpu,utilization.memory,temperature.gpu,power.draw \
        --format=csv > "${run_dir}/system/gpu_stats.csv" 2>&1
    nvidia-smi -q > "${run_dir}/system/nvidia-smi-full.txt" 2>&1
    nvidia-smi mig -lgip > "${run_dir}/system/mig_topology.txt" 2>&1
    nvidia-smi mig -lgi  >> "${run_dir}/system/mig_topology.txt" 2>&1
    nvidia-smi mig -lci  >> "${run_dir}/system/mig_topology.txt" 2>&1
    lscpu > "${run_dir}/system/lscpu.txt" 2>&1
    cat /proc/meminfo > "${run_dir}/system/meminfo.txt" 2>&1
    python --version > "${run_dir}/system/packages.txt" 2>&1
    pip show vllm torch transformers >> "${run_dir}/system/packages.txt" 2>&1
    env | grep -E "CUDA|VLLM|PYTORCH|NCCL|GPU" | sort \
        > "${run_dir}/system/env_vars.txt" 2>&1
    set -e
}

classify_error() {
    local logfile="$1"
    set +e
    if   grep -q "out of memory\|CUBLAS_STATUS_ALLOC_FAILED\|CUDA out of memory" \
              "${logfile}" 2>/dev/null; then echo "OOM"
    elif grep -q "illegal memory access"   "${logfile}" 2>/dev/null; then echo "ILLEGAL_MEMORY_ACCESS"
    elif grep -q "Segmentation fault\|SIGSEGV" "${logfile}" 2>/dev/null; then echo "SEGFAULT"
    elif grep -q "EngineCore\|EngineDeadError" "${logfile}" 2>/dev/null; then echo "ENGINE_CORE_FATAL"
    elif grep -q "Timeout\|timed out"      "${logfile}" 2>/dev/null; then echo "TIMEOUT"
    elif grep -q "compilation\|inductor"   "${logfile}" 2>/dev/null; then echo "COMPILE_FAILED"
    elif grep -q "RuntimeError"            "${logfile}" 2>/dev/null; then echo "RUNTIME_ERROR"
    else echo "UNKNOWN"
    fi
    set -e
}

# ---------------------------------------------------------------------------
# Core benchmark runner — one individual run
# run_single EXP_ID RUN_IDX BATCH_SIZE DTYPE DESC MLFLOW_EXTRA_PARAMS
# ---------------------------------------------------------------------------
run_single() {
    local exp_id="$1"
    local run_idx="$2"
    local batch_size="$3"
    local dtype="$4"
    local desc="$5"
    local run_dir="${RESULTS_DIR}/${exp_id}/run_${run_idx}"

    mkdir -p "${run_dir}/system" "${run_dir}/mlperf_logs"

    info "  Run ${run_idx}/${RUNS_PER_EXP}: ${exp_id} (BS=${batch_size})"

    set +e
    nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu,power.draw \
        --format=csv,noheader > "${run_dir}/system/gpu_pre_run.txt" 2>&1
    set -e

    capture_system_info "${run_dir}"

    local exit_code=0
    local start_time; start_time=$(date +%s)
    cd "${INFERENCE_DIR}"

    set +e
    python -u - 2>&1 <<PYCODE | tee "${run_dir}/run.log"
import mlflow
import subprocess
import os
import sys
import time
sys.path.insert(0, '.')

# Import parse_summary_file from main if available, else define inline
try:
    from main import parse_summary_file
except Exception:
    import re
    def parse_summary_file(path):
        metrics = {}
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    m = re.match(r'^([\w\s/]+):\s*([\d.]+)', line)
                    if m:
                        key = m.group(1).strip().replace(' ', '_').lower()
                        try:
                            metrics[key] = float(m.group(2))
                        except ValueError:
                            metrics[key] = m.group(2)
                    if 'Tokens per second' in line:
                        try:
                            metrics['tokens_per_second'] = float(line.split(':')[-1].strip())
                        except Exception:
                            pass
                    if 'Samples per second' in line:
                        try:
                            metrics['samples_per_second'] = float(line.split(':')[-1].strip())
                        except Exception:
                            pass
        except Exception as e:
            metrics['parse_error'] = str(e)
        return metrics

run_name = '${exp_id}_run${run_idx}'
run_dir  = '${run_dir}'
exp_id   = '${exp_id}'
run_idx  = ${run_idx}

mlflow.set_tracking_uri('${MLFLOW_TRACKING_URI}')
mlflow.set_experiment('${MLFLOW_EXPERIMENT_NAME}')

start_time = time.time()
mlflow.start_run(run_name=run_name)

try:
    # ── Parameters ──────────────────────────────────────────────────────────
    mlflow.log_param('exp_id',              exp_id)
    mlflow.log_param('run_index',           run_idx)
    mlflow.log_param('batch_size',          ${batch_size})
    mlflow.log_param('dtype',               '${dtype}')
    mlflow.log_param('description',         '${desc}')
    mlflow.log_param('scenario',            '${SCENARIO}')
    mlflow.log_param('total_sample_count',  ${TOTAL_SAMPLE_COUNT})
    mlflow.log_param('gpu_count',           ${GPU_COUNT})
    mlflow.log_param('tensor_parallel_size',${GPU_COUNT})
    mlflow.log_param('model_path',          '${CHECKPOINT_PATH}')
    mlflow.log_param('dataset_path',        '${DATASET_PATH}')
    mlflow.log_param('runs_per_exp',        ${RUNS_PER_EXP})

    # ── Version tags ─────────────────────────────────────────────────────────
    def safe_ver(pkg):
        try:
            return __import__(pkg).__version__
        except Exception:
            return 'unavailable'
    mlflow.set_tag('vllm_version',          safe_ver('vllm'))
    mlflow.set_tag('torch_version',         safe_ver('torch'))
    mlflow.set_tag('transformers_version',  safe_ver('transformers'))

    # ── Hardware tags ─────────────────────────────────────────────────────────
    try:
        gpu_model   = subprocess.check_output(
            ['nvidia-smi','--query-gpu=name','--format=csv,noheader,nounits']
        ).decode().strip().split('\n')[0]
        cpu_lines   = subprocess.check_output(['lscpu']).decode().split('\n')
        cpu_model   = next((l.split(':')[1].strip() for l in cpu_lines
                            if 'Model name' in l), 'unknown')
        driver_line = subprocess.check_output(['nvidia-smi']).decode().split('\n')[2].strip()
        hostname    = subprocess.check_output(['hostname']).decode().strip()
        mlflow.set_tag('gpu_model',   gpu_model)
        mlflow.set_tag('cpu_model',   cpu_model)
        mlflow.set_tag('nvidia_driver', driver_line)
        mlflow.set_tag('host',        hostname)
    except Exception as e:
        mlflow.set_tag('hw_tag_error', str(e))

    mlflow.set_tag('exp_id',    exp_id)
    mlflow.set_tag('run_index', str(run_idx))
    mlflow.set_tag('dtype',     '${dtype}')
    mlflow.set_tag('scenario',  '${SCENARIO}')

    # ── Run benchmark ─────────────────────────────────────────────────────────
    cmd = [
        'python', 'main.py',
        '--scenario',            '${SCENARIO}',
        '--model-path',          '${CHECKPOINT_PATH}',
        '--batch-size',          '${batch_size}',
        '--dtype',               '${dtype}',
        '--user-conf',           '${USER_CONF}',
        '--total-sample-count',  '${TOTAL_SAMPLE_COUNT}',
        '--dataset-path',        '${DATASET_PATH}',
        '--output-log-dir',      f'{run_dir}/mlperf_logs',
        '--tensor-parallel-size','${GPU_COUNT}',
        '--vllm',
    ]
    proc = subprocess.run(cmd, check=False)
    ret_code = proc.returncode

    duration_sec = time.time() - start_time
    mlflow.log_metric('duration_sec', duration_sec)
    mlflow.log_param('exit_code', ret_code)

    # ── Parse and log metrics ─────────────────────────────────────────────────
    summary_path = f'{run_dir}/mlperf_logs/mlperf_log_summary.txt'
    if os.path.exists(summary_path):
        metrics = parse_summary_file(summary_path)
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
            else:
                mlflow.log_param(k, str(v))
    else:
        mlflow.set_tag('summary_missing', 'true')

    # ── Artifacts ─────────────────────────────────────────────────────────────
    # mlflow.log_artifacts(run_dir)  # Moved to after post-run captures

finally:
    # mlflow.end_run()  # Moved to after artifacts
    run_id = mlflow.active_run().info.run_id
    print(f"RUN_ID:{run_id}")
PYCODE
    exit_code="${PIPESTATUS[0]}"
    set -e

    local end_time; end_time=$(date +%s)
    local duration=$(( end_time - start_time ))

    set +e
    nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu,power.draw \
        --format=csv,noheader > "${run_dir}/system/gpu_post_run.txt" 2>&1
    cp "${INFERENCE_DIR}"/mlperf_log_* "${run_dir}/mlperf_logs/" 2>/dev/null
    set -e

    # Extract run_id from log
    run_id=$(grep "RUN_ID:" "${run_dir}/run.log" | awk -F: '{print $2}' | tr -d '\n')

    # Log artifacts and end run after post-run captures
    if [[ -n "${run_id}" ]]; then
    python -c "
import mlflow
mlflow.set_tracking_uri('${MLFLOW_TRACKING_URI}')
mlflow.set_experiment('${MLFLOW_EXPERIMENT_NAME}')
mlflow.start_run(run_id='${run_id}')
mlflow.log_artifacts('${run_dir}')
mlflow.end_run()
"
    else
        warn "MLflow run_id missing; skipping artifact upload"
    fi

    # Extract results
    local tokens samples valid
    set +e
    tokens=$(grep  "Tokens per second:"  "${run_dir}/run.log" 2>/dev/null | tail -1 | awk '{print $NF}')
    samples=$(grep "Samples per second:" "${run_dir}/run.log" 2>/dev/null | tail -1 | awk '{print $NF}')
    valid=$(grep   "Result is"           "${run_dir}/run.log" 2>/dev/null | tail -1 | awk '{print $NF}')
    set -e
    tokens="${tokens:-FAILED}"
    samples="${samples:-FAILED}"
    valid="${valid:-FAILED}"

    # Write per-run result file
    cat > "${run_dir}/result.txt" <<RESEOF
Experiment:       ${exp_id}
Run:              ${run_idx}/${RUNS_PER_EXP}
Description:      ${desc}
Batch Size:       ${batch_size}
Dtype:            ${dtype}
Tokens/sec:       ${tokens}
Samples/sec:      ${samples}
Valid:            ${valid}
Duration:         ${duration}s
Exit Code:        ${exit_code}
RESEOF

    if [[ "${tokens}" != "FAILED" ]]; then
        success "    run_${run_idx}: ${tokens} tok/s  [${valid}]  (${duration}s)"
    else
        local err_type; err_type=$(classify_error "${run_dir}/run.log")
        error   "    run_${run_idx}: FAILED (${err_type})  (${duration}s)"
        case "${err_type}" in
            OOM)                   warn "    → OOM: try reducing batch_size or gpu_memory_utilization" ;;
            SEGFAULT)              warn "    → Segfault: check PYTORCH_CUDA_ALLOC_CONF" ;;
            ENGINE_CORE_FATAL)     warn "    → Engine fatal: vLLM version incompatibility with this flag" ;;
            ILLEGAL_MEMORY_ACCESS) warn "    → ILLEGAL_MEM: max_num_batched_tokens may exceed engine limit" ;;
            COMPILE_FAILED)        warn "    → Compile failed: torch.compile incompatibility" ;;
        esac
    fi

    # Cooldown between runs
    log "    Cooldown 30s..."; sleep 30
    cleanup_gpu
}

# ---------------------------------------------------------------------------
# Per-experiment runner — calls run_single RUNS_PER_EXP times, then aggregates
# ---------------------------------------------------------------------------
run_experiment() {
    local exp_id="$1"
    local batch_size="$2"
    local dtype="$3"
    local sut_func="$4"
    local desc="$5"

    # Phase-level and single-experiment filtering
    local target="${TARGET:-all}"
    case "${target}" in
        all) ;;
        *)
            [[ "${target}" == "${exp_id}" ]] || return 0 ;;
    esac

    local exp_dir="${RESULTS_DIR}/${exp_id}"
    mkdir -p "${exp_dir}"

    log "========================================================"
    log "  ${exp_id}: ${desc}"
    log "  Batch=${batch_size} | Dtype=${dtype} | Runs=${RUNS_PER_EXP}"
    log "========================================================"

    # Write SUT to inference dir (same SUT for all runs of this experiment)
    ${sut_func} > "${exp_dir}/SUT_VLLM.py"
    cp "${exp_dir}/SUT_VLLM.py" "${INFERENCE_DIR}/SUT_VLLM.py"

    # Persist metadata for this experiment
    cat > "${exp_dir}/experiment_meta.txt" <<METAEOF
exp_id:       ${exp_id}
description:  ${desc}
batch_size:   ${batch_size}
dtype:        ${dtype}
runs:         ${RUNS_PER_EXP}
timestamp:    $(date -u +%Y-%m-%dT%H:%M:%SZ)
sut_func:     ${sut_func}
METAEOF

    # Run RUNS_PER_EXP times
    for run_idx in $(seq 1 "${RUNS_PER_EXP}"); do
        run_single "${exp_id}" "${run_idx}" "${batch_size}" "${dtype}" "${desc}"
    done

    # Aggregate results across runs
    aggregate_experiment "${exp_id}" "${desc}"
}

# ---------------------------------------------------------------------------
# Aggregation — compute mean/stdev/min/max for an experiment
# ---------------------------------------------------------------------------
aggregate_experiment() {
    local exp_id="$1"
    local desc="$2"
    local exp_dir="${RESULTS_DIR}/${exp_id}"

    python3 - <<PYAGG
import os, statistics, glob

exp_dir = '${exp_dir}'
exp_id  = '${exp_id}'
desc    = '${desc}'

tok_vals   = []
valid_list = []

for run_idx in range(1, ${RUNS_PER_EXP} + 1):
    rfile = os.path.join(exp_dir, f'run_{run_idx}', 'result.txt')
    if not os.path.exists(rfile):
        continue
    tok_str = valid_str = ''
    with open(rfile) as f:
        for line in f:
            if line.startswith('Tokens/sec:'):
                tok_str = line.split(':', 1)[1].strip()
            if line.startswith('Valid:'):
                valid_str = line.split(':', 1)[1].strip()
    if tok_str and tok_str != 'FAILED':
        try:
            tok_vals.append(float(tok_str))
            valid_list.append(valid_str)
        except ValueError:
            pass

# Load baseline mean if available
baseline_mean = None
baseline_agg = os.path.join('${RESULTS_DIR}', 'exp_00', 'aggregate.txt')
if os.path.exists(baseline_agg):
    with open(baseline_agg) as f:
        for line in f:
            if line.lower().startswith('mean tok/s'):
                try:
                    baseline_mean = float(line.split(':',1)[1].strip())
                except Exception:
                    baseline_mean = None

if tok_vals:
    mean_tok   = statistics.mean(tok_vals)
    stdev_tok  = statistics.stdev(tok_vals) if len(tok_vals) > 1 else 0.0
    min_tok    = min(tok_vals)
    max_tok    = max(tok_vals)
    cv_pct     = (stdev_tok / mean_tok * 100) if mean_tok else 0.0
    valid_rate = sum(1 for v in valid_list if 'VALID' in v.upper() and 'INVALID' not in v.upper())
    speedup = None
    if baseline_mean and mean_tok:
        speedup = (mean_tok / baseline_mean - 1.0) * 100
    agg = [
        f"Experiment:    {exp_id}",
        f"Description:   {desc}",
        f"Runs:          {len(tok_vals)}/{${RUNS_PER_EXP}} succeeded",
        f"Mean tok/s:    {mean_tok:.2f}",
        f"Stdev tok/s:   {stdev_tok:.2f}",
        f"CV%:           {cv_pct:.1f}%",
        f"Min tok/s:     {min_tok:.2f}",
        f"Max tok/s:     {max_tok:.2f}",
        f"All values:    {[f'{v:.2f}' for v in tok_vals]}",
        f"Valid rate:    {valid_rate}/{len(valid_list)}",
    ]
    if speedup is not None and exp_id != 'exp_00':
        agg.append(f"Speedup vs exp_00: {speedup:+.2f}%")
else:
    agg = [
        f"Experiment:    {exp_id}",
        f"Description:   {desc}",
        f"Runs:          0/{${RUNS_PER_EXP}} succeeded",
        f"Mean tok/s:    FAILED",
    ]

agg_text = "\n".join(agg) + "\n"
agg_path = os.path.join(exp_dir, 'aggregate.txt')
with open(agg_path, 'w') as f:
    f.write(agg_text)
print(agg_text)
PYAGG
}

# =============================================================================
# SUT GENERATOR FUNCTIONS
#
# Every SUT below reproduces the OG baseline faithfully in everything EXCEPT
# the one flag under test.
#
# BASELINE FLAGS pinned in every SUT unless the experiment changes that flag:
#   dtype="bfloat16"
#   quantization=None                      ← OG has no quantization
#   tensor_parallel_size=1                 ← MIG has 1 GPU
#   gpu_memory_utilization=0.90            ← vLLM default, matches OG
#   max_model_len=131072                   ← Llama 3.1 default, matches OG
#   enable_prefix_caching=False            ← explicitly off
#   enable_chunked_prefill=False           ← explicitly off
#   enforce_eager=False                    ← CUDA graphs enabled
#   cpu_offload_gb=0
#   scheduler_delay_factor=0.0
#   max_num_batched_tokens: NOT SET        ← vLLM auto (matches OG)
#   async_scheduling: NOT SET              ← vLLM default
#   num_scheduler_steps: NOT SET           ← vLLM default (1)
#   compilation_config: NOT SET            ← no compile
#   VLLM_ATTENTION_BACKEND: NOT SET        ← default FA3
#   PYTORCH_CUDA_ALLOC_CONF: NOT SET       ← OG never sets this
#   Sort by length: NO                     ← OG processes in loadgen order
#   engine: LLM (sync) — structurally same as OG, avoids async overhead
# =============================================================================

# ── exp_00: OG BASELINE ───────────────────────────────────────────────────────
sut_exp00() {
cat << 'SUTEOF'
# EXP_00: OG BASELINE
# Exact reproduction of SUT_VLLM_OG.py with sync LLM engine.
# AsyncLLMEngine replaced with LLM (sync) for structural consistency across
# all experiments. All other parameters reproduce OG defaults exactly.
# NO quantization. NO sorting. NO special flags.
# This is the reference against which every experiment is measured.
import os
import array
import threading
import queue
import logging
import torch
from vllm import LLM, SamplingParams
import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-SUT")


class SUT:
    def __init__(self, model_path=None, dtype="bfloat16", batch_size=None,
                 total_sample_count=13368, dataset_path=None,
                 use_cached_outputs=False, workers=1, tensor_parallel_size=1):
        self.model_path = model_path or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.batch_size = batch_size if batch_size else 16   # OG default
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        # OG has no _set_hardware_optimizations — no env vars set here
        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="bfloat16",                  # OG dtype
            quantization=None,                 # OG: NO quantization
            tensor_parallel_size=1,            # MIG: 1 GPU
            gpu_memory_utilization=0.90,       # vLLM default (OG never sets this)
            max_model_len=131072,              # Llama 3.1 default (OG never sets this)
            enable_prefix_caching=False,       # explicitly off
            enable_chunked_prefill=False,      # explicitly off
            enforce_eager=False,               # CUDA graphs
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
            # max_num_batched_tokens: NOT SET  (vLLM auto)
            # async_scheduling:       NOT SET  (vLLM default)
            # num_scheduler_steps:    NOT SET  (vLLM default = 1)
            # compilation_config:     NOT SET  (no compile)
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start()
            self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        for w in self.worker_threads:
            w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [
                self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [
                list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(
                        qitem[i].id, bi[0], bi[1], n_tokens)])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
        # OG: NO sorting — process in loadgen-supplied order
        log.info(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[:self.batch_size])
            query_samples = query_samples[self.batch_size:]
        log.info("IssueQuery done")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self): return self.qsl
    def predict(self, **kwargs): raise NotImplementedError
    def flush_queries(self): pass
    def __del__(self): pass


class SUTServer(SUT): pass
SUTEOF
}

# =============================================================================
# Shared baseline for all experiments — copy-paste anchor.
# Every experiment below is a complete SUT with exactly ONE line changed.
# The changed line is marked: # ← CHANGE (from baseline: <old value>)
# All other pinned flags are explicitly annotated with their baseline value.
# =============================================================================

# ── exp_01: +quantization="fp8" ──────────────────────────────────────────────
sut_exp01() {
cat << 'SUTEOF'
# EXP_01: +quantization="fp8"
# ONLY CHANGE from baseline: quantization=None -> "fp8"
# FP8 compresses 8B weights from ~16GB (BF16) to ~3.5GB.
# At BS=16 (memory-bandwidth bound), halving the weight footprint
# should give roughly proportional throughput gain.
# Everything else is identical to exp_00.
import os
import array
import threading
import queue
import logging
import torch
from vllm import LLM, SamplingParams
import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-SUT")


class SUT:
    def __init__(self, model_path=None, dtype="bfloat16", batch_size=None,
                 total_sample_count=13368, dataset_path=None,
                 use_cached_outputs=False, workers=1, tensor_parallel_size=1):
        self.model_path = model_path or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.batch_size = batch_size if batch_size else 16
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="float16",                   # fp8 quant forces float16
            quantization="fp8",                # ← CHANGE (from baseline: None)
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,       # baseline value
            max_model_len=131072,              # baseline value
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start()
            self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        for w in self.worker_threads:
            w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [
                self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [
                list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(
                        qitem[i].id, bi[0], bi[1], n_tokens)])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
        # NO sorting — baseline behaviour
        log.info(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[:self.batch_size])
            query_samples = query_samples[self.batch_size:]
        log.info("IssueQuery done")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self): return self.qsl
    def predict(self, **kwargs): raise NotImplementedError
    def flush_queries(self): pass
    def __del__(self): pass


class SUTServer(SUT): pass
SUTEOF
}

# ── exp_02: batch_size=512 ────────────────────────────────────────────────────
# NOTE: SUT is identical to exp_00. Only the CLI --batch-size argument changes.
# This is documented explicitly: the batch_size comes from the CLI arg, not
# the SUT constructor default. The SUT below sets default=512 to ensure
# it is used even if CLI parsing differs.
sut_exp02() {
cat << 'SUTEOF'
# EXP_02: batch_size=512
# ONLY CHANGE from baseline: batch_size 16 -> 512
# SUT is otherwise identical to exp_00.
# CLI --batch-size 512 also passed. Both paths agree.
# At BS=512 the GPU transitions from memory-bound to compute-bound.
# NO quantization. NO sorting. Everything else baseline.
import os
import array
import threading
import queue
import logging
import torch
from vllm import LLM, SamplingParams
import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-SUT")


class SUT:
    def __init__(self, model_path=None, dtype="bfloat16", batch_size=None,
                 total_sample_count=13368, dataset_path=None,
                 use_cached_outputs=False, workers=1, tensor_parallel_size=1):
        self.model_path = model_path or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.batch_size = batch_size if batch_size else 512  # ← CHANGE (baseline: 16)
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="bfloat16",                  # baseline value
            quantization=None,                 # baseline value: NO quant
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,       # baseline value
            max_model_len=131072,              # baseline value
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start()
            self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        for w in self.worker_threads:
            w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [
                self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [
                list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(
                        qitem[i].id, bi[0], bi[1], n_tokens)])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
        # NO sorting — baseline behaviour
        log.info(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[:self.batch_size])
            query_samples = query_samples[self.batch_size:]
        log.info("IssueQuery done")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self): return self.qsl
    def predict(self, **kwargs): raise NotImplementedError
    def flush_queries(self): pass
    def __del__(self): pass


class SUTServer(SUT): pass
SUTEOF
}

# ── exp_03: batch_size=1024 ───────────────────────────────────────────────────
sut_exp03() {
cat << 'SUTEOF'
# EXP_03: batch_size=1024
# ONLY CHANGE from baseline: batch_size 16 -> 1024
# SUT is otherwise identical to exp_00.
# Larger batch than exp_02 — tests whether 2x more batching helps vs BS=512.
# At BS=1024: 13368/1024 = ~13 batches. KV cache at 131072 slots may OOM.
# NO quantization. NO sorting. Everything else baseline.
import os
import array
import threading
import queue
import logging
import torch
from vllm import LLM, SamplingParams
import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-SUT")


class SUT:
    def __init__(self, model_path=None, dtype="bfloat16", batch_size=None,
                 total_sample_count=13368, dataset_path=None,
                 use_cached_outputs=False, workers=1, tensor_parallel_size=1):
        self.model_path = model_path or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.batch_size = batch_size if batch_size else 1024  # ← CHANGE (baseline: 16)
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="bfloat16",                  # baseline value
            quantization=None,                 # baseline value: NO quant
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,       # baseline value
            max_model_len=131072,              # baseline value — may OOM at BS=1024
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start()
            self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        for w in self.worker_threads:
            w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [
                self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [
                list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(
                        qitem[i].id, bi[0], bi[1], n_tokens)])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
        log.info(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[:self.batch_size])
            query_samples = query_samples[self.batch_size:]
        log.info("IssueQuery done")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self): return self.qsl
    def predict(self, **kwargs): raise NotImplementedError
    def flush_queries(self): pass
    def __del__(self): pass


class SUTServer(SUT): pass
SUTEOF
}

# ── exp_04: max_model_len=2668 ────────────────────────────────────────────────
sut_exp04() {
cat << 'SUTEOF'
# EXP_04: max_model_len=2668
# ONLY CHANGE from baseline: max_model_len 131072 -> 2668
# 2668 = dataset max input (2540) + max output (128).
# This reduces KV cache slot from 8192MB to 166MB per sequence,
# enabling 316 concurrent sequences instead of 6 at 0.90 utilization.
# Still BS=16, NO quant, NO sort — isolates the KV cache density effect.
import os
import array
import threading
import queue
import logging
import torch
from vllm import LLM, SamplingParams
import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-SUT")


class SUT:
    def __init__(self, model_path=None, dtype="bfloat16", batch_size=None,
                 total_sample_count=13368, dataset_path=None,
                 use_cached_outputs=False, workers=1, tensor_parallel_size=1):
        self.model_path = model_path or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.batch_size = batch_size if batch_size else 16   # baseline value
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="bfloat16",                  # baseline value
            quantization=None,                 # baseline value: NO quant
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,       # baseline value
            max_model_len=2668,                # ← CHANGE (from baseline: 131072)
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start()
            self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        for w in self.worker_threads:
            w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [
                self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [
                list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(
                        qitem[i].id, bi[0], bi[1], n_tokens)])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
        # NO sorting — baseline behaviour
        log.info(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[:self.batch_size])
            query_samples = query_samples[self.batch_size:]
        log.info("IssueQuery done")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self): return self.qsl
    def predict(self, **kwargs): raise NotImplementedError
    def flush_queries(self): pass
    def __del__(self): pass


class SUTServer(SUT): pass
SUTEOF
}

# ── exp_05: sort_by_input_length ──────────────────────────────────────────────
sut_exp05() {
cat << 'SUTEOF'
# EXP_05: sort_by_input_length
# ONLY CHANGE from baseline: issue_queries() now sorts by input length
# (shortest first) before batching. The LLM() constructor is unchanged.
# At BS=16 the gain will be modest — variance in a 16-sample batch is lower
# than in a 512-sample batch. But this isolates the pure sorting effect.
# NO quantization. NO KV cache resize. Everything else baseline.
import os
import array
import threading
import queue
import logging
import torch
from vllm import LLM, SamplingParams
import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-SUT")


class SUT:
    def __init__(self, model_path=None, dtype="bfloat16", batch_size=None,
                 total_sample_count=13368, dataset_path=None,
                 use_cached_outputs=False, workers=1, tensor_parallel_size=1):
        self.model_path = model_path or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.batch_size = batch_size if batch_size else 16
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="bfloat16",
            quantization=None,                 # baseline: NO quant
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,       # baseline value
            max_model_len=131072,              # baseline value
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start()
            self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        for w in self.worker_threads:
            w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [
                self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [
                list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(
                        qitem[i].id, bi[0], bi[1], n_tokens)])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
        # ← CHANGE: sort by input length (shortest first) before batching
        log.info(f"IssueQuery started with {len(query_samples)} samples")
        query_samples = sorted(
            query_samples,
            key=lambda q: len(self.data_object.input_ids[q.index])
        )
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[:self.batch_size])
            query_samples = query_samples[self.batch_size:]
        log.info("IssueQuery done")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self): return self.qsl
    def predict(self, **kwargs): raise NotImplementedError
    def flush_queries(self): pass
    def __del__(self): pass


class SUTServer(SUT): pass
SUTEOF
}

# ── exp_06: gpu_memory_utilization=0.95 ──────────────────────────────────────
sut_exp06() {
cat << 'SUTEOF'
# EXP_06: gpu_memory_utilization=0.95
# ONLY CHANGE from baseline: gpu_memory_utilization 0.90 -> 0.95
# Allocates ~5% more HBM3e for KV cache (~3.5GB on 71GB MIG slice).
# At BS=16 this is unlikely to matter (KV demand is tiny), but isolates
# whether even the allocator hint change has any effect.
# NO quantization. NO sorting. Everything else baseline.
import os
import array
import threading
import queue
import logging
import torch
from vllm import LLM, SamplingParams
import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-SUT")


class SUT:
    def __init__(self, model_path=None, dtype="bfloat16", batch_size=None,
                 total_sample_count=13368, dataset_path=None,
                 use_cached_outputs=False, workers=1, tensor_parallel_size=1):
        self.model_path = model_path or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.batch_size = batch_size if batch_size else 16
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="bfloat16",
            quantization=None,                 # baseline: NO quant
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,       # ← CHANGE (from baseline: 0.90)
            max_model_len=131072,              # baseline value
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start()
            self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        for w in self.worker_threads:
            w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [
                self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [
                list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(
                        qitem[i].id, bi[0], bi[1], n_tokens)])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
        log.info(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[:self.batch_size])
            query_samples = query_samples[self.batch_size:]
        log.info("IssueQuery done")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self): return self.qsl
    def predict(self, **kwargs): raise NotImplementedError
    def flush_queries(self): pass
    def __del__(self): pass


class SUTServer(SUT): pass
SUTEOF
}

# ── exp_07: chunked_prefill=True + max_num_batched_tokens=65536 ──────────────
sut_exp07() {
cat << 'SUTEOF'
# EXP_07: chunked_prefill=True + max_num_batched_tokens=65536
# TWO FLAGS CHANGED — justified: these are operationally inseparable.
# chunked_prefill without a token budget is undefined in vLLM V1.
# max_num_batched_tokens without chunked_prefill has no chunking effect.
# They must be tested as a pair.
# NOTE: gpu_memory_utilization dropped 0.90 -> 0.95 to accommodate activation
# memory spike from 65536-token prefill batches.
# NO quantization. NO sorting. Everything else baseline.
import os
import array
import threading
import queue
import logging
import torch
from vllm import LLM, SamplingParams
import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-SUT")


class SUT:
    def __init__(self, model_path=None, dtype="bfloat16", batch_size=None,
                 total_sample_count=13368, dataset_path=None,
                 use_cached_outputs=False, workers=1, tensor_parallel_size=1):
        self.model_path = model_path or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.batch_size = batch_size if batch_size else 16
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="bfloat16",
            quantization=None,                 # baseline: NO quant
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,       # raised from 0.90 for activation headroom
            max_model_len=131072,              # baseline value
            max_num_batched_tokens=65536,      # ← CHANGE (from baseline: not set)
            enable_prefix_caching=False,
            enable_chunked_prefill=True,       # ← CHANGE (from baseline: False)
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start()
            self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        for w in self.worker_threads:
            w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [
                self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [
                list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(
                        qitem[i].id, bi[0], bi[1], n_tokens)])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
        log.info(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[:self.batch_size])
            query_samples = query_samples[self.batch_size:]
        log.info("IssueQuery done")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self): return self.qsl
    def predict(self, **kwargs): raise NotImplementedError
    def flush_queries(self): pass
    def __del__(self): pass


class SUTServer(SUT): pass
SUTEOF
}

# ── exp_08: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ─────────────────
sut_exp08() {
cat << 'SUTEOF'
# EXP_08: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ONLY CHANGE from baseline: set expandable_segments allocator env var.
# This prevents fragmentation-induced segfaults in long runs.
# At BS=16 with few batches, this is unlikely to change throughput.
# Tests whether the allocator hint itself has any cost or benefit.
# NO quantization. NO sorting. Everything else baseline.
import os
import array
import threading
import queue
import logging
import torch
from vllm import LLM, SamplingParams
import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-SUT")

# ← CHANGE: set before any CUDA allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class SUT:
    def __init__(self, model_path=None, dtype="bfloat16", batch_size=None,
                 total_sample_count=13368, dataset_path=None,
                 use_cached_outputs=False, workers=1, tensor_parallel_size=1):
        self.model_path = model_path or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.batch_size = batch_size if batch_size else 16
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="bfloat16",
            quantization=None,                 # baseline: NO quant
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,       # baseline value
            max_model_len=131072,              # baseline value
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start()
            self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        for w in self.worker_threads:
            w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [
                self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [
                list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(
                        qitem[i].id, bi[0], bi[1], n_tokens)])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
        log.info(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[:self.batch_size])
            query_samples = query_samples[self.batch_size:]
        log.info("IssueQuery done")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self): return self.qsl
    def predict(self, **kwargs): raise NotImplementedError
    def flush_queries(self): pass
    def __del__(self): pass


class SUTServer(SUT): pass
SUTEOF
}

# ── exp_09: enable_prefix_caching=False explicit ────────────────────────────
sut_exp09() {
cat << 'SUTEOF'
# EXP_09: enable_prefix_caching=False (explicitly confirmed off)
# NO CHANGE IN BEHAVIOUR — this is a sanity / control experiment.
# vLLM 0.10.0 may default prefix_caching=True. The baseline already sets it
# to False, but this run exists to confirm the baseline is correct and that
# the explicit=False overhead is zero.
# Expected result: within noise of exp_00. If different, the baseline
# was NOT correctly setting prefix_caching=False.
import os
import array
import threading
import queue
import logging
import torch
from vllm import LLM, SamplingParams
import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-SUT")


class SUT:
    def __init__(self, model_path=None, dtype="bfloat16", batch_size=None,
                 total_sample_count=13368, dataset_path=None,
                 use_cached_outputs=False, workers=1, tensor_parallel_size=1):
        self.model_path = model_path or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.batch_size = batch_size if batch_size else 16
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="bfloat16",
            quantization=None,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=131072,
            enable_prefix_caching=False,       # explicitly confirmed off
            enable_chunked_prefill=False,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start()
            self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        for w in self.worker_threads:
            w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [
                self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [
                list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(
                        qitem[i].id, bi[0], bi[1], n_tokens)])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
        log.info(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[:self.batch_size])
            query_samples = query_samples[self.batch_size:]
        log.info("IssueQuery done")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self): return self.qsl
    def predict(self, **kwargs): raise NotImplementedError
    def flush_queries(self): pass
    def __del__(self): pass


class SUTServer(SUT): pass
SUTEOF
}

# ── exp_10: async_scheduling=True ─────────────────────────────────────────────
sut_exp10() {
cat << 'SUTEOF'
# EXP_10: async_scheduling=True [vLLM 0.10.0]
# ONLY CHANGE from baseline: async_scheduling not set -> True
# Overlaps Python scheduler with GPU execution. At BS=16 (~836 batches),
# there is meaningful scheduler work between batches, unlike BS=1024 (~13 batches).
# Previous result at BS=1024 was -3.4%, but that had stacked configs.
# This isolates the scheduler overhead at OG batch size.
import os
import array
import threading
import queue
import logging
import torch
from vllm import LLM, SamplingParams
import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-SUT")


class SUT:
    def __init__(self, model_path=None, dtype="bfloat16", batch_size=None,
                 total_sample_count=13368, dataset_path=None,
                 use_cached_outputs=False, workers=1, tensor_parallel_size=1):
        self.model_path = model_path or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.batch_size = batch_size if batch_size else 16
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="bfloat16",
            quantization=None,                 # baseline: NO quant
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,       # baseline value
            max_model_len=131072,              # baseline value
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
            async_scheduling=True,             # ← CHANGE (from baseline: not set)
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start()
            self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        for w in self.worker_threads:
            w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [
                self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [
                list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(
                        qitem[i].id, bi[0], bi[1], n_tokens)])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
        log.info(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[:self.batch_size])
            query_samples = query_samples[self.batch_size:]
        log.info("IssueQuery done")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self): return self.qsl
    def predict(self, **kwargs): raise NotImplementedError
    def flush_queries(self): pass
    def __del__(self): pass


class SUTServer(SUT): pass
SUTEOF
}

# ── exp_11: num_scheduler_steps=2 ────────────────────────────────────────────
sut_exp11() {
cat << 'SUTEOF'
# EXP_11: num_scheduler_steps=2 [vLLM 0.10.0]
# ONLY CHANGE from baseline: num_scheduler_steps not set -> 2
# Runs 2 decode steps per scheduler invocation. At BS=16, there are 836 batches
# and many decode steps per batch (up to 128). The GIL-release frequency is halved.
# At steps=4 (stacked run) we saw -11.9%. Steps=2 is the conservative probe.
import os
import array
import threading
import queue
import logging
import torch
from vllm import LLM, SamplingParams
import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-SUT")


class SUT:
    def __init__(self, model_path=None, dtype="bfloat16", batch_size=None,
                 total_sample_count=13368, dataset_path=None,
                 use_cached_outputs=False, workers=1, tensor_parallel_size=1):
        self.model_path = model_path or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.batch_size = batch_size if batch_size else 16
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="bfloat16",
            quantization=None,                 # baseline: NO quant
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,       # baseline value
            max_model_len=131072,              # baseline value
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
            num_scheduler_steps=2,             # ← CHANGE (from baseline: not set)
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start()
            self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        for w in self.worker_threads:
            w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [
                self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [
                list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(
                        qitem[i].id, bi[0], bi[1], n_tokens)])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
        log.info(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[:self.batch_size])
            query_samples = query_samples[self.batch_size:]
        log.info("IssueQuery done")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self): return self.qsl
    def predict(self, **kwargs): raise NotImplementedError
    def flush_queries(self): pass
    def __del__(self): pass


class SUTServer(SUT): pass
SUTEOF
}

# ── exp_12: num_scheduler_steps=4 ────────────────────────────────────────────
sut_exp12() {
cat << 'SUTEOF'
# EXP_12: num_scheduler_steps=4 [vLLM 0.10.0]
# ONLY CHANGE from baseline: num_scheduler_steps not set -> 4
# Previous stacked result was -11.9%. This is a clean isolated measurement.
# At BS=16 with many short batches, the wasted-step penalty may be smaller
# than at BS=1024 (fewer sequences finishing mid-stride per batch).
import os
import array
import threading
import queue
import logging
import torch
from vllm import LLM, SamplingParams
import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-SUT")


class SUT:
    def __init__(self, model_path=None, dtype="bfloat16", batch_size=None,
                 total_sample_count=13368, dataset_path=None,
                 use_cached_outputs=False, workers=1, tensor_parallel_size=1):
        self.model_path = model_path or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.batch_size = batch_size if batch_size else 16
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="bfloat16",
            quantization=None,                 # baseline: NO quant
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,       # baseline value
            max_model_len=131072,              # baseline value
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
            num_scheduler_steps=4,             # ← CHANGE (from baseline: not set)
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start()
            self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        for w in self.worker_threads:
            w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [
                self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [
                list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(
                        qitem[i].id, bi[0], bi[1], n_tokens)])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
        log.info(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[:self.batch_size])
            query_samples = query_samples[self.batch_size:]
        log.info("IssueQuery done")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self): return self.qsl
    def predict(self, **kwargs): raise NotImplementedError
    def flush_queries(self): pass
    def __del__(self): pass


class SUTServer(SUT): pass
SUTEOF
}

# ── exp_13: compilation_config=3 ─────────────────────────────────────────────
sut_exp13() {
cat << 'SUTEOF'
# EXP_13: compilation_config=3 [vLLM 0.10.0]
# ONLY CHANGE from baseline: compilation_config not set -> 3
# torch.compile level 3. First batch incurs 2-5min compile overhead.
# At BS=16 there are 836 batches — the compile cost is more likely to be
# amortised than at BS=1024 (13 batches). Interesting comparison point.
# Previous stacked result at BS=1024 was -2.5%. This isolates it at BS=16.
import os
import array
import threading
import queue
import logging
import torch
from vllm import LLM, SamplingParams
import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-SUT")


class SUT:
    def __init__(self, model_path=None, dtype="bfloat16", batch_size=None,
                 total_sample_count=13368, dataset_path=None,
                 use_cached_outputs=False, workers=1, tensor_parallel_size=1):
        self.model_path = model_path or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.batch_size = batch_size if batch_size else 16
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="bfloat16",
            quantization=None,                 # baseline: NO quant
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,       # baseline value
            max_model_len=131072,              # baseline value
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
            compilation_config=3,              # ← CHANGE (from baseline: not set)
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start()
            self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        for w in self.worker_threads:
            w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [
                self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [
                list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(
                        qitem[i].id, bi[0], bi[1], n_tokens)])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
        log.info(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[:self.batch_size])
            query_samples = query_samples[self.batch_size:]
        log.info("IssueQuery done")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self): return self.qsl
    def predict(self, **kwargs): raise NotImplementedError
    def flush_queries(self): pass
    def __del__(self): pass


class SUTServer(SUT): pass
SUTEOF
}

# ── exp_14: VLLM_ATTENTION_BACKEND=FLASHINFER ────────────────────────────────
sut_exp14() {
cat << 'SUTEOF'
# EXP_14: VLLM_ATTENTION_BACKEND=FLASHINFER [vLLM 0.10.0]
# ONLY CHANGE from baseline: VLLM_ATTENTION_BACKEND not set -> FLASHINFER
# Set via env var before LLM() is called so vLLM sees it at init time.
# Previous result (stacked): OOM at BS=1024, -12% at BS=512.
# This run is at BS=16, which should avoid OOM. Isolates kernel performance.
# All LLM() params identical to baseline.
import os
import array
import threading
import queue
import logging
import torch
from vllm import LLM, SamplingParams
import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-SUT")

# ← CHANGE: set before LLM() init
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"


class SUT:
    def __init__(self, model_path=None, dtype="bfloat16", batch_size=None,
                 total_sample_count=13368, dataset_path=None,
                 use_cached_outputs=False, workers=1, tensor_parallel_size=1):
        self.model_path = model_path or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.batch_size = batch_size if batch_size else 16
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="bfloat16",
            quantization=None,                 # baseline: NO quant
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,       # baseline value
            max_model_len=131072,              # baseline value
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
            # No LLM() changes — the backend is controlled by env var only
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start()
            self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        for w in self.worker_threads:
            w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [
                self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [
                list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(
                        qitem[i].id, bi[0], bi[1], n_tokens)])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
        log.info(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[:self.batch_size])
            query_samples = query_samples[self.batch_size:]
        log.info("IssueQuery done")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self): return self.qsl
    def predict(self, **kwargs): raise NotImplementedError
    def flush_queries(self): pass
    def __del__(self): pass


class SUTServer(SUT): pass
SUTEOF
}

# =============================================================================
# FINAL SUMMARY — aggregates all experiments into one table with stats
# =============================================================================
print_final_summary() {
    log "========================================================"
    log "ISOLATION ABLATION — FINAL SUMMARY"
    log "Baseline: OG SUT (BF16, BS=16, no quant, no sort)"
    log "Each experiment: 4 independent runs vs baseline"
    log "========================================================"

    # Collect baseline mean
    local baseline_mean="N/A"
    local baseline_agg="${RESULTS_DIR}/exp_00/aggregate.txt"
    if [[ -f "${baseline_agg}" ]]; then
        baseline_mean=$(grep "^Mean tok/s:" "${baseline_agg}" \
            | awk '{print $NF}' | grep -v FAILED || echo "N/A")
    fi

    printf "\n%-10s %-40s %-12s %-10s %-8s %-8s %-8s %-8s %-6s\n" \
        "Exp" "Description" "Mean tok/s" "vs Base" "Stdev" "CV%" "Min" "Max" "Valid"
    printf "%-10s %-40s %-12s %-10s %-8s %-8s %-8s %-8s %-6s\n" \
        "---" "-----------" "----------" "--------" "-----" "---" "---" "---" "-----"

    for agg_file in "${RESULTS_DIR}"/exp_*/aggregate.txt; do
        [[ -f "${agg_file}" ]] || continue
        local exp_id mean stdev cv min_v max_v valid_r desc delta
        exp_id=$( grep "^Experiment:"  "${agg_file}" | cut -d: -f2- | xargs)
        mean=$(   grep "^Mean tok/s:"  "${agg_file}" | awk '{print $NF}')
        stdev=$(  grep "^Stdev tok/s:" "${agg_file}" | awk '{print $NF}')
        cv=$(     grep "^CV%:"         "${agg_file}" | awk '{print $NF}')
        min_v=$(  grep "^Min tok/s:"   "${agg_file}" | awk '{print $NF}')
        max_v=$(  grep "^Max tok/s:"   "${agg_file}" | awk '{print $NF}')
        valid_r=$(grep "^Valid rate:"  "${agg_file}" | cut -d: -f2- | xargs)
        desc=$(   grep "^Description:" "${agg_file}" | cut -d: -f2- | xargs \
            | cut -c1-40)

        if [[ "${mean}" != "FAILED" && "${mean}" != "N/A" && \
              "${baseline_mean}" != "N/A" && "${baseline_mean}" != "FAILED" ]]; then
            delta=$(python3 -c \
                "print(f'{(${mean}/${baseline_mean}-1)*100:+.1f}%')" 2>/dev/null \
                || echo "N/A")
        else
            delta="N/A"
        fi

        printf "%-10s %-40s %-12s %-10s %-8s %-8s %-8s %-8s %-6s\n" \
            "${exp_id}" "${desc:0:40}" "${mean}" "${delta}" \
            "${stdev}" "${cv}" "${min_v}" "${max_v}" "${valid_r}"
    done

    # Write CSV
    local csv="${RESULTS_DIR}/isolation_summary.csv"
    echo "exp_id,description,mean_tok_s,vs_baseline_pct,stdev_tok_s,cv_pct,min_tok_s,max_tok_s,valid_rate,runs" \
        > "${csv}"

    for agg_file in "${RESULTS_DIR}"/exp_*/aggregate.txt; do
        [[ -f "${agg_file}" ]] || continue
        local exp_id mean stdev cv min_v max_v valid_r desc runs delta
        exp_id=$( grep "^Experiment:"  "${agg_file}" | cut -d: -f2- | xargs)
        mean=$(   grep "^Mean tok/s:"  "${agg_file}" | awk '{print $NF}')
        stdev=$(  grep "^Stdev tok/s:" "${agg_file}" | awk '{print $NF}')
        cv=$(     grep "^CV%:"         "${agg_file}" | awk '{print $NF}')
        min_v=$(  grep "^Min tok/s:"   "${agg_file}" | awk '{print $NF}')
        max_v=$(  grep "^Max tok/s:"   "${agg_file}" | awk '{print $NF}')
        valid_r=$(grep "^Valid rate:"  "${agg_file}" | cut -d: -f2- | xargs)
        runs=$(   grep "^Runs:"        "${agg_file}" | cut -d: -f2- | xargs)
        desc=$(   grep "^Description:" "${agg_file}" | cut -d: -f2- | xargs)
        if [[ "${mean}" != "FAILED" && "${mean}" != "N/A" && \
              "${baseline_mean}" != "N/A" && "${baseline_mean}" != "FAILED" ]]; then
            delta=$(python3 -c \
                "print(f'{(${mean}/${baseline_mean}-1)*100:.2f}')" 2>/dev/null \
                || echo "N/A")
        else
            delta="N/A"
        fi
        echo "${exp_id},\"${desc}\",${mean},${delta},${stdev},${cv},${min_v},${max_v},\"${valid_r}\",\"${runs}\"" \
            >> "${csv}"
    done

    log "========================================================"
    log "CSV written to: ${csv}"
    log "Results dir:    ${RESULTS_DIR}"
    log "MLflow exp:     ${MLFLOW_EXPERIMENT_NAME}"
    log "========================================================"
}

# =============================================================================
# MAIN
# =============================================================================
main() {
    TARGET="${1:-all}"

    log "========================================================"
    log "vLLM Isolation Ablation Study"
    log "$(date)"
    log "========================================================"
    log "Baseline cmd: python main.py --batch-size 16 --dtype bfloat16 --vllm"
    log "Runs/exp:     ${RUNS_PER_EXP}"
    log "Experiments:  15 (1 baseline + 14 isolated changes)"
    log "Total runs:   $((RUNS_PER_EXP * 15)) (~$((RUNS_PER_EXP * 15 * 700 / 3600)) hours)"
    log "Results dir:  ${RESULTS_DIR}"
    log "MLflow exp:   ${MLFLOW_EXPERIMENT_NAME}"
    log "Target:       ${TARGET}"
    log "========================================================"
    echo ""

    mkdir -p "${RESULTS_DIR}"
    [[ -d "${INFERENCE_DIR}" ]] || {
        error "INFERENCE_DIR not found: ${INFERENCE_DIR}"; exit 1; }

    # Back up whatever SUT is currently live
    [[ -f "${INFERENCE_DIR}/SUT_VLLM.py" ]] && \
        cp "${INFERENCE_DIR}/SUT_VLLM.py" \
           "${RESULTS_DIR}/SUT_VLLM_original_backup.py"

    # ------------------------------------------------------------------
    # EXP_00: OG BASELINE  (BS=16 — matching the given baseline command)
    # ------------------------------------------------------------------
    run_experiment "exp_00" 16  "bfloat16" "sut_exp00" \
        "OG baseline: BF16 BS=16 no-quant no-sort. Reference for all deltas."

    # ------------------------------------------------------------------
    # ISOLATED CHANGES — each vs exp_00, BS=16 unless noted
    # ------------------------------------------------------------------

    # exp_01: FP8 quantization only
    run_experiment "exp_01" 16  "bfloat16" "sut_exp01" \
        "+quantization=fp8 only. All else=baseline."

    # exp_02: batch_size=512 only (SUT + CLI both say 512)
    run_experiment "exp_02" 512 "bfloat16" "sut_exp02" \
        "+batch_size=512 only. No quant, no sort, all else=baseline."

    # exp_03: batch_size=1024 only
    run_experiment "exp_03" 1024 "bfloat16" "sut_exp03" \
        "+batch_size=1024 only. No quant, no sort, all else=baseline."

    # exp_04: max_model_len=2668 only
    run_experiment "exp_04" 16  "bfloat16" "sut_exp04" \
        "+max_model_len=2668 only. No quant, no sort, all else=baseline."

    # exp_05: sort by input length only
    run_experiment "exp_05" 16  "bfloat16" "sut_exp05" \
        "+sort_by_input_length only. No quant, no KV resize, all else=baseline."

    # exp_06: gpu_memory_utilization=0.95 only
    run_experiment "exp_06" 16  "bfloat16" "sut_exp06" \
        "+gpu_mem_util=0.95 only. All else=baseline."

    # exp_07: chunked_prefill + max_num_batched_tokens (inseparable pair)
    run_experiment "exp_07" 16  "bfloat16" "sut_exp07" \
        "+chunked_prefill=True + max_num_batched_tokens=65536. All else=baseline."

    # exp_08: expandable_segments only
    run_experiment "exp_08" 16  "bfloat16" "sut_exp08" \
        "+PYTORCH_CUDA_ALLOC_CONF=expandable_segments only. All else=baseline."

    # exp_09: prefix_caching=False explicit (sanity/control)
    run_experiment "exp_09" 16  "bfloat16" "sut_exp09" \
        "prefix_caching=False explicit (control). Expected: ~=baseline."

    # exp_10: async_scheduling=True only [vLLM 0.10.0]
    run_experiment "exp_10" 16  "bfloat16" "sut_exp10" \
        "+async_scheduling=True only [vLLM 0.10.0]. All else=baseline."

    # exp_11: num_scheduler_steps=2 only [vLLM 0.10.0]
    run_experiment "exp_11" 16  "bfloat16" "sut_exp11" \
        "+num_scheduler_steps=2 only [vLLM 0.10.0]. All else=baseline."

    # exp_12: num_scheduler_steps=4 only [vLLM 0.10.0]
    run_experiment "exp_12" 16  "bfloat16" "sut_exp12" \
        "+num_scheduler_steps=4 only [vLLM 0.10.0]. All else=baseline."

    # exp_13: compilation_config=3 only [vLLM 0.10.0]
    run_experiment "exp_13" 16  "bfloat16" "sut_exp13" \
        "+compilation_config=3 only [vLLM 0.10.0]. All else=baseline."

    # exp_14: FlashInfer backend only [vLLM 0.10.0]
    run_experiment "exp_14" 16  "bfloat16" "sut_exp14" \
        "+VLLM_ATTENTION_BACKEND=FLASHINFER only [vLLM 0.10.0]. All else=baseline."

    # ------------------------------------------------------------------
    # Restore the original SUT and print final summary
    # ------------------------------------------------------------------
    sut_exp00 > "${INFERENCE_DIR}/SUT_VLLM.py"
    log "Restored baseline SUT to ${INFERENCE_DIR}/SUT_VLLM.py"

    print_final_summary
}

main "$@"