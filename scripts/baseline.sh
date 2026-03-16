#!/bin/bash
# =============================================================================
# vLLM MLPerf Ablation Study — Llama-3.1-8B-Instruct on H200 MIG 71GB
# Offline Scenario — ONE variable at a time vs fixed baseline
#
# Design principle:
#   - exp_00 is the BASELINE — established optimal config, run first
#   - Every experiment in Phase A/B adds exactly ONE thing to exp_00
#   - Flags not under test are ALWAYS set explicitly to their baseline value
#   - Phase C runs only logical combinations of individually-positive results
#   - Never rely on vLLM defaults for any flag
#
# BASELINE CONFIG (exp_00):
#   quantization="fp8"              always on
#   tensor_parallel_size=1          always on
#   batch_size=1024                 always on
#   max_model_len=2668              always on (dataset max 2540 + output 128)
#   gpu_memory_utilization=0.95     baseline (tested higher in exp_A4)
#   enable_chunked_prefill=False    baseline OFF (tested in exp_A1, exp_A3)
#   max_num_batched_tokens not set  baseline (tested in exp_A2, exp_A3)
#   enable_prefix_caching=False     baseline OFF (tested in exp_A5)
#   scheduler_delay_factor=0.0      baseline (tested in exp_A6)
#   enforce_eager=False             always on (CUDA graphs)
#   cpu_offload_gb=0                always on
#   sort by input length            always on (zero-cost, confirmed win)
#   expandable_segments:True        always on (required for stability)
#
# Phase A — Isolated vLLM config knobs:
#   exp_A1: +enable_chunked_prefill=True
#   exp_A2: +max_num_batched_tokens=65536
#   exp_A3: +chunked_prefill=True + max_num_batched_tokens=65536 (logical pair)
#   exp_A4: +gpu_memory_utilization=0.98
#   exp_A5: +enable_prefix_caching=True   (expect regression on CNN DailyMail)
#   exp_A6: +scheduler_delay_factor=0.1
#
# Phase B — vLLM 0.10.0 features (each vs clean baseline):
#   exp_B1: +async_scheduling=True
#   exp_B2: +num_scheduler_steps=2
#   exp_B3: +num_scheduler_steps=4
#   exp_B4: +compilation_config=3
#   exp_B5: +VLLM_ATTENTION_BACKEND=FLASHINFER
#
# Phase C — combinations (run after A+B, only if both components positive):
#   exp_C1: +async_scheduling + compilation_config=3
#   exp_C2: +async_scheduling + best num_scheduler_steps from B2/B3
#   exp_C3: +chunked_prefill + max_num_batched_tokens + async_scheduling
#
# Usage:
#   chmod +x run_ablation.sh
#   ./run_ablation.sh              # run all experiments
#   ./run_ablation.sh exp_A1       # run one experiment
#   ./run_ablation.sh exp_B        # run all Phase B experiments
#   ./run_ablation.sh exp_C        # run all Phase C experiments
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INFERENCE_DIR="${INFERENCE_DIR:-${REPO_ROOT}/llama3.1-8b}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${INFERENCE_DIR}/model}"
DATASET_PATH="${DATASET_PATH:-${INFERENCE_DIR}/dataset}"
GPU_COUNT="${GPU_COUNT:-1}"
RESULTS_BASE="${REPO_ROOT}/results/throughput/single_knob_tests/single_knob_runs"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RESULTS_DIR="${RESULTS_BASE}/${TIMESTAMP}"
TOTAL_SAMPLE_COUNT=13368
SCENARIO="Offline"
USER_CONF="${INFERENCE_DIR}/user.conf"

# MLflow configuration
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
MLFLOW_EXPERIMENT_NAME="llama3.1-8b_baseline-${TIMESTAMP}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log()     { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"; }
success() { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓${NC} $*"; }
warn()    { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠${NC} $*"; }
error()   { echo -e "${RED}[$(date '+%H:%M:%S')] ✗${NC} $*"; }

cleanup_gpu() {
    log "Cleaning up GPU state..."
    set +e
    pkill -f "vllm" 2>/dev/null; pkill -f "SUT_VLLM" 2>/dev/null
    sleep 5; nvidia-smi --gpu-reset 2>/dev/null; sleep 3
    set -e
}

capture_system_info() {
    local run_dir="$1"
    mkdir -p "${run_dir}/system"
    set +e
    nvidia-smi > "${run_dir}/system/nvidia-smi.txt" 2>&1
    nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
               --format=csv > "${run_dir}/system/gpu_stats.csv" 2>&1
    nvidia-smi -q > "${run_dir}/system/nvidia-smi-full.txt" 2>&1
    nvidia-smi mig -lgip > "${run_dir}/system/mig_topology.txt" 2>&1
    nvidia-smi mig -lgi >> "${run_dir}/system/mig_topology.txt" 2>&1
    nvidia-smi mig -lci >> "${run_dir}/system/mig_topology.txt" 2>&1
    cat /proc/driver/nvidia/version > "${run_dir}/system/driver_version.txt" 2>&1
    nvcc --version >> "${run_dir}/system/driver_version.txt" 2>&1
    lscpu > "${run_dir}/system/lscpu.txt" 2>&1
    cat /proc/meminfo > "${run_dir}/system/meminfo.txt" 2>&1
    python --version > "${run_dir}/system/packages.txt" 2>&1
    pip show vllm torch transformers >> "${run_dir}/system/packages.txt" 2>&1
    pip list | grep -E "vllm|torch|cuda|nvidia|mlperf" >> "${run_dir}/system/packages.txt" 2>&1
    env | grep -E "CUDA|VLLM|PYTORCH|NCCL|GPU" | sort > "${run_dir}/system/env_vars.txt" 2>&1
    set -e
}

classify_error() {
    local logfile="$1"
    set +e
    if   grep -q "out of memory\|CUBLAS_STATUS_ALLOC_FAILED\|CUDA out of memory" "${logfile}" 2>/dev/null; then echo "OOM"
    elif grep -q "illegal memory access"                 "${logfile}" 2>/dev/null; then echo "ILLEGAL_MEMORY_ACCESS"
    elif grep -q "Segmentation fault\|SIGSEGV"           "${logfile}" 2>/dev/null; then echo "SEGFAULT"
    elif grep -q "EngineCore\|EngineDeadError"           "${logfile}" 2>/dev/null; then echo "ENGINE_CORE_FATAL"
    elif grep -q "Timeout\|timed out"                    "${logfile}" 2>/dev/null; then echo "TIMEOUT"
    elif grep -q "compilation\|torch.compile\|inductor"  "${logfile}" 2>/dev/null; then echo "COMPILE_FAILED"
    elif grep -q "RuntimeError"                          "${logfile}" 2>/dev/null; then echo "RUNTIME_ERROR"
    else echo "UNKNOWN"
    fi
    set -e
}

run_benchmark() {
    local run_dir="$1" batch_size="$2" dtype="$3" desc="$4"
    mkdir -p "${run_dir}/mlperf_logs"
    log "Running: $(basename ${run_dir})  batch=${batch_size}"
    log "  ${desc}"
    set +e
    nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu \
        --format=csv,noheader > "${run_dir}/system/gpu_pre_run.txt" 2>&1
    set -e

    # Run MLflow integration with main.py in a single Python session
    run_name="$(basename ${run_dir})"
    local exit_code=0
    local start_time; start_time=$(date +%s)
    cd "${INFERENCE_DIR}"
    set +e
    timeout 1800 python -u - 2>&1 <<PYCODE | tee "${run_dir}/run.log"
import mlflow
import subprocess
import os
import sys
import time
sys.path.insert(0, '.')
from main import parse_summary_file

run_name='${run_name}'
run_dir='${run_dir}'

mlflow.set_tracking_uri('${MLFLOW_TRACKING_URI}')
mlflow.set_experiment('${MLFLOW_EXPERIMENT_NAME}')

start_time = time.time()
mlflow.start_run(run_name=run_name)

try:
    # Parameters
    mlflow.log_param('batch_size', ${batch_size})
    mlflow.log_param('dtype', '${dtype}')
    mlflow.log_param('description', '${desc}')
    mlflow.log_param('scenario', '${SCENARIO}')
    mlflow.log_param('total_sample_count', ${TOTAL_SAMPLE_COUNT})
    mlflow.log_param('gpu_count', ${GPU_COUNT})
    mlflow.log_param('tensor_parallel_size', ${GPU_COUNT})
    mlflow.log_param('model_path', '${CHECKPOINT_PATH}')
    mlflow.log_param('dataset_path', '${DATASET_PATH}')
    mlflow.log_param('user_conf', '${USER_CONF}')
    mlflow.log_param('run_dir', run_dir)

    # Versions
    def safe_import(pkg_name):
        try:
            mod = __import__(pkg_name)
            return getattr(mod, '__version__', 'unknown')
        except Exception:
            return 'unavailable'
    mlflow.set_tag('vllm_version', safe_import('vllm'))
    mlflow.set_tag('torch_version', safe_import('torch'))
    mlflow.set_tag('transformers_version', safe_import('transformers'))

    # Hardware tags
    gpu_model = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits']).decode().strip().split('\n')[0]
    cpu_info = subprocess.check_output(['lscpu']).decode()
    cpu_model = [line.split(':')[1].strip() for line in cpu_info.split('\n') if 'Model name' in line][0]
    driver_info = subprocess.check_output(['nvidia-smi']).decode().split('\n')[2].strip()
    mlflow.set_tag('gpu_model', gpu_model)
    mlflow.set_tag('cpu_model', cpu_model)
    mlflow.set_tag('nvidia_driver', driver_info)
    mlflow.set_tag('host', subprocess.check_output(['hostname']).decode().strip())
    mlflow.set_tag('batch_size', '${batch_size}')
    mlflow.set_tag('dtype', '${dtype}')
    mlflow.set_tag('scenario', '${SCENARIO}')
    mlflow.set_tag('exp_name', run_name)

    # Run benchmark
    cmd = [
        'python', 'main.py',
        '--scenario', '${SCENARIO}',
        '--model-path', '${CHECKPOINT_PATH}',
        '--batch-size', '${batch_size}',
        '--dtype', '${dtype}',
        '--user-conf', '${USER_CONF}',
        '--total-sample-count', '${TOTAL_SAMPLE_COUNT}',
        '--dataset-path', '${DATASET_PATH}',
        '--output-log-dir', f"{run_dir}/mlperf_logs",
        '--tensor-parallel-size', '${GPU_COUNT}',
        '--vllm'
    ]
    proc = subprocess.run(cmd, check=False)
    ret_code = proc.returncode

    # Duration
    duration_sec = time.time() - start_time
    mlflow.log_metric('duration_sec', duration_sec)
    mlflow.log_param('exit_code', ret_code)

    # Parse and log metrics
    summary_path = f"{run_dir}/mlperf_logs/mlperf_log_summary.txt"
    if os.path.exists(summary_path):
        metrics = parse_summary_file(summary_path)
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value)
            else:
                mlflow.log_param(metric_name, str(metric_value))
    else:
        mlflow.set_tag('summary_missing', True)

    # Artifacts
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
    nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu \
        --format=csv,noheader > "${run_dir}/system/gpu_post_run.txt" 2>&1
    cp "${INFERENCE_DIR}"/mlperf_log_* "${run_dir}/mlperf_logs/" 2>/dev/null
    set -e

    # Extract run_id from log
    run_id=$(grep "RUN_ID:" "${run_dir}/run.log" | awk -F: '{print $2}' | tr -d '\n')

    # Log artifacts and end run after post-run captures
    python -c "
import mlflow
mlflow.set_tracking_uri('${MLFLOW_TRACKING_URI}')
mlflow.set_experiment('${MLFLOW_EXPERIMENT_NAME}')
mlflow.start_run(run_id='${run_id}')
mlflow.log_artifacts('${run_dir}')
mlflow.end_run()
"

    cat > "${run_dir}/metadata.json" <<METAEOF
{"experiment":"$(basename ${run_dir})","batch_size":${batch_size},"dtype":"${dtype}","description":"${desc}","exit_code":${exit_code},"duration_seconds":${duration},"timestamp":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
METAEOF

    local tokens samples valid
    set +e
    tokens=$(grep  "Tokens per second:"  "${run_dir}/run.log" 2>/dev/null | awk '{print $NF}')
    samples=$(grep "Samples per second:" "${run_dir}/run.log" 2>/dev/null | awk '{print $NF}')
    valid=$(grep   "Result is"           "${run_dir}/run.log" 2>/dev/null | awk '{print $NF}')
    set -e
    tokens="${tokens:-N/A}"; samples="${samples:-N/A}"; valid="${valid:-N/A}"

    if grep -q "MLPerf Results Summary" "${run_dir}/run.log" 2>/dev/null; then
        cat > "${run_dir}/result.txt" <<RESEOF
Experiment:       $(basename ${run_dir})
Description:      ${desc}
Batch Size:       ${batch_size}
Dtype:            ${dtype}
Tokens/sec:       ${tokens}
Samples/sec:      ${samples}
Valid:            ${valid}
Duration:         ${duration}s
Exit Code:        ${exit_code}
RESEOF
        success "$(basename ${run_dir}): ${tokens} tok/s  [${valid}]"
    else
        local err_type; err_type=$(classify_error "${run_dir}/run.log")
        cat > "${run_dir}/result.txt" <<RESEOF
Experiment:       $(basename ${run_dir})
Description:      ${desc}
Batch Size:       ${batch_size}
Dtype:            ${dtype}
Tokens/sec:       FAILED
Samples/sec:      FAILED
Valid:            FAILED
Error Type:       ${err_type}
Duration:         ${duration}s
Exit Code:        ${exit_code}
RESEOF
        error "$(basename ${run_dir}): FAILED (${err_type})"
        case "${err_type}" in
            OOM)                  warn "OOM — try reducing batch_size or gpu_memory_utilization" ;;
            SEGFAULT)             warn "Segfault — check PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" ;;
            ENGINE_CORE_FATAL)    warn "EngineCore fatal — check vLLM version compatibility for this flag" ;;
            ILLEGAL_MEMORY_ACCESS) warn "Illegal memory access — max_num_batched_tokens may be too high" ;;
            COMPILE_FAILED)       warn "torch.compile failed — try compilation_config={\"level\":3,\"use_inductor\":false}" ;;
        esac
    fi
    return 0
}

run_exp() {
    local exp_id="$1" exp_name="$2" batch_size="$3" dtype="$4" sut_func="$5" desc="$6"
    # Support phase-level filtering: e.g. "exp_B" matches exp_B1, exp_B2 ...
    local target="${1:-all}"  # not used — use global TARGET
    case "${TARGET}" in
        all) ;;
        exp_A|exp_B|exp_C)
            [[ "${exp_id}" == ${TARGET}* ]] || return 0 ;;
        *)
            [[ "${TARGET}" == "${exp_id}" ]] || return 0 ;;
    esac

    local run_dir="${RESULTS_DIR}/${exp_id}_${exp_name}"
    mkdir -p "${run_dir}/system"
    log "========================================================"
    log "  ${exp_id}: ${exp_name}"
    log "========================================================"
    ${sut_func} > "${run_dir}/SUT_VLLM.py"
    cp "${run_dir}/SUT_VLLM.py" "${INFERENCE_DIR}/SUT_VLLM.py"
    capture_system_info "${run_dir}"
    run_benchmark "${run_dir}" "${batch_size}" "${dtype}" "${desc}"
    log "Cooling down 30s..."; sleep 30; cleanup_gpu; sleep 10
}

# =============================================================================
# SUT GENERATOR FUNCTIONS
# =============================================================================
# Every SUT shares the same boilerplate. The only differences are:
#   1. load_model() — exactly one flag differs from baseline
#   2. _set_hardware_optimizations() — only exp_B5 adds FLASHINFER env var
#   3. Comments documenting what is under test and why everything else is pinned
#
# BASELINE VALUES (pinned in every SUT unless the experiment changes that flag):
#   dtype="float16"                    (fp8 quant forces this regardless of bfloat16)
#   quantization="fp8"
#   tensor_parallel_size=1
#   gpu_memory_utilization=0.95
#   max_model_len=2668
#   max_num_seqs=512
#   enable_prefix_caching=False
#   enable_chunked_prefill=False
#   enforce_eager=False
#   cpu_offload_gb=0
#   scheduler_delay_factor=0.0
#   batch_size=1024 (via CLI)
#   sort by input length (in issue_queries)
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# =============================================================================

# Shared boilerplate blocks to avoid repetition in the heredocs
_SUT_IMPORTS='import os, array, threading, queue, logging
import torch, numpy as np
from vllm import LLM, SamplingParams
import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-SUT")'

_SUT_INIT='    def __init__(self, model_path=None, dtype="bfloat16", batch_size=None,
                 total_sample_count=13368, dataset_path=None,
                 use_cached_outputs=False, workers=1, tensor_parallel_size=1):
        self.model_path = model_path or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.batch_size = batch_size if batch_size else 1024
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self._set_hardware_optimizations()
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
                                   total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
                                   self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam,
                                   self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
            ignore_eos=False, skip_special_tokens=True,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue(maxsize=self.batch_size * 4)
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()'

# ── exp_00: BASELINE ──────────────────────────────────────────────────────────
sut_exp00() {
  printf '%s\n' \
'# EXP_00: BASELINE
# Fixed config. Every other experiment changes exactly ONE thing from this.
#
# Pinned flags (OFF unless the experiment under test changes them):
#   enable_chunked_prefill=False     — tested in exp_A1, exp_A3
#   max_num_batched_tokens not set   — tested in exp_A2, exp_A3
#   gpu_memory_utilization=0.95      — tested in exp_A4
#   enable_prefix_caching=False      — tested in exp_A5
#   scheduler_delay_factor=0.0       — tested in exp_A6
#   async_scheduling not set         — tested in exp_B1
#   num_scheduler_steps not set      — tested in exp_B2, exp_B3
#   compilation_config not set       — tested in exp_B4
#   VLLM_ATTENTION_BACKEND default   — tested in exp_B5
import os, array, threading, queue, logging
import torch, numpy as np
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
        self.batch_size = batch_size if batch_size else 1024
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self._set_hardware_optimizations()
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
                                   total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
                                   self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam,
                                   self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
            ignore_eos=False, skip_special_tokens=True,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue(maxsize=self.batch_size * 4)
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def _set_hardware_optimizations(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        # VLLM_ATTENTION_BACKEND not set — default Flash Attention 3

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="float16",               # fp8 quant forces float16 regardless
            quantization="fp8",            # always on
            tensor_parallel_size=1,        # always on (8B fits on 1 MIG slice)
            gpu_memory_utilization=0.95,   # BASELINE (exp_A4 tests 0.98)
            max_model_len=2668,            # always on (dataset max 2540 + output 128)
            max_num_seqs=512,              # always on
            enable_prefix_caching=False,   # BASELINE OFF (exp_A5 tests True)
            enable_chunked_prefill=False,  # BASELINE OFF (exp_A1/A3 tests True)
            enforce_eager=False,           # always on (CUDA graphs)
            cpu_offload_gb=0,              # always on
            scheduler_delay_factor=0.0,    # BASELINE (exp_A6 tests 0.1)
            # max_num_batched_tokens not set  — BASELINE (exp_A2/A3 tests 65536)
            # async_scheduling not set        — BASELINE (exp_B1 tests True)
            # num_scheduler_steps not set     — BASELINE (exp_B2/B3 tests 2/4)
            # compilation_config not set      — BASELINE (exp_B4 tests 3)
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start(); self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers): self.query_queue.put(None)
        for w in self.worker_threads: w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None: break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids
            )
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)
                ])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
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

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items(): log.info(f"  {k}: {v}")

class SUTServer(SUT): pass
'
}

# =============================================================================
# PHASE A: Isolated vLLM config knobs
# =============================================================================

# ── exp_A1: +enable_chunked_prefill=True ─────────────────────────────────────
sut_expA1() {
  printf '%s\n' \
'# EXP_A1: +enable_chunked_prefill=True
# CHANGE from baseline: enable_chunked_prefill=False -> True
# All other flags identical to exp_00.
#
# What this tests: whether chunked prefill helps in the offline scenario.
# Chunked prefill splits long prompts into chunks and interleaves them with
# decode steps. Designed for server scenarios to reduce TTFT. For offline
# batch inference it may add scheduler overhead with no latency benefit.
# Note: NOT setting max_num_batched_tokens — exp_A2 tests that in isolation,
# exp_A3 tests the logical pair together.
import os, array, threading, queue, logging
import torch, numpy as np
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
        self.batch_size = batch_size if batch_size else 1024
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self._set_hardware_optimizations()
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
                                   total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
                                   self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam,
                                   self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
            ignore_eos=False, skip_special_tokens=True,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue(maxsize=self.batch_size * 4)
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def _set_hardware_optimizations(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="float16",
            quantization="fp8",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668,
            max_num_seqs=512,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,   # CHANGE: baseline=False
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
            # max_num_batched_tokens not set — tested separately in exp_A2/A3
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start(); self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers): self.query_queue.put(None)
        for w in self.worker_threads: w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None: break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids
            )
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)
                ])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
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

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items(): log.info(f"  {k}: {v}")

class SUTServer(SUT): pass
'
}

# ── exp_A2: +max_num_batched_tokens=65536 ────────────────────────────────────
sut_expA2() {
  printf '%s\n' \
'# EXP_A2: +max_num_batched_tokens=65536
# CHANGE from baseline: max_num_batched_tokens not set -> 65536
# All other flags identical to exp_00.
#
# What this tests: whether capping prefill batch size at 65536 tokens helps
# in isolation (without chunked prefill). The dataset mean input is 870 tokens,
# so 65536 / 870 ~ 75 sequences can prefill per step. Hard ceiling is 65536
# in vLLM V1 engine — exceeding it causes ILLEGAL_MEMORY_ACCESS.
# enable_chunked_prefill=False — prefill still runs as single pass, but the
# token budget per scheduler step is now bounded.
import os, array, threading, queue, logging
import torch, numpy as np
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
        self.batch_size = batch_size if batch_size else 1024
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self._set_hardware_optimizations()
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
                                   total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
                                   self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam,
                                   self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
            ignore_eos=False, skip_special_tokens=True,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue(maxsize=self.batch_size * 4)
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def _set_hardware_optimizations(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="float16",
            quantization="fp8",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668,
            max_num_seqs=512,
            max_num_batched_tokens=65536,  # CHANGE: baseline=not set
            enable_prefix_caching=False,
            enable_chunked_prefill=False,  # kept OFF — exp_A3 tests the combination
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start(); self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers): self.query_queue.put(None)
        for w in self.worker_threads: w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None: break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids
            )
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)
                ])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
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

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items(): log.info(f"  {k}: {v}")

class SUTServer(SUT): pass
'
}

# ── exp_A3: +chunked_prefill=True + max_num_batched_tokens=65536 ──────────────
sut_expA3() {
  printf '%s\n' \
'# EXP_A3: +enable_chunked_prefill=True + max_num_batched_tokens=65536
# CHANGE from baseline: both chunked_prefill and max_num_batched_tokens together.
# All other flags identical to exp_00.
#
# This is the only Phase A experiment that changes TWO flags. Justified because
# chunked prefill and max_num_batched_tokens are operationally inseparable:
# chunked prefill without a token budget cap is undefined behaviour in vLLM V1,
# and max_num_batched_tokens without chunked prefill does not control chunking.
# Comparing exp_A3 vs exp_A1 isolates the effect of the token budget.
# Comparing exp_A3 vs exp_A2 isolates the effect of chunking.
# Comparing exp_A3 vs exp_00 gives the combined effect (which we saw as -4.3%
# in the previous stacked run — now we can attribute it cleanly).
import os, array, threading, queue, logging
import torch, numpy as np
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
        self.batch_size = batch_size if batch_size else 1024
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self._set_hardware_optimizations()
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
                                   total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
                                   self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam,
                                   self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
            ignore_eos=False, skip_special_tokens=True,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue(maxsize=self.batch_size * 4)
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def _set_hardware_optimizations(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="float16",
            quantization="fp8",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668,
            max_num_seqs=512,
            max_num_batched_tokens=65536,  # CHANGE: baseline=not set
            enable_prefix_caching=False,
            enable_chunked_prefill=True,   # CHANGE: baseline=False
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start(); self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers): self.query_queue.put(None)
        for w in self.worker_threads: w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None: break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids
            )
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)
                ])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
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

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items(): log.info(f"  {k}: {v}")

class SUTServer(SUT): pass
'
}

# ── exp_A4: +gpu_memory_utilization=0.98 ─────────────────────────────────────
sut_expA4() {
  printf '%s\n' \
'# EXP_A4: +gpu_memory_utilization=0.98
# CHANGE from baseline: gpu_memory_utilization=0.95 -> 0.98
# All other flags identical to exp_00.
#
# What this tests: whether the extra 3% of HBM3e (~2.1GB on 71GB) improves
# throughput. At 0.95 we have 51.55 GiB KV cache (from previous logs).
# At 0.98 this grows, allowing more concurrent sequences in the KV cache.
# Risk: activation memory spikes during large batches may OOM. If that happens
# the classifier will catch it and log OOM.
import os, array, threading, queue, logging
import torch, numpy as np
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
        self.batch_size = batch_size if batch_size else 1024
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self._set_hardware_optimizations()
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
                                   total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
                                   self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam,
                                   self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
            ignore_eos=False, skip_special_tokens=True,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue(maxsize=self.batch_size * 4)
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def _set_hardware_optimizations(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="float16",
            quantization="fp8",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.98,   # CHANGE: baseline=0.95
            max_model_len=2668,
            max_num_seqs=512,
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
            w.start(); self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers): self.query_queue.put(None)
        for w in self.worker_threads: w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None: break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids
            )
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)
                ])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
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

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items(): log.info(f"  {k}: {v}")

class SUTServer(SUT): pass
'
}

# ── exp_A5: +enable_prefix_caching=True ──────────────────────────────────────
sut_expA5() {
  printf '%s\n' \
'# EXP_A5: +enable_prefix_caching=True
# CHANGE from baseline: enable_prefix_caching=False -> True
# All other flags identical to exp_00.
#
# What this tests: whether prefix caching helps (or hurts) on CNN DailyMail.
# CNN DailyMail articles are all unique — zero cache hit rate expected.
# Hypothesis: result is either flat (cache miss, no cost) or a small regression
# (cache lookup overhead per request with zero hits).
# This experiment confirms the hypothesis rigorously rather than assuming it.
import os, array, threading, queue, logging
import torch, numpy as np
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
        self.batch_size = batch_size if batch_size else 1024
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self._set_hardware_optimizations()
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
                                   total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
                                   self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam,
                                   self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
            ignore_eos=False, skip_special_tokens=True,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue(maxsize=self.batch_size * 4)
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def _set_hardware_optimizations(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="float16",
            quantization="fp8",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668,
            max_num_seqs=512,
            enable_prefix_caching=True,    # CHANGE: baseline=False
            enable_chunked_prefill=False,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start(); self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers): self.query_queue.put(None)
        for w in self.worker_threads: w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None: break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids
            )
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)
                ])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
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

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items(): log.info(f"  {k}: {v}")

class SUTServer(SUT): pass
'
}

# ── exp_A6: +scheduler_delay_factor=0.1 ──────────────────────────────────────
sut_expA6() {
  printf '%s\n' \
'# EXP_A6: +scheduler_delay_factor=0.1
# CHANGE from baseline: scheduler_delay_factor=0.0 -> 0.1
# All other flags identical to exp_00.
#
# What this tests: whether a small scheduler delay improves batching.
# scheduler_delay_factor adds a wait proportional to the last step duration
# before scheduling the next batch. Gives late-arriving requests a chance
# to join the current batch, potentially increasing batch fullness.
# In the offline scenario all queries arrive at t=0, so this is unlikely
# to help. Expected: flat or small regression from wasted wait time.
# Testing 0.1 (10% of last step duration) as a minimal probe.
import os, array, threading, queue, logging
import torch, numpy as np
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
        self.batch_size = batch_size if batch_size else 1024
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self._set_hardware_optimizations()
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
                                   total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
                                   self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam,
                                   self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
            ignore_eos=False, skip_special_tokens=True,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue(maxsize=self.batch_size * 4)
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def _set_hardware_optimizations(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="float16",
            quantization="fp8",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668,
            max_num_seqs=512,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.1,    # CHANGE: baseline=0.0
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start(); self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers): self.query_queue.put(None)
        for w in self.worker_threads: w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None: break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids
            )
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)
                ])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
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

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items(): log.info(f"  {k}: {v}")

class SUTServer(SUT): pass
'
}

# =============================================================================
# PHASE B: vLLM 0.10.0 features — each vs clean baseline
# =============================================================================

# ── exp_B1: +async_scheduling=True ───────────────────────────────────────────
sut_expB1() {
  printf '%s\n' \
'# EXP_B1: +async_scheduling=True  [vLLM 0.10.0]
# CHANGE from baseline: async_scheduling not set -> True
# All other flags identical to exp_00.
#
# What this tests: whether overlapping Python scheduler with GPU execution
# helps at BS=1024. Async scheduling prepares batch N+1 while GPU runs batch N.
# Previous stacked result: -3.4% (but against a different baseline config).
# This run gives a clean isolated measurement.
# If positive: include in Phase C combinations.
import os, array, threading, queue, logging
import torch, numpy as np
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
        self.batch_size = batch_size if batch_size else 1024
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self._set_hardware_optimizations()
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
                                   total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
                                   self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam,
                                   self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
            ignore_eos=False, skip_special_tokens=True,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue(maxsize=self.batch_size * 4)
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def _set_hardware_optimizations(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="float16",
            quantization="fp8",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668,
            max_num_seqs=512,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
            async_scheduling=True,         # CHANGE: baseline=not set
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start(); self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers): self.query_queue.put(None)
        for w in self.worker_threads: w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None: break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids
            )
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)
                ])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
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

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items(): log.info(f"  {k}: {v}")

class SUTServer(SUT): pass
'
}

# ── exp_B2: +num_scheduler_steps=2 ───────────────────────────────────────────
sut_expB2() {
  printf '%s\n' \
'# EXP_B2: +num_scheduler_steps=2  [vLLM 0.10.0]
# CHANGE from baseline: num_scheduler_steps not set -> 2
# All other flags identical to exp_00.
#
# What this tests: whether 2 decode steps per scheduler call reduces GIL
# pressure without the wasted-step penalty seen at steps=4.
# At steps=4 we saw -11.9% (sequences finishing mid-stride waste extra steps).
# At steps=2 the waste is halved. This is the conservative probe.
# If positive, compare against steps=4 (exp_B3) to find the optimal value.
import os, array, threading, queue, logging
import torch, numpy as np
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
        self.batch_size = batch_size if batch_size else 1024
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self._set_hardware_optimizations()
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
                                   total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
                                   self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam,
                                   self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
            ignore_eos=False, skip_special_tokens=True,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue(maxsize=self.batch_size * 4)
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def _set_hardware_optimizations(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="float16",
            quantization="fp8",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668,
            max_num_seqs=512,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
            num_scheduler_steps=2,         # CHANGE: baseline=not set (default=1)
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start(); self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers): self.query_queue.put(None)
        for w in self.worker_threads: w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None: break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids
            )
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)
                ])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
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

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items(): log.info(f"  {k}: {v}")

class SUTServer(SUT): pass
'
}

# ── exp_B3: +num_scheduler_steps=4 ───────────────────────────────────────────
sut_expB3() {
  printf '%s\n' \
'# EXP_B3: +num_scheduler_steps=4  [vLLM 0.10.0]
# CHANGE from baseline: num_scheduler_steps not set -> 4
# All other flags identical to exp_00.
#
# Repeat of the previous stacked run (which gave -11.9% VALID).
# Now measured cleanly vs the proper baseline (exp_00).
# The expected result is still a regression due to wasted decode steps when
# sequences finish mid-stride. Keeping it to confirm the number cleanly.
import os, array, threading, queue, logging
import torch, numpy as np
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
        self.batch_size = batch_size if batch_size else 1024
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self._set_hardware_optimizations()
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
                                   total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
                                   self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam,
                                   self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
            ignore_eos=False, skip_special_tokens=True,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue(maxsize=self.batch_size * 4)
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def _set_hardware_optimizations(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="float16",
            quantization="fp8",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668,
            max_num_seqs=512,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
            num_scheduler_steps=4,         # CHANGE: baseline=not set (default=1)
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start(); self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers): self.query_queue.put(None)
        for w in self.worker_threads: w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None: break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids
            )
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)
                ])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
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

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items(): log.info(f"  {k}: {v}")

class SUTServer(SUT): pass
'
}

# ── exp_B4: +compilation_config=3 ────────────────────────────────────────────
sut_expB4() {
  printf '%s\n' \
'# EXP_B4: +compilation_config=3  [vLLM 0.10.0]
# CHANGE from baseline: compilation_config not set -> 3
# All other flags identical to exp_00.
#
# torch.compile level 3. First batch takes ~2-5min to compile.
# Previous stacked result: -2.5% INVALID.
# Running cleanly vs exp_00 baseline.
# Warning: the FA3 + FP8 dtype error seen previously was specifically with
# kv_cache_dtype set — which we do NOT set here. The baseline uses auto KV
# dtype so that blocker should not apply. If it does crash, error type will
# be ENGINE_CORE_FATAL and the classifier will catch it.
import os, array, threading, queue, logging
import torch, numpy as np
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
        self.batch_size = batch_size if batch_size else 1024
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self._set_hardware_optimizations()
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
                                   total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
                                   self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam,
                                   self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
            ignore_eos=False, skip_special_tokens=True,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue(maxsize=self.batch_size * 4)
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def _set_hardware_optimizations(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="float16",
            quantization="fp8",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668,
            max_num_seqs=512,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
            compilation_config=3,          # CHANGE: baseline=not set
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start(); self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers): self.query_queue.put(None)
        for w in self.worker_threads: w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None: break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids
            )
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)
                ])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
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

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items(): log.info(f"  {k}: {v}")

class SUTServer(SUT): pass
'
}

# ── exp_B5: +VLLM_ATTENTION_BACKEND=FLASHINFER ───────────────────────────────
sut_expB5() {
  printf '%s\n' \
'# EXP_B5: +VLLM_ATTENTION_BACKEND=FLASHINFER  [vLLM 0.10.0]
# CHANGE from baseline: VLLM_ATTENTION_BACKEND not set -> FLASHINFER
# All other flags identical to exp_00.
#
# FlashInfer v0.3.0 ships with vLLM 0.10.x. Previous OOM at BS=1024 and
# BS=512 may have been caused by a previous experiment leaving GPU state dirty.
# Running clean (after GPU reset and 30s cooldown from exp_B4) at BS=1024.
# The env var is set in _set_hardware_optimizations() before LLM() is called
# so vLLM sees it at init time. All LLM() params identical to baseline.
import os, array, threading, queue, logging
import torch, numpy as np
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
        self.batch_size = batch_size if batch_size else 1024
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self._set_hardware_optimizations()
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
                                   total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
                                   self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam,
                                   self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
            ignore_eos=False, skip_special_tokens=True,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue(maxsize=self.batch_size * 4)
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def _set_hardware_optimizations(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"  # CHANGE: baseline=not set

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="float16",
            quantization="fp8",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668,
            max_num_seqs=512,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
            # No other changes — isolates the attention backend
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start(); self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers): self.query_queue.put(None)
        for w in self.worker_threads: w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None: break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids
            )
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)
                ])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
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

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items(): log.info(f"  {k}: {v}")

class SUTServer(SUT): pass
'
}

# =============================================================================
# PHASE C: Logical combinations — only run after reviewing Phase A+B results
# =============================================================================

# ── exp_C1: +async_scheduling + compilation_config=3 ─────────────────────────
sut_expC1() {
  printf '%s\n' \
'# EXP_C1: +async_scheduling=True + compilation_config=3
# CHANGE from baseline: async_scheduling + compilation_config=3 combined.
# PRECONDITION: only run if BOTH exp_B1 and exp_B4 were individually positive.
# Rationale: async scheduling and torch.compile target different bottlenecks.
# compile fuses kernel launches; async scheduling overlaps CPU scheduling with
# GPU execution. They should be additive if both genuinely help.
import os, array, threading, queue, logging
import torch, numpy as np
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
        self.batch_size = batch_size if batch_size else 1024
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self._set_hardware_optimizations()
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
                                   total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
                                   self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam,
                                   self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
            ignore_eos=False, skip_special_tokens=True,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue(maxsize=self.batch_size * 4)
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def _set_hardware_optimizations(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="float16",
            quantization="fp8",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668,
            max_num_seqs=512,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
            async_scheduling=True,         # CHANGE: from exp_B1
            compilation_config=3,          # CHANGE: from exp_B4
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start(); self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers): self.query_queue.put(None)
        for w in self.worker_threads: w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None: break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids
            )
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)
                ])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
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

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items(): log.info(f"  {k}: {v}")

class SUTServer(SUT): pass
'
}

# ── exp_C2: +async_scheduling + num_scheduler_steps=best ─────────────────────
sut_expC2() {
  printf '%s\n' \
'# EXP_C2: +async_scheduling=True + num_scheduler_steps=2
# CHANGE from baseline: async_scheduling + num_scheduler_steps=2 combined.
# PRECONDITION: run if exp_B1 (async) OR exp_B2 (steps=2) was positive.
# Uses steps=2 (conservative). If exp_B3 beat exp_B2, change to steps=4 here.
# These two features target different overhead sources:
#   async_scheduling: CPU-GPU pipeline parallelism
#   num_scheduler_steps: reduces Python scheduler call frequency
# They may be additive. If steps=2 was already a regression, skip this.
import os, array, threading, queue, logging
import torch, numpy as np
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
        self.batch_size = batch_size if batch_size else 1024
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self._set_hardware_optimizations()
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
                                   total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
                                   self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam,
                                   self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
            ignore_eos=False, skip_special_tokens=True,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue(maxsize=self.batch_size * 4)
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def _set_hardware_optimizations(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="float16",
            quantization="fp8",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668,
            max_num_seqs=512,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
            async_scheduling=True,         # CHANGE: from exp_B1
            num_scheduler_steps=2,         # CHANGE: from exp_B2 (use best of B2/B3)
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start(); self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers): self.query_queue.put(None)
        for w in self.worker_threads: w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None: break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids
            )
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)
                ])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
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

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items(): log.info(f"  {k}: {v}")

class SUTServer(SUT): pass
'
}

# ── exp_C3: best Phase A + best Phase B ──────────────────────────────────────
sut_expC3() {
  printf '%s\n' \
'# EXP_C3: Best Phase A config + best Phase B feature
# CHANGE from baseline: populate this after reviewing A+B results.
# Template: chunked_prefill + batched_tokens (if A3 won) + async_scheduling (if B1 won).
# This is intentionally left as a template — fill in the winners after Phase A+B.
# Currently uses: chunked_prefill=True + max_num_batched_tokens=65536 + async_scheduling=True
# as the most likely logical combination based on prior knowledge.
# Adjust before running based on actual A+B outcomes.
import os, array, threading, queue, logging
import torch, numpy as np
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
        self.batch_size = batch_size if batch_size else 1024
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self._set_hardware_optimizations()
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
                                   total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
                                   self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam,
                                   self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
            ignore_eos=False, skip_special_tokens=True,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue(maxsize=self.batch_size * 4)
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def _set_hardware_optimizations(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    def load_model(self):
        # !! FILL IN WINNERS FROM PHASE A + B BEFORE RUNNING !!
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="float16",
            quantization="fp8",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668,
            max_num_seqs=512,
            max_num_batched_tokens=65536,  # from exp_A3 (if positive)
            enable_prefix_caching=False,
            enable_chunked_prefill=True,   # from exp_A3 (if positive)
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
            async_scheduling=True,         # from exp_B1 (if positive)
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start(); self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers): self.query_queue.put(None)
        for w in self.worker_threads: w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None: break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids
            )
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)
                ])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
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

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items(): log.info(f"  {k}: {v}")

class SUTServer(SUT): pass
'
}

# =============================================================================
# MAIN
# =============================================================================
main() {
    TARGET="${1:-all}"
    log "vLLM Ablation Study — $(date)"
    log "Results : ${RESULTS_DIR}"
    log "Harness : ${INFERENCE_DIR}"
    log "Target  : ${TARGET}"
    echo ""
    log "Experiment matrix:"
    log "  exp_00           BASELINE (run first, reference for all others)"
    log "  exp_A1–A6        Phase A: isolated vLLM config knobs"
    log "  exp_B1–B5        Phase B: vLLM 0.10.0 features"
    log "  exp_C1–C3        Phase C: combinations (run after reviewing A+B)"
    echo ""

    mkdir -p "${RESULTS_DIR}"
    [[ -d "${INFERENCE_DIR}" ]] || { error "INFERENCE_DIR not found: ${INFERENCE_DIR}"; exit 1; }
    [[ -f "${INFERENCE_DIR}/SUT_VLLM.py" ]] && \
        cp "${INFERENCE_DIR}/SUT_VLLM.py" "${RESULTS_DIR}/SUT_VLLM_original_backup.py"

    # ── BASELINE ──────────────────────────────────────────────────────
    run_exp "exp_00" "BASELINE" \
        1024 "bfloat16" "sut_exp00" \
        "BASELINE: FP8+TP1+BS1024+max_len2668+sort+expandable_segments. All other flags OFF."

    # ── PHASE A ───────────────────────────────────────────────────────
    run_exp "exp_A1" "chunked_prefill" \
        1024 "bfloat16" "sut_expA1" \
        "+enable_chunked_prefill=True only. All other flags=baseline."

    run_exp "exp_A2" "batched_tokens_65536" \
        1024 "bfloat16" "sut_expA2" \
        "+max_num_batched_tokens=65536 only. chunked_prefill=False. All other flags=baseline."

    run_exp "exp_A3" "chunked_plus_batched_tokens" \
        1024 "bfloat16" "sut_expA3" \
        "+chunked_prefill=True + max_num_batched_tokens=65536. Logical pair. All other flags=baseline."

    run_exp "exp_A4" "gpu_mem_0.98" \
        1024 "bfloat16" "sut_expA4" \
        "+gpu_memory_utilization=0.98 only. All other flags=baseline."

    run_exp "exp_A5" "prefix_cache_on" \
        1024 "bfloat16" "sut_expA5" \
        "+enable_prefix_caching=True only. Expect flat or regression on CNN DailyMail."

    run_exp "exp_A6" "scheduler_delay_0.1" \
        1024 "bfloat16" "sut_expA6" \
        "+scheduler_delay_factor=0.1 only. All other flags=baseline."

    # ── PHASE B ───────────────────────────────────────────────────────
    run_exp "exp_B1" "async_scheduling" \
        1024 "bfloat16" "sut_expB1" \
        "+async_scheduling=True only [vLLM 0.10.0]. All other flags=baseline."

    run_exp "exp_B2" "sched_steps_2" \
        1024 "bfloat16" "sut_expB2" \
        "+num_scheduler_steps=2 only [vLLM 0.10.0]. All other flags=baseline."

    run_exp "exp_B3" "sched_steps_4" \
        1024 "bfloat16" "sut_expB3" \
        "+num_scheduler_steps=4 only [vLLM 0.10.0]. All other flags=baseline."

    run_exp "exp_B4" "compile_lvl3" \
        1024 "bfloat16" "sut_expB4" \
        "+compilation_config=3 only [vLLM 0.10.0]. First batch ~2-5min compile. All other flags=baseline."

    run_exp "exp_B5" "flashinfer" \
        1024 "bfloat16" "sut_expB5" \
        "+VLLM_ATTENTION_BACKEND=FLASHINFER only [vLLM 0.10.0]. All other flags=baseline."

    # ── PHASE C ───────────────────────────────────────────────────────
    # NOTE: Phase C runs automatically but you should review A+B results first.
    # If neither source experiment was positive, the combination won't help either.
    # Edit exp_C3 SUT before running to reflect actual A+B winners.

    run_exp "exp_C1" "async_plus_compile" \
        1024 "bfloat16" "sut_expC1" \
        "+async_scheduling + compilation_config=3. Run only if B1 AND B4 both positive."

    run_exp "exp_C2" "async_plus_sched_steps_2" \
        1024 "bfloat16" "sut_expC2" \
        "+async_scheduling + num_scheduler_steps=2. Run only if B1 OR B2 positive."

    run_exp "exp_C3" "best_A_plus_best_B" \
        1024 "bfloat16" "sut_expC3" \
        "Best Phase A + best Phase B. EDIT SUT BEFORE RUNNING based on A+B results."

    # ── Restore baseline SUT ──────────────────────────────────────────
    sut_exp00 > "${INFERENCE_DIR}/SUT_VLLM.py"
    log "Restored BASELINE SUT to ${INFERENCE_DIR}/SUT_VLLM.py"

    # ── Summary ───────────────────────────────────────────────────────
    echo ""
    log "========================================================"
    log "ABLATION STUDY SUMMARY (vs BASELINE exp_00)"
    log "========================================================"
    local baseline_tok="N/A"
    local baseline_file="${RESULTS_DIR}/exp_00_BASELINE/result.txt"
    [[ -f "${baseline_file}" ]] && \
        baseline_tok=$(grep "^Tokens/sec:" "${baseline_file}" | cut -d: -f2- | xargs)

    printf "%-42s %-12s %-10s %-10s %-8s\n" \
        "Experiment" "Tokens/sec" "vs Baseline" "Valid" "Status"
    printf "%-42s %-12s %-10s %-10s %-8s\n" \
        "----------" "----------" "-----------" "-----" "------"

    for rf in "${RESULTS_DIR}"/exp_*/result.txt; do
        [[ -f "${rf}" ]] || continue
        local exp tok valid status delta
        exp=$(   grep "^Experiment:"  "${rf}" | cut -d: -f2- | xargs)
        tok=$(   grep "^Tokens/sec:"  "${rf}" | cut -d: -f2- | xargs)
        valid=$( grep "^Valid:"       "${rf}" | cut -d: -f2- | xargs)
        status="OK"; [[ "${tok}" == "FAILED" ]] && status="FAILED"
        if [[ "${tok}" != "FAILED" && "${tok}" != "N/A" && \
              "${baseline_tok}" != "N/A" && "${baseline_tok}" != "FAILED" ]]; then
            delta=$(python3 -c "print(f'{(${tok}/${baseline_tok}-1)*100:+.1f}%')" 2>/dev/null || echo "N/A")
        else
            delta="N/A"
        fi
        printf "%-42s %-12s %-10s %-10s %-8s\n" \
            "${exp}" "${tok}" "${delta}" "${valid}" "${status}"
    done

    log "========================================================"
    log "Full results: ${RESULTS_DIR}"

    # ── CSV ───────────────────────────────────────────────────────────
    local csv="${RESULTS_DIR}/ablation_summary.csv"
    echo "experiment,tokens_per_sec,samples_per_sec,vs_baseline_pct,valid,exit_code,duration_s,description" \
        > "${csv}"
    for rf in "${RESULTS_DIR}"/exp_*/result.txt; do
        [[ -f "${rf}" ]] || continue
        local exp tok samp valid ec dur desc delta
        exp=$(  grep "^Experiment:"  "${rf}" | cut -d: -f2- | xargs)
        tok=$(  grep "^Tokens/sec:"  "${rf}" | cut -d: -f2- | xargs)
        samp=$( grep "^Samples/sec:" "${rf}" | cut -d: -f2- | xargs || echo "N/A")
        valid=$(grep "^Valid:"       "${rf}" | cut -d: -f2- | xargs)
        ec=$(   grep "^Exit Code:"   "${rf}" | cut -d: -f2- | xargs)
        dur=$(  grep "^Duration:"    "${rf}" | cut -d: -f2- | tr -d 's' | xargs)
        desc=$( grep "^Description:" "${rf}" | cut -d: -f2- | xargs)
        if [[ "${tok}" != "FAILED" && "${tok}" != "N/A" && \
              "${baseline_tok}" != "N/A" && "${baseline_tok}" != "FAILED" ]]; then
            delta=$(python3 -c "print(f'{(${tok}/${baseline_tok}-1)*100:+.2f}')" 2>/dev/null || echo "N/A")
        else
            delta="N/A"
        fi
        echo "${exp},${tok},${samp},${delta},${valid},${ec},${dur},\"${desc}\"" >> "${csv}"
    done
    log "CSV: ${csv}"
}

main "$@"