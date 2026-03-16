#!/bin/bash
# =============================================================================
# vLLM MLPerf Benchmark Automation Script
# Llama-3.1-8B-Instruct on H200 MIG 71GB — Offline Scenario
#
# Usage:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh              # run all 15 experiments
#   ./run_experiments.sh exp_12       # run one experiment
#
# exp_01–exp_11: Baseline tuning progression (2945 tok/s on vLLM 0.10.0)
# exp_12–exp_15: vLLM 0.10.0 specific experiments
#
#   exp_01: Baseline — BF16, TP=1, BS=16, no quantization
#   exp_02: +FP8 quantization
#   exp_03: +tensor_parallel_size=1 explicit
#   exp_04: +gpu_memory_utilization=0.98
#   exp_05: +batch_size=512  (same SUT as exp_04, CLI only)
#   exp_06: +max_model_len=2668
#   exp_07: +max_num_batched_tokens=65536, gpu_mem->0.95, chunked_prefill=True
#   exp_08: +expandable_segments:True (segfault root cause fix)
#   exp_09: +sort queries by input length
#   exp_10: +enable_prefix_caching=False explicit
#   exp_11: +batch_size=1024  (2945 tok/s baseline on vLLM 0.10.0)
#   exp_12: +async_scheduling=True
#   exp_13: +num_scheduler_steps=4
#   exp_14: +compilation_config=3
#   exp_15: FlashInfer v0.3.0 retest
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INFERENCE_DIR="${INFERENCE_DIR:-${REPO_ROOT}/llama3.1-8b}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${INFERENCE_DIR}/model}"
DATASET_PATH="${DATASET_PATH:-${INFERENCE_DIR}/dataset}"
GPU_COUNT="${GPU_COUNT:-1}"
RESULTS_BASE="${REPO_ROOT}/results/throughput/progression_full_stack"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RESULTS_DIR="${RESULTS_BASE}/${TIMESTAMP}"
TOTAL_SAMPLE_COUNT=13368
SCENARIO="Offline"
USER_CONF="${INFERENCE_DIR}/user.conf"

# MLflow configuration
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
MLFLOW_EXPERIMENT_NAME="llama3.1-8b_experiments-${TIMESTAMP}"

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
    nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv \
               > "${run_dir}/system/gpu_stats.csv" 2>&1
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
    
    # Run main.py with MLflow integration
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

    # Extract metrics from summary file for display and result tracking
    local tokens samples valid min_latency max_latency mean_latency p50_latency p90_latency p95_latency p99_latency
    set +e
    summary_file="${run_dir}/mlperf_logs/mlperf_log_summary.txt"
    if [[ -f "${summary_file}" ]]; then
        tokens=$(grep "Tokens per second:" "${summary_file}" 2>/dev/null | awk '{print $NF}')
        samples=$(grep "Samples per second:" "${summary_file}" 2>/dev/null | awk '{print $NF}')
        valid=$(grep "Result is" "${summary_file}" 2>/dev/null | awk '{print $NF}')
        min_latency=$(grep "Min latency (ns)" "${summary_file}" 2>/dev/null | awk '{print $(NF-1)}' || echo "N/A")
        max_latency=$(grep "Max latency (ns)" "${summary_file}" 2>/dev/null | awk '{print $(NF-1)}' || echo "N/A")
        mean_latency=$(grep "Mean latency (ns)" "${summary_file}" 2>/dev/null | awk '{print $(NF-1)}' || echo "N/A")
        p50_latency=$(grep "50.00 percentile latency (ns)" "${summary_file}" 2>/dev/null | awk '{print $(NF-1)}' || echo "N/A")
        p90_latency=$(grep "90.00 percentile latency (ns)" "${summary_file}" 2>/dev/null | awk '{print $(NF-1)}' || echo "N/A")
        p95_latency=$(grep "95.00 percentile latency (ns)" "${summary_file}" 2>/dev/null | awk '{print $(NF-1)}' || echo "N/A")
        p99_latency=$(grep "99.00 percentile latency (ns)" "${summary_file}" 2>/dev/null | awk '{print $(NF-1)}' || echo "N/A")
    fi
    set -e
    tokens="${tokens:-N/A}"; samples="${samples:-N/A}"; valid="${valid:-N/A}"
    min_latency="${min_latency:-N/A}"; max_latency="${max_latency:-N/A}"; mean_latency="${mean_latency:-N/A}"
    p50_latency="${p50_latency:-N/A}"; p90_latency="${p90_latency:-N/A}"; p95_latency="${p95_latency:-N/A}"; p99_latency="${p99_latency:-N/A}"

    if grep -q "Result is.*VALID" "${run_dir}/mlperf_logs/mlperf_log_summary.txt" 2>/dev/null; then
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
        local err_type; err_type=$(classify_error "${run_dir}/mlperf_logs/mlperf_log_summary.txt")
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
    fi
    return 0
}

classify_error() {
    local logfile="$1"
    set +e
    if   grep -q "out of memory\|CUBLAS_STATUS_ALLOC_FAILED\|CUDA out of memory" "${logfile}" 2>/dev/null; then echo "OOM"
    elif grep -q "illegal memory access"        "${logfile}" 2>/dev/null; then echo "ILLEGAL_MEMORY_ACCESS"
    elif grep -q "Segmentation fault\|SIGSEGV"  "${logfile}" 2>/dev/null; then echo "SEGFAULT"
    elif grep -q "EngineCore failed"            "${logfile}" 2>/dev/null; then echo "ENGINE_INIT_FAILED"
    elif grep -q "Timeout\|timed out"           "${logfile}" 2>/dev/null; then echo "TIMEOUT"
    elif grep -q "compilation\|torch.compile\|inductor" "${logfile}" 2>/dev/null; then echo "COMPILE_FAILED"
    else echo "UNKNOWN"
    fi
    set -e
}

run_with_oom_retry() {
    local run_dir="$1" batch_size="$2" dtype="$3" desc="$4"
    run_benchmark "${run_dir}" "${batch_size}" "${dtype}" "${desc}"
    if grep -q "^Tokens/sec:.*FAILED" "${run_dir}/result.txt" 2>/dev/null; then
        local err_type; err_type=$(classify_error "${run_dir}/run.log")
        case "${err_type}" in
            OOM)
                local retry_batch=$(( batch_size / 2 ))
                warn "OOM -- retrying at batch_size=${retry_batch}..."
                local retry_dir="${run_dir}_oom_retry_bs${retry_batch}"
                mkdir -p "${retry_dir}/system"
                set +e; cp "${run_dir}/SUT_VLLM.py" "${retry_dir}/SUT_VLLM.py" 2>/dev/null
                cp -r "${run_dir}/system/." "${retry_dir}/system/" 2>/dev/null; set -e
                run_benchmark "${retry_dir}" "${retry_batch}" "${dtype}" "${desc} [OOM retry]"
                ;;
            SEGFAULT)           warn "Segfault -- check PYTORCH_CUDA_ALLOC_CONF (must be expandable_segments:True)" ;;
            ENGINE_INIT_FAILED) warn "Engine init failed -- max_num_batched_tokens too high for V1 engine" ;;
            ILLEGAL_MEMORY_ACCESS) warn "Illegal memory access -- max_num_batched_tokens must be <=65536" ;;
            COMPILE_FAILED)     warn "torch.compile failed -- try compilation_config={\"level\": 3, \"use_inductor\": false}" ;;
        esac
    fi
}

print_summary() {
    echo ""
    log "========================================================"
    log "EXPERIMENT SUMMARY"
    log "========================================================"
    printf "%-56s %-14s %-10s %-8s\n" "Experiment" "Tokens/sec" "Valid" "Status"
    printf "%-56s %-14s %-10s %-8s\n" "----------" "----------" "-----" "------"
    for rf in "${RESULTS_DIR}"/exp_*/result.txt; do
        [[ -f "${rf}" ]] || continue
        local exp tokens valid status
        exp=$(   grep "^Experiment:"  "${rf}" | cut -d: -f2- | xargs)
        tokens=$(grep "^Tokens/sec:"  "${rf}" | cut -d: -f2- | xargs)
        valid=$( grep "^Valid:"       "${rf}" | cut -d: -f2- | xargs)
        status="OK"; [[ "${tokens}" == "FAILED" ]] && status="FAILED"
        printf "%-56s %-14s %-10s %-8s\n" "${exp}" "${tokens}" "${valid}" "${status}"
    done
    log "========================================================"
    log "Full results: ${RESULTS_DIR}"
    echo ""
}

sut_exp01() {
  printf '%s\n' '# EXP 01: BASELINE — BF16, TP=1, BS=16 (base command default), no quantization.
# All optimisation flags pinned explicitly to off/zero from the start.
# We never rely on vLLM defaults.
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
        self.batch_size = batch_size if batch_size else 16
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
                                   total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
                                   self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam,
                                   self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(temperature=0.0, top_p=1, top_k=1,
                                              seed=42, max_tokens=128, min_tokens=1)
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
            tensor_parallel_size=self.tensor_parallel_size,
            enable_prefix_caching=False,   # pinned off
            enable_chunked_prefill=False,  # pinned off
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
            outputs = self.model.generate(prompt_token_ids=input_ids_tensor,
                                          sampling_params=self.sampling_params, use_tqdm=False)
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(pred_output_tokens,
                                                             query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)])
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

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items(): log.info(f"  {k}: {v}")

class SUTServer(SUT): pass
'
}

sut_exp02() {
  printf '%s\n' '# EXP 02: +FP8 quantization
# CHANGE: quantization=fp8 — weights 8.5GB -> 3.47GB
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
        self.batch_size = batch_size if batch_size else 16
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
                                   total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
                                   self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam,
                                   self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(temperature=0.0, top_p=1, top_k=1,
                                              seed=42, max_tokens=128, min_tokens=1)
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
            quantization="fp8",            # CHANGE: weights 8.5GB -> 3.47GB
            tensor_parallel_size=self.tensor_parallel_size,
            enable_prefix_caching=False,   # pinned off
            enable_chunked_prefill=False,  # pinned off
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
            outputs = self.model.generate(prompt_token_ids=input_ids_tensor,
                                          sampling_params=self.sampling_params, use_tqdm=False)
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(pred_output_tokens,
                                                             query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)])
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

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items(): log.info(f"  {k}: {v}")

class SUTServer(SUT): pass
'
}

sut_exp03() {
  printf '%s\n' '# EXP 03: +tensor_parallel_size=1 explicit
# CHANGE: TP=1 removes inter-GPU comm. 8B model fits on single MIG slice.
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
        self.batch_size = batch_size if batch_size else 16
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
                                   total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
                                   self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam,
                                   self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(temperature=0.0, top_p=1, top_k=1,
                                              seed=42, max_tokens=128, min_tokens=1)
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
            quantization="fp8",
            tensor_parallel_size=1,        # CHANGE: explicit TP=1
            enable_prefix_caching=False,   # pinned off
            enable_chunked_prefill=False,  # pinned off
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
            outputs = self.model.generate(prompt_token_ids=input_ids_tensor,
                                          sampling_params=self.sampling_params, use_tqdm=False)
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(pred_output_tokens,
                                                             query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)])
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

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items(): log.info(f"  {k}: {v}")

class SUTServer(SUT): pass
'
}

sut_exp04() {
  printf '%s\n' '# EXP 04: +gpu_memory_utilization=0.98
# CHANGE: gpu_memory_utilization 0.90 -> 0.98. More VRAM for KV cache.
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
        self.batch_size = batch_size if batch_size else 16
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
                                   total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
                                   self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam,
                                   self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(temperature=0.0, top_p=1, top_k=1,
                                              seed=42, max_tokens=128, min_tokens=1)
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
            quantization="fp8",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.98,   # CHANGE: was default 0.90
            enable_prefix_caching=False,   # pinned off
            enable_chunked_prefill=False,  # pinned off
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
            outputs = self.model.generate(prompt_token_ids=input_ids_tensor,
                                          sampling_params=self.sampling_params, use_tqdm=False)
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(pred_output_tokens,
                                                             query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)])
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

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items(): log.info(f"  {k}: {v}")

class SUTServer(SUT): pass
'
}

sut_exp06() {
  printf '%s\n' '# EXP 06: +max_model_len=2668
# CHANGE: max_model_len 131072 -> 2668 (dataset max 2540 + max_output 128).
# Default wastes 97pct of each KV slot. At 2668: ~148 concurrent seqs vs ~3.
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
        self.batch_size = batch_size if batch_size else 512
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
        self.sampling_params = SamplingParams(temperature=0.0, top_p=1, top_k=1,
                                              seed=42, max_tokens=128, min_tokens=1, ignore_eos=False, skip_special_tokens=True)
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
            dtype="bfloat16",
            quantization="fp8",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.98,
            max_model_len=2668,            # CHANGE: dataset minimum (2540+128)
            enable_prefix_caching=False,   # pinned off
            enable_chunked_prefill=False,  # pinned off
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
            outputs = self.model.generate(prompt_token_ids=input_ids_tensor,
                                          sampling_params=self.sampling_params, use_tqdm=False)
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(pred_output_tokens,
                                                             query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)])
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

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items(): log.info(f"  {k}: {v}")

class SUTServer(SUT): pass
'
}

sut_exp07() {
  printf '%s\n' '# EXP 07: +max_num_batched_tokens=65536, gpu_mem_util 0.98->0.95, chunked_prefill=True
# CHANGE 1: max_num_batched_tokens=65536 — removes prefill bottleneck.
# CHANGE 2: gpu_memory_utilization 0.98->0.95 (REQUIRED — activation memory spike).
# CHANGE 3: enable_chunked_prefill=True. enable_prefix_caching still pinned off.
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
        self.batch_size = batch_size if batch_size else 512
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
        self.sampling_params = SamplingParams(temperature=0.0, top_p=1, top_k=1,
                                              seed=42, max_tokens=128, min_tokens=1, ignore_eos=False, skip_special_tokens=True)
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
            dtype="bfloat16",
            quantization="fp8",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,   # CHANGE: 0.98->0.95 (required with batched tokens)
            max_model_len=2668,
            max_num_batched_tokens=65536,  # CHANGE: removes prefill bottleneck
            max_num_seqs=512,
            enable_prefix_caching=False,   # pinned off
            enable_chunked_prefill=True,   # CHANGE: enabled
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
            outputs = self.model.generate(prompt_token_ids=input_ids_tensor,
                                          sampling_params=self.sampling_params, use_tqdm=False)
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(pred_output_tokens,
                                                             query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)])
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

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items(): log.info(f"  {k}: {v}")

class SUTServer(SUT): pass
'
}

sut_exp08() {
  printf '%s\n' '# EXP 08: +expandable_segments:True (SEGFAULT ROOT CAUSE FIX)
# CHANGE 1: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   max_split_size_mb:512 caused fragmentation crash after ~30 batches.
# CHANGE 2: dtype=float16 explicit. CHANGE 3: cpu_offload_gb=0 confirmed.
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
        self.batch_size = batch_size if batch_size else 512
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
        self.sampling_params = SamplingParams(temperature=0.0, top_p=1, top_k=1,
                                              seed=42, max_tokens=128, min_tokens=1, ignore_eos=False, skip_special_tokens=True)
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
            dtype="float16",               # CHANGE: explicit (bfloat16 silently cast for FP8)
            quantization="fp8",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668,
            max_num_batched_tokens=65536,
            max_num_seqs=512,
            enable_prefix_caching=False,   # pinned off
            enable_chunked_prefill=True,
            enforce_eager=False,
            cpu_offload_gb=0,              # CHANGE: confirmed 0 (was 10 in some variants)
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
            outputs = self.model.generate(prompt_token_ids=input_ids_tensor,
                                          sampling_params=self.sampling_params, use_tqdm=False)
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(pred_output_tokens,
                                                             query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)])
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

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items(): log.info(f"  {k}: {v}")

class SUTServer(SUT): pass
'
}

sut_exp09() {
  printf '%s\n' '# EXP 09: +sort queries by input token length
# CHANGE: issue_queries sorts by len(input_ids) before batching.
# Dataset token lengths: min=79, max=2540, mean=870 — high variance.
# Observed: mean latency -24pct, throughput +0.7pct.
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
        self.batch_size = batch_size if batch_size else 512
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
        self.sampling_params = SamplingParams(temperature=0.0, top_p=1, top_k=1,
                                              seed=42, max_tokens=128, min_tokens=1, ignore_eos=False, skip_special_tokens=True)
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
            dtype="float16",               # CHANGE: explicit (bfloat16 silently cast for FP8)
            quantization="fp8",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668,
            max_num_batched_tokens=65536,
            max_num_seqs=512,
            enable_prefix_caching=False,   # pinned off
            enable_chunked_prefill=True,
            enforce_eager=False,
            cpu_offload_gb=0,              # CHANGE: confirmed 0 (was 10 in some variants)
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
            outputs = self.model.generate(prompt_token_ids=input_ids_tensor,
                                          sampling_params=self.sampling_params, use_tqdm=False)
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(pred_output_tokens,
                                                             query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)])
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

sut_exp10_final() {
  printf '%s\n' '# EXP 10 / FINAL CONFIG: enable_prefix_caching=False explicit
# CHANGE: enable_prefix_caching=False now stated explicitly (was pinned off).
# CNN DailyMail articles are all unique — zero cache hit rate on this dataset.
# exp_11 reuses this SUT with batch_size=1024 (confirmed optimal at 2945 tok/s).
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
        self.sampling_params = SamplingParams(temperature=0.0, top_p=1, top_k=1,
                                              seed=42, max_tokens=128, min_tokens=1, ignore_eos=False, skip_special_tokens=True)
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
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=0.95,
            max_model_len=2668,
            max_num_batched_tokens=65536,
            max_num_seqs=512,
            enable_prefix_caching=False,   # CHANGE exp_10: explicit (zero hit rate on CNN DailyMail)
            enable_chunked_prefill=True,
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
            outputs = self.model.generate(prompt_token_ids=input_ids_tensor,
                                          sampling_params=self.sampling_params, use_tqdm=False)
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(pred_output_tokens,
                                                             query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)])
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

sut_exp12() {
  printf '%s\n' '# EXP 12: +async_scheduling=True  [vLLM 0.10.0]
# CHANGE: async_scheduling=True — overlaps Python scheduler with GPU execution.
# Eliminates scheduler dead-time between decode steps.
# Baseline entering this experiment: 2945 tok/s on vLLM 0.10.0.
# IMPORTANT: verify ROUGE scores still pass 99pct targets after enabling.
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
        self.sampling_params = SamplingParams(temperature=0.0, top_p=1, top_k=1,
                                              seed=42, max_tokens=128, min_tokens=1, ignore_eos=False, skip_special_tokens=True)
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
        # CHANGE: async_scheduling=True
        # New in vLLM 0.10.0. Overlaps Python scheduler with GPU execution.
        # Next batch prepared while current batch runs on GPU.
        # IMPORTANT: verify ROUGE scores still pass 99pct targets after enabling.
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="float16",
            quantization="fp8",
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=0.95,
            max_model_len=2668,
            max_num_batched_tokens=65536,
            max_num_seqs=512,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
            async_scheduling=True,         # CHANGE: overlap scheduler with GPU
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
            outputs = self.model.generate(prompt_token_ids=input_ids_tensor,
                                          sampling_params=self.sampling_params, use_tqdm=False)
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(pred_output_tokens,
                                                             query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)])
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

sut_exp13() {
  printf '%s\n' '# EXP 13: +num_scheduler_steps=4  [vLLM 0.10.0]
# CHANGE: num_scheduler_steps=4 — 4 decode steps per scheduler call.
# Reduces Python GIL pressure and scheduler overhead by ~4x.
# Different from async_scheduling: batches calls vs overlapping them.
# Can combine with async_scheduling. Safe with sorted batches at BS=1024.
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
        self.sampling_params = SamplingParams(temperature=0.0, top_p=1, top_k=1,
                                              seed=42, max_tokens=128, min_tokens=1, ignore_eos=False, skip_special_tokens=True)
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
        # CHANGE: num_scheduler_steps=4
        # Runs 4 decode steps per scheduler invocation instead of 1.
        # Reduces Python GIL pressure and scheduler overhead by ~4x.
        # Different from async_scheduling: batches calls vs overlapping them.
        # Can be combined with async_scheduling for additive gain.
        # Safe with sorted batches at BS=1024 (uniform lengths, minimal step waste).
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="float16",
            quantization="fp8",
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=0.95,
            max_model_len=2668,
            max_num_batched_tokens=65536,
            max_num_seqs=512,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
            num_scheduler_steps=4,         # CHANGE: 4 decode steps per scheduler call
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
            outputs = self.model.generate(prompt_token_ids=input_ids_tensor,
                                          sampling_params=self.sampling_params, use_tqdm=False)
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(pred_output_tokens,
                                                             query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)])
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

sut_exp14() {
    printf '%s\n' '# EXP 14: +compilation_config=3  [vLLM 0.10.0]
# CHANGE: compilation_config=3 — torch.compile level 3 (production level).
# Fuses CUDA kernels for decode loop. First batch: 2-5min compile.
# 13368 samples / BS=1024 = ~13 batches — cost amortised after batch 1.
# If fails: try compilation_config={"level": 3, "use_inductor": False}
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
        self.sampling_params = SamplingParams(temperature=0.0, top_p=1, top_k=1,
                                              seed=42, max_tokens=128, min_tokens=1, ignore_eos=False, skip_special_tokens=True)
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
        # CHANGE: compilation_config=3 (torch.compile level 3)
        # Fuses CUDA kernels for the decode loop. Level 3 = production-recommended.
        # First batch: 2-5min compile time. All subsequent batches: fused kernels.
        # 13368 samples / BS=1024 = ~13 batches, cost amortised after batch 1.
        # If startup fails: try compilation_config={"level": 3, "use_inductor": False}
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="float16",
            quantization="fp8",
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=0.95,
            max_model_len=2668,
            max_num_batched_tokens=65536,
            max_num_seqs=512,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
            compilation_config=3,          # CHANGE: torch.compile level 3
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
            outputs = self.model.generate(prompt_token_ids=input_ids_tensor,
                                          sampling_params=self.sampling_params, use_tqdm=False)
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(pred_output_tokens,
                                                             query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)])
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

sut_exp15() {
    printf '%s\n' '# EXP 15: FlashInfer v0.3.0 retest  [vLLM 0.10.0]
# CHANGE: VLLM_ATTENTION_BACKEND=FLASHINFER (set in _set_hardware_optimizations).
# FlashInfer upgraded to v0.3.0 in vLLM 0.10.x. Previously regressed on 0.8.x.
# 0.3.x decode kernel substantially rewritten — clean retest on same config.
# All LLM() params identical to baseline to isolate the backend change.
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
        self.sampling_params = SamplingParams(temperature=0.0, top_p=1, top_k=1,
                                              seed=42, max_tokens=128, min_tokens=1, ignore_eos=False, skip_special_tokens=True)
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
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"  # CHANGE: FlashInfer v0.3.0

    def load_model(self):
        # CHANGE: VLLM_ATTENTION_BACKEND=FLASHINFER (set in _set_hardware_optimizations)
        # FlashInfer upgraded to v0.3.0 in vLLM 0.10.x.
        # Previously regressed on 0.8.x. 0.3.x decode kernel substantially rewritten.
        # All LLM() params identical to baseline to isolate the backend change.
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="float16",
            quantization="fp8",
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=0.95,
            max_model_len=2668,
            max_num_batched_tokens=65536,
            max_num_seqs=512,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
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
            outputs = self.model.generate(prompt_token_ids=input_ids_tensor,
                                          sampling_params=self.sampling_params, use_tqdm=False)
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(pred_output_tokens,
                                                             query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)])
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


main() {
    local target_exp="${1:-all}"
    log "vLLM MLPerf Benchmark Automation — $(date)"
    log "Results : ${RESULTS_DIR}"
    log "Harness : ${INFERENCE_DIR}"
    log "Target  : ${target_exp}"
    mkdir -p "${RESULTS_DIR}"
    [[ -d "${INFERENCE_DIR}" ]] || { error "INFERENCE_DIR not found: ${INFERENCE_DIR}"; exit 1; }
    [[ -f "${INFERENCE_DIR}/SUT_VLLM.py" ]] && \
        cp "${INFERENCE_DIR}/SUT_VLLM.py" "${RESULTS_DIR}/SUT_VLLM_original_backup.py"

    run_exp() {
        local exp_id="$1" exp_name="$2" batch_size="$3" dtype="$4" sut_func="$5" desc="$6"
        [[ "${target_exp}" != "all" && "${target_exp}" != "${exp_id}" ]] && return 0
        local run_dir="${RESULTS_DIR}/${exp_id}_${exp_name}"
        mkdir -p "${run_dir}/system"
        log "========================================================"; log "  ${exp_id}: ${exp_name}"; log "========================================================"
        ${sut_func} > "${run_dir}/SUT_VLLM.py"
        cp "${run_dir}/SUT_VLLM.py" "${INFERENCE_DIR}/SUT_VLLM.py"
        capture_system_info "${run_dir}"
        run_with_oom_retry "${run_dir}" "${batch_size}" "${dtype}" "${desc}"
        log "Cooling down 30s..."; sleep 30; cleanup_gpu; sleep 10
    }

    # ── exp_01–exp_04 ─────────────────────────────────────────────────
    run_exp "exp_01" "baseline_bf16_bs16" \
        16 "bfloat16" "sut_exp01" "Baseline: BF16, TP=1, BS=16, all flags pinned explicitly"
    run_exp "exp_02" "fp8_quantization" \
        16 "bfloat16" "sut_exp02" "+FP8 quantization: weights 8.5GB -> 3.47GB"
    run_exp "exp_03" "tensor_parallel_1" \
        16 "bfloat16" "sut_exp03" "+TP=1 explicit: removes inter-GPU comm"
    run_exp "exp_04" "gpu_mem_util_0.98" \
        16 "bfloat16" "sut_exp04" "+gpu_memory_utilization=0.98"

    # ── exp_05: same SUT as exp_04, batch_size via CLI only ───────────
    if [[ "${target_exp}" == "all" || "${target_exp}" == "exp_05" ]]; then
        local exp05_dir="${RESULTS_DIR}/exp_05_batch_size_512"
        mkdir -p "${exp05_dir}/system"
        log "========================================================"; log "  exp_05: batch_size_512"; log "========================================================"
        sut_exp04 > "${exp05_dir}/SUT_VLLM.py"
        cp "${exp05_dir}/SUT_VLLM.py" "${INFERENCE_DIR}/SUT_VLLM.py"
        capture_system_info "${exp05_dir}"
        run_with_oom_retry "${exp05_dir}" 512 "bfloat16" "batch_size 16->512: saturates GPU compute"
        log "Cooling down 30s..."; sleep 30; cleanup_gpu; sleep 10
    fi

    # ── exp_06–exp_09 ─────────────────────────────────────────────────
    run_exp "exp_06" "max_model_len_2668" \
        512 "bfloat16" "sut_exp06" "+max_model_len=2668: dataset minimum. 148x concurrency"
    run_exp "exp_07" "batched_tokens_65536" \
        512 "bfloat16" "sut_exp07" "+max_num_batched_tokens=65536 + gpu_mem->0.95 + chunked_prefill"
    run_exp "exp_08" "expandable_segments" \
        512 "bfloat16" "sut_exp08" "+expandable_segments:True: root cause fix for segfaults"
    run_exp "exp_09" "sort_by_length" \
        512 "bfloat16" "sut_exp09" "+sort by input length: latency -24pct, throughput +0.7pct"

    # ── exp_10: prefix_caching=False explicit at BS=512 ───────────────
    if [[ "${target_exp}" == "all" || "${target_exp}" == "exp_10" ]]; then
        local exp10_dir="${RESULTS_DIR}/exp_10_no_prefix_cache_bs512"
        mkdir -p "${exp10_dir}/system"
        log "========================================================"; log "  exp_10: no_prefix_cache_bs512"; log "========================================================"
        sut_exp10_final > "${exp10_dir}/SUT_VLLM.py"
        cp "${exp10_dir}/SUT_VLLM.py" "${INFERENCE_DIR}/SUT_VLLM.py"
        capture_system_info "${exp10_dir}"
        run_with_oom_retry "${exp10_dir}" 512 "bfloat16" \
            "+enable_prefix_caching=False explicit: zero hit rate on CNN DailyMail"
        log "Cooling down 30s..."; sleep 30; cleanup_gpu; sleep 10
    fi

    # ── exp_11: final config, BS=1024 ────────────────────────────────
    if [[ "${target_exp}" == "all" || "${target_exp}" == "exp_11" ]]; then
        local exp11_dir="${RESULTS_DIR}/exp_11_batch_1024_final_2945tok"
        mkdir -p "${exp11_dir}/system"
        log "========================================================"; log "  exp_11: batch_1024 FINAL — 2945 tok/s on vLLM 0.10.0"; log "========================================================"
        sut_exp10_final > "${exp11_dir}/SUT_VLLM.py"
        cp "${exp11_dir}/SUT_VLLM.py" "${INFERENCE_DIR}/SUT_VLLM.py"
        capture_system_info "${exp11_dir}"
        run_with_oom_retry "${exp11_dir}" 1024 "bfloat16" \
            "batch_size 512->1024: +3.4pct. 2945 tok/s on vLLM 0.10.0."
        log "Cooling down 30s..."; sleep 30; cleanup_gpu; sleep 10
    fi

    # ── exp_12–exp_15: vLLM 0.10.0 ───────────────────────────────────
    run_exp "exp_12" "async_scheduling" \
        1024 "bfloat16" "sut_exp12" \
        "+async_scheduling=True: overlap scheduler with GPU (new in vLLM 0.10.0)"

    run_exp "exp_13" "num_scheduler_steps_4" \
        1024 "bfloat16" "sut_exp13" \
        "+num_scheduler_steps=4: 4 decode steps per scheduler call, reduces GIL pressure"

    run_exp "exp_14" "torch_compile_lvl3" \
        1024 "bfloat16" "sut_exp14" \
        "+compilation_config=3: torch.compile level 3 (first batch ~2-5min compile)"

    run_exp "exp_15" "flashinfer_v0.3_retest" \
        1024 "bfloat16" "sut_exp15" \
        "FlashInfer v0.3.0 retest: upgraded in vLLM 0.10.x, previously regressed on 0.8.x"

    # ── restore exp_11 final config as active SUT ─────────────────────
    sut_exp10_final > "${INFERENCE_DIR}/SUT_VLLM.py"
    log "Restored exp_11 final SUT to ${INFERENCE_DIR}/SUT_VLLM.py"

    print_summary

    local csv="${RESULTS_DIR}/results_summary.csv"
    echo "experiment,batch_size,dtype,tokens_per_sec,samples_per_sec,valid,exit_code,duration_s,description" > "${csv}"
    for rf in "${RESULTS_DIR}"/exp_*/result.txt; do
        [[ -f "${rf}" ]] || continue
        local exp bs dt tok samp val ec dur desc
        exp=$(  grep "^Experiment:"  "${rf}" | cut -d: -f2- | xargs)
        bs=$(   grep "^Batch Size:"  "${rf}" | cut -d: -f2- | xargs)
        dt=$(   grep "^Dtype:"       "${rf}" | cut -d: -f2- | xargs)
        tok=$(  grep "^Tokens/sec:"  "${rf}" | cut -d: -f2- | xargs)
        samp=$( grep "^Samples/sec:" "${rf}" | cut -d: -f2- | xargs || echo "N/A")
        val=$(  grep "^Valid:"       "${rf}" | cut -d: -f2- | xargs)
        ec=$(   grep "^Exit Code:"   "${rf}" | cut -d: -f2- | xargs)
        dur=$(  grep "^Duration:"    "${rf}" | cut -d: -f2- | tr -d 's' | xargs)
        desc=$( grep "^Description:" "${rf}" | cut -d: -f2- | xargs)
        echo "${exp},${bs},${dt},${tok},${samp},${val},${ec},${dur},\"${desc}\"" >> "${csv}"
    done
    log "CSV written to: ${csv}"
}

main "$@"