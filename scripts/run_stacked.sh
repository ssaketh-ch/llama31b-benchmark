#!/bin/bash
# =============================================================================
# vLLM 0.17.1 V1 Engine — MLPerf stacked Study
# Llama-3.1-8B-Instruct on H200 MIG 71GB slice
# Offline Scenario — ONE variable at a time vs fixed baseline
#
# ─── vLLM 0.17.1 V1 API FACTS ────────────────────────────────────────────────
# - generate() requires TokensPrompt(prompt_token_ids=...) wrapper
# - compilation_config= accepts bare int 0/1/2/3
# - async scheduling ON by default; disable with disable_async_output_proc=True
# - num_scheduler_steps REMOVED in V1
# - async_scheduling kwarg REMOVED in V1
# - attention_backend= is a first-class LLM() kwarg (not env var)
# - speculative_config= dict for spec decode; V1 ONLY supports ngram method
# - enable_prefix_caching= default is True in V1 (APC always on)
# - disable_cascade_attention= controls cascade attention (new V1 feature)
# - max_num_batched_tokens default on H100/H200 >=70GB is 16384
# - skip_tokenizer_init=True valid V1 performance kwarg
#
# ─── BASELINE CONFIG (exp_00) ─────────────────────────────────────────────────
#   quantization="fp8"                  always on
#   dtype="float16"                     required with fp8 quant
#   tensor_parallel_size=1
#   batch_size=1024
#   max_model_len=2668                  dataset max 2540 + 128 output
#   max_num_seqs=512
#   gpu_memory_utilization=0.90         safe for 71 GB MIG
#   max_num_batched_tokens=16384        V1 default for H200
#   enable_prefix_caching=True          V1 default; explicit
#   disable_async_output_proc=False     async ON; explicit
#   compilation_config=2                V1 default O2; explicit
#   enforce_eager=False                 CUDA graphs ON
#   cpu_offload_gb=0
#   attention_backend="FLASH_ATTN"      FA3 on H200; explicit
#   sort by input length                always on
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   always on
#
# ─── PHASE A — Core engine knobs ─────────────────────────────────────────────
#   exp_A1: gmu 0.90 -> 0.95
#   exp_A2: max_num_batched_tokens 16384 -> 8192
#   exp_A3: max_num_batched_tokens 16384 -> 32768
#   exp_A4: max_num_seqs 512 -> 1024
#   exp_A5: block_size -> 32 (default 16)
#   exp_A6: enable_prefix_caching False (disable APC)
#   exp_A7: skip_tokenizer_init=True
#
# ─── PHASE B — Compilation and attention ─────────────────────────────────────
#   exp_B1: compilation_config=3 (O3 aggressive)
#   exp_B2: compilation_config=0 (O0 no compile)
#   exp_B3: disable_async_output_proc=True (measure async output cost)
#   exp_B4: attention_backend="FLASHINFER"
#   exp_B5: disable_cascade_attention=False (enable cascade heuristic)
#
# ─── PHASE C — ngram speculative decoding ────────────────────────────────────
#   exp_C1: ngram 3 tokens, lookup_max=3, gmu=0.78
#   exp_C2: ngram 5 tokens, lookup_max=4, gmu=0.78
#   exp_C3: ngram 5 tokens, lookup_max=4, gmu=0.82 (OOM probe)
#   exp_C4: ngram 3 tokens, O3 compile, gmu=0.78 (if B1 positive)
#
# ─── PHASE D — Combinations ──────────────────────────────────────────────────
#   exp_D1: best Phase A + best Phase B (no spec decode)
#   exp_D2: best Phase A + best Phase C (with spec decode)
#
# Usage:
#   ./run_stacked.sh              # all experiments
#   ./run_stacked.sh exp_A1       # single experiment
#   ./run_stacked.sh exp_A        # all Phase A
#   ./run_stacked.sh exp_C        # all Phase C
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_DIR="${INFERENCE_DIR:-${SCRIPT_DIR}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${SCRIPT_DIR}/model}"
DATASET_PATH="${DATASET_PATH:-${SCRIPT_DIR}/data}"
GPU_COUNT="${GPU_COUNT:-1}"
RESULTS_BASE="${SCRIPT_DIR}/stacked_results"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RESULTS_DIR="${RESULTS_BASE}/${TIMESTAMP}"
TOTAL_SAMPLE_COUNT=13368
SCENARIO="Offline"
USER_CONF="${INFERENCE_DIR}/user.conf"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:////${SCRIPT_DIR}/mlflow.db}"
MLFLOW_EXPERIMENT_NAME="llama3.1-8b_stacked_v17-${TIMESTAMP}"
export GIT_PYTHON_REFRESH=quiet

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
log()     { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"; }
success() { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓${NC} $*"; }
warn()    { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠${NC} $*"; }
error()   { echo -e "${RED}[$(date '+%H:%M:%S')] ✗${NC} $*"; }

cleanup_gpu() {
    log "Resetting GPU state..."
    set +e
    pkill -f "vllm" 2>/dev/null; sleep 2
    pkill -f "SUT_VLLM" 2>/dev/null; sleep 2
    pkill -f "main.py" 2>/dev/null; sleep 2
    nvidia-smi --gpu-reset 2>/dev/null
    sleep 10
    set -e
}

capture_system_info() {
    local run_dir="$1"; mkdir -p "${run_dir}/system"
    set +e
    nvidia-smi > "${run_dir}/system/nvidia-smi.txt" 2>&1
    nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv > "${run_dir}/system/gpu_stats.csv" 2>&1
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
    local logfile="$1"; set +e
    if   grep -q "out of memory\|CUBLAS_STATUS_ALLOC_FAILED\|CUDA out of memory" "${logfile}" 2>/dev/null; then echo "OOM"
    elif grep -q "illegal memory access"                  "${logfile}" 2>/dev/null; then echo "ILLEGAL_MEMORY_ACCESS"
    elif grep -q "Segmentation fault\|SIGSEGV"            "${logfile}" 2>/dev/null; then echo "SEGFAULT"
    elif grep -q "EngineCore\|EngineDeadError"            "${logfile}" 2>/dev/null; then echo "ENGINE_CORE_FATAL"
    elif grep -q "NotImplementedError.*[Ss]pec"           "${logfile}" 2>/dev/null; then echo "SPEC_NOT_SUPPORTED"
    elif grep -q "Timeout\|timed out"                     "${logfile}" 2>/dev/null; then echo "TIMEOUT"
    elif grep -q "compilation\|torch.compile\|inductor"   "${logfile}" 2>/dev/null; then echo "COMPILE_FAILED"
    elif grep -q "RuntimeError"                           "${logfile}" 2>/dev/null; then echo "RUNTIME_ERROR"
    else echo "UNKNOWN"; fi
    set -e
}

run_benchmark() {
    local run_dir="$1" batch_size="$2" dtype="$3" desc="$4" exp_id="$5"
    mkdir -p "${run_dir}/system" "${run_dir}/mlperf_logs"
    log "Running: ${exp_id}  batch=${batch_size}"

    set +e
    nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader > "${run_dir}/system/gpu_pre_run.txt" 2>&1
    set -e

    local exit_code=0
    local start_time; start_time=$(date +%s)
    cd "${INFERENCE_DIR}"
    set +e
    timeout 1800 python -u - 2>&1 <<PYCODE | tee "${run_dir}/run.log"
import mlflow, subprocess, os, sys, re, time
sys.path.insert(0, '.')

def parse_summary_file(path):
    metrics = {}
    try:
        with open(path) as f:
            for line in f:
                m = re.match(r'^([\w\s/]+):\s*([\d.]+)', line.strip())
                if m:
                    key = m.group(1).strip().replace(' ', '_').lower()
                    try:    metrics[key] = float(m.group(2))
                    except ValueError: metrics[key] = m.group(2)
    except Exception as e:
        metrics['parse_error'] = str(e)
    return metrics

run_name = '${exp_id}'
run_dir  = '${run_dir}'

mlflow.set_tracking_uri('${MLFLOW_TRACKING_URI}')
mlflow.set_experiment('${MLFLOW_EXPERIMENT_NAME}')
start_ts = time.time()
mlflow.start_run(run_name=run_name)
try:
    mlflow.log_param('exp_id',               '${exp_id}')
    mlflow.log_param('batch_size',           ${batch_size})
    mlflow.log_param('dtype',                '${dtype}')
    mlflow.log_param('description',          '${desc}')
    mlflow.log_param('scenario',             '${SCENARIO}')
    mlflow.log_param('total_sample_count',   ${TOTAL_SAMPLE_COUNT})
    mlflow.log_param('gpu_count',            ${GPU_COUNT})
    mlflow.log_param('tensor_parallel_size', ${GPU_COUNT})
    mlflow.log_param('model_path',           '${CHECKPOINT_PATH}')
    mlflow.log_param('dataset_path',         '${DATASET_PATH}')
    mlflow.log_param('user_conf',            '${USER_CONF}')
    mlflow.log_param('stacked_type',        'isolated')
    mlflow.log_param('vllm_target_version',  '0.17.1')
    mlflow.log_param('engine',               'V1')
    mlflow.log_param('gpu_slice_gb',         71)

    def safe_ver(pkg):
        try: return __import__(pkg).__version__
        except: return 'unavailable'
    mlflow.set_tag('vllm_version',         safe_ver('vllm'))
    mlflow.set_tag('torch_version',        safe_ver('torch'))
    mlflow.set_tag('transformers_version', safe_ver('transformers'))
    try:
        gpu  = subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader,nounits']).decode().strip().split('\n')[0]
        cpu  = next((l.split(':')[1].strip() for l in subprocess.check_output(['lscpu']).decode().split('\n') if 'Model name' in l), 'unknown')
        drv  = subprocess.check_output(['nvidia-smi']).decode().split('\n')[2].strip()
        host = subprocess.check_output(['hostname']).decode().strip()
        mlflow.set_tag('gpu_model', gpu); mlflow.set_tag('cpu_model', cpu)
        mlflow.set_tag('nvidia_driver', drv); mlflow.set_tag('host', host)
    except Exception as e:
        mlflow.set_tag('hw_tag_error', str(e))

    mlflow.set_tag('exp_id', '${exp_id}')
    mlflow.set_tag('dtype',  '${dtype}')
    mlflow.set_tag('run_type', 'baseline' if '${exp_id}' == 'exp_00' else 'stacked')
    mlflow.set_tag('vllm_engine', 'V1')

    try:
        pg = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,temperature.gpu,power.draw,utilization.gpu','--format=csv,noheader,nounits']).decode().strip().split('\n')[0].split(',')
        for k, v in zip(['mem_used_mb','mem_free_mb','temp_c','power_w','util_gpu_pct'], pg):
            try: mlflow.log_metric(f'gpu_pre_{k}', float(v.strip()))
            except: pass
    except: pass

    bench_cmd = [
        'python','main.py',
        '--scenario',            '${SCENARIO}',
        '--model-path',          '${CHECKPOINT_PATH}',
        '--batch-size',          '${batch_size}',
        '--dtype',               '${dtype}',
        '--user-conf',           '${USER_CONF}',
        '--total-sample-count',  '${TOTAL_SAMPLE_COUNT}',
        '--dataset-path',        '${DATASET_PATH}',
        '--output-log-dir',      os.path.join(run_dir, 'mlperf_logs'),
        '--tensor-parallel-size','${GPU_COUNT}',
        '--vllm',
    ]
    print(f"[bench] {' '.join(bench_cmd)}", flush=True)
    proc = subprocess.run(bench_cmd, check=False)
    mlflow.log_param('bench_exit_code', proc.returncode)

    try:
        pg2 = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.sm,clocks.mem','--format=csv']).decode()
        with open(os.path.join(run_dir, 'system', 'gpu_post_run.csv'), 'w') as f: f.write(pg2)
        lines2 = [l.strip() for l in pg2.strip().split('\n') if l.strip()]
        if len(lines2) >= 2:
            hdrs = [h.strip().lower().replace(' ','_').replace('.','_') for h in lines2[0].split(',')]
            vals = [v.strip() for v in lines2[1].split(',')]
            for h, v in zip(hdrs, vals):
                try: mlflow.log_metric(f'gpu_post_{h}', float(''.join(c for c in v if c in '0123456789.')))
                except: pass
    except Exception as e:
        mlflow.set_tag('gpu_post_run_error', str(e))

    summary_candidates = [
        os.path.join(run_dir, 'mlperf_logs', 'mlperf_log_summary.txt'),
        os.path.join('${INFERENCE_DIR}', 'mlperf_log_summary.txt'),
    ]
    summary_path = next((p for p in summary_candidates if os.path.exists(p)), None)
    if summary_path:
        sm = parse_summary_file(summary_path)
        for k, v in sm.items():
            if isinstance(v, (int, float)): mlflow.log_metric(f'mlperf_{k}', v)
            else: mlflow.log_param(f'mlperf_{k}', str(v))
    else:
        mlflow.set_tag('summary_missing', 'true')

    mlflow.log_metric('duration_sec', time.time() - start_ts)

    for subdir, ap in [(os.path.join(run_dir,'system'),'system'),(os.path.join(run_dir,'mlperf_logs'),'mlperf_logs')]:
        if os.path.isdir(subdir): mlflow.log_artifacts(subdir, artifact_path=ap)
    for fname in ('run.log','SUT_VLLM.py','result.txt','experiment_meta.txt'):
        fpath = os.path.join(run_dir, fname)
        if os.path.exists(fpath): mlflow.log_artifact(fpath)
finally:
    mlflow.end_run()
PYCODE
    exit_code="${PIPESTATUS[0]}"; set -e
    local end_time; end_time=$(date +%s)
    local duration=$(( end_time - start_time ))
    set +e
    nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader > "${run_dir}/system/gpu_post_run.txt" 2>&1
    cp "${INFERENCE_DIR}"/mlperf_log_* "${run_dir}/mlperf_logs/" 2>/dev/null
    local tokens samples valid
    tokens=$(grep  "Tokens per second:"  "${run_dir}/run.log" 2>/dev/null | tail -1 | awk '{print $NF}')
    samples=$(grep "Samples per second:" "${run_dir}/run.log" 2>/dev/null | tail -1 | awk '{print $NF}')
    valid=$(grep   "Result is"           "${run_dir}/run.log" 2>/dev/null | tail -1 | awk '{print $NF}')
    set -e
    tokens="${tokens:-FAILED}"; samples="${samples:-FAILED}"; valid="${valid:-FAILED}"
    cat > "${run_dir}/result.txt" <<RESEOF
Experiment:   ${exp_id}
Description:  ${desc}
Batch Size:   ${batch_size}
Dtype:        ${dtype}
Tokens/sec:   ${tokens}
Samples/sec:  ${samples}
Valid:        ${valid}
Duration:     ${duration}s
Exit Code:    ${exit_code}
RESEOF
    if [[ "${tokens}" != "FAILED" ]]; then
        success "${exp_id}: ${tokens} tok/s  [${valid}]  (${duration}s)"
    else
        local err_type; err_type=$(classify_error "${run_dir}/run.log")
        echo "Error Type:   ${err_type}" >> "${run_dir}/result.txt"
        error "${exp_id}: FAILED (${err_type})  (${duration}s)"
        case "${err_type}" in
            OOM)                warn "OOM — lower gpu_memory_utilization or max_num_seqs" ;;
            SPEC_NOT_SUPPORTED) warn "Spec decode method unsupported — only ngram valid in V1" ;;
            ENGINE_CORE_FATAL)  warn "Engine fatal — flag incompatible with V1" ;;
            COMPILE_FAILED)     warn "Compile failed — try compilation_config=2" ;;
        esac
    fi
}

run_exp() {
    local exp_id="$1" exp_name="$2" batch_size="$3" dtype="$4" sut_func="$5" desc="$6"
    case "${TARGET}" in
        all) ;;
        exp_A|exp_B|exp_C|exp_D) [[ "${exp_id}" == ${TARGET}* ]] || return 0 ;;
        *) [[ "${TARGET}" == "${exp_id}" ]] || return 0 ;;
    esac
    local run_dir="${RESULTS_DIR}/${exp_id}_${exp_name}"
    mkdir -p "${run_dir}/system" "${run_dir}/mlperf_logs"
    log "========================================================"
    log "  ${exp_id}: ${exp_name}"
    log "  ${desc}"
    log "========================================================"
    ${sut_func} > "${run_dir}/SUT_VLLM.py"
    cp "${run_dir}/SUT_VLLM.py" "${INFERENCE_DIR}/SUT_VLLM.py"
    cat > "${run_dir}/experiment_meta.txt" <<METAEOF
exp_id:       ${exp_id}
description:  ${desc}
batch_size:   ${batch_size}
dtype:        ${dtype}
timestamp:    $(date -u +%Y-%m-%dT%H:%M:%SZ)
sut_func:     ${sut_func}
vllm_version: 0.17.1
engine:       V1
gpu_slice_gb: 71
METAEOF
    capture_system_info "${run_dir}"
    run_benchmark "${run_dir}" "${batch_size}" "${dtype}" "${desc}" "${exp_id}"
    log "Cooling down 30s..."; sleep 30; cleanup_gpu; sleep 10
}

# =============================================================================
# SUT generator — vLLM 0.17.1 V1 API
# All SUTs use TokensPrompt wrapper (V1 API — raw list of lists removed)
# =============================================================================
_emit_sut_v17() {
    local header_comment="$1" load_model_body="$2"
    cat <<SUTEOF
${header_comment}
import array, logging, os, queue, threading, time
import torch
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
from vllm.inputs import TokensPrompt
import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-SUT")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision("high")


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
        log.info("Loading model (vLLM 0.17.1 V1)...")
${load_model_body}
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries, daemon=True)
            worker.start(); self.worker_threads[j] = worker

    def stop(self):
        for _ in range(self.num_workers): self.query_queue.put(None)
        for worker in self.worker_threads: worker.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None: break
            query_ids = [q.index for q in qitem]
            tik1 = time.time()
            # vLLM V1 API: TokensPrompt wrapper required; raw list of lists removed
            prompts = [TokensPrompt(prompt_token_ids=self.data_object.input_ids[q.index])
                       for q in qitem]
            tik2 = time.time()
            outputs = self.model.generate(prompts, sampling_params=self.sampling_params)
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            tik3 = time.time()
            processed_output = self.data_object.postProcess(pred_output_tokens,
                                                            query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)])
            tok = time.time()
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")
                log.info(f"\tBatchMaker: {tik2-tik1:.4f}s | Inference: {tik3-tik2:.4f}s | Post: {tok-tik3:.4f}s | Total: {tok-tik1:.4f}s")

    def issue_queries(self, query_samples):
        log.info(f"IssueQuery: {len(query_samples)} samples")
        query_samples = sorted(query_samples,
                               key=lambda q: len(self.data_object.input_ids[q.index]))
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
SUTEOF
}

# ── BASELINE ──────────────────────────────────────────────────────────────────
sut_exp00() { _emit_sut_v17 \
'# EXP_00: BASELINE — vLLM 0.17.1 V1, 71GB MIG
# Memory: gmu=0.82 → ~58GB to vLLM | FP8 weights ~8GB | KV cache ~45GB
# V1 defaults made explicit: APC=True, asyncOut=True, O2, FLASH_ATTN' \
'        self.model = LLM(
            self.model_path,
            dtype="float16",                     # fp8 quant requires float16
            quantization="fp8",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,         # BASELINE — safe for 71GB MIG
            max_model_len=2668,
            max_num_seqs=512,
            max_num_batched_tokens=16384,        # V1 H200 default; explicit
            enable_prefix_caching=True,          # V1 APC default; explicit
            disable_async_output_proc=False,     # async output ON; explicit
            compilation_config=2,                # V1 O2 default; explicit
            enforce_eager=False,
            cpu_offload_gb=0,
            attention_backend="FLASH_ATTN",      # FA3 on H200; explicit
        )'; }

# ── PHASE A ───────────────────────────────────────────────────────────────────
sut_expA1() { _emit_sut_v17 \
'# EXP_A1: +gpu_memory_utilization=0.95
# CHANGE: gmu 0.90->0.95. ~4GB more for KV cache. Risk: activation OOM at BS=1024.' \
'        self.model = LLM(
            self.model_path,
            dtype="float16", quantization="fp8", tensor_parallel_size=1,
            gpu_memory_utilization=0.95,         # CHANGE: baseline=0.90
            max_model_len=2668, max_num_seqs=512, max_num_batched_tokens=16384,
            enable_prefix_caching=True, disable_async_output_proc=False,
            compilation_config=2, enforce_eager=False, cpu_offload_gb=0,
            attention_backend="FLASH_ATTN",
        )'; }

sut_expA2() { _emit_sut_v17 \
'# EXP_A2: +max_num_batched_tokens=8192
# CHANGE: 16384->8192. Halves per-step token budget. May reduce prefill/decode interference.' \
'        self.model = LLM(
            self.model_path,
            dtype="float16", quantization="fp8", tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668, max_num_seqs=512,
            max_num_batched_tokens=8192,         # CHANGE: baseline=16384
            enable_prefix_caching=True, disable_async_output_proc=False,
            compilation_config=2, enforce_eager=False, cpu_offload_gb=0,
            attention_backend="FLASH_ATTN",
        )'; }

sut_expA3() { _emit_sut_v17 \
'# EXP_A3: +max_num_batched_tokens=32768
# CHANGE: 16384->32768. More tokens per step; better prefill arithmetic intensity.' \
'        self.model = LLM(
            self.model_path,
            dtype="float16", quantization="fp8", tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668, max_num_seqs=512,
            max_num_batched_tokens=32768,        # CHANGE: baseline=16384
            enable_prefix_caching=True, disable_async_output_proc=False,
            compilation_config=2, enforce_eager=False, cpu_offload_gb=0,
            attention_backend="FLASH_ATTN",
        )'; }

sut_expA4() { _emit_sut_v17 \
'# EXP_A4: +max_num_seqs=1024
# CHANGE: 512->1024 concurrent sequences. FP8 KV at gmu=0.95 fits ~1024 seqs.
# Risk: tight on 71GB. OOM classifier will catch it.' \
'        self.model = LLM(
            self.model_path,
            dtype="float16", quantization="fp8", tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668,
            max_num_seqs=1024,                   # CHANGE: baseline=512
            max_num_batched_tokens=16384,
            enable_prefix_caching=True, disable_async_output_proc=False,
            compilation_config=2, enforce_eager=False, cpu_offload_gb=0,
            attention_backend="FLASH_ATTN",
        )'; }

sut_expA5() { _emit_sut_v17 \
'# EXP_A5: +block_size=32
# CHANGE: default 16->32. Larger KV blocks; better memory locality for longer seqs.
# Downside: more internal fragmentation for short sequences.' \
'        self.model = LLM(
            self.model_path,
            dtype="float16", quantization="fp8", tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668, max_num_seqs=512,
            max_num_batched_tokens=16384,
            block_size=32,                       # CHANGE: baseline=default(16)
            enable_prefix_caching=True, disable_async_output_proc=False,
            compilation_config=2, enforce_eager=False, cpu_offload_gb=0,
            attention_backend="FLASH_ATTN",
        )'; }

sut_expA6() { _emit_sut_v17 \
'# EXP_A6: -enable_prefix_caching=False (disable APC)
# CHANGE: True->False. V1 has APC on by default. CNN/DM shares instruction prefix
# across all 13368 prompts — disabling measures the intra-batch caching value.' \
'        self.model = LLM(
            self.model_path,
            dtype="float16", quantization="fp8", tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668, max_num_seqs=512, max_num_batched_tokens=16384,
            enable_prefix_caching=False,         # CHANGE: baseline=True
            disable_async_output_proc=False,
            compilation_config=2, enforce_eager=False, cpu_offload_gb=0,
            attention_backend="FLASH_ATTN",
        )'; }

sut_expA7() { _emit_sut_v17 \
'# EXP_A7: +skip_tokenizer_init=True
# CHANGE: not set->True. Skips tokeniser init; valid as dataset is pre-tokenised.
# Reduces CPU overhead; small startup speedup.' \
'        self.model = LLM(
            self.model_path,
            dtype="float16", quantization="fp8", tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668, max_num_seqs=512, max_num_batched_tokens=16384,
            enable_prefix_caching=True, disable_async_output_proc=False,
            compilation_config=2, enforce_eager=False, cpu_offload_gb=0,
            attention_backend="FLASH_ATTN",
            skip_tokenizer_init=True,            # CHANGE: baseline=not set
        )'; }

# ── PHASE B ───────────────────────────────────────────────────────────────────
sut_expB1() { _emit_sut_v17 \
'# EXP_B1: +compilation_config=3 (O3 aggressive)
# CHANGE: 2->3. Per docs: "currently equal to O2, may add experimental opts in future."
# Establishes whether O3 differs from O2 on 0.17.1 in practice.' \
'        self.model = LLM(
            self.model_path,
            dtype="float16", quantization="fp8", tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668, max_num_seqs=512, max_num_batched_tokens=16384,
            enable_prefix_caching=True, disable_async_output_proc=False,
            compilation_config=3,                # CHANGE: baseline=2
            enforce_eager=False, cpu_offload_gb=0,
            attention_backend="FLASH_ATTN",
        )'; }

sut_expB2() { _emit_sut_v17 \
'# EXP_B2: +compilation_config=0 (O0 — no compile, baseline for compile overhead)
# CHANGE: 2->0. Disables all torch.compile and CUDA graph capture.
# Expected large regression — quantifies the combined value of O2 compilation.' \
'        self.model = LLM(
            self.model_path,
            dtype="float16", quantization="fp8", tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668, max_num_seqs=512, max_num_batched_tokens=16384,
            enable_prefix_caching=True, disable_async_output_proc=False,
            compilation_config=0,                # CHANGE: baseline=2 (O0=no compile)
            enforce_eager=True, cpu_offload_gb=0,
            attention_backend="FLASH_ATTN",
        )'; }

sut_expB3() { _emit_sut_v17 \
'# EXP_B3: +disable_async_output_proc=True (serialise output processing)
# CHANGE: False->True. V1 default runs output processing async with next GPU step.
# Measures the async output processing benefit for this workload.' \
'        self.model = LLM(
            self.model_path,
            dtype="float16", quantization="fp8", tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668, max_num_seqs=512, max_num_batched_tokens=16384,
            enable_prefix_caching=True,
            disable_async_output_proc=True,      # CHANGE: baseline=False
            compilation_config=2, enforce_eager=False, cpu_offload_gb=0,
            attention_backend="FLASH_ATTN",
        )'; }

sut_expB4() { _emit_sut_v17 \
'# EXP_B4: +attention_backend="FLASHINFER"
# CHANGE: FLASH_ATTN->FLASHINFER. In V1 this is a first-class LLM() kwarg.
# Isolated measurement to confirm the direction.' \
'        self.model = LLM(
            self.model_path,
            dtype="float16", quantization="fp8", tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668, max_num_seqs=512, max_num_batched_tokens=16384,
            enable_prefix_caching=True, disable_async_output_proc=False,
            compilation_config=2, enforce_eager=False, cpu_offload_gb=0,
            attention_backend="FLASHINFER",      # CHANGE: baseline=FLASH_ATTN
        )'; }

sut_expB5() { _emit_sut_v17 \
'# EXP_B5: +disable_cascade_attention=False (enable cascade attention heuristic)
# CHANGE: not set (default True=disabled)->False=enabled.
# V1 cascade attention uses hierarchical computation for long sequences / large batches.
# Will only activate when the built-in heuristic deems it beneficial.' \
'        self.model = LLM(
            self.model_path,
            dtype="float16", quantization="fp8", tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668, max_num_seqs=512, max_num_batched_tokens=16384,
            enable_prefix_caching=True, disable_async_output_proc=False,
            compilation_config=2, enforce_eager=False, cpu_offload_gb=0,
            attention_backend="FLASH_ATTN",
            disable_cascade_attention=False,     # CHANGE: default=True(disabled)
        )'; }

# ── PHASE C — ngram speculative decoding ──────────────────────────────────────
# V1 ONLY supports ngram spec decode. Draft model raises NotImplementedError.
# CNN/DailyMail is input-grounded → high ngram acceptance rate expected.
# Research shows up to 2.8x speedup on CNN/DM vs 1.5x on ShareGPT with ngram.
# gpu_memory_utilization reduced to 0.78 for spec decode verification buffers.

sut_expC1() { _emit_sut_v17 \
'# EXP_C1: ngram spec decode — 3 tokens, lookup_max=3 (conservative probe)
# gmu=0.78 for spec decode buffer headroom on 71GB MIG.' \
'        self.model = LLM(
            self.model_path,
            dtype="float16", quantization="fp8", tensor_parallel_size=1,
            gpu_memory_utilization=0.78,         # reduced for spec decode buffers
            max_model_len=2668, max_num_seqs=512, max_num_batched_tokens=16384,
            enable_prefix_caching=True, disable_async_output_proc=False,
            compilation_config=2, enforce_eager=False, cpu_offload_gb=0,
            attention_backend="FLASH_ATTN",
            speculative_config={               # CHANGE: ngram spec decode
                "method": "ngram",
                "num_speculative_tokens": 3,
                "ngram_prompt_lookup_max": 3,
                "ngram_prompt_lookup_min": 1,
            },
        )'; }

sut_expC2() { _emit_sut_v17 \
'# EXP_C2: ngram spec decode — 5 tokens, lookup_max=4 (moderate)
# More aggressive; expects higher throughput if acceptance rate stays above ~0.7.' \
'        self.model = LLM(
            self.model_path,
            dtype="float16", quantization="fp8", tensor_parallel_size=1,
            gpu_memory_utilization=0.78,
            max_model_len=2668, max_num_seqs=512, max_num_batched_tokens=16384,
            enable_prefix_caching=True, disable_async_output_proc=False,
            compilation_config=2, enforce_eager=False, cpu_offload_gb=0,
            attention_backend="FLASH_ATTN",
            speculative_config={
                "method": "ngram",
                "num_speculative_tokens": 5,
                "ngram_prompt_lookup_max": 4,
                "ngram_prompt_lookup_min": 1,
            },
        )'; }

sut_expC3() { _emit_sut_v17 \
'# EXP_C3: ngram 5 tokens, lookup_max=4, gmu=0.82 (OOM probe)
# Same spec config as C2 but baseline gmu. Tests if 0.78 was unnecessarily conservative.' \
'        self.model = LLM(
            self.model_path,
            dtype="float16", quantization="fp8", tensor_parallel_size=1,
            gpu_memory_utilization=0.82,         # back to baseline gmu — OOM probe
            max_model_len=2668, max_num_seqs=512, max_num_batched_tokens=16384,
            enable_prefix_caching=True, disable_async_output_proc=False,
            compilation_config=2, enforce_eager=False, cpu_offload_gb=0,
            attention_backend="FLASH_ATTN",
            speculative_config={
                "method": "ngram",
                "num_speculative_tokens": 5,
                "ngram_prompt_lookup_max": 4,
                "ngram_prompt_lookup_min": 1,
            },
        )'; }

sut_expC4() { _emit_sut_v17 \
'# EXP_C4: ngram 3 tokens + O3 compile (run only if B1 positive)
# Tests whether more aggressively fused verification kernels amplify spec decode gain.' \
'        self.model = LLM(
            self.model_path,
            dtype="float16", quantization="fp8", tensor_parallel_size=1,
            gpu_memory_utilization=0.78,
            max_model_len=2668, max_num_seqs=512, max_num_batched_tokens=16384,
            enable_prefix_caching=True, disable_async_output_proc=False,
            compilation_config=3,                # from exp_B1 (if positive)
            enforce_eager=False, cpu_offload_gb=0,
            attention_backend="FLASH_ATTN",
            speculative_config={
                "method": "ngram",
                "num_speculative_tokens": 3,
                "ngram_prompt_lookup_max": 3,
                "ngram_prompt_lookup_min": 1,
            },
        )'; }

# ── PHASE D — Combinations ────────────────────────────────────────────────────
sut_expD1() { _emit_sut_v17 \
'# EXP_D1: Best Phase A + best Phase B (no spec decode)
# TEMPLATE — edit before running. Currently: A1(gmu=0.95) + A4(seqs=1024) + B1(O3).' \
'        # !! VERIFY A+B WINNERS BEFORE RUNNING — EDIT THIS SUT !!
        self.model = LLM(
            self.model_path,
            dtype="float16", quantization="fp8", tensor_parallel_size=1,
            gpu_memory_utilization=0.95,         # from exp_A1 (if positive)
            max_model_len=2668,
            max_num_seqs=1024,                   # from exp_A4 (if positive)
            max_num_batched_tokens=16384,
            enable_prefix_caching=True, disable_async_output_proc=False,
            compilation_config=3,                # from exp_B1 (if positive)
            enforce_eager=False, cpu_offload_gb=0,
            attention_backend="FLASH_ATTN",
        )'; }

sut_expD2() { _emit_sut_v17 \
'# EXP_D2: Best Phase A + best Phase C (with spec decode)
# TEMPLATE — edit before running. gmu=0.78 for spec decode buffers regardless.' \
'        # !! VERIFY A+C WINNERS BEFORE RUNNING — EDIT THIS SUT to suit your needs.
        self.model = LLM(
            self.model_path,
            dtype="float16", quantization="fp8", tensor_parallel_size=1,
            gpu_memory_utilization=0.78,         # conservative for spec decode
            max_model_len=2668, max_num_seqs=512, max_num_batched_tokens=16384,
            enable_prefix_caching=True, disable_async_output_proc=False,
            compilation_config=2, enforce_eager=False, cpu_offload_gb=0,
            attention_backend="FLASH_ATTN",
            speculative_config={               # from best of C1/C2
                "method": "ngram",
                "num_speculative_tokens": 5,
                "ngram_prompt_lookup_max": 4,
                "ngram_prompt_lookup_min": 1,
            },
        )'; }

# =============================================================================
# SUMMARY + CSV
# =============================================================================
print_final_summary() {
    log "========================================================"
    log "STACKED STUDY SUMMARY — vLLM 0.17.1 V1, 71GB MIG"
    log "========================================================"
    local baseline_tok="N/A"
    local baseline_file; baseline_file=$(find "${RESULTS_DIR}" -name "result.txt" -path "*/exp_00_*" | head -1)
    [[ -f "${baseline_file}" ]] && baseline_tok=$(grep "^Tokens/sec:" "${baseline_file}" | cut -d: -f2- | xargs)

    printf "\n%-44s %-12s %-10s %-10s %-8s\n" "Experiment" "Tokens/sec" "vs Baseline" "Valid" "Status"
    printf "%-44s %-12s %-10s %-10s %-8s\n"   "----------" "----------" "-----------" "-----" "------"
    for rf in "${RESULTS_DIR}"/exp_*/result.txt; do
        [[ -f "${rf}" ]] || continue
        local exp tok valid status delta
        exp=$(  grep "^Experiment:"  "${rf}" | cut -d: -f2- | xargs)
        tok=$(  grep "^Tokens/sec:"  "${rf}" | cut -d: -f2- | xargs)
        valid=$(grep "^Valid:"       "${rf}" | cut -d: -f2- | xargs)
        status="OK"; [[ "${tok}" == "FAILED" ]] && status="FAILED"
        if [[ "${tok}" != "FAILED" && "${tok}" != "N/A" && "${baseline_tok}" != "N/A" && "${baseline_tok}" != "FAILED" ]]; then
            delta=$(python3 -c "print(f'{(${tok}/${baseline_tok}-1)*100:+.1f}%')" 2>/dev/null || echo "N/A")
        else delta="N/A"; fi
        printf "%-44s %-12s %-10s %-10s %-8s\n" "${exp}" "${tok}" "${delta}" "${valid}" "${status}"
    done

    local csv="${RESULTS_DIR}/stacked_summary_v17.csv"
    echo "experiment,tokens_per_sec,samples_per_sec,vs_baseline_pct,valid,exit_code,duration_s,description" > "${csv}"
    for rf in "${RESULTS_DIR}"/exp_*/result.txt; do
        [[ -f "${rf}" ]] || continue
        local exp tok samp valid ec dur desc delta
        exp=$(  grep "^Experiment:"  "${rf}" | cut -d: -f2- | xargs)
        tok=$(  grep "^Tokens/sec:"  "${rf}" | cut -d: -f2- | xargs)
        samp=$( grep "^Samples/sec:" "${rf}" | cut -d: -f2- | xargs 2>/dev/null || echo "N/A")
        valid=$(grep "^Valid:"       "${rf}" | cut -d: -f2- | xargs)
        ec=$(   grep "^Exit Code:"   "${rf}" | cut -d: -f2- | xargs)
        dur=$(  grep "^Duration:"    "${rf}" | cut -d: -f2- | tr -d 's' | xargs)
        desc=$( grep "^Description:" "${rf}" | cut -d: -f2- | xargs)
        if [[ "${tok}" != "FAILED" && "${tok}" != "N/A" && "${baseline_tok}" != "N/A" && "${baseline_tok}" != "FAILED" ]]; then
            delta=$(python3 -c "print(f'{(${tok}/${baseline_tok}-1)*100:+.2f}')" 2>/dev/null || echo "N/A")
        else delta="N/A"; fi
        echo "${exp},${tok},${samp},${delta},${valid},${ec},${dur},\"${desc}\"" >> "${csv}"
    done
    log "Results: ${RESULTS_DIR} | CSV: ${csv} | MLflow: ${MLFLOW_EXPERIMENT_NAME}"
}

# =============================================================================
# MAIN
# =============================================================================
main() {
    TARGET="${1:-all}"
    log "========================================================"
    log "vLLM 0.17.1 V1 stacked — H200 MIG 71GB | TARGET=${TARGET}"
    log "========================================================"
    echo ""
    mkdir -p "${RESULTS_DIR}"
    [[ -d "${INFERENCE_DIR}" ]] || { error "INFERENCE_DIR not found: ${INFERENCE_DIR}"; exit 1; }
    [[ -f "${INFERENCE_DIR}/SUT_VLLM.py" ]] && cp "${INFERENCE_DIR}/SUT_VLLM.py" "${RESULTS_DIR}/SUT_VLLM_original_backup.py"

    run_exp "exp_00" "BASELINE"              1024 "bfloat16" "sut_exp00"  "BASELINE: FP8+BS1024+gmu0.90+len2668+APC+asyncOut+O2+FLASH_ATTN."
    run_exp "exp_A1" "gmu_0.95"              1024 "bfloat16" "sut_expA1"  "+gpu_memory_utilization=0.95."
    run_exp "exp_A2" "batched_tokens_8192"   1024 "bfloat16" "sut_expA2"  "+max_num_batched_tokens=8192 (halved)."
    run_exp "exp_A3" "batched_tokens_32768"  1024 "bfloat16" "sut_expA3"  "+max_num_batched_tokens=32768 (doubled)."
    run_exp "exp_A4" "max_seqs_1024"         1024 "bfloat16" "sut_expA4"  "+max_num_seqs=1024."
    run_exp "exp_A5" "block_size_32"         1024 "bfloat16" "sut_expA5"  "+block_size=32."
    run_exp "exp_A6" "prefix_cache_off"      1024 "bfloat16" "sut_expA6"  "-enable_prefix_caching=False (disable APC)."
    run_exp "exp_A7" "skip_tokenizer"        1024 "bfloat16" "sut_expA7"  "+skip_tokenizer_init=True."
    run_exp "exp_B1" "compile_O3"            1024 "bfloat16" "sut_expB1"  "+compilation_config=3 (O3)."
    run_exp "exp_B2" "compile_O0"            1024 "bfloat16" "sut_expB2"  "+compilation_config=0 (O0 no compile)."
    run_exp "exp_B3" "no_async_output"       1024 "bfloat16" "sut_expB3"  "+disable_async_output_proc=True."
    run_exp "exp_B4" "flashinfer"            1024 "bfloat16" "sut_expB4"  "+attention_backend=FLASHINFER (V1 kwarg)."
    run_exp "exp_B5" "cascade_attention"     1024 "bfloat16" "sut_expB5"  "+disable_cascade_attention=False."
    run_exp "exp_C1" "ngram_3tok_lmax3"      1024 "bfloat16" "sut_expC1"  "+ngram spec: 3 tokens, lookup_max=3, gmu=0.78."
    run_exp "exp_C2" "ngram_5tok_lmax4"      1024 "bfloat16" "sut_expC2"  "+ngram spec: 5 tokens, lookup_max=4, gmu=0.78."
    run_exp "exp_C3" "ngram_5tok_gmu82"      1024 "bfloat16" "sut_expC3"  "+ngram spec: 5 tokens, gmu=0.82 (OOM probe)."
    run_exp "exp_C4" "ngram_3tok_O3"         1024 "bfloat16" "sut_expC4"  "+ngram spec + O3 (run only if B1 positive)."
    run_exp "exp_D1" "best_A_plus_best_B"    1024 "bfloat16" "sut_expD1"  "Best A+B. EDIT SUT FIRST."
    run_exp "exp_D2" "best_A_plus_best_C"    1024 "bfloat16" "sut_expD2"  "Best A+C. EDIT SUT FIRST."

    sut_exp00 > "${INFERENCE_DIR}/SUT_VLLM.py"
    log "Restored BASELINE SUT to ${INFERENCE_DIR}/SUT_VLLM.py"
    print_final_summary
}

main "$@"