#!/bin/bash
# =============================================================================
# vLLM v0.17.1 Isolation Ablation — Llama-3.1-8B on H200 MIG 71GB
# Offline Scenario — ONE change vs production baseline per experiment.
#
# ─── DESIGN PRINCIPLE ────────────────────────────────────────────────────────
# The baseline (exp_00) is the PRODUCTION baseline — exactly what is run in
# practice: LLM(model_path, dtype=dtype, tensor_parallel_size=TP) with no
# extra flags, letting vLLM V1 use all its defaults, at --batch-size 16.
# Each subsequent experiment keeps the same bare LLM() call but overrides
# exactly ONE parameter to measure its individual impact.
#
# ─── PRODUCTION BASELINE (exp_00) ────────────────────────────────────────────
# LLM(model_path, dtype=dtype, tensor_parallel_size=TP)
#   → vLLM V1 defaults apply: prefix_caching=ON, async_output=ON, O2 compile
#     chunked_prefill ON, max_num_batched_tokens = V1 offline default (~16384)
#
# ─── V1 FEATURES TESTED BY DISABLING (baseline has them ON) ─────────────────
#  exp_01   enable_prefix_caching=False     (V1 default: True  → test OFF)
#  exp_02   disable_async_output_proc=True  (V1 default: False → test async OFF)
#  exp_03   compilation_config=O1           (V1 default: O2   → test O1)
#  exp_04   compilation_config=O0           (V1 default: O2   → test no-compile)
#  exp_05   compilation_config=O3           (V1 default: O2   → test O3)
#  exp_06   max_num_batched_tokens=2668      (neutralise chunked prefill = max_model_len)
#  exp_07   async_scheduling=True           (V1 default: False → test experimental ON)
#
# ─── SCHEDULER / MEMORY SWEEP ────────────────────────────────────────────────
#  exp_08   max_num_batched_tokens=8192
#  exp_09   max_num_batched_tokens=65536
#
# ─── QUANTISATION ─────────────────────────────────────────────────────────────
#  exp_10   quantization="fp8"
#  exp_11   kv_cache_dtype="fp8"
#  exp_12   quantization="fp8" + kv_cache_dtype="fp8"
#
# ─── BATCHING & SCHEDULING ────────────────────────────────────────────────────
#  exp_13   batch_size=1024
#  exp_14   batch_size=2048
#  exp_15   sort_by_input_length (at baseline batch=16)
#  exp_16   gpu_memory_utilization=0.95
#
# ─── KERNEL / TOKENISER ───────────────────────────────────────────────────────
#  exp_17   attention_backend="FLASHINFER"
#  exp_18   skip_tokenizer_init=True
#
# ─── REFERENCE ────────────────────────────────────────────────────────────────
#  exp_00b  RIGHT-SIZED KV: production + max_model_len=2668 (shows KV benefit)
#
# Usage:
#   chmod +x run_isloation.sh
#   ./run_isloation.sh
#   TARGET=exp_04 ./run_isloation.sh
#   RUNS_PER_EXP=2 ./run_isloation.sh
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCE_DIR="${INFERENCE_DIR:-${SCRIPT_DIR}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${SCRIPT_DIR}/model}"
DATASET_PATH="${DATASET_PATH:-${SCRIPT_DIR}/data}"
GPU_COUNT="${GPU_COUNT:-1}"
TOTAL_SAMPLE_COUNT=13368
SCENARIO="Offline"
USER_CONF="${INFERENCE_DIR}/user.conf"
RUNS_PER_EXP="${RUNS_PER_EXP:-1}"
RESULTS_BASE="${SCRIPT_DIR}/isolation_results_v17"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RESULTS_DIR="${RESULTS_BASE}/${TIMESTAMP}"
# Default to local SQLite store — no MLflow server required.
# Override: MLFLOW_TRACKING_URI=http://localhost:5000 ./isolation.sh
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:////${SCRIPT_DIR}/mlflow.db}"
MLFLOW_EXPERIMENT_NAME="llama3.1-8b_isolation_v17-${TIMESTAMP}"
# Silence MLflow's git-not-found warning (git is not installed in the container)
export GIT_PYTHON_REFRESH=quiet

# TARGET=exp_04 runs from exp_04 onwards (continue mode, default).
# TARGET=exp_04 RUN_SCOPE=only runs ONLY exp_04.
RUN_SCOPE="${RUN_SCOPE:-continue}"
TARGET_REACHED=0

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
log()     { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"; }
success() { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓${NC} $*"; }
warn()    { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠${NC} $*"; }
error()   { echo -e "${RED}[$(date '+%H:%M:%S')] ✗${NC} $*"; }
info()    { echo -e "${CYAN}[$(date '+%H:%M:%S')] ○${NC} $*"; }

cleanup_gpu() {
    log "Resetting GPU state (30s cooldown)..."
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
    nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,\
utilization.gpu,utilization.memory,temperature.gpu,power.draw \
        --format=csv > "${run_dir}/system/gpu_stats.csv" 2>&1
    nvidia-smi mig -lgip > "${run_dir}/system/mig_topology.txt" 2>&1
    python --version > "${run_dir}/system/packages.txt" 2>&1
    pip show vllm torch transformers >> "${run_dir}/system/packages.txt" 2>&1
    set -e
}

classify_error() {
    local logfile="$1"; set +e
    if   grep -q "out of memory\|CUDA out of memory"    "${logfile}" 2>/dev/null; then echo "OOM"
    elif grep -q "illegal memory access"                "${logfile}" 2>/dev/null; then echo "ILLEGAL_MEM"
    elif grep -q "Segmentation fault"                   "${logfile}" 2>/dev/null; then echo "SEGFAULT"
    elif grep -q "EngineCore\|EngineDeadError"          "${logfile}" 2>/dev/null; then echo "ENGINE_FATAL"
    elif grep -q "torch.compile\|compilation\|inductor" "${logfile}" 2>/dev/null; then echo "COMPILE_FAIL"
    elif grep -q "RuntimeError"                         "${logfile}" 2>/dev/null; then echo "RUNTIME_ERROR"
    else echo "UNKNOWN"; fi
    set -e
}

run_single() {
    local exp_id="$1" run_idx="$2" batch_size="$3" dtype="$4" desc="$5"
    local run_dir="${RESULTS_DIR}/${exp_id}/run_${run_idx}"
    mkdir -p "${run_dir}/system" "${run_dir}/mlperf_logs"
    info "  Run ${run_idx}/${RUNS_PER_EXP}: ${exp_id} (BS=${batch_size})"
    capture_system_info "${run_dir}"

    local exit_code=0
    local start_time; start_time=$(date +%s)
    cd "${INFERENCE_DIR}"
    set +e
    timeout 7200 python -u - 2>&1 <<PYCODE | tee "${run_dir}/run.log"
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

run_name = '${exp_id}_run${run_idx}'
run_dir  = '${run_dir}'
exp_id   = '${exp_id}'
run_idx  = ${run_idx}

mlflow.set_tracking_uri('${MLFLOW_TRACKING_URI}')
mlflow.set_experiment('${MLFLOW_EXPERIMENT_NAME}')
start_ts = time.time()
mlflow.start_run(run_name=run_name)
try:
    # ── Parameters ──────────────────────────────────────────────────────────
    mlflow.log_param('exp_id',               exp_id)
    mlflow.log_param('run_index',            run_idx)
    mlflow.log_param('batch_size',           ${batch_size})
    mlflow.log_param('dtype',                '${dtype}')
    mlflow.log_param('description',          '${desc}')
    mlflow.log_param('scenario',             '${SCENARIO}')
    mlflow.log_param('total_sample_count',   ${TOTAL_SAMPLE_COUNT})
    mlflow.log_param('gpu_count',            ${GPU_COUNT})
    mlflow.log_param('tensor_parallel_size', ${GPU_COUNT})
    mlflow.log_param('model_path',           '${CHECKPOINT_PATH}')
    mlflow.log_param('dataset_path',         '${DATASET_PATH}')
    mlflow.log_param('runs_per_exp',         ${RUNS_PER_EXP})
    mlflow.log_param('mode',                 'isolation_performance')

    # ── Version / hardware tags ──────────────────────────────────────────────
    def safe_ver(pkg):
        try: return __import__(pkg).__version__
        except: return 'unavailable'
    mlflow.set_tag('vllm_version',         safe_ver('vllm'))
    mlflow.set_tag('torch_version',        safe_ver('torch'))
    mlflow.set_tag('transformers_version', safe_ver('transformers'))
    try:
        gpu   = subprocess.check_output(
            ['nvidia-smi','--query-gpu=name','--format=csv,noheader,nounits']
        ).decode().strip().split('\n')[0]
        cpu   = next((l.split(':')[1].strip() for l in
                      subprocess.check_output(['lscpu']).decode().split('\n')
                      if 'Model name' in l), 'unknown')
        drv   = subprocess.check_output(['nvidia-smi']).decode().split('\n')[2].strip()
        host  = subprocess.check_output(['hostname']).decode().strip()
        mlflow.set_tag('gpu_model',     gpu)
        mlflow.set_tag('cpu_model',     cpu)
        mlflow.set_tag('nvidia_driver', drv)
        mlflow.set_tag('host',          host)
    except Exception as e:
        mlflow.set_tag('hw_tag_error', str(e))

    mlflow.set_tag('exp_id',    exp_id)
    mlflow.set_tag('run_index', str(run_idx))
    mlflow.set_tag('dtype',     '${dtype}')
    mlflow.set_tag('scenario',  '${SCENARIO}')
    mlflow.set_tag('mode',      'isolation_performance')

    # ── Run type ─────────────────────────────────────────────────────────────
    if exp_id in ('exp_00', 'exp_00b'):
        run_type = 'baseline'
    else:
        run_type = 'isolation'
    mlflow.set_tag('run_type', run_type)

    # ── Argument tags (all bench_cmd args) ───────────────────────────────────
    mlflow.set_tag('arg_scenario',             '${SCENARIO}')
    mlflow.set_tag('arg_model_path',           '${CHECKPOINT_PATH}')
    mlflow.set_tag('arg_batch_size',           '${batch_size}')
    mlflow.set_tag('arg_dtype',                '${dtype}')
    mlflow.set_tag('arg_user_conf',            '${USER_CONF}')
    mlflow.set_tag('arg_total_sample_count',   '${TOTAL_SAMPLE_COUNT}')
    mlflow.set_tag('arg_dataset_path',         '${DATASET_PATH}')
    mlflow.set_tag('arg_output_log_dir',       os.path.join(run_dir, 'mlperf_logs'))
    mlflow.set_tag('arg_tensor_parallel_size', '${GPU_COUNT}')
    mlflow.set_tag('arg_vllm',                 'true')

    # ── Pre-run GPU snapshot ──────────────────────────────────────────────────
    try:
        pg = subprocess.check_output([
            'nvidia-smi','--query-gpu=memory.used,memory.free,temperature.gpu,'
            'power.draw,utilization.gpu','--format=csv,noheader,nounits'
        ]).decode().strip().split('\n')[0].split(',')
        for k, v in zip(['mem_used_mb','mem_free_mb','temp_c','power_w','util_gpu_pct'], pg):
            try: mlflow.log_metric(f'gpu_pre_{k}', float(v.strip()))
            except: pass
    except: pass

    # ── Run benchmark ─────────────────────────────────────────────────────────
    bench_cmd = [
        'python', 'main.py',
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

    # ── Post-run GPU snapshot ─────────────────────────────────────────────────
    try:
        pg2 = subprocess.check_output([
            'nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,'
            'utilization.gpu,utilization.memory,temperature.gpu,power.draw,'
            'clocks.sm,clocks.mem','--format=csv'
        ]).decode()
        with open(os.path.join(run_dir, 'system', 'gpu_post_run.csv'), 'w') as f:
            f.write(pg2)
        lines2 = [l.strip() for l in pg2.strip().split('\n') if l.strip()]
        if len(lines2) >= 2:
            hdrs = [h.strip().lower().replace(' ','_').replace('.','_')
                    for h in lines2[0].split(',')]
            vals = [v.strip() for v in lines2[1].split(',')]
            for h, v in zip(hdrs, vals):
                try: mlflow.log_metric(f'gpu_post_{h}',
                         float(''.join(c for c in v if c in '0123456789.')))
                except: pass
    except Exception as e:
        mlflow.set_tag('gpu_post_run_error', str(e))

    # ── Parse and log mlperf summary ─────────────────────────────────────────
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


    # ── Ensure all logs are present before artifact logging ───────────────
    import glob, shutil
    mlperf_log_files = glob.glob(os.path.join('${INFERENCE_DIR}', 'mlperf_log_*'))
    mlperf_logs_dir = os.path.join(run_dir, 'mlperf_logs')
    os.makedirs(mlperf_logs_dir, exist_ok=True)
    for f in mlperf_log_files:
        try:
            shutil.copy2(f, mlperf_logs_dir)
        except Exception as e:
            mlflow.set_tag('mlperf_log_copy_error', str(e))

    # ── Artifacts ─────────────────────────────────────────────────────────
    for subdir, ap in [
        (os.path.join(run_dir, 'system'),      'system'),
        (os.path.join(run_dir, 'mlperf_logs'), 'mlperf_logs'),
    ]:
        if os.path.isdir(subdir):
            mlflow.log_artifacts(subdir, artifact_path=ap)
    for fname in ('run.log', 'SUT_VLLM.py', 'result.txt', 'experiment_meta.txt'):
        fpath = os.path.join(run_dir, fname)
        if os.path.exists(fpath):
            mlflow.log_artifact(fpath)

finally:
    mlflow.end_run()
PYCODE
    exit_code="${PIPESTATUS[0]}"; set -e
    local end_time; end_time=$(date +%s)
    local duration=$(( end_time - start_time ))
    set +e
    cp "${INFERENCE_DIR}"/mlperf_log_* "${run_dir}/mlperf_logs/" 2>/dev/null
    local tokens samples valid
    tokens=$(grep  "Tokens per second:"  "${run_dir}/run.log" 2>/dev/null | tail -1 | awk '{print $NF}')
    samples=$(grep "Samples per second:" "${run_dir}/run.log" 2>/dev/null | tail -1 | awk '{print $NF}')
    valid=$(grep   "Result is"           "${run_dir}/run.log" 2>/dev/null | tail -1 | awk '{print $NF}')
    set -e
    tokens="${tokens:-FAILED}"; samples="${samples:-FAILED}"; valid="${valid:-FAILED}"
    cat > "${run_dir}/result.txt" <<RESEOF
Experiment:   ${exp_id}
Run:          ${run_idx}/${RUNS_PER_EXP}
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
        success "    run_${run_idx}: ${tokens} tok/s  [${valid}]  (${duration}s)"
    else
        local err_type; err_type=$(classify_error "${run_dir}/run.log")
        error "    run_${run_idx}: FAILED (${err_type})  (${duration}s)"
        case "${err_type}" in
            OOM)         warn "    → OOM: reduce batch_size" ;;
            ENGINE_FATAL)warn "    → Engine fatal: flag incompatible with V1" ;;
            COMPILE_FAIL)warn "    → Compile failed: try lower compilation_config level" ;;
        esac
    fi
    log "    Cooldown 30s..."; sleep 30; cleanup_gpu
}

run_experiment() {
    local exp_id="$1" batch_size="$2" dtype="$3" sut_func="$4" desc="$5"

    # ── TARGET / RUN_SCOPE filtering ─────────────────────────────────────────
    # Default (TARGET=all): run every experiment.
    # TARGET=exp_04 RUN_SCOPE=continue (default): run exp_04 and all after it.
    # TARGET=exp_04 RUN_SCOPE=only: run ONLY exp_04.
    local target="${TARGET:-all}"
    if [[ "${target}" != "all" ]]; then
        if [[ "${RUN_SCOPE}" == "only" ]]; then
            [[ "${target}" == "${exp_id}" ]] || return 0
        else
            if [[ "${TARGET_REACHED}" -eq 0 ]]; then
                [[ "${target}" == "${exp_id}" ]] && TARGET_REACHED=1 || return 0
            fi
        fi
    fi

    local exp_dir="${RESULTS_DIR}/${exp_id}"; mkdir -p "${exp_dir}"
    log "========================================================"
    log "  ${exp_id}: ${desc}"
    log "  Batch=${batch_size} | Dtype=${dtype} | Runs=${RUNS_PER_EXP}"
    log "========================================================"
    ${sut_func} > "${exp_dir}/SUT_VLLM.py"
    cp "${exp_dir}/SUT_VLLM.py" "${INFERENCE_DIR}/SUT_VLLM.py"
    cat > "${exp_dir}/experiment_meta.txt" <<METAEOF
exp_id:       ${exp_id}
description:  ${desc}
batch_size:   ${batch_size}
dtype:        ${dtype}
runs:         ${RUNS_PER_EXP}
timestamp:    $(date -u +%Y-%m-%dT%H:%M:%SZ)
vllm_version: 0.17.1
METAEOF
    for run_idx in $(seq 1 "${RUNS_PER_EXP}"); do
        run_single "${exp_id}" "${run_idx}" "${batch_size}" "${dtype}" "${desc}"
    done
    aggregate_experiment "${exp_id}" "${desc}"
}

aggregate_experiment() {
    local exp_id="$1" desc="$2"
    local exp_dir="${RESULTS_DIR}/${exp_id}"
    python3 - <<PYAGG
import os, statistics
exp_dir = '${exp_dir}'; exp_id = '${exp_id}'; desc = '${desc}'
tok_vals, valid_list = [], []
for run_idx in range(1, ${RUNS_PER_EXP} + 1):
    rfile = os.path.join(exp_dir, f'run_{run_idx}', 'result.txt')
    if not os.path.exists(rfile): continue
    tok_str = valid_str = ''
    with open(rfile) as f:
        for line in f:
            if line.startswith('Tokens/sec:'): tok_str   = line.split(':',1)[1].strip()
            if line.startswith('Valid:'):      valid_str = line.split(':',1)[1].strip()
    if tok_str and tok_str != 'FAILED':
        try: tok_vals.append(float(tok_str)); valid_list.append(valid_str)
        except ValueError: pass
baseline_mean = None
baseline_agg = os.path.join('${RESULTS_DIR}', 'exp_00', 'aggregate.txt')
if os.path.exists(baseline_agg):
    with open(baseline_agg) as f:
        for line in f:
            if line.lower().startswith('mean tok/s'):
                try: baseline_mean = float(line.split(':',1)[1].strip())
                except: pass
if tok_vals:
    mean_tok  = statistics.mean(tok_vals)
    stdev_tok = statistics.stdev(tok_vals) if len(tok_vals) > 1 else 0.0
    cv_pct    = (stdev_tok / mean_tok * 100) if mean_tok else 0.0
    valid_rate= sum(1 for v in valid_list if 'VALID' in v.upper() and 'INVALID' not in v.upper())
    speedup   = f'{(mean_tok / baseline_mean - 1.0) * 100:+.2f}%' if baseline_mean else 'N/A'
    lines = [
        f"Experiment:    {exp_id}", f"Description:   {desc}",
        f"Runs:          {len(tok_vals)}/{${RUNS_PER_EXP}} succeeded",
        f"Mean tok/s:    {mean_tok:.2f}", f"Stdev tok/s:   {stdev_tok:.2f}",
        f"CV%:           {cv_pct:.1f}%",
        f"Min tok/s:     {min(tok_vals):.2f}", f"Max tok/s:     {max(tok_vals):.2f}",
        f"All values:    {[f'{v:.2f}' for v in tok_vals]}",
        f"Valid rate:    {valid_rate}/{len(valid_list)}",
    ]
    if 'exp_00' not in exp_id and speedup != 'N/A':
        lines.append(f"Speedup vs exp_00: {speedup}")
else:
    lines = [f"Experiment:    {exp_id}", f"Description:   {desc}",
             f"Runs:          0/{${RUNS_PER_EXP}} succeeded", f"Mean tok/s:    FAILED"]
txt = "\n".join(lines) + "\n"
with open(os.path.join(exp_dir, 'aggregate.txt'), 'w') as f: f.write(txt)
print(txt)
PYAGG
}

# =============================================================================
# SUT GENERATOR FUNCTIONS
#
# Baseline (exp_00): LLM(model_path, dtype=dtype, tensor_parallel_size=TP)
#   Exactly the original production SUT — vLLM V1 applies its own defaults:
#     enable_prefix_caching=True  (APC always ON in V1)
#     disable_async_output_proc=False  (async output ON)
#     compilation_config=O2  (torch.compile + FULL_AND_PIECEWISE CGs)
#     chunked_prefill=ON  (hardcoded, but controllable via max_num_batched_tokens)
#
# Each experiment: identical to baseline LLM() + exactly ONE explicit override.
# generate() uses TokensPrompt (vLLM V1 API; prompt_token_ids kwarg removed in V1).
# =============================================================================

_emit_sut() {
    local comment="$1" batch_size_default="$2" sort_flag="$3" load_model_body="$4"
    cat <<SUTEOF
${comment}
import array, logging, queue, threading, time
import torch
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
from vllm.inputs import TokensPrompt
import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-SUT")


class SUT:
    def __init__(
        self,
        model_path=None,
        dtype="bfloat16",
        batch_size=None,
        total_sample_count=13368,
        dataset_path=None,
        use_cached_outputs=False,
        workers=1,
        tensor_parallel_size=8,
    ):
        self.model_path = model_path or "meta-llama/Meta-Llama-3.1-8B-Instruct"

        if not batch_size:
            batch_size = ${batch_size_default}
        self.batch_size = batch_size

        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size

        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."

        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path,
            dataset_path=self.dataset_path,
            total_sample_count=total_sample_count,
            dtype=dtype,
        )
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam,
        )

        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1,
            seed=42, max_tokens=128, min_tokens=1,
        )

        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()

        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
${load_model_body}
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        for worker in self.worker_threads:
            worker.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break

            query_ids = [q.index for q in qitem]

            tik1 = time.time()
            # Build TokensPrompt list (vLLM V1 API; prompt_token_ids kwarg removed)
            prompts = [
                TokensPrompt(prompt_token_ids=self.data_object.input_ids[q.index])
                for q in qitem
            ]
            tik2 = time.time()
            outputs = self.model.generate(prompts, sampling_params=self.sampling_params)
            pred_output_tokens = []
            for output in outputs:
                pred_output_tokens.append(list(output.outputs[0].token_ids))
            tik3 = time.time()

            processed_output = self.data_object.postProcess(
                pred_output_tokens,
                query_id_list=query_ids,
            )
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                response = [
                    lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)
                ]
                lg.QuerySamplesComplete(response)

            tok = time.time()
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")
                if tik1:
                    log.info(f"\tBatchMaker time: {tik2 - tik1}")
                    log.info(f"\tInference time: {tik3 - tik2}")
                    log.info(f"\tPostprocess time: {tok - tik3}")
                    log.info(f"\t==== Total time: {tok - tik1}")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self):
        return self.qsl

    def predict(self, **kwargs):
        raise NotImplementedError

    def issue_queries(self, query_samples):
        log.info(f"IssueQuery started with {len(query_samples)} samples")
SUTEOF
    if [[ "${sort_flag}" == "sort" ]]; then
        echo "        query_samples = sorted(query_samples, key=lambda q: len(self.data_object.input_ids[q.index]))"
    fi
    cat <<SUTEOF2
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[: self.batch_size])
            query_samples = query_samples[self.batch_size :]
        log.info(f"IssueQuery done")

    def flush_queries(self):
        pass

    def __del__(self):
        pass

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items():
            log.info(f"  {k}: {v}")


class SUTServer(SUT): pass
SUTEOF2
}

# ── REUSABLE BASELINE LLM BLOCK — production: bare LLM(), vLLM picks defaults ─
_BASELINE_LLM='        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
        )'

# ── exp_00: PRODUCTION BASELINE ─────────────────────────────────────────────
sut_exp00() {
    _emit_sut \
        "# EXP_00: PRODUCTION BASELINE — stock vLLM V1 defaults, no extra flags.
# Matches the real production run: LLM(model_path, dtype, TP) at --batch-size 16.
# vLLM V1 will apply its own defaults: prefix_caching=ON, async_output=ON, O2 compile.
# Every experiment changes exactly ONE thing from this." \
        16 nosort "${_BASELINE_LLM}"
}

# ── exp_00b: RIGHT-SIZED KV (reference) ─────────────────────────────────────
sut_exp00b() {
    _emit_sut \
        "# EXP_00b: RIGHT-SIZED KV — production baseline + max_model_len=2668.
# ONLY CHANGE from exp_00: max_model_len explicitly set to 2668 (input 2540 + output 128).
# Shows the throughput gain from right-sizing the KV cache vs vLLM's 131072 default." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=2668,             # <- CHANGE: right-size KV (vLLM default: 131072)
            max_num_batched_tokens=2668,    # match max_model_len to neutralise chunked prefill
        )'
}

# ── exp_01: -enable_prefix_caching=False (disable V1 default APC) ─────────────
sut_exp01() {
    _emit_sut \
        "# EXP_01: enable_prefix_caching=False (disable V1 default APC)
# ONLY CHANGE from production baseline: enable_prefix_caching explicitly False.
# V1 default: True. MLPerf Offline has zero prefix sharing; tests if APC overhead hurts." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            enable_prefix_caching=False,    # <- CHANGE: V1 default True -> test False
        )'
}

# ── exp_02: disable_async_output_proc=True (disable V1 default async) ─────────
sut_exp02() {
    _emit_sut \
        "# EXP_02: disable_async_output_proc=True (disable V1 default async output)
# ONLY CHANGE from production baseline: disable_async_output_proc explicitly True.
# V1 default: False (async ON). Tests cost of disabling overlapped post-processing." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            async_output=False, # <- CHANGE: V1 default True -> async OFF
        )'
}

# ── exp_03: compilation_config=O1 (below V1 default O2) ──────────────────────
sut_exp03() {
    _emit_sut \
        "# EXP_03: compilation_config=O1 (one level below V1 default O2)
# ONLY CHANGE from production baseline: compilation_config explicitly O1.
# O1: piecewise CUDA graphs only. V1 default is O2 (full+piecewise)." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            compilation_config=CompilationConfig(level=1), # <- CHANGE: V1 default O2 -> O1
        )'
}

# ── exp_04: compilation_config=O0 (no compile vs V1 default O2) ───────────────
sut_exp04() {
    _emit_sut \
        "# EXP_04: compilation_config=O0 (no compile vs V1 default O2)
# ONLY CHANGE from production baseline: compilation_config explicitly O0.
# O0: no torch.compile, no CUDA graphs. Tests full cost of V1 compilation." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            compilation_config=CompilationConfig(level=0), # <- CHANGE: V1 default O2 -> O0
        )'
}

# ── exp_05: compilation_config=O3 (above V1 default O2) ──────────────────────
sut_exp05() {
    _emit_sut \
        "# EXP_05: compilation_config=O3 (above V1 default O2)
# ONLY CHANGE from production baseline: compilation_config explicitly O3.
# O3: max optimisation. Tests if exceeding the V1 default yields further gain." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            compilation_config=CompilationConfig(level=3), # <- CHANGE: V1 default O2 -> O3
        )'
}

# ── exp_06: max_num_batched_tokens=2668 (neutralise chunked prefill) ──────────
sut_exp06() {
    _emit_sut \
        "# EXP_06: max_num_batched_tokens=2668 (neutralise chunked prefill)
# ONLY CHANGE from production baseline: pin max_num_batched_tokens=max_model_len=2668.
# Forces scheduler to process one full sequence per step (V0 behaviour).
# Tests the overhead of V1's chunked prefill scheduler." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=2668,
            max_num_batched_tokens=2668,    # <- CHANGE: neutralise chunked prefill
        )'
}

# ── exp_07: async_scheduling=True (experimental) ─────────────────────────────
sut_exp07() {
    _emit_sut \
        "# EXP_07: async_scheduling=True (experimental, V1 default False)
# ONLY CHANGE from production baseline: enable the experimental async scheduler.
# Overlaps Python scheduling with GPU execution." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            async_scheduling=True,          # <- CHANGE: V1 default False -> True
        )'
}

# ── exp_08: max_num_batched_tokens=8192 ──────────────────────────────────────
sut_exp08() {
    _emit_sut \
        "# EXP_08: max_num_batched_tokens=8192 (right-sized token budget)
# ONLY CHANGE from production baseline: pin max_model_len=2668 + max_num_batched_tokens=8192.
# Tests intermediate scheduler token budget." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=2668,
            max_num_batched_tokens=8192,    # <- CHANGE: test 8192 token budget
        )'
}

# ── exp_09: max_num_batched_tokens=65536 ─────────────────────────────────────
sut_exp09() {
    _emit_sut \
        "# EXP_09: max_num_batched_tokens=65536 (large token budget)
# ONLY CHANGE from production baseline: pin max_model_len=2668 + max_num_batched_tokens=65536.
# Tests if a very large scheduler token budget beyond V1 default helps throughput." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=2668,
            max_num_batched_tokens=65536,   # <- CHANGE: large token budget
        )'
}

# ── exp_10: FP8 weight quantisation ──────────────────────────────────────────
sut_exp10() {
    _emit_sut \
        "# EXP_10: quantization=fp8 (weight quant only, KV stays BF16)
# ONLY CHANGE from production baseline: quantization=fp8. Weights ~16 GB -> ~8 GB.
# dtype must be float16 for fp8 quant." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype="float16",                # fp8 requires float16
            tensor_parallel_size=self.tensor_parallel_size,
            quantization="fp8",             # <- CHANGE: test fp8 weight quant
        )'
}

# ── exp_11: FP8 KV cache ─────────────────────────────────────────────────────
sut_exp11() {
    _emit_sut \
        "# EXP_11: kv_cache_dtype=fp8 (KV quant only, weights stay BF16)
# ONLY CHANGE from production baseline: kv_cache_dtype=fp8. Halves KV footprint." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            kv_cache_dtype="fp8",           # <- CHANGE: test fp8 KV cache
        )'
}

# ── exp_12: FP8 weights + FP8 KV ─────────────────────────────────────────────
sut_exp12() {
    _emit_sut \
        "# EXP_12: quantization=fp8 + kv_cache_dtype=fp8 (both quant)
# TWO CHANGES from production baseline (inseparable memory pair). Max compression." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype="float16",                # fp8 requires float16
            tensor_parallel_size=self.tensor_parallel_size,
            quantization="fp8",             # <- CHANGE 1: fp8 weights
            kv_cache_dtype="fp8",           # <- CHANGE 2: fp8 KV cache
        )'
}

# ── exp_13: batch_size=1024 ───────────────────────────────────────────────────
sut_exp13() { _emit_sut "# EXP_13: +batch_size=1024. All engine pins = baseline." 1024 nosort "${_BASELINE_LLM}"; }

# ── exp_14: batch_size=2048 ───────────────────────────────────────────────────
sut_exp14() { _emit_sut "# EXP_14: +batch_size=2048. All engine pins = baseline." 2048 nosort "${_BASELINE_LLM}"; }

# ── exp_15: sort_by_input_length ─────────────────────────────────────────────
sut_exp15() { _emit_sut "# EXP_15: sort_by_input_length. Engine unchanged from production baseline." 16 sort "${_BASELINE_LLM}"; }

# ── exp_16: gpu_memory_utilization=0.95 ──────────────────────────────────────
sut_exp16() {
    _emit_sut \
        "# EXP_16: gpu_memory_utilization=0.95
# ONLY CHANGE from production baseline: gpu_memory_utilization=0.95 (+3.5 GB HBM)." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=0.95,    # <- CHANGE: test 0.95 vs vLLM default 0.90
        )'
}

# ── exp_17: attention_backend=FLASHINFER ─────────────────────────────────────
sut_exp17() {
    _emit_sut \
        "# EXP_17: attention_backend=FLASHINFER
# ONLY CHANGE from production baseline: attention_backend=FLASHINFER.
# V1 default on H200 is FlashAttention3. Tests FlashInfer vs FA3 tiling." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            attention_backend="FLASHINFER",  # <- CHANGE: test FlashInfer vs V1 default FA3
        )'
}

# ── exp_18: skip_tokenizer_init=True ─────────────────────────────────────────
sut_exp18() {
    _emit_sut \
        "# EXP_18: skip_tokenizer_init=True
# ONLY CHANGE from production baseline: skip_tokenizer_init=True.
# MLPerf passes pre-tokenised IDs. Saves ~500 MB RAM and ~10s startup.
# WARNING: verify dataset.py postProcess() does NOT call the tokeniser." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            skip_tokenizer_init=True,       # <- CHANGE: test skipping tokenizer init
        )'
}

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print_final_summary() {
    log "========================================================"
    log "ISOLATION ABLATION v0.17.1 — FINAL SUMMARY"
    log "Baseline (exp_00): production — stock vLLM V1 defaults, BS=16"
    log "========================================================"
    local baseline_mean="N/A"
    local baseline_agg="${RESULTS_DIR}/exp_00/aggregate.txt"
    [[ -f "${baseline_agg}" ]] && \
        baseline_mean=$(grep "^Mean tok/s:" "${baseline_agg}" | awk '{print $NF}' || echo "N/A")

    printf "\n%-12s %-50s %-12s %-10s %-6s\n" \
        "Exp" "Description" "Mean tok/s" "vs exp_00" "Valid"
    printf "%-12s %-50s %-12s %-10s %-6s\n" \
        "---" "-----------" "----------" "---------" "-----"
    for agg_file in "${RESULTS_DIR}"/*/aggregate.txt; do
        [[ -f "${agg_file}" ]] || continue
        local exp_id mean valid_r desc delta
        exp_id=$(grep "^Experiment:"  "${agg_file}" | cut -d: -f2- | xargs)
        mean=$(  grep "^Mean tok/s:"  "${agg_file}" | awk '{print $NF}')
        valid_r=$(grep "^Valid rate:" "${agg_file}" | cut -d: -f2- | xargs)
        desc=$(  grep "^Description:" "${agg_file}" | cut -d: -f2- | xargs | cut -c1-50)
        if [[ "${mean}" != "FAILED" && "${baseline_mean}" != "N/A" ]]; then
            delta=$(python3 -c "print(f'{(${mean}/${baseline_mean}-1)*100:+.1f}%')" 2>/dev/null || echo "N/A")
        else
            delta="N/A"
        fi
        printf "%-12s %-50s %-12s %-10s %-6s\n" "${exp_id}" "${desc}" "${mean}" "${delta}" "${valid_r}"
    done

    local csv="${RESULTS_DIR}/isolation_summary_v17.csv"
    echo "exp_id,description,mean_tok_s,vs_baseline_pct,stdev,cv_pct,min,max,valid_rate,runs" > "${csv}"
    for agg_file in "${RESULTS_DIR}"/*/aggregate.txt; do
        [[ -f "${agg_file}" ]] || continue
        local e mean stdev cv min_v max_v valid_r desc runs delta
        e=$(    grep "^Experiment:"  "${agg_file}" | cut -d: -f2- | xargs)
        mean=$( grep "^Mean tok/s:"  "${agg_file}" | awk '{print $NF}')
        stdev=$(grep "^Stdev tok/s:" "${agg_file}" | awk '{print $NF}')
        cv=$(   grep "^CV%:"         "${agg_file}" | awk '{print $NF}')
        min_v=$(grep "^Min tok/s:"   "${agg_file}" | awk '{print $NF}')
        max_v=$(grep "^Max tok/s:"   "${agg_file}" | awk '{print $NF}')
        valid_r=$(grep "^Valid rate:" "${agg_file}" | cut -d: -f2- | xargs)
        runs=$( grep "^Runs:"        "${agg_file}" | cut -d: -f2- | xargs)
        desc=$( grep "^Description:" "${agg_file}" | cut -d: -f2- | xargs)
        [[ "${mean}" != "FAILED" && "${baseline_mean}" != "N/A" ]] && \
            delta=$(python3 -c "print(f'{(${mean}/${baseline_mean}-1)*100:.2f}')" 2>/dev/null || echo "N/A") || delta="N/A"
        echo "${e},\"${desc}\",${mean},${delta},${stdev},${cv},${min_v},${max_v},\"${valid_r}\",\"${runs}\"" >> "${csv}"
    done

    log "CSV:     ${csv}"
    log "Results: ${RESULTS_DIR}"
    log "MLflow:  ${MLFLOW_EXPERIMENT_NAME}"
}

# =============================================================================
# MAIN
# =============================================================================
main() {
    TARGET="${1:-all}"
    log "========================================================"
    log "vLLM v0.17.1 Isolation Ablation — H200 MIG 71GB"
    log "Baseline: production (stock vLLM V1 defaults) | BS=16"
    log "Experiments: 20 | Runs/exp: ${RUNS_PER_EXP} | Total: $((RUNS_PER_EXP*20)) runs"
    log "TARGET=${TARGET} | RUN_SCOPE=${RUN_SCOPE}"
    log "Usage:"
    log "  ./isolation.sh                          # all experiments"
    log "  ./isolation.sh exp_04                   # exp_04 onwards (continue)"
    log "  RUN_SCOPE=only ./isolation.sh exp_04    # ONLY exp_04"
    log "  RUNS_PER_EXP=2 ./isolation.sh           # 2 runs per exp"
    log "========================================================"

    mkdir -p "${RESULTS_DIR}"
    [[ -d "${INFERENCE_DIR}" ]] || { error "INFERENCE_DIR not found: ${INFERENCE_DIR}"; exit 1; }
    [[ -f "${INFERENCE_DIR}/SUT_VLLM.py" ]] && \
        cp "${INFERENCE_DIR}/SUT_VLLM.py" "${RESULTS_DIR}/SUT_VLLM_original_backup.py"

    run_experiment "exp_00"  16   "bfloat16" "sut_exp00"  "PRODUCTION BASELINE: stock vLLM V1 defaults, BS=16"
    run_experiment "exp_00b" 16   "bfloat16" "sut_exp00b" "RIGHT-SIZED KV: max_model_len=2668 on production baseline"

    run_experiment "exp_01"  16   "bfloat16" "sut_exp01"  "-prefix_caching=False (disable V1 default APC)"
    run_experiment "exp_02"  16   "bfloat16" "sut_exp02"  "-async_output_proc OFF (disable V1 default async)"
    run_experiment "exp_03"  16   "bfloat16" "sut_exp03"  "compilation_config=O1 (below V1 default O2)"
    run_experiment "exp_04"  16   "bfloat16" "sut_exp04"  "compilation_config=O0 (no compile vs V1 default O2)"
    run_experiment "exp_05"  16   "bfloat16" "sut_exp05"  "compilation_config=O3 (above V1 default O2)"
    run_experiment "exp_06"  16   "bfloat16" "sut_exp06"  "max_num_batched_tokens=2668 (neutralise chunked prefill)"
    run_experiment "exp_07"  16   "bfloat16" "sut_exp07"  "async_scheduling=True (experimental)"

    run_experiment "exp_08"  16   "bfloat16" "sut_exp08"  "max_num_batched_tokens=8192"
    run_experiment "exp_09"  16   "bfloat16" "sut_exp09"  "max_num_batched_tokens=65536"

    run_experiment "exp_10"  16   "bfloat16" "sut_exp10"  "quantization=fp8 weight only"
    run_experiment "exp_11"  16   "bfloat16" "sut_exp11"  "kv_cache_dtype=fp8 KV only"
    run_experiment "exp_12"  16   "bfloat16" "sut_exp12"  "fp8 weights + fp8 KV"

    run_experiment "exp_13"  1024 "bfloat16" "sut_exp13"  "+batch_size=1024"
    run_experiment "exp_14"  2048 "bfloat16" "sut_exp14"  "+batch_size=2048"
    run_experiment "exp_15"  16   "bfloat16" "sut_exp15"  "sort_by_input_length"
    run_experiment "exp_16"  16   "bfloat16" "sut_exp16"  "gpu_memory_utilization=0.95"

    run_experiment "exp_17"  16   "bfloat16" "sut_exp17"  "attention_backend=FLASHINFER"
    run_experiment "exp_18"  16   "bfloat16" "sut_exp18"  "skip_tokenizer_init=True"

    sut_exp00 > "${INFERENCE_DIR}/SUT_VLLM.py"
    log "Restored baseline SUT (exp_00)"
    print_final_summary
}

main "$@"