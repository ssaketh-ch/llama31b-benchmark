#!/bin/bash
# =============================================================================
# vLLM v0.17.1 — Quantisation · Prefix Cache · Chunked-Prefill
# Llama-3.1-8B-Instruct on H200 MIG 71GB | Offline Scenario
#
# KEY DIMENSIONS UNDER STUDY
# ──────────────────────────
#  (Q)  Quantisation   – bf16 vs dynamic fp8 (W8A8) / fp8 KV / both / int8 W8A8
#  (P)  Prefix cache   – V1 default ON vs explicit OFF
#  (CP) Chunked prefill + token budget sweep
#         enable_chunked_prefill=False (hard off; must have MNBT > max_model_len)
#         enable_chunked_prefill=True  + max_num_batched_tokens sweep
#             Smaller MNBT → better ITL (fewer prefills slowing decodes)
#             Larger  MNBT → better TTFT (more prefill tokens per batch)
#             MNBT = max_model_len → ~V0 scheduling (still prioritizes decodes)
#             Docs recommend MNBT > 8192 for throughput on small models / large GPUs
#
# ─── NOTE ON QUANTISATION ────────────────────────────────────────────────────
# Only DYNAMIC quantisation is used here — no pre-quantised checkpoint needed.
#   fp8  (W8A8): quantization="fp8"   — natively supported on H200 (Hopper SM9.0)
#   int8 (W8A8): quantization="int8"  — natively supported on H200
# AWQ, GPTQ, INT4 require a pre-quantised checkpoint and are NOT tested here.
#
# ─── BASELINE (baseline) ─────────────────────────────────────────────────────
#   dtype=bfloat16, no quantisation
#   max_model_len=2668  (input 2540 + output 128)
#   max_num_batched_tokens=16384  
#   enable_chunked_prefill=True   ← V1 default; explicit
#   enable_prefix_caching=True    ← V1 default; explicit
#   batch_size=16
#
# ─── QUANTISATION GROUP (Q) ──────────────────────────────────────────────────
#   quant_fp8_weights      quantization="fp8"                           dynamic fp8 W8A8 (float16)
#   quant_fp8_kv           kv_cache_dtype="fp8"                         fp8 KV cache only (bf16 weights)
#   quant_fp8_full         quantization="fp8" + kv_cache_dtype="fp8"    full fp8: weights + KV
#   quant_int8_weights     quantization="int8"                          dynamic int8 W8A8
#
# ─── PREFIX CACHING GROUP (P) ────────────────────────────────────────────────
#   prefix_off   enable_prefix_caching=False  APC disabled
#
# ─── CHUNKED PREFILL GROUP (CP) ──────────────────────────────────────────────
# CP OFF (one experiment — MNBT must be > max_model_len=2668 to avoid crash):
#   cp_off       enable_chunked_prefill=False + MNBT=16384
#
# CP ON + MNBT sweep (enable_chunked_prefill=True, all others same as baseline):
#   cp_on_mnbt_2668    MNBT=2668   ← =max_model_len  → ~V0 scheduling (still prioritizes decodes)
#   cp_on_mnbt_4096    MNBT=4096
#   cp_on_mnbt_8192    MNBT=8192   ← docs-recommended minimum for throughput
#   cp_on_mnbt_16384   MNBT=16384  ← V1 default (same as baseline, explicit reference)
#   cp_on_mnbt_32768   MNBT=32768
#   cp_on_mnbt_65536   MNBT=65536
#
# Usage:
#   ./quant_prefill.sh                                    # all experiments
#   TARGET=quant_fp8_kv ./quant_prefill.sh                      # quant_fp8_kv onwards (continue)
#   TARGET=quant_fp8_kv RUN_SCOPE=only ./quant_prefill.sh       # only quant_fp8_kv
#   RUNS_PER_EXP=2 ./quant_prefill.sh                     # 2 runs per experiment
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
RESULTS_BASE="${SCRIPT_DIR}/quant_prefill_results"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RESULTS_DIR="${RESULTS_BASE}/${TIMESTAMP}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:////${SCRIPT_DIR}/mlflow.db}"
MLFLOW_EXPERIMENT_NAME="llama3.1-8b_quant_prefill-${TIMESTAMP}"
export GIT_PYTHON_REFRESH=quiet

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
    pkill -f "vllm"     2>/dev/null; sleep 2
    pkill -f "SUT_VLLM" 2>/dev/null; sleep 2
    pkill -f "main.py"  2>/dev/null; sleep 2
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

_tracking_uri = '${MLFLOW_TRACKING_URI}'
if _tracking_uri.startswith('http'):
    import socket, urllib.parse as _up
    _p = _up.urlparse(_tracking_uri)
    try:
        _s = socket.create_connection((_p.hostname, _p.port or 80), timeout=5)
        _s.close()
    except OSError as _e:
        print(f'[warn] MLflow HTTP unreachable ({_e}). Falling back to SQLite.', flush=True)
        _tracking_uri = 'sqlite:////${SCRIPT_DIR}/mlflow.db'
mlflow.set_tracking_uri(_tracking_uri)
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
    mlflow.log_param('mode',                 'quant_prefill_study')

    # ── Version / hardware tags ──────────────────────────────────────────────
    def safe_ver(pkg):
        try: return __import__(pkg).__version__
        except: return 'unavailable'
    mlflow.set_tag('vllm_version',         safe_ver('vllm'))
    mlflow.set_tag('torch_version',        safe_ver('torch'))
    mlflow.set_tag('transformers_version', safe_ver('transformers'))
    try:
        gpu  = subprocess.check_output(
            ['nvidia-smi','--query-gpu=name','--format=csv,noheader,nounits']
        ).decode().strip().split('\n')[0]
        cpu  = next((l.split(':')[1].strip() for l in
                     subprocess.check_output(['lscpu']).decode().split('\n')
                     if 'Model name' in l), 'unknown')
        drv  = subprocess.check_output(['nvidia-smi']).decode().split('\n')[2].strip()
        host = subprocess.check_output(['hostname']).decode().strip()
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
    mlflow.set_tag('mode',      'quant_prefill_study')

    run_type = 'baseline' if exp_id == 'baseline' else 'ablation'
    mlflow.set_tag('run_type', run_type)

    # ── Argument tags ────────────────────────────────────────────────────────
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
            OOM)         warn "    → OOM: try smaller batch_size or lower gpu_memory_utilization" ;;
            ENGINE_FATAL)warn "    → Engine fatal: flag incompatible with V1" ;;
        esac
    fi
    log "    Cooldown 30s..."; sleep 30; cleanup_gpu
}

run_experiment() {
    local exp_id="$1" batch_size="$2" dtype="$3" sut_func="$4" desc="$5"

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
baseline_agg = os.path.join('${RESULTS_DIR}', 'baseline', 'aggregate.txt')
baseline_agg = os.path.join('${RESULTS_DIR}', 'baseline', 'aggregate.txt')
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
    if exp_id != 'baseline' and speedup != 'N/A':
        lines.append(f"Speedup vs baseline: {speedup}")
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
# generate() uses TokensPrompt (vLLM V1 API).
# =============================================================================

_emit_sut() {
    local comment="$1" batch_size_default="$2" load_model_body="$3"
    cat <<SUTEOF
${comment}
import array, logging, queue, threading, time
import torch
from vllm import LLM, SamplingParams
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
        query_samples = sorted(query_samples, key=lambda q: len(self.data_object.input_ids[q.index]))
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
SUTEOF
}

# =============================================================================
# BASELINE LLM BLOCK
# dtype=bfloat16, no quant
# enable_chunked_prefill=True (V1 default), max_num_batched_tokens=16384
# enable_prefix_caching=True (V1 default), max_model_len=2668
# =============================================================================
_BASELINE_LLM='        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=2668,
            enable_chunked_prefill=True,
            max_num_batched_tokens=16384,
            enable_prefix_caching=True,
        )'

## Baseline: bf16, no quant, CP ON
baseline() {
    _emit_sut \
        "# BASELINE: bf16, no quant, prefix ON, chunked_prefill ON, MNBT=16384, max_model_len=2668\n# All Q/P/CP experiments measure delta vs this." \
        16 "${_BASELINE_LLM}"
}

# =============================================================================
# GROUP Q — QUANTISATION
# All dynamic (no pre-quantised checkpoint needed). H200 (Hopper SM9.0) supports:
#   fp8  W8A8: quantization="fp8"  — requires dtype=float16
#   int8 W8A8: quantization="int8" — fused scaled-mm, bf16 activations okay
#   fp8 KV:   kv_cache_dtype="fp8" — separate from weight quant
# AWQ / GPTQ / INT4 need a pre-quantised checkpoint → not tested here.
# =============================================================================

## Quant: fp8 W8A8 weights
quant_fp8_weights() {
    _emit_sut \
        "# EXP_Q1: quantization=fp8 (dynamic W8A8; weights + activations fp8)
# CHANGE from baseline: quantization=fp8, dtype must be float16 (fp8 requires it).
# No pre-quantised checkpoint needed — vLLM quantises on load." \
        16 \
        '        self.model = LLM(
            self.model_path,
            dtype="float16",                # required: fp8 W8A8 cannot use bfloat16
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=2668,
            enable_chunked_prefill=True,
            max_num_batched_tokens=16384,
            enable_prefix_caching=True,
            quantization="fp8",             # <- CHANGE: dynamic fp8 W8A8
        )'
}

## Quant: fp8 KV cache only
quant_fp8_kv() {
    _emit_sut \
        "# EXP_Q2: kv_cache_dtype=fp8 (KV cache only; weights stay bfloat16)
# CHANGE from baseline: KV activations stored in fp8 — halves KV memory footprint.
# Compute dtype stays bf16; no checkpoint needed." \
        16 \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=2668,
            enable_chunked_prefill=True,
            max_num_batched_tokens=16384,
            enable_prefix_caching=True,
            kv_cache_dtype="fp8",           # <- CHANGE: fp8 KV cache only
        )'
}

## Quant: fp8 W8A8 weights + fp8 KV
quant_fp8_full() {
    _emit_sut \
        "# EXP_Q3: quantization=fp8 + kv_cache_dtype=fp8 (full fp8)
# CHANGE from baseline: both weight compute (W8A8) and KV cache in fp8.
# Maximum memory compression. dtype=float16 required by fp8 W8A8." \
        16 \
        '        self.model = LLM(
            self.model_path,
            dtype="float16",                # required: fp8 W8A8 cannot use bfloat16
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=2668,
            enable_chunked_prefill=True,
            max_num_batched_tokens=16384,
            enable_prefix_caching=True,
            quantization="fp8",             # <- CHANGE 1: fp8 W8A8 weights
            kv_cache_dtype="fp8",           # <- CHANGE 2: fp8 KV cache
        )'
}

## Quant: int8 W8A8 weights
quant_int8_weights() {
    _emit_sut \
        "# EXP_Q4: quantization=bitsandbytes (int8 emulation; dynamic W8A8)
# CHANGE from baseline: bitsandbytes int8 emulation (dynamic W8A8)\n# No pre-quantised checkpoint needed. H200 supports bitsandbytes int8 emulation natively." \
        16 \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=2668,
            enable_chunked_prefill=True,
            max_num_batched_tokens=16384,
            enable_prefix_caching=True,
            quantization="bitsandbytes",    # <- CHANGE: bitsandbytes int8 emulation
        )'
}

# =============================================================================
# GROUP P — PREFIX CACHING
# =============================================================================

## Prefix caching ON
prefix_on() {
    _emit_sut \
        "# EXP_P1: enable_prefix_caching=True (explicit; same as baseline)
# Identical to baseline — reference showing the V1 default explicitly." \
        16 \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=2668,
            enable_chunked_prefill=True,
            max_num_batched_tokens=16384,
            enable_prefix_caching=True,     # <- explicit ON (V1 default)
        )'
}

## Prefix caching OFF
prefix_off() {
    _emit_sut \
        "# EXP_P2: enable_prefix_caching=False (APC disabled)
# CHANGE from baseline: prefix caching off. MLPerf Offline has zero prefix sharing;
# tests whether APC bookkeeping overhead matters on a non-sharing workload." \
        16 \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=2668,
            enable_chunked_prefill=True,
            max_num_batched_tokens=16384,
            enable_prefix_caching=False,    # <- CHANGE: APC off
        )'
}

# =============================================================================
# GROUP CP — CHUNKED PREFILL + MAX_NUM_BATCHED_TOKENS SWEEP
#
# enable_chunked_prefill is the binary control:
#   OFF: one shot — MNBT must be > max_model_len (2668) or vLLM crashes at startup.
#        Use MNBT=16384 (well above max_model_len) for CP OFF.
#   ON:  tune performance with MNBT (chunked prefill is always available):
#        MNBT = max_model_len (2668) → ~V0 scheduling (still prioritizes decodes)
#        MNBT < "large" → better inter-token latency (ITL)
#        MNBT > 8192    → recommended for throughput (per vLLM docs)
#        MNBT = V1 default (16384) → baseline
# =============================================================================

## Chunked prefill OFF
cp_off() {
    _emit_sut \
        "# EXP_CP_OFF: enable_chunked_prefill=False
# CHANGE from baseline: chunked prefill hard-disabled.
# MNBT must be > max_model_len=2668 to avoid startup crash; using 16384.
# vLLM processes each prefill as a single un-chunked pass." \
        16 \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=2668,
            enable_chunked_prefill=False,   # <- CHANGE: CP disabled
            max_num_batched_tokens=16384,   # must be > max_model_len when CP OFF
            enable_prefix_caching=True,
        )'
}

# ── CP ON + MNBT sweep ────────────────────────────────────────────────────────
_sut_cp_on() {
    local budget="$1" label="$2"
    _emit_sut \
        "# EXP_CP${label}: enable_chunked_prefill=True + max_num_batched_tokens=${budget}
# CHANGE from baseline: MNBT=${budget} (baseline is 16384).
# Tunes chunked-prefill scheduler token budget." \
        16 \
        "        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=2668,
            enable_chunked_prefill=True,
            max_num_batched_tokens=${budget},    # <- CHANGE: MNBT ${budget}
            enable_prefix_caching=True,
        )"
}

## Chunked prefill ON + MNBT sweep
cp_on_mnbt_2668()   { _sut_cp_on 2668  1; }   # MNBT=2668 → ~V0 scheduling
cp_on_mnbt_4096()   { _sut_cp_on 4096  2; }
cp_on_mnbt_8192()   { _sut_cp_on 8192  3; }
cp_on_mnbt_16384()  { _sut_cp_on 16384 4; }
cp_on_mnbt_32768()  { _sut_cp_on 32768 5; }
cp_on_mnbt_65536()  { _sut_cp_on 65536 6; }

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print_final_summary() {
    log "========================================================"
    log "QUANT · PREFIX CACHE · CHUNKED PREFILL STUDY — FINAL SUMMARY"
    log "Baseline (baseline): bf16, no quant, prefix ON, CP ON, MNBT=16384, BS=16"
    log "========================================================"
    local baseline_mean="N/A"
    local baseline_agg="${RESULTS_DIR}/baseline/aggregate.txt"
    [[ -f "${baseline_agg}" ]] && \
        baseline_mean=$(grep "^Mean tok/s:" "${baseline_agg}" | awk '{print $NF}' || echo "N/A")

    printf "\n%-18s %-55s %-12s %-10s %-6s\n" \
        "Exp" "Description" "Mean tok/s" "vs baseline" "Valid"
    printf "%-18s %-55s %-12s %-10s %-6s\n" \
        "---" "-----------" "----------" "---------" "-----"
    for agg_file in "${RESULTS_DIR}"/*/aggregate.txt; do
        [[ -f "${agg_file}" ]] || continue
        local exp_id mean valid_r desc delta
        exp_id=$(grep "^Experiment:"  "${agg_file}" | cut -d: -f2- | xargs)
        mean=$(  grep "^Mean tok/s:"  "${agg_file}" | awk '{print $NF}')
        valid_r=$(grep "^Valid rate:" "${agg_file}" | cut -d: -f2- | xargs)
        desc=$(  grep "^Description:" "${agg_file}" | cut -d: -f2- | xargs | cut -c1-55)
        if [[ "${mean}" != "FAILED" && "${baseline_mean}" != "N/A" ]]; then
            delta=$(python3 -c "print(f'{(${mean}/${baseline_mean}-1)*100:+.1f}%')" 2>/dev/null || echo "N/A")
        else
            delta="N/A"
        fi
        printf "%-12s %-55s %-12s %-10s %-6s\n" "${exp_id}" "${desc}" "${mean}" "${delta}" "${valid_r}"
    done

    local csv="${RESULTS_DIR}/quant_prefill_summary.csv"    echo "exp_id,description,mean_tok_s,vs_baseline_pct,stdev,cv_pct,min,max,valid_rate,runs" > "${csv}"
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
    local n_exps=14
    log "========================================================"
    log "vLLM v0.17.1 — Quant · Prefix Cache · Chunked Prefill"
    log "H200 MIG 71GB | Llama-3.1-8B-Instruct | ${SCENARIO}"
    log "Experiments: ${n_exps} | Runs/exp: ${RUNS_PER_EXP} | Total: $((RUNS_PER_EXP * n_exps)) runs"
    log "TARGET=${TARGET} | RUN_SCOPE=${RUN_SCOPE}"
    log "Usage:"
    log "  ./quant_prefill.sh                                # all experiments"
    log "  ./quant_prefill.sh quant_fp8_kv                   # quant_fp8_kv onwards"
    log "  RUN_SCOPE=only ./quant_prefill.sh cp_on_mnbt_8192 # only cp_on_mnbt_8192"
    log "  RUNS_PER_EXP=2 ./quant_prefill.sh                # 2 runs each"
    log "========================================================"

    mkdir -p "${RESULTS_DIR}"
    [[ -d "${INFERENCE_DIR}" ]] || { error "INFERENCE_DIR not found: ${INFERENCE_DIR}"; exit 1; }
    [[ -f "${INFERENCE_DIR}/SUT_VLLM.py" ]] && \
        cp "${INFERENCE_DIR}/SUT_VLLM.py" "${RESULTS_DIR}/SUT_VLLM_original_backup.py"

    # Priority experiments
    run_experiment "baseline"    16 "bfloat16" "baseline"      "Baseline: bf16, no quant, prefix ON, CP ON, MNBT=16384, BS=16"
    run_experiment "quant_fp8_weights"    16 "bfloat16" "quant_fp8_weights"     "Quant: fp8 W8A8 weights (float16 compute), BS=16"
    run_experiment "quant_int8_weights"  16 "bfloat16" "quant_int8_weights"   "Quant: int8 W8A8 weights, BS=16"
    run_experiment "prefix_off"   16 "bfloat16" "prefix_off"     "Prefix caching OFF (APC disabled), BS=16"
    run_experiment "cp_off" 16 "bfloat16" "cp_off" "Chunked prefill OFF (MNBT=16384), BS=16"
    run_experiment "cp_on_mnbt_2668"   16 "bfloat16" "cp_on_mnbt_2668"    "CP ON + MNBT=2668 (~V0 scheduling), BS=16"
    run_experiment "cp_on_mnbt_8192"   16 "bfloat16" "cp_on_mnbt_8192"    "CP ON + MNBT=8192 (docs-recommended min), BS=16"
    run_experiment "cp_on_mnbt_32768"  16 "bfloat16" "cp_on_mnbt_32768"   "CP ON + MNBT=32768, BS=16"

    # Remaining experiments
    run_experiment "quant_fp8_kv"        16 "bfloat16" "quant_fp8_kv"         "Quant: fp8 KV cache only (bf16 weights), BS=16"
    run_experiment "quant_fp8_full"      16 "bfloat16" "quant_fp8_full"       "Quant: fp8 W8A8 weights + fp8 KV (full fp8), BS=16"
    run_experiment "cp_on_mnbt_4096"   16 "bfloat16" "cp_on_mnbt_4096"    "CP ON + MNBT=4096, BS=16"
    run_experiment "cp_on_mnbt_16384"  16 "bfloat16" "cp_on_mnbt_16384"   "CP ON + MNBT=16384 (V1 default; baseline), BS=16"
    run_experiment "cp_on_mnbt_65536"  16 "bfloat16" "cp_on_mnbt_65536"   "CP ON + MNBT=65536, BS=16"

    baseline > "${INFERENCE_DIR}/SUT_VLLM.py"
    log "Restored baseline SUT (baseline)"
    print_final_summary
}

main "$@"
