#!/bin/bash
#SBATCH --job-name=llama3-8b-accuracy
#SBATCH --output=llama3-8b-accuracy-%j.out
#SBATCH --error=llama3-8b-accuracy-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:mig-3g.71gb:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=mig_nodes
#SBATCH --chdir=/home/saketh-msc/inference/language/llama3.1-8b

# =============================================================================
# vLLM MLPerf Accuracy Ablation — Llama-3.1-8B-Instruct on H200 MIG 71GB
# Offline Scenario — ONE change vs baseline, 1 run per experiment
#
# BASELINE (exp_00):
#   dtype="bfloat16", quantization=None, tensor_parallel_size=1
#   gpu_memory_utilization=0.90, max_model_len=131072
#   enforce_eager=False (CUDA graphs ON), enable_prefix_caching=False
#   enable_chunked_prefill=False, max_num_batched_tokens=131072
#   num_scheduler_steps=1, compilation_config=None, scheduler_delay_factor=0.0
#   SamplingParams: temperature=0.0, top_k=1, top_p=1, seed=42, max_tokens=128, min_tokens=1
#   batch_size=16, sort_by_length=False, input_truncation=False
#
# EXPERIMENT MAP:
#   exp_00  BASELINE
#   exp_01  temperature=0.0 explicit
#   exp_02  min_tokens=0
#   exp_03  skip_special_tokens=True explicit
#   exp_04  ignore_eos=False explicit
#   exp_05  max_tokens=256
#   exp_06  max_tokens=64  (negative control)
#   exp_07  batch_size=512
#   exp_08  batch_size=1024
#   exp_09  dtype=float16
#   exp_10  FP8 quantization (fp8 + float16)
#   exp_11  sort_by_input_length
#   exp_12  max_model_len=2668  (input truncated to 2540)
#   exp_13  max_model_len=1500  (input truncated to 1372)
#   exp_14  max_model_len=1000  (input truncated to 872)
#   exp_15  enable_prefix_caching=True
#   exp_16  chunked_prefill=True + batched_tokens=65536
#   exp_17  num_scheduler_steps=4
#   exp_18  compilation_config=3
#   exp_C1  COMBO: FP8 + max_model_len=2668
#   exp_C2  COMBO: FP8 + batch_size=1024
#   exp_C3  COMBO: full optimal config (FP8+BS1024+len2668+sort)
#
# Usage:
#   ./run_accuracy_ablation.sh              # all experiments
#   ./run_accuracy_ablation.sh exp_12       # start at exp_12, continue
#   RUN_SCOPE=only ./run_accuracy_ablation.sh exp_12  # run only exp_12
#   RUNS_PER_EXP=2 ./run_accuracy_ablation.sh
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
INFERENCE_DIR="${INFERENCE_DIR:-/home/saketh-msc/inference/language/llama3.1-8b}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/saketh-msc/inference/language/llama3.1-8b/model}"
DATASET_PATH="${DATASET_PATH:-/home/saketh-msc/inference/language/llama3.1-8b/data}"
# evaluation.py takes the CNN/DailyMail JSON file directly (different from DATASET_PATH)
# This is the path shown in the run logs: dataset/llama3-1-8b-cnn-eval.uri/cnn_eval.json
EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-/home/saketh-msc/inference/language/llama3.1-8b/dataset/llama3-1-8b-cnn-eval.uri/cnn_eval.json}"
GPU_COUNT="${GPU_COUNT:-1}"
TOTAL_SAMPLE_COUNT=13368
SCENARIO="Offline"
USER_CONF="${INFERENCE_DIR}/user.conf"
RUNS_PER_EXP="${RUNS_PER_EXP:-1}"

RESULTS_BASE="${SUBMIT_DIR}/accuracy_results"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RESULTS_DIR="${RESULTS_BASE}/${TIMESTAMP}"

MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
MLFLOW_EXPERIMENT_NAME="llama3.1-8b_accuracy-${TIMESTAMP}"

RUN_SCOPE="${RUN_SCOPE:-continue}"
TARGET_REACHED=0

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; MAGENTA='\033[0;35m'; NC='\033[0m'
log()     { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"; }
success() { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓${NC} $*"; }
warn()    { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠${NC} $*"; }
error()   { echo -e "${RED}[$(date '+%H:%M:%S')] ✗${NC} $*"; }
info()    { echo -e "${CYAN}[$(date '+%H:%M:%S')] ○${NC} $*"; }
rouge_log(){ echo -e "${MAGENTA}[$(date '+%H:%M:%S')] ◆${NC} $*"; }

cleanup_gpu() {
    log "Resetting GPU (30s cooldown)..."
    set +e
    pkill -f "vllm" 2>/dev/null; sleep 2
    pkill -f "SUT_VLLM" 2>/dev/null; sleep 2
    pkill -f "main.py" 2>/dev/null; sleep 2
    nvidia-smi --gpu-reset 2>/dev/null
    sleep 10
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
    nvidia-smi mig -lgi  >> "${run_dir}/system/mig_topology.txt" 2>&1
    nvidia-smi mig -lci  >> "${run_dir}/system/mig_topology.txt" 2>&1
    lscpu > "${run_dir}/system/lscpu.txt" 2>&1
    cat /proc/meminfo > "${run_dir}/system/meminfo.txt" 2>&1
    python --version > "${run_dir}/system/packages.txt" 2>&1
    pip show vllm torch transformers rouge-score >> "${run_dir}/system/packages.txt" 2>&1
    env | grep -E "CUDA|VLLM|PYTORCH|NCCL|GPU" | sort > "${run_dir}/system/env_vars.txt" 2>&1
    set -e
}

classify_error() {
    local logfile="$1"
    set +e
    if   grep -q "longer than the maximum model length" "${logfile}" 2>/dev/null; then echo "INPUT_TOO_LONG"
    elif grep -q "out of memory\|CUBLAS_STATUS_ALLOC_FAILED\|CUDA out of memory" "${logfile}" 2>/dev/null; then echo "OOM"
    elif grep -q "illegal memory access" "${logfile}" 2>/dev/null; then echo "ILLEGAL_MEMORY_ACCESS"
    elif grep -q "Segmentation fault\|SIGSEGV" "${logfile}" 2>/dev/null; then echo "SEGFAULT"
    elif grep -q "EngineCore\|EngineDeadError" "${logfile}" 2>/dev/null; then echo "ENGINE_CORE_FATAL"
    elif grep -q "Timeout\|timed out" "${logfile}" 2>/dev/null; then echo "TIMEOUT"
    elif grep -q "compilation\|inductor" "${logfile}" 2>/dev/null; then echo "COMPILE_FAILED"
    elif grep -q "evaluation.py\|rouge" "${logfile}" 2>/dev/null; then echo "EVAL_FAILED"
    else echo "UNKNOWN"
    fi
    set -e
}

# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------
run_single() {
    local exp_id="$1" run_idx="$2" batch_size="$3" dtype="$4" desc="$5"
    local run_dir="${RESULTS_DIR}/${exp_id}/run_${run_idx}"
    mkdir -p "${run_dir}/system" "${run_dir}/mlperf_logs" "${run_dir}/run_outputs"
    info "  Run ${run_idx}/${RUNS_PER_EXP}: ${exp_id} (BS=${batch_size}, dtype=${dtype})"

    set +e
    nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu,power.draw \
        --format=csv,noheader > "${run_dir}/system/gpu_pre_run.txt" 2>&1
    set -e
    capture_system_info "${run_dir}"

    local start_time; start_time=$(date +%s)
    cd "${INFERENCE_DIR}"
    mkdir -p "${INFERENCE_DIR}/run_outputs"

    set +e
    timeout 7200 python -u - 2>&1 <<PYCODE | tee "${run_dir}/run.log"
import mlflow, subprocess, os, sys, json, re, ast, time, shutil
sys.path.insert(0, '.')

def parse_evaluation_output(text):
    """
    evaluation.py prints:
      {'rouge1': '38.8001', 'rouge2': '15.9451', 'rougeL': '24.5198',
       'rougeLsum': '35.8487', 'gen_len': 8164530, 'gen_num': 13368}
    Values are STRING-quoted floats. We must coerce them to float/int
    so isinstance(v, (int,float)) is True for MLflow logging.
    """
    metrics = {}
    text = text.strip()
    if not text: return metrics
    text = re.sub(r'np\.int64\((\d+)\)', r'\1', text)
    text = re.sub(r'np\.float64\(([\d.]+)\)', r'\1', text)
    # Strategy 1: JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and parsed: metrics = parsed
    except: pass
    # Strategy 2: Python dict literal (handles single-quoted strings)
    if not metrics:
        try:
            m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
            if m:
                parsed = ast.literal_eval(m.group(0))
                if isinstance(parsed, dict) and parsed: metrics = parsed
        except: pass
    # Strategy 3: key: value lines
    if not metrics:
        for line in text.split('\n'):
            m = re.match(r'(\w+)\s*[=:]\s*([\d.]+)', line.strip())
            if m:
                try: metrics[m.group(1)] = float(m.group(2))
                except ValueError: pass
    # Coerce all string-quoted numbers to float/int
    coerced = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            coerced[k] = v
        elif isinstance(v, str):
            try: coerced[k] = int(v)
            except ValueError:
                try: coerced[k] = float(v)
                except ValueError: coerced[k] = v
        else:
            coerced[k] = v
    return coerced

def parse_summary_file(path):
    metrics = {}
    try:
        with open(path) as f:
            for line in f:
                m = re.match(r'^([\w\s/]+):\s*([\d.]+)', line.strip())
                if m:
                    key = m.group(1).strip().replace(' ', '_').lower()
                    try: metrics[key] = float(m.group(2))
                    except ValueError: metrics[key] = m.group(2)
    except Exception as e:
        metrics['parse_error'] = str(e)
    return metrics

run_name = '${exp_id}_run${run_idx}'
run_dir  = '${run_dir}'

mlflow.set_tracking_uri('${MLFLOW_TRACKING_URI}')
mlflow.set_experiment('${MLFLOW_EXPERIMENT_NAME}')
start_time = time.time()
mlflow.start_run(run_name=run_name)

try:
    mlflow.log_param('exp_id',             '${exp_id}')
    mlflow.log_param('run_index',          ${run_idx})
    mlflow.log_param('batch_size',         ${batch_size})
    mlflow.log_param('dtype',              '${dtype}')
    mlflow.log_param('description',        '${desc}')
    mlflow.log_param('scenario',           '${SCENARIO}')
    mlflow.log_param('total_sample_count', ${TOTAL_SAMPLE_COUNT})
    mlflow.log_param('gpu_count',          ${GPU_COUNT})
    mlflow.log_param('model_path',         '${CHECKPOINT_PATH}')
    mlflow.log_param('runs_per_exp',       ${RUNS_PER_EXP})
    mlflow.log_param('mode',               'accuracy')

    def safe_ver(pkg):
        try: return __import__(pkg).__version__
        except: return 'unavailable'
    mlflow.set_tag('vllm_version',         safe_ver('vllm'))
    mlflow.set_tag('torch_version',        safe_ver('torch'))
    mlflow.set_tag('transformers_version', safe_ver('transformers'))
    mlflow.set_tag('exp_id',   '${exp_id}')
    mlflow.set_tag('scenario', '${SCENARIO}')
    mlflow.set_tag('mode',     'accuracy')

    try:
        gpu_model = subprocess.check_output(
            ['nvidia-smi','--query-gpu=name','--format=csv,noheader,nounits']
        ).decode().strip().split('\n')[0]
        cpu_lines = subprocess.check_output(['lscpu']).decode().split('\n')
        cpu_model = next((l.split(':')[1].strip() for l in cpu_lines if 'Model name' in l), 'unknown')
        driver_line = subprocess.check_output(['nvidia-smi']).decode().split('\n')[2].strip()
        mlflow.set_tag('gpu_model',     gpu_model)
        mlflow.set_tag('cpu_model',     cpu_model)
        mlflow.set_tag('nvidia_driver', driver_line)
        mlflow.set_tag('host', subprocess.check_output(['hostname']).decode().strip())
    except Exception as e:
        mlflow.set_tag('hw_tag_error', str(e))

    # Step 1: Run benchmark
    accuracy_log_dir = os.path.join(run_dir, 'mlperf_logs')
    os.makedirs(accuracy_log_dir, exist_ok=True)
    bench_cmd = [
        'python', 'main.py',
        '--scenario',            '${SCENARIO}',
        '--model-path',          '${CHECKPOINT_PATH}',
        '--batch-size',          '${batch_size}',
        '--accuracy',
        '--dtype',               '${dtype}',
        '--user-conf',           '${USER_CONF}',
        '--total-sample-count',  '${TOTAL_SAMPLE_COUNT}',
        '--dataset-path',        '${DATASET_PATH}',
        '--output-log-dir',      accuracy_log_dir,
        '--tensor-parallel-size','${GPU_COUNT}',
        '--vllm',
    ]
    print(f"[bench] {' '.join(bench_cmd)}", flush=True)
    bench_proc = subprocess.run(bench_cmd, check=False)
    mlflow.log_param('bench_exit_code', bench_proc.returncode)

    # Step 2: Post-run GPU snapshot
    try:
        post_gpu = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.sm,clocks.mem',
            '--format=csv'
        ]).decode()
        with open(os.path.join(run_dir, 'system', 'gpu_post_run.csv'), 'w') as f:
            f.write(post_gpu)
        lines = [l.strip() for l in post_gpu.strip().split('\n') if l.strip()]
        if len(lines) >= 2:
            hdrs = [h.strip().lower().replace(' ','_').replace('.','_') for h in lines[0].split(',')]
            vals = [v.strip() for v in lines[1].split(',')]
            for h, v in zip(hdrs, vals):
                try: mlflow.log_metric(f'gpu_post_{h}', float(''.join(c for c in v if c in '0123456789.')))
                except: pass
    except Exception as e:
        mlflow.set_tag('gpu_post_run_error', str(e))

    # Step 3: Parse mlperf summary
    summary_path = os.path.join(accuracy_log_dir, 'mlperf_log_summary.txt')
    if os.path.exists(summary_path):
        sm = parse_summary_file(summary_path)
        for k, v in sm.items():
            if isinstance(v, (int, float)): mlflow.log_metric(f'mlperf_{k}', v)
            else: mlflow.log_param(f'mlperf_{k}', str(v))
        print(f"[summary] Parsed {len(sm)} metrics", flush=True)
    else:
        mlflow.set_tag('summary_missing', 'true')

    # Step 4: Run evaluation.py → ROUGE
    accuracy_json = os.path.join(accuracy_log_dir, 'mlperf_log_accuracy.json')
    rouge_metrics = {}
    if os.path.exists(accuracy_json):
        eval_cmd = ['python', 'evaluation.py',
                    '--mlperf-accuracy-file', accuracy_json,
                    '--dataset-file', '${EVAL_DATASET_PATH}', '--dtype', 'int32']
        print(f"[eval] {' '.join(eval_cmd)}", flush=True)
        eval_result = subprocess.run(eval_cmd, capture_output=True, text=True, check=False)
        eval_stdout = eval_result.stdout.strip()
        eval_stderr = eval_result.stderr.strip()
        with open(os.path.join(run_dir, 'evaluation_output.txt'), 'w') as f:
            f.write(f"=== STDOUT ===\n{eval_stdout}\n\n=== STDERR ===\n{eval_stderr}\n\n=== EXIT: {eval_result.returncode} ===\n")
        mlflow.log_param('eval_exit_code', eval_result.returncode)
        rouge_metrics = parse_evaluation_output(eval_stdout)
        if not rouge_metrics and eval_stderr:
            rouge_metrics = parse_evaluation_output(eval_stderr)
        if rouge_metrics:
            print(f"[eval] ROUGE: {rouge_metrics}", flush=True)
            for k, v in rouge_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
                    print(f"  {k}: {v}", flush=True)
            mlflow.set_tag('evaluation_status', 'success')
        else:
            print(f"[eval] WARNING: could not parse ROUGE. stdout: {eval_stdout[:300]}", flush=True)
            mlflow.set_tag('evaluation_status', 'parse_failed')
            mlflow.set_tag('eval_stdout_preview', eval_stdout[:300])
    else:
        mlflow.set_tag('evaluation_status', 'skipped_no_json')

    # Step 5: Copy run_outputs
    inf_out = '${INFERENCE_DIR}/run_outputs'
    if os.path.isdir(inf_out):
        for fname in os.listdir(inf_out):
            try: shutil.copy2(os.path.join(inf_out, fname), os.path.join(run_dir, 'run_outputs', fname))
            except: pass

    # Step 6: Duration + artifacts
    mlflow.log_metric('duration_sec', time.time() - start_time)
    system_dir = os.path.join(run_dir, 'system')
    mlperf_dir = os.path.join(run_dir, 'mlperf_logs')
    run_out_dir = os.path.join(run_dir, 'run_outputs')
    if os.path.isdir(system_dir): mlflow.log_artifacts(system_dir, artifact_path='system')
    if os.path.isdir(mlperf_dir): mlflow.log_artifacts(mlperf_dir, artifact_path='mlperf_logs')
    if os.path.isdir(run_out_dir) and os.listdir(run_out_dir):
        mlflow.log_artifacts(run_out_dir, artifact_path='run_outputs')
    for fname in ('run.log', 'evaluation_output.txt', 'SUT_VLLM.py', 'result.txt', 'experiment_meta.txt'):
        fpath = os.path.join(run_dir, fname)
        if os.path.exists(fpath): mlflow.log_artifact(fpath)
    if rouge_metrics:
        with open(os.path.join(run_dir, 'rouge_metrics.json'), 'w') as f:
            json.dump(rouge_metrics, f, indent=2)

finally:
    mlflow.end_run()
PYCODE
    local exit_code="${PIPESTATUS[0]}"
    set -e

    local end_time; end_time=$(date +%s)
    local duration=$(( end_time - start_time ))

    set +e
    cp "${INFERENCE_DIR}"/mlperf_log_* "${run_dir}/mlperf_logs/" 2>/dev/null
    set -e

    # Extract ROUGE values from run log
    # evaluation.py outputs: {'rouge1': '38.8001', 'rouge2': '15.9451', ...}
    local rouge1 rouge2 rougeL rougeLsum gen_len tokens valid
    set +e
    rouge1=$(    grep -oP "rouge1['\"]?\s*:\s*['\"]?\K[\d.]+"    "${run_dir}/run.log" 2>/dev/null | tail -1)
    rouge2=$(    grep -oP "rouge2['\"]?\s*:\s*['\"]?\K[\d.]+"    "${run_dir}/run.log" 2>/dev/null | tail -1)
    rougeL=$(    grep -oP "rougeL['\"]?\s*:\s*['\"]?\K[\d.]+"    "${run_dir}/run.log" 2>/dev/null | grep -v "sum" | tail -1)
    rougeLsum=$( grep -oP "rougeLsum['\"]?\s*:\s*['\"]?\K[\d.]+" "${run_dir}/run.log" 2>/dev/null | tail -1)
    gen_len=$(   grep -oP "gen_len['\"]?\s*:\s*\K[\d.]+"         "${run_dir}/run.log" 2>/dev/null | tail -1)
    tokens=$(    grep "Tokens per second:" "${run_dir}/run.log" 2>/dev/null | tail -1 | awk '{print $NF}')
    valid=$(     grep "Result is"          "${run_dir}/run.log" 2>/dev/null | tail -1 | awk '{print $NF}')
    set -e

    rouge1="${rouge1:-N/A}"; rouge2="${rouge2:-N/A}"; rougeL="${rougeL:-N/A}"
    rougeLsum="${rougeLsum:-N/A}"; gen_len="${gen_len:-N/A}"
    tokens="${tokens:-N/A}"; valid="${valid:-N/A}"

    cat > "${run_dir}/result.txt" <<RESEOF
Experiment:       ${exp_id}
Run:              ${run_idx}/${RUNS_PER_EXP}
Description:      ${desc}
Batch Size:       ${batch_size}
Dtype:            ${dtype}
Mode:             accuracy
rouge1:           ${rouge1}
rouge2:           ${rouge2}
rougeL:           ${rougeL}
rougeLsum:        ${rougeLsum}
gen_len:          ${gen_len}
Tokens/sec:       ${tokens}
Valid:            ${valid}
Duration:         ${duration}s
Exit Code:        ${exit_code}
RESEOF

    if [[ "${rouge1}" != "N/A" ]]; then
        rouge_log "    run_${run_idx}: rouge1=${rouge1}  rouge2=${rouge2}  rougeL=${rougeL}  gen_len=${gen_len}"
        success "    run_${run_idx}: ${tokens} tok/s  [${valid}]  (${duration}s)"
    elif grep -q "MLPerf Results Summary" "${run_dir}/run.log" 2>/dev/null; then
        warn "    run_${run_idx}: benchmark ran but ROUGE not found — check evaluation_output.txt"
    else
        local err_type; err_type=$(classify_error "${run_dir}/run.log")
        error "    run_${run_idx}: FAILED (${err_type})  (${duration}s)"
        case "${err_type}" in
            INPUT_TOO_LONG)        warn "    → INPUT_TOO_LONG: truncation in process_queries() failed" ;;
            OOM)                   warn "    → OOM: reduce batch_size or gpu_memory_utilization" ;;
            ENGINE_CORE_FATAL)     warn "    → Engine fatal: vLLM version incompatibility" ;;
            ILLEGAL_MEMORY_ACCESS) warn "    → ILLEGAL_MEM: max_num_batched_tokens too high" ;;
            COMPILE_FAILED)        warn "    → Compile failed: torch.compile issue" ;;
            EVAL_FAILED)           warn "    → EVAL: check evaluation_output.txt" ;;
        esac
    fi

    log "    Cooldown 30s..."; sleep 30
    cleanup_gpu
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

    local exp_dir="${RESULTS_DIR}/${exp_id}"
    mkdir -p "${exp_dir}"
    log "========================================================"
    log "  ${exp_id}: ${desc}"
    log "  BS=${batch_size} | dtype=${dtype} | runs=${RUNS_PER_EXP}"
    log "========================================================"

    ${sut_func} > "${exp_dir}/SUT_VLLM.py"
    cp "${exp_dir}/SUT_VLLM.py" "${INFERENCE_DIR}/SUT_VLLM.py"

    cat > "${exp_dir}/experiment_meta.txt" <<METAEOF
exp_id:       ${exp_id}
description:  ${desc}
batch_size:   ${batch_size}
dtype:        ${dtype}
mode:         accuracy
runs:         ${RUNS_PER_EXP}
timestamp:    $(date -u +%Y-%m-%dT%H:%M:%SZ)
sut_func:     ${sut_func}
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
import os, statistics, json
exp_dir = '${exp_dir}'; exp_id = '${exp_id}'
rouge_keys = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'gen_len']
rouge_data = {k: [] for k in rouge_keys}; tok_vals = []
for run_idx in range(1, ${RUNS_PER_EXP} + 1):
    rfile = os.path.join(exp_dir, f'run_{run_idx}', 'result.txt')
    if not os.path.exists(rfile): continue
    lines = {}
    with open(rfile) as f:
        for line in f:
            if ':' in line: k, _, v = line.partition(':'); lines[k.strip()] = v.strip()
    for k in rouge_keys:
        v = lines.get(k, 'N/A')
        if v not in ('N/A', 'FAILED', ''):
            try: rouge_data[k].append(float(v))
            except: pass
    tok = lines.get('Tokens/sec', 'N/A')
    if tok not in ('N/A', 'FAILED', ''):
        try: tok_vals.append(float(tok))
        except: pass
lines_out = [f"Experiment:  {exp_id}", "ROUGE SCORES:"]; agg_dict = {'exp_id': exp_id, 'description': '${desc}'}
for k in rouge_keys:
    vals = rouge_data[k]
    if vals:
        mean = statistics.mean(vals); stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
        lines_out.append(f"  {k:12s}: mean={mean:.4f}  stdev={stdev:.4f}")
        agg_dict[f'{k}_mean'] = mean; agg_dict[f'{k}_stdev'] = stdev
    else:
        lines_out.append(f"  {k:12s}: N/A"); agg_dict[f'{k}_mean'] = None
if tok_vals:
    lines_out.append(f"Throughput:  {statistics.mean(tok_vals):.2f} tok/s")
    agg_dict['mean_tok_s'] = statistics.mean(tok_vals)
text = '\n'.join(lines_out); print(text)
with open(os.path.join(exp_dir, 'aggregate.txt'), 'w') as f: f.write(text + '\n')
with open(os.path.join(exp_dir, 'aggregate.json'), 'w') as f: json.dump(agg_dict, f, indent=2)
PYAGG
}


# =============================================================================
# SUT GENERATOR FUNCTIONS
#
# BASELINE load_model() — pinned in EVERY SUT (including experiments):
#   dtype="bfloat16", quantization=None, tensor_parallel_size=1
#   gpu_memory_utilization=0.90, max_model_len=131072
#   enforce_eager=False         (CUDA graphs ON — vLLM default)
#   enable_prefix_caching=False (vLLM default)
#   enable_chunked_prefill=False (MUST be explicit — vLLM default=None=AUTO on H200)
#   max_num_batched_tokens=131072 (explicit when chunked=False)
#   num_scheduler_steps=1       (vLLM default)
#   compilation_config=None     (no torch.compile)
#   scheduler_delay_factor=0.0
#   cpu_offload_gb=0
#
# INPUT TRUNCATION (exp_12/13/14/C1/C3):
#   vLLM raises ValueError (not silent truncation) when input > max_model_len - max_tokens.
#   Truncating SUTs slice input_ids to (max_model_len - 128) in process_queries().
#
# FP8 NOTE (exp_10/C1/C2/C3):
#   quantization="fp8" forces dtype="float16" in vLLM 0.10.0. Inseparable pair.
# =============================================================================


sut_exp00() {
cat << 'SUTEOF'
# exp_00: BASELINE — all flags pinned at vLLM defaults. Reference ROUGE.
import os, array, threading, queue, logging, torch
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
            self.data_object.perf_count, self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42, max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype                  = "bfloat16",
            quantization           = None,
            tensor_parallel_size   = 1,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enforce_eager          = False,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens = 131072,
            num_scheduler_steps    = 1,
            compilation_config     = None,
            scheduler_delay_factor = 0.0,
            cpu_offload_gb         = 0,
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
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(
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

sut_exp01() {
cat << 'SUTEOF'
# exp_01: temperature=0.0 explicit (ONE change)
CHANGE: not set → 0.0
EXPECTED: identical ROUGE
import os, array, threading, queue, logging, torch
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
            self.data_object.perf_count, self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0,  # <- CHANGE: explicit
            top_p=1, top_k=1, seed=42, max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype                  = "bfloat16",
            quantization           = None,
            tensor_parallel_size   = 1,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enforce_eager          = False,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens = 131072,
            num_scheduler_steps    = 1,
            compilation_config     = None,
            scheduler_delay_factor = 0.0,
            cpu_offload_gb         = 0,
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
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(
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

sut_exp02() {
cat << 'SUTEOF'
# exp_02: min_tokens=0 (ONE change)
CHANGE: 1 → 0
EXPECTED: identical ROUGE if no sample terminates at 0
import os, array, threading, queue, logging, torch
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
            self.data_object.perf_count, self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42, max_tokens=128,
            min_tokens=0,  # <- CHANGE (baseline: 1),
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype                  = "bfloat16",
            quantization           = None,
            tensor_parallel_size   = 1,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enforce_eager          = False,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens = 131072,
            num_scheduler_steps    = 1,
            compilation_config     = None,
            scheduler_delay_factor = 0.0,
            cpu_offload_gb         = 0,
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
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(
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

sut_exp03() {
cat << 'SUTEOF'
# exp_03: skip_special_tokens=True explicit (ONE change)
CHANGE: not set → True (vLLM default IS True — confirming no override)
EXPECTED: identical ROUGE
import os, array, threading, queue, logging, torch
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
            self.data_object.perf_count, self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42, max_tokens=128, min_tokens=1,
            skip_special_tokens=True,  # <- CHANGE: explicit (vLLM default=True),
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype                  = "bfloat16",
            quantization           = None,
            tensor_parallel_size   = 1,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enforce_eager          = False,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens = 131072,
            num_scheduler_steps    = 1,
            compilation_config     = None,
            scheduler_delay_factor = 0.0,
            cpu_offload_gb         = 0,
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
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(
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

sut_exp04() {
cat << 'SUTEOF'
# exp_04: ignore_eos=False explicit (ONE change)
CHANGE: not set → False (vLLM default IS False — confirming)
EXPECTED: identical ROUGE
import os, array, threading, queue, logging, torch
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
            self.data_object.perf_count, self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42, max_tokens=128, min_tokens=1,
            ignore_eos=False,  # <- CHANGE: explicit (vLLM default=False),
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype                  = "bfloat16",
            quantization           = None,
            tensor_parallel_size   = 1,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enforce_eager          = False,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens = 131072,
            num_scheduler_steps    = 1,
            compilation_config     = None,
            scheduler_delay_factor = 0.0,
            cpu_offload_gb         = 0,
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
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(
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

sut_exp05() {
cat << 'SUTEOF'
# exp_05: max_tokens=256 (ONE change)
CHANGE: 128 → 256
EXPECTED: ROUGE may improve if baseline is truncating valid summaries
import os, array, threading, queue, logging, torch
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
            self.data_object.perf_count, self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42,
            max_tokens=256, min_tokens=1,  # <- CHANGE (baseline: 128),
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype                  = "bfloat16",
            quantization           = None,
            tensor_parallel_size   = 1,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enforce_eager          = False,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens = 131072,
            num_scheduler_steps    = 1,
            compilation_config     = None,
            scheduler_delay_factor = 0.0,
            cpu_offload_gb         = 0,
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
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(
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

sut_exp06() {
cat << 'SUTEOF'
# exp_06: max_tokens=64 — NEGATIVE CONTROL (ONE change)
CHANGE: 128 → 64
EXPECTED: ROUGE degrades. Confirms 128 is not already in degraded zone.
import os, array, threading, queue, logging, torch
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
            self.data_object.perf_count, self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42,
            max_tokens=64, min_tokens=1,  # <- CHANGE (baseline: 128),
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype                  = "bfloat16",
            quantization           = None,
            tensor_parallel_size   = 1,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enforce_eager          = False,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens = 131072,
            num_scheduler_steps    = 1,
            compilation_config     = None,
            scheduler_delay_factor = 0.0,
            cpu_offload_gb         = 0,
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
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(
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

sut_exp07() {
cat << 'SUTEOF'
# exp_07: batch_size=512 (ONE change)
CHANGE: 16 → 512
EXPECTED: identical ROUGE if greedy decode is truly independent per-sequence
import os, array, threading, queue, logging, torch
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
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
            self.data_object.perf_count, self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42, max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype                  = "bfloat16",
            quantization           = None,
            tensor_parallel_size   = 1,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enforce_eager          = False,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens = 131072,
            num_scheduler_steps    = 1,
            compilation_config     = None,
            scheduler_delay_factor = 0.0,
            cpu_offload_gb         = 0,
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
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(
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

sut_exp08() {
cat << 'SUTEOF'
# exp_08: batch_size=1024 — STRONGEST NON-DETERMINISM TEST (ONE change)
CHANGE: 16 → 1024
EXPECTED: identical ROUGE. Any delta explains INVALID throughput results.
import os, array, threading, queue, logging, torch
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
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
            self.data_object.perf_count, self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42, max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype                  = "bfloat16",
            quantization           = None,
            tensor_parallel_size   = 1,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enforce_eager          = False,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens = 131072,
            num_scheduler_steps    = 1,
            compilation_config     = None,
            scheduler_delay_factor = 0.0,
            cpu_offload_gb         = 0,
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
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(
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

sut_exp11() {
cat << 'SUTEOF'
# exp_11: sort_by_input_length (ONE change)
CHANGE: unsorted → shortest-first
EXPECTED: identical ROUGE. Any delta = batch-level interaction from sorting.
import os, array, threading, queue, logging, torch
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
            self.data_object.perf_count, self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42, max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype                  = "bfloat16",
            quantization           = None,
            tensor_parallel_size   = 1,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enforce_eager          = False,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens = 131072,
            num_scheduler_steps    = 1,
            compilation_config     = None,
            scheduler_delay_factor = 0.0,
            cpu_offload_gb         = 0,
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
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(
                    qitem[i].id, bi[0], bi[1], n_tokens)])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
        log.info(f"IssueQuery started with {len(query_samples)} samples")
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

class SUTServer(SUT): pass
SUTEOF
}

sut_exp15() {
cat << 'SUTEOF'
# exp_15: enable_prefix_caching=True (ONE change)
CHANGE: False → True
EXPECTED: identical ROUGE on unique articles (no shared prefixes)
gpu_memory_utilization STAYS 0.90 — was 0.95 before, that was a bug.
import os, array, threading, queue, logging, torch
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
            self.data_object.perf_count, self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42, max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype                  = "bfloat16",
            quantization           = None,
            tensor_parallel_size   = 1,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enforce_eager          = False,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens = 131072,
            num_scheduler_steps    = 1,
            compilation_config     = None,
            scheduler_delay_factor = 0.0,
            cpu_offload_gb         = 0,
            enable_prefix_caching  = True,    # <- CHANGE (baseline: False)
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
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(
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

sut_exp17() {
cat << 'SUTEOF'
# exp_17: num_scheduler_steps=4 (ONE change)
CHANGE: 1 → 4
EXPECTED: may change outputs — delayed EOS detection runs extra decode steps.
Directly explains INVALID results in throughput runs with steps=4.
import os, array, threading, queue, logging, torch
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
            self.data_object.perf_count, self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42, max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype                  = "bfloat16",
            quantization           = None,
            tensor_parallel_size   = 1,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enforce_eager          = False,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens = 131072,
            num_scheduler_steps    = 1,
            compilation_config     = None,
            scheduler_delay_factor = 0.0,
            cpu_offload_gb         = 0,
            num_scheduler_steps    = 4,    # <- CHANGE (baseline: 1)
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
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(
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

sut_exp18() {
cat << 'SUTEOF'
# exp_18: compilation_config=3 (ONE change)
CHANGE: None → 3
EXPECTED: may change outputs — fused kernel FP accumulation order differs.
First batch takes 2-5min to compile.
import os, array, threading, queue, logging, torch
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
            self.data_object.perf_count, self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42, max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype                  = "bfloat16",
            quantization           = None,
            tensor_parallel_size   = 1,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enforce_eager          = False,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens = 131072,
            num_scheduler_steps    = 1,
            compilation_config     = None,
            scheduler_delay_factor = 0.0,
            cpu_offload_gb         = 0,
            compilation_config     = 3,    # <- CHANGE (baseline: None)
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
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(
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

sut_exp09() {
cat << 'SUTEOF'
# exp_09: dtype=float16 (ONE change)
CHANGE: bfloat16 → float16
Any ROUGE delta = dtype-dependent rounding changes token selection under greedy decode.
import os, array, threading, queue, logging, torch
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
            self.data_object.perf_count, self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42, max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype                  = "float16",   # <- CHANGE
            quantization           = None,
            tensor_parallel_size   = 1,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enforce_eager          = False,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens = 131072,
            num_scheduler_steps    = 1,
            compilation_config     = None,
            scheduler_delay_factor = 0.0,
            cpu_offload_gb         = 0,
            dtype                  = "float16",   # <- CHANGE (baseline: bfloat16)
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
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(
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

sut_exp10() {
cat << 'SUTEOF'
# exp_10: FP8 quantization (ONE logical change)
CHANGE: quantization None → fp8; dtype bfloat16 → float16 (forced by FP8 path)
These are inseparable in vLLM 0.10.0.
EXPECTED: some ROUGE degradation. Quantifies accuracy cost of FP8 speedup.
import os, array, threading, queue, logging, torch
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
            self.data_object.perf_count, self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42, max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype                  = "float16",   # <- CHANGE (forced by FP8 path)
            quantization           = "fp8",        # <- CHANGE
            tensor_parallel_size   = 1,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enforce_eager          = False,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens = 131072,
            num_scheduler_steps    = 1,
            compilation_config     = None,
            scheduler_delay_factor = 0.0,
            cpu_offload_gb         = 0,
            dtype                  = "float16",   # <- CHANGE (forced by FP8 path)
            quantization           = "fp8",        # <- CHANGE
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
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(
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

sut_exp12() {
cat << 'SUTEOF'
# exp_12: max_model_len=2668 (ONE change)
CHANGE: 131072 → 2668 (dataset max 2540 + output 128)
INPUT TRUNCATION: inputs sliced to 2540 tokens to prevent vLLM ValueError.
EXPECTED: identical ROUGE if dataset truly has max 2540 tokens after tokenisation.
import os, array, threading, queue, logging, torch
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
        # Inputs truncated to self.max_input_len tokens.
        # vLLM raises ValueError (not silent truncation) for oversized inputs.
        # truncation_len = max_model_len(2668) - max_tokens(128) = 2540
        self.max_input_len = 2540
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
            self.data_object.perf_count, self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42, max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype                  = "bfloat16",
            quantization           = None,
            tensor_parallel_size   = 1,
            gpu_memory_utilization = 0.90,
            max_model_len          = 2668,   # <- CHANGE
            enforce_eager          = False,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens = 2668,   # = max_model_len when chunked=False
            num_scheduler_steps    = 1,
            compilation_config     = None,
            scheduler_delay_factor = 0.0,
            cpu_offload_gb         = 0,
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
            # Truncate to max_input_len to prevent vLLM ValueError
            input_ids_tensor = [
                self.data_object.input_ids[q.index][:self.max_input_len]
                for q in qitem
            ]
            outputs = self.model.generate(prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params, use_tqdm=False)
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(
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

sut_exp13() {
cat << 'SUTEOF'
# exp_13: max_model_len=1500 (ONE change)
CHANGE: 131072 → 1500
INPUT TRUNCATION: inputs sliced to 1372 tokens. ~50% of CNN/DailyMail samples affected.
EXPECTED: measurable ROUGE degradation. Quantifies accuracy cost of moderate KV right-sizing.
import os, array, threading, queue, logging, torch
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
        # Inputs truncated to self.max_input_len tokens.
        # vLLM raises ValueError (not silent truncation) for oversized inputs.
        # truncation_len = max_model_len(1500) - max_tokens(128) = 1372
        self.max_input_len = 1372
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
            self.data_object.perf_count, self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42, max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype                  = "bfloat16",
            quantization           = None,
            tensor_parallel_size   = 1,
            gpu_memory_utilization = 0.90,
            max_model_len          = 1500,   # <- CHANGE
            enforce_eager          = False,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens = 1500,   # = max_model_len when chunked=False
            num_scheduler_steps    = 1,
            compilation_config     = None,
            scheduler_delay_factor = 0.0,
            cpu_offload_gb         = 0,
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
            # Truncate to max_input_len to prevent vLLM ValueError
            input_ids_tensor = [
                self.data_object.input_ids[q.index][:self.max_input_len]
                for q in qitem
            ]
            outputs = self.model.generate(prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params, use_tqdm=False)
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(
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

sut_exp14() {
cat << 'SUTEOF'
# exp_14: max_model_len=1000 (ONE change)
CHANGE: 131072 → 1000
INPUT TRUNCATION: inputs sliced to 872 tokens. Mean input is 870 — ~50% affected.
EXPECTED: clear ROUGE drop. Completes truncation curve: {1000, 1500, 2668, 131072}.
import os, array, threading, queue, logging, torch
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
        # Inputs truncated to self.max_input_len tokens.
        # vLLM raises ValueError (not silent truncation) for oversized inputs.
        # truncation_len = max_model_len(1000) - max_tokens(128) = 872
        self.max_input_len = 872
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
            self.data_object.perf_count, self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42, max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype                  = "bfloat16",
            quantization           = None,
            tensor_parallel_size   = 1,
            gpu_memory_utilization = 0.90,
            max_model_len          = 1000,   # <- CHANGE
            enforce_eager          = False,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens = 1000,   # = max_model_len when chunked=False
            num_scheduler_steps    = 1,
            compilation_config     = None,
            scheduler_delay_factor = 0.0,
            cpu_offload_gb         = 0,
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
            # Truncate to max_input_len to prevent vLLM ValueError
            input_ids_tensor = [
                self.data_object.input_ids[q.index][:self.max_input_len]
                for q in qitem
            ]
            outputs = self.model.generate(prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params, use_tqdm=False)
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(
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

sut_exp16() {
cat << 'SUTEOF'
# exp_16: chunked_prefill=True + max_num_batched_tokens=65536 (ONE logical change)
# CHANGE: enable_chunked_prefill False → True; max_num_batched_tokens 131072 → 65536
# These are an inseparable pair (chunked without cap = unbounded; cap without chunked = no effect).
# gpu_memory_utilization STAYS 0.90 (was incorrectly 0.95 before — that was a silent second change).
# EXPECTED: identical ROUGE if chunked attention is numerically equivalent to full-sequence.
import os, array, threading, queue, logging, torch
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
            self.data_object.perf_count, self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42, max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model (exp_16: chunked_prefill=True, tokens=65536)...")
        self.model = LLM(
            self.model_path,
            dtype                  = "bfloat16",
            quantization           = None,
            tensor_parallel_size   = 1,
            gpu_memory_utilization = 0.90,   # stays 0.90 (was 0.95 before — bug fixed)
            max_model_len          = 131072,
            enforce_eager          = False,
            enable_prefix_caching  = False,
            enable_chunked_prefill = True,    # <- CHANGE (baseline: False)
            max_num_batched_tokens = 65536,   # <- CHANGE (baseline: 131072)
            num_scheduler_steps    = 1,
            compilation_config     = None,
            scheduler_delay_factor = 0.0,
            cpu_offload_gb         = 0,
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
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(
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

sut_exp_C1() {
cat << 'SUTEOF'
# exp_C1: COMBO — FP8 + max_model_len=2668
CHANGES: fp8+float16 AND max_model_len=2668 with input truncation
This is the primary throughput config. The key accuracy question.
import os, array, threading, queue, logging, torch
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
        # Inputs truncated to self.max_input_len tokens.
        # vLLM raises ValueError (not silent truncation) for oversized inputs.
        # truncation_len = max_model_len(2668) - max_tokens(128) = 2540
        self.max_input_len = 2540
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
            self.data_object.perf_count, self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42, max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype                  = "float16",   # <- FP8 path forces float16
            quantization           = None,
            tensor_parallel_size   = 1,
            gpu_memory_utilization = 0.90,
            max_model_len          = 2668,   # <- CHANGE
            enforce_eager          = False,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens = 2668,   # = max_model_len when chunked=False
            num_scheduler_steps    = 1,
            compilation_config     = None,
            scheduler_delay_factor = 0.0,
            cpu_offload_gb         = 0,
            dtype                  = "float16",   # forced by FP8
            quantization           = "fp8",
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
            # Truncate to max_input_len to prevent vLLM ValueError
            input_ids_tensor = [
                self.data_object.input_ids[q.index][:self.max_input_len]
                for q in qitem
            ]
            outputs = self.model.generate(prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params, use_tqdm=False)
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(
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

sut_exp_C2() {
cat << 'SUTEOF'
# exp_C2: COMBO — FP8 + batch_size=1024
CHANGES: fp8+float16 AND batch_size=1024
Tests FP8 + large batch interaction for output non-determinism.
import os, array, threading, queue, logging, torch
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
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
            self.data_object.perf_count, self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42, max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype                  = "float16",   # <- FP8 path forces float16
            quantization           = None,
            tensor_parallel_size   = 1,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enforce_eager          = False,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens = 131072,
            num_scheduler_steps    = 1,
            compilation_config     = None,
            scheduler_delay_factor = 0.0,
            cpu_offload_gb         = 0,
            dtype                  = "float16",   # forced by FP8
            quantization           = "fp8",
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
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(
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

sut_exp_C3() {
cat << 'SUTEOF'
# exp_C3: COMBO — full optimal config (FP8+BS1024+len2668+sort)
THE FINAL ANSWER: does the best throughput config pass accuracy?
All throughput optimisations combined.
import os, array, threading, queue, logging, torch
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
        # Inputs truncated to self.max_input_len tokens.
        # vLLM raises ValueError (not silent truncation) for oversized inputs.
        # truncation_len = max_model_len(2668) - max_tokens(128) = 2540
        self.max_input_len = 2540
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(self.data_object.total_sample_count,
            self.data_object.perf_count, self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42, max_tokens=128, min_tokens=1,
        )
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype                  = "float16",   # <- FP8 path forces float16
            quantization           = None,
            tensor_parallel_size   = 1,
            gpu_memory_utilization = 0.90,
            max_model_len          = 2668,   # <- CHANGE
            enforce_eager          = False,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens = 2668,   # = max_model_len when chunked=False
            num_scheduler_steps    = 1,
            compilation_config     = None,
            scheduler_delay_factor = 0.0,
            cpu_offload_gb         = 0,
            dtype                  = "float16",   # forced by FP8
            quantization           = "fp8",
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
            # Truncate to max_input_len to prevent vLLM ValueError
            input_ids_tensor = [
                self.data_object.input_ids[q.index][:self.max_input_len]
                for q in qitem
            ]
            outputs = self.model.generate(prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params, use_tqdm=False)
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(
                    qitem[i].id, bi[0], bi[1], n_tokens)])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def issue_queries(self, query_samples):
        log.info(f"IssueQuery started with {len(query_samples)} samples")
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

class SUTServer(SUT): pass
SUTEOF
}

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print_final_summary() {
    log "========================================================"
    log "ACCURACY ABLATION — FINAL SUMMARY"
    log "Primary: ROUGE (rouge1, rouge2, rougeL, rougeLsum)"
    log "========================================================"

    local baseline_r1="N/A" baseline_r2="N/A" baseline_rL="N/A"
    local baseline_agg="${RESULTS_DIR}/exp_00/aggregate.txt"
    if [[ -f "${baseline_agg}" ]]; then
        baseline_r1=$(grep "rouge1\b"  "${baseline_agg}" | grep -oP "mean=\K[\d.]+" | head -1 || echo "N/A")
        baseline_r2=$(grep "rouge2\b"  "${baseline_agg}" | grep -oP "mean=\K[\d.]+" | head -1 || echo "N/A")
        baseline_rL=$(grep "rougeL\b"  "${baseline_agg}" | grep -oP "mean=\K[\d.]+" | head -1 || echo "N/A")
    fi

    printf "\n%-10s %-34s %-8s %-8s %-8s %-8s\n" "Exp" "Description" "r1" "Δr1" "rL" "ΔrL"
    printf "%-10s %-34s %-8s %-8s %-8s %-8s\n"   "---" "-----------" "--" "---" "--" "---"

    for agg_file in "${RESULTS_DIR}"/exp_*/aggregate.txt; do
        [[ -f "${agg_file}" ]] || continue
        local exp_id r1 rL dr1 drL
        exp_id=$(grep "^Experiment:" "${agg_file}" | cut -d: -f2- | xargs)
        desc=$(  grep -m1 "" "${agg_file}" | cut -c1-34)
        r1=$(    grep "rouge1\b"  "${agg_file}" | grep -oP "mean=\K[\d.]+" | head -1 || echo "N/A")
        rL=$(    grep "rougeL\b"  "${agg_file}" | grep -oP "mean=\K[\d.]+" | head -1 || echo "N/A")
        dr1="N/A"; drL="N/A"
        if [[ "${r1}" != "N/A" && "${baseline_r1}" != "N/A" ]]; then
            dr1=$(python3 -c "print(f'{${r1}-${baseline_r1}:+.4f}')" 2>/dev/null || echo "N/A")
        fi
        if [[ "${rL}" != "N/A" && "${baseline_rL}" != "N/A" ]]; then
            drL=$(python3 -c "print(f'{${rL}-${baseline_rL}:+.4f}')" 2>/dev/null || echo "N/A")
        fi
        printf "%-10s %-34s %-8s %-8s %-8s %-8s\n" "${exp_id}" "${desc}" "${r1}" "${dr1}" "${rL}" "${drL}"
    done

    local csv="${RESULTS_DIR}/accuracy_summary.csv"
    echo "exp_id,description,rouge1,rouge1_delta,rouge2,rouge2_delta,rougeL,rougeL_delta,rougeLsum,gen_len,tok_s" > "${csv}"
    for agg_json in "${RESULTS_DIR}"/exp_*/aggregate.json; do
        [[ -f "${agg_json}" ]] || continue
        python3 - "${agg_json}" "${baseline_r1}" "${baseline_r2}" "${baseline_rL}" >> "${csv}" 2>/dev/null << 'PYCSV'
import sys, json
path, b_r1, b_r2, b_rL = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
with open(path) as f: d = json.load(f)
def delta(v, b):
    try: return f"{float(v)-float(b):+.4f}"
    except: return "N/A"
def fmt(v): return f"{v:.4f}" if isinstance(v, float) else "N/A"
r1=d.get('rouge1_mean'); r2=d.get('rouge2_mean'); rL=d.get('rougeL_mean')
rLs=d.get('rougeLsum_mean'); gl=d.get('gen_len_mean'); tok=d.get('mean_tok_s')
print(f"{d.get('exp_id','')},\"{d.get('description','')}\","
      f"{fmt(r1)},{delta(r1,b_r1)},{fmt(r2)},{delta(r2,b_r2)},"
      f"{fmt(rL)},{delta(rL,b_rL)},{fmt(rLs)},{fmt(gl)},{fmt(tok)}")
PYCSV
    done
    log "CSV: ${csv}"
    log "Baseline: rouge1=${baseline_r1}  rouge2=${baseline_r2}  rougeL=${baseline_rL}"
}

# =============================================================================
# MAIN
# =============================================================================
main() {
    TARGET="${1:-all}"
    [[ "${RUN_SCOPE}" == "continue" || "${RUN_SCOPE}" == "only" ]] || RUN_SCOPE="continue"
    [[ "${TARGET}" == "all" ]] && TARGET_REACHED=1 || TARGET_REACHED=0

    log "========================================================"
    log "vLLM Accuracy Ablation — $(date)"
    log "RUNS_PER_EXP=${RUNS_PER_EXP} | 22 experiments (18 isolated + 3 combos)"
    log "Results: ${RESULTS_DIR} | MLflow: ${MLFLOW_EXPERIMENT_NAME}"
    log ""
    log "FIXES vs previous version:"
    log "  • ROUGE string values coerced to float — MLflow now logs them correctly"
    log "  • max_model_len experiments use input truncation — no more ValueError"
    log "  • RUNS_PER_EXP=1 (accuracy is deterministic with seed=42)"
    log "  • FP8 experiments added: exp_10, exp_C1, exp_C2, exp_C3"
    log "  • exp_16 gpu_memory_utilization bug fixed (0.95→0.90)"
    log "  • enable_chunked_prefill=False + max_num_batched_tokens=131072 explicit in all SUTs"
    log "========================================================"

    mkdir -p "${RESULTS_DIR}"
    [[ -d "${INFERENCE_DIR}" ]] || { error "INFERENCE_DIR not found: ${INFERENCE_DIR}"; exit 1; }
    [[ -f "${INFERENCE_DIR}/SUT_VLLM.py" ]] && \
        cp "${INFERENCE_DIR}/SUT_VLLM.py" "${RESULTS_DIR}/SUT_VLLM_original_backup.py"
    mkdir -p "${INFERENCE_DIR}/run_outputs"

    # Baseline
    run_experiment "exp_00" 16   "bfloat16" "sut_exp00" \
        "BASELINE: BF16 BS=16 all flags pinned. Reference ROUGE."

    # Control / greedy decode verification
    run_experiment "exp_01" 16   "bfloat16" "sut_exp01" \
        "temperature=0.0 explicit. Confirm greedy decode."
    run_experiment "exp_02" 16   "bfloat16" "sut_exp02" \
        "min_tokens=0. Allow empty outputs."
    run_experiment "exp_03" 16   "bfloat16" "sut_exp03" \
        "skip_special_tokens=True explicit. Confirm no hidden special tokens."
    run_experiment "exp_04" 16   "bfloat16" "sut_exp04" \
        "ignore_eos=False explicit. Confirm EOS stopping."

    # Output length sensitivity
    run_experiment "exp_05" 16   "bfloat16" "sut_exp05" \
        "max_tokens=256. Longer output budget."
    run_experiment "exp_06" 16   "bfloat16" "sut_exp06" \
        "max_tokens=64. Shorter output — negative control."

    # Batch size / non-determinism
    run_experiment "exp_07" 512  "bfloat16" "sut_exp07" \
        "batch_size=512. Non-determinism test."
    run_experiment "exp_08" 1024 "bfloat16" "sut_exp08" \
        "batch_size=1024. Strongest non-determinism test."

    # dtype / quantization
    run_experiment "exp_09" 16   "float16"  "sut_exp09" \
        "dtype=float16. Different rounding vs bfloat16."
    run_experiment "exp_10" 16   "float16"  "sut_exp10" \
        "FP8 quantization (fp8 weights + float16 activations)."

    # Processing order
    run_experiment "exp_11" 16   "bfloat16" "sut_exp11" \
        "sort_by_input_length. Should be accuracy-neutral."

    # KV cache truncation curve (all use input truncation)
    run_experiment "exp_12" 16   "bfloat16" "sut_exp12" \
        "max_model_len=2668. Input truncated to 2540 tokens."
    run_experiment "exp_13" 16   "bfloat16" "sut_exp13" \
        "max_model_len=1500. Input truncated to 1372 tokens."
    run_experiment "exp_14" 16   "bfloat16" "sut_exp14" \
        "max_model_len=1000. Input truncated to 872 tokens."

    # Feature correctness
    run_experiment "exp_15" 16   "bfloat16" "sut_exp15" \
        "enable_prefix_caching=True. Should equal baseline on unique articles."
    run_experiment "exp_16" 16   "bfloat16" "sut_exp16" \
        "chunked_prefill=True + batched_tokens=65536. Correctness check."

    # vLLM 0.10.0 correctness
    run_experiment "exp_17" 16   "bfloat16" "sut_exp17" \
        "num_scheduler_steps=4. Delayed EOS — any output change?"
    run_experiment "exp_18" 16   "bfloat16" "sut_exp18" \
        "compilation_config=3. Fused kernels numerical diff check."

    # Combination experiments
    run_experiment "exp_C1" 16   "float16"  "sut_exp_C1" \
        "COMBO: FP8 + max_model_len=2668. Throughput config accuracy check."
    run_experiment "exp_C2" 1024 "float16"  "sut_exp_C2" \
        "COMBO: FP8 + batch_size=1024. FP8+large batch interaction."
    run_experiment "exp_C3" 1024 "float16"  "sut_exp_C3" \
        "COMBO: Full optimal (FP8+BS1024+len2668+sort). Final accuracy answer."

    # Restore baseline
    sut_exp00 > "${INFERENCE_DIR}/SUT_VLLM.py"
    log "Restored baseline SUT"

    print_final_summary
}

main "$@"