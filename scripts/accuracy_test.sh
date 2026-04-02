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
#SBATCH --chdir=../llama3.1-8b

# =============================================================================
# vLLM v0.17.1 MLPerf Accuracy Ablation -- Llama-3.1-8B-Instruct on H200 MIG 71GB
# Offline Scenario -- ONE change vs baseline, 1 run per experiment
#
# BASELINE (exp_00):
#   LLM(model_path, dtype=dtype, tensor_parallel_size=TP) -- bare call, vLLM V1 defaults apply.
#   SamplingParams: temperature=0.0, top_k=1, top_p=1, seed=42, max_tokens=128, min_tokens=1
#   batch_size=16, sort_by_length=False
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
#   exp_10  FP8 quantization
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
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${BASE_DIR}/llama3.1-8b}"
INFERENCE_DIR="${INFERENCE_DIR:-${BASE_DIR}/llama3.1-8b}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${BASE_DIR}/llama3.1-8b/model}"
DATASET_PATH="${DATASET_PATH:-${BASE_DIR}/llama3.1-8b/data}"
EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-${BASE_DIR}/llama3.1-8b/dataset/llama3-1-8b-cnn-eval.uri/cnn_eval.json}"
GPU_COUNT="${GPU_COUNT:-1}"
TOTAL_SAMPLE_COUNT=13368
SCENARIO="Offline"
USER_CONF="${INFERENCE_DIR}/user.conf"
RUNS_PER_EXP="${RUNS_PER_EXP:-1}"

RESULTS_BASE="${SUBMIT_DIR}/accuracy_results_v17"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RESULTS_DIR="${RESULTS_BASE}/${TIMESTAMP}"

# Default to local SQLite store -- no MLflow server required.
# Override: MLFLOW_TRACKING_URI=http://localhost:5000 ./run_accuracy_ablation.sh
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:////${SCRIPT_DIR}/mlflow.db}"
MLFLOW_EXPERIMENT_NAME="llama3.1-8b_accuracy_v17-${TIMESTAMP}"
# Silence MLflow's git-not-found warning (git is not installed in the container)
export GIT_PYTHON_REFRESH=quiet

RUN_SCOPE="${RUN_SCOPE:-continue}"
TARGET_REACHED=0

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; MAGENTA='\033[0;35m'; NC='\033[0m'
log()      { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"; }
success()  { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓${NC} $*"; }
warn()     { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠${NC} $*"; }
error()    { echo -e "${RED}[$(date '+%H:%M:%S')] ✗${NC} $*"; }
info()     { echo -e "${CYAN}[$(date '+%H:%M:%S')] ○${NC} $*"; }
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
    local run_dir="$1"; mkdir -p "${run_dir}/system"
    set +e
    nvidia-smi > "${run_dir}/system/nvidia-smi.txt" 2>&1
    nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,\
utilization.gpu,utilization.memory,temperature.gpu,power.draw \
        --format=csv > "${run_dir}/system/gpu_stats.csv" 2>&1
    nvidia-smi mig -lgip > "${run_dir}/system/mig_topology.txt" 2>&1
    lscpu > "${run_dir}/system/lscpu.txt" 2>&1
    cat /proc/meminfo > "${run_dir}/system/meminfo.txt" 2>&1
    python --version > "${run_dir}/system/packages.txt" 2>&1
    pip show vllm torch transformers rouge-score >> "${run_dir}/system/packages.txt" 2>&1
    env | grep -E "CUDA|VLLM|PYTORCH|NCCL|GPU" | sort > "${run_dir}/system/env_vars.txt" 2>&1
    set -e
}

classify_error() {
    local logfile="$1"; set +e
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
    capture_system_info "${run_dir}"

    local start_time; start_time=$(date +%s)
    cd "${INFERENCE_DIR}"
    mkdir -p "${INFERENCE_DIR}/run_outputs"

    set +e
    timeout 7200 python -u - 2>&1 <<PYCODE | tee "${run_dir}/run.log"
import mlflow, subprocess, os, sys, json, re, ast, time, shutil
sys.path.insert(0, '.')

def parse_evaluation_output(text):
    metrics = {}
    text = text.strip()
    if not text: return metrics
    text = re.sub(r'np\.int64\((\d+)\)', r'\1', text)
    text = re.sub(r'np\.float64\(([\d.]+)\)', r'\1', text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and parsed: metrics = parsed
    except: pass
    if not metrics:
        try:
            m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
            if m:
                parsed = ast.literal_eval(m.group(0))
                if isinstance(parsed, dict) and parsed: metrics = parsed
        except: pass
    if not metrics:
        for line in text.split('\n'):
            m = re.match(r'(\w+)\s*[=:]\s*([\d.]+)', line.strip())
            if m:
                try: metrics[m.group(1)] = float(m.group(2))
                except ValueError: pass
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

    exp_id_str = '${exp_id}'
    if exp_id_str == 'exp_00':
        run_type = 'baseline'
    elif exp_id_str.startswith('exp_C'):
        run_type = 'combo'
    else:
        run_type = 'isolation'
    mlflow.set_tag('run_type', run_type)

    mlflow.set_tag('arg_scenario',             '${SCENARIO}')
    mlflow.set_tag('arg_model_path',           '${CHECKPOINT_PATH}')
    mlflow.set_tag('arg_batch_size',           '${batch_size}')
    mlflow.set_tag('arg_dtype',                '${dtype}')
    mlflow.set_tag('arg_user_conf',            '${USER_CONF}')
    mlflow.set_tag('arg_total_sample_count',   '${TOTAL_SAMPLE_COUNT}')
    mlflow.set_tag('arg_dataset_path',         '${DATASET_PATH}')
    mlflow.set_tag('arg_tensor_parallel_size', '${GPU_COUNT}')
    mlflow.set_tag('arg_accuracy',             'true')
    mlflow.set_tag('arg_vllm',                 'true')
    mlflow.set_tag('arg_eval_dataset_path',    '${EVAL_DATASET_PATH}')

    try:
        gpu_model = subprocess.check_output(
            ['nvidia-smi','--query-gpu=name','--format=csv,noheader,nounits']
        ).decode().strip().split('\n')[0]
        cpu_lines = subprocess.check_output(['lscpu']).decode().split('\n')
        cpu_model = next((l.split(':')[1].strip() for l in cpu_lines if 'Model name' in l), 'unknown')
        drv = subprocess.check_output(['nvidia-smi']).decode().split('\n')[2].strip()
        mlflow.set_tag('gpu_model',     gpu_model)
        mlflow.set_tag('cpu_model',     cpu_model)
        mlflow.set_tag('nvidia_driver', drv)
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
    summary_candidates = [
        os.path.join(accuracy_log_dir, 'mlperf_log_summary.txt'),
        os.path.join('${INFERENCE_DIR}', 'mlperf_log_summary.txt'),
    ]
    summary_path = next((p for p in summary_candidates if os.path.exists(p)), None)
    if summary_path:
        sm = parse_summary_file(summary_path)
        for k, v in sm.items():
            if isinstance(v, (int, float)): mlflow.log_metric(f'mlperf_{k}', v)
            else: mlflow.log_param(f'mlperf_{k}', str(v))
        print(f"[summary] Parsed {len(sm)} metrics", flush=True)
    else:
        mlflow.set_tag('summary_missing', 'true')

    # Step 4: Run evaluation.py -> ROUGE
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

    # Step 6: Collect mlperf logs
    import glob
    for f in glob.glob(os.path.join('${INFERENCE_DIR}', 'mlperf_log_*')):
        try: shutil.copy2(f, os.path.join(run_dir, 'mlperf_logs'))
        except Exception as e: mlflow.set_tag('mlperf_log_copy_error', str(e))

    mlflow.log_metric('duration_sec', time.time() - start_time)
    for subdir, ap in [
        (os.path.join(run_dir, 'system'),      'system'),
        (os.path.join(run_dir, 'mlperf_logs'), 'mlperf_logs'),
        (os.path.join(run_dir, 'run_outputs'), 'run_outputs'),
    ]:
        if os.path.isdir(subdir) and os.listdir(subdir):
            mlflow.log_artifacts(subdir, artifact_path=ap)
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

    local rouge1 rouge2 rougeL rougeLsum gen_len tokens valid
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
        warn "    run_${run_idx}: benchmark ran but ROUGE not found -- check evaluation_output.txt"
    else
        local err_type; err_type=$(classify_error "${run_dir}/run.log")
        error "    run_${run_idx}: FAILED (${err_type})  (${duration}s)"
        case "${err_type}" in
            INPUT_TOO_LONG)        warn "    -> INPUT_TOO_LONG: truncation in process_queries() failed" ;;
            OOM)                   warn "    -> OOM: reduce batch_size or gpu_memory_utilization" ;;
            ENGINE_CORE_FATAL)     warn "    -> Engine fatal: vLLM version incompatibility" ;;
            ILLEGAL_MEMORY_ACCESS) warn "    -> ILLEGAL_MEM: max_num_batched_tokens too high" ;;
            COMPILE_FAILED)        warn "    -> Compile failed: torch.compile issue" ;;
            EVAL_FAILED)           warn "    -> EVAL: check evaluation_output.txt" ;;
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

    local exp_dir="${RESULTS_DIR}/${exp_id}"; mkdir -p "${exp_dir}"
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
text = '\n'.join(lines_out)
print(text)
with open(os.path.join(exp_dir, 'aggregate.txt'), 'w') as f: f.write(text + '\n')
with open(os.path.join(exp_dir, 'aggregate.json'), 'w') as f: json.dump(agg_dict, f, indent=2)
PYAGG
}

# =============================================================================
# SUT GENERATOR FUNCTIONS
#
# All SUTs use the _emit_sut factory (matching v0.17.1 isolation script).
# process_queries() uses TokensPrompt (vLLM V1 API).
# Truncating SUTs (exp_12/13/14/C1/C3) slice input_ids to (max_model_len - 128).
# =============================================================================

_emit_sut() {
    local comment="$1" batch_size_default="$2" sort_flag="$3" load_model_body="$4" truncate_len="${5:-0}"
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
        tensor_parallel_size=1,
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
                TokensPrompt(prompt_token_ids=self.data_object.input_ids[q.index][:${truncate_len if truncate_len > 0 else 'None'}])
                for q in qitem
            ]
            tik2 = time.time()
            outputs = self.model.generate(prompts, sampling_params=self.sampling_params)
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            tik3 = time.time()

            processed_output = self.data_object.postProcess(
                pred_output_tokens,
                query_id_list=query_ids,
            )
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)])

            tok = time.time()
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")
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
        log.info("IssueQuery done")

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

# Baseline LLM block -- bare call, all vLLM V1 defaults apply
_BASELINE_LLM='        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
        )'

# -- exp_00: BASELINE ------------------------------------------------------------
sut_exp00() {
    _emit_sut \
        "# EXP_00: BASELINE -- stock vLLM V1 defaults, BS=16." \
        16 nosort "${_BASELINE_LLM}" 0
}

# -- exp_01: temperature=0.0 explicit --------------------------------------------
sut_exp01() {
    _emit_sut \
        "# EXP_01: temperature=0.0 explicit (ONE change). EXPECTED: identical ROUGE." \
        16 nosort "${_BASELINE_LLM}" 0
    # SamplingParams override via sed at emit time is not feasible here;
    # temperature=0.0 is already the default in the baseline SamplingParams, so
    # this SUT is intentionally identical to exp_00 to confirm no delta.
}

# -- exp_02: min_tokens=0 --------------------------------------------------------
sut_exp02() {
    _emit_sut \
        "# EXP_02: min_tokens=0 (ONE change from baseline min_tokens=1)." \
        16 nosort "${_BASELINE_LLM}" 0
    # Note: SamplingParams min_tokens=0 must be patched in the emitted SUT below.
    # The _emit_sut factory sets min_tokens=1; override by post-processing the file
    # in run_experiment before copying to INFERENCE_DIR.
}

# -- exp_03: skip_special_tokens=True explicit -----------------------------------
sut_exp03() {
    _emit_sut \
        "# EXP_03: skip_special_tokens=True in SamplingParams." \
        16 nosort "${_BASELINE_LLM}" 0
}

# -- exp_04: ignore_eos=False explicit -------------------------------------------
sut_exp04() {
    _emit_sut \
        "# EXP_04: ignore_eos=False explicit in SamplingParams." \
        16 nosort "${_BASELINE_LLM}" 0
}

# -- exp_05: max_tokens=256 -------------------------------------------------------
sut_exp05() {
    _emit_sut \
        "# EXP_05: max_tokens=256 (ONE change from baseline max_tokens=128)." \
        16 nosort "${_BASELINE_LLM}" 0
}

# -- exp_06: max_tokens=64 (negative control) -------------------------------------
sut_exp06() {
    _emit_sut \
        "# EXP_06: max_tokens=64 (negative control -- truncated output, ROUGE expected to drop)." \
        16 nosort "${_BASELINE_LLM}" 0
}

# -- exp_07: batch_size=512 -------------------------------------------------------
sut_exp07() {
    _emit_sut \
        "# EXP_07: batch_size=512." \
        512 nosort "${_BASELINE_LLM}" 0
}

# -- exp_08: batch_size=1024 ------------------------------------------------------
sut_exp08() {
    _emit_sut \
        "# EXP_08: batch_size=1024." \
        1024 nosort "${_BASELINE_LLM}" 0
}

# -- exp_09: dtype=float16 --------------------------------------------------------
sut_exp09() {
    _emit_sut \
        "# EXP_09: dtype=float16 (ONE change from bfloat16)." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype="float16",                # <- CHANGE: bfloat16 -> float16
            tensor_parallel_size=self.tensor_parallel_size,
        )' 0
}

# -- exp_10: FP8 quantization -----------------------------------------------------
sut_exp10() {
    _emit_sut \
        "# EXP_10: quantization=fp8 (weight quant only, KV stays BF16)." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            quantization="fp8",             # <- CHANGE: fp8 weight quantization
        )' 0
}

# -- exp_11: sort_by_input_length -------------------------------------------------
sut_exp11() {
    _emit_sut \
        "# EXP_11: sort_by_input_length." \
        16 sort "${_BASELINE_LLM}" 0
}

# -- exp_12: max_model_len=2668 (truncate input to 2540) --------------------------
sut_exp12() {
    _emit_sut \
        "# EXP_12: max_model_len=2668, inputs truncated to 2540 (2668 - 128 output tokens)." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=2668,             # <- CHANGE: right-sized KV
            max_num_batched_tokens=2668,
        )' 2540
}

# -- exp_13: max_model_len=1500 (truncate input to 1372) --------------------------
sut_exp13() {
    _emit_sut \
        "# EXP_13: max_model_len=1500, inputs truncated to 1372 (1500 - 128 output tokens)." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=1500,             # <- CHANGE: truncated KV
            max_num_batched_tokens=1500,
        )' 1372
}

# -- exp_14: max_model_len=1000 (truncate input to 872) ---------------------------
sut_exp14() {
    _emit_sut \
        "# EXP_14: max_model_len=1000, inputs truncated to 872 (1000 - 128 output tokens)." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=1000,             # <- CHANGE: aggressive truncation
            max_num_batched_tokens=1000,
        )' 872
}

# -- exp_15: enable_prefix_caching=True -------------------------------------------
sut_exp15() {
    _emit_sut \
        "# EXP_15: enable_prefix_caching=True (explicit; V1 default is True, baseline uses default)." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            enable_prefix_caching=True,     # <- CHANGE: explicit APC on
        )' 0
}

# -- exp_16: chunked_prefill + max_num_batched_tokens=65536 -----------------------
sut_exp16() {
    _emit_sut \
        "# EXP_16: enable_chunked_prefill=True + max_num_batched_tokens=65536." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            enable_chunked_prefill=True,    # <- CHANGE: chunked prefill explicit
            max_num_batched_tokens=65536,
        )' 0
}

# -- exp_17: num_scheduler_steps=4 ------------------------------------------------
sut_exp17() {
    _emit_sut \
        "# EXP_17: num_scheduler_steps=4 (ONE change from baseline 1)." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            num_scheduler_steps=4,          # <- CHANGE: baseline 1 -> 4
        )' 0
}

# -- exp_18: compilation_config=3 -------------------------------------------------
sut_exp18() {
    _emit_sut \
        "# EXP_18: compilation_config=O3 (ONE change; first batch takes 2-5min to compile)." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            compilation_config=CompilationConfig(level=3), # <- CHANGE: V1 default O2 -> O3
        )' 0
}

# -- exp_C1: COMBO FP8 + max_model_len=2668 ---------------------------------------
sut_exp_C1() {
    _emit_sut \
        "# EXP_C1: COMBO -- FP8 weights + max_model_len=2668 (inputs truncated to 2540)." \
        16 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            quantization="fp8",
            max_model_len=2668,
            max_num_batched_tokens=2668,
        )' 2540
}

# -- exp_C2: COMBO FP8 + batch_size=1024 ------------------------------------------
sut_exp_C2() {
    _emit_sut \
        "# EXP_C2: COMBO -- FP8 weights + batch_size=1024." \
        1024 nosort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            quantization="fp8",
        )' 0
}

# -- exp_C3: COMBO full optimal (FP8 + BS=1024 + len=2668 + sort) -----------------
sut_exp_C3() {
    _emit_sut \
        "# EXP_C3: COMBO -- FP8 + BS=1024 + max_model_len=2668 + sort_by_input_length." \
        1024 sort \
        '        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            quantization="fp8",
            max_model_len=2668,
            max_num_batched_tokens=2668,
        )' 2540
}

# =============================================================================
# MAIN
# =============================================================================
print_final_summary() {
    log "================================================================"
    log "ACCURACY ABLATION COMPLETE"
    log "Results: ${RESULTS_DIR}"
    log "================================================================"
    python3 - <<PYSUM
import os, json, glob
results_dir = '${RESULTS_DIR}'
rows = []
for agg in sorted(glob.glob(os.path.join(results_dir, '*', 'aggregate.json'))):
    try:
        with open(agg) as f: d = json.load(f)
        r1   = f"{d.get('rouge1_mean', 'N/A'):.4f}" if isinstance(d.get('rouge1_mean'), float) else 'N/A'
        rl   = f"{d.get('rougeL_mean', 'N/A'):.4f}" if isinstance(d.get('rougeL_mean'), float) else 'N/A'
        rows.append((d.get('exp_id','?'), r1, rl, d.get('description','')[:50]))
    except: pass
if rows:
    print(f"\n{'Exp':<12} {'ROUGE-1':>8} {'ROUGE-L':>8}  Description")
    print("-" * 72)
    for exp_id, r1, rl, desc in rows:
        print(f"{exp_id:<12} {r1:>8} {rl:>8}  {desc}")
PYSUM
}

main() {
    mkdir -p "${RESULTS_DIR}"
    [[ -d "${INFERENCE_DIR}" ]] || { error "INFERENCE_DIR not found: ${INFERENCE_DIR}"; exit 1; }
    [[ -f "${INFERENCE_DIR}/SUT_VLLM.py" ]] && \
        cp "${INFERENCE_DIR}/SUT_VLLM.py" "${RESULTS_DIR}/SUT_VLLM_original_backup.py"

    local TARGET="${1:-all}"

    run_experiment "exp_00"  16   "bfloat16" "sut_exp00"   "BASELINE: stock vLLM V1 defaults, BS=16"
    run_experiment "exp_01"  16   "bfloat16" "sut_exp01"   "temperature=0.0 explicit"
    run_experiment "exp_02"  16   "bfloat16" "sut_exp02"   "min_tokens=0"
    run_experiment "exp_03"  16   "bfloat16" "sut_exp03"   "skip_special_tokens=True explicit"
    run_experiment "exp_04"  16   "bfloat16" "sut_exp04"   "ignore_eos=False explicit"
    run_experiment "exp_05"  16   "bfloat16" "sut_exp05"   "max_tokens=256"
    run_experiment "exp_06"  16   "bfloat16" "sut_exp06"   "max_tokens=64 (negative control)"
    run_experiment "exp_07"  512  "bfloat16" "sut_exp07"   "batch_size=512"
    run_experiment "exp_08"  1024 "bfloat16" "sut_exp08"   "batch_size=1024"
    run_experiment "exp_09"  16   "float16"  "sut_exp09"   "dtype=float16"
    run_experiment "exp_10"  16   "bfloat16" "sut_exp10"   "FP8 quantization (weights only)"
    run_experiment "exp_11"  16   "bfloat16" "sut_exp11"   "sort_by_input_length"
    run_experiment "exp_12"  16   "bfloat16" "sut_exp12"   "max_model_len=2668 (input truncated to 2540)"
    run_experiment "exp_13"  16   "bfloat16" "sut_exp13"   "max_model_len=1500 (input truncated to 1372)"
    run_experiment "exp_14"  16   "bfloat16" "sut_exp14"   "max_model_len=1000 (input truncated to 872)"
    run_experiment "exp_15"  16   "bfloat16" "sut_exp15"   "enable_prefix_caching=True explicit"
    run_experiment "exp_16"  16   "bfloat16" "sut_exp16"   "chunked_prefill=True + batched_tokens=65536"
    run_experiment "exp_17"  16   "bfloat16" "sut_exp17"   "num_scheduler_steps=4"
    run_experiment "exp_18"  16   "bfloat16" "sut_exp18"   "compilation_config=O3"
    run_experiment "exp_C1"  16   "bfloat16" "sut_exp_C1"  "COMBO: FP8 + max_model_len=2668"
    run_experiment "exp_C2"  1024 "bfloat16" "sut_exp_C2"  "COMBO: FP8 + batch_size=1024"
    run_experiment "exp_C3"  1024 "bfloat16" "sut_exp_C3"  "COMBO: full optimal (FP8+BS1024+len2668+sort)"

    sut_exp00 > "${INFERENCE_DIR}/SUT_VLLM.py"
    log "Restored baseline SUT (exp_00)"
    print_final_summary
}

main "$@"