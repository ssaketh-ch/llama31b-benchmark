#!/bin/bash
#SBATCH --job-name=llama3-8b-server
#SBATCH --output=llama3-8b-server-%j.out
#SBATCH --error=llama3-8b-server-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:mig-3g.71gb:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=mig_nodes
#SBATCH --chdir=/home/saketh-msc/inference/language/llama3.1-8b

# =============================================================================
# vLLM MLPerf Server Scenario Ablation — Llama-3.1-8B-Instruct / H200 MIG 71GB
#
# DESIGN PRINCIPLES
# ─────────────────
# 1. ONE change vs baseline per experiment.
#    Combinations (exp_C*) are run only after all individual results are known.
#
# 2. EVERY vLLM knob is pinned EXPLICITLY in EVERY experiment — including
#    flags that are not being tested. This prevents vLLM auto-settings from
#    producing phantom results. For example:
#      • enable_chunked_prefill=False is set in every non-chunked experiment
#        because vLLM 0.10 may auto-enable it when max_num_batched_tokens is set
#      • enforce_eager=False (CUDA Graphs ON) is set in every experiment except
#        the one explicitly testing CUDA Graphs OFF
#      • max_num_seqs=256 (vLLM default) is pinned in every experiment not
#        testing max_num_seqs, so the concurrency level is known
#      • preemption_mode="recompute" (vLLM default) is pinned everywhere except
#        the experiment testing preemption_mode=swap
#      • scheduler_delay_factor=0.0, num_scheduler_steps=1, async_scheduling=False
#        are all pinned in every non-testing experiment
#      • PYTORCH_CUDA_ALLOC_CONF is explicitly UNSET before every experiment
#        except exp_17 which tests expandable_segments
#
# 3. SUT: every experiment uses the canonical SUTServer with AsyncLLMEngine.
#    Only load_model() is overridden. This ensures lg.FirstTokenComplete()
#    is always called correctly and TTFT / TPOT metrics are non-zero.
#
# 4. Metrics captured:
#      Throughput : Completed tokens/sec, Completed samples/sec
#      Validity   : VALID/INVALID, TTFT early stopping, TPOT early stopping
#      E2E latency: mean/p50/p90/p95/p97/p99/p99.9/min/max (ns → ms in result.txt)
#      TTFT       : mean/p50/p90/p95/p97/p99/p99.9/min/max first-token latency
#      TPOT       : mean/p50/p90/p95/p97/p99/p99.9/min/max time-per-output-token
#      Config     : target_qps, ttft_constraint_ns, tpot_constraint_ns
#
# EXPERIMENT MATRIX
# ─────────────────────────────────────────────────────────────────────────────
# BASELINE (exp_00)
#   All flags explicit. No knobs changed.
#   AsyncEngineArgs:
#     dtype=bfloat16, quantization=None, gpu_mem=0.90, max_model_len=131072,
#     enable_prefix_caching=False, enable_chunked_prefill=False,
#     enforce_eager=False (CUDA Graphs ON), max_num_seqs=256,
#     scheduler_delay_factor=0.0, preemption_mode=recompute,
#     num_scheduler_steps=1, async_scheduling=False
#     max_num_batched_tokens: NOT SET (vLLM auto when chunked_prefill=False)
#   env: PYTORCH_CUDA_ALLOC_CONF unset
#
# INDIVIDUAL (one change each, all others pinned to baseline):
#   exp_01  enforce_eager=True         CUDA Graphs OFF  (regression control)
#   exp_02  chunked_prefill=T + 8192   TTFT sweep fine
#   exp_03  chunked_prefill=T + 16384  TTFT sweep medium
#   exp_04  chunked_prefill=T + 32768  TTFT sweep coarse
#   exp_05  max_model_len=2668         KV density
#   exp_06  scheduler_delay=0.05       batch accumulation sweep
#   exp_07  scheduler_delay=0.10
#   exp_08  scheduler_delay=0.20
#   exp_09  max_num_seqs=128           concurrency sweep (restrict)
#   exp_10  max_num_seqs=512           concurrency sweep (expand)
#   exp_11  max_num_seqs=1024          concurrency sweep (high)
#   exp_12  preemption_mode=recompute  control (= baseline, confirms pinning)
#   exp_13  preemption_mode=swap       preemption mode comparison
#   exp_14  num_scheduler_steps=2      multi-step sweep
#   exp_15  num_scheduler_steps=4
#   exp_16  async_scheduling=True      async CPU/GPU overlap
#   exp_17  expandable_segments=True   allocator (env var only)
#   exp_18  fp8 + float16              quantization (inseparable pair)
#
# COMBINATIONS (run after all individual results known):
#   exp_C1  chunked_16384 + max_len=2668
#   exp_C2  chunked_16384 + async_scheduling
#   exp_C3  max_len=2668 + max_num_seqs=512
#   exp_C4  EDIT BEFORE RUNNING — insert winners from exp_01..18
#
# Usage:
#   chmod +x server_run.sh
#   ./server_run.sh                      # all experiments
#   TARGET=exp_05 ./server_run.sh        # single experiment
#   RUNS_PER_EXP=2 ./server_run.sh       # fewer runs (debug)
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
INFERENCE_DIR="${INFERENCE_DIR:-/home/saketh-msc/inference/language/llama3.1-8b}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/saketh-msc/inference/language/llama3.1-8b/model}"
DATASET_PATH="${DATASET_PATH:-/home/saketh-msc/inference/language/llama3.1-8b/data}"
GPU_COUNT="${GPU_COUNT:-1}"
TOTAL_SAMPLE_COUNT=13368
SCENARIO="Server"
USER_CONF="${INFERENCE_DIR}/user.conf"
RUNS_PER_EXP="${RUNS_PER_EXP:-4}"   # was set twice (4 then 1) in original — fixed

RESULTS_BASE="${SUBMIT_DIR}/server_results"
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
RESULTS_DIR="${RESULTS_BASE}/${TIMESTAMP}"

MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
MLFLOW_EXPERIMENT_NAME="llama3.1-8b_server-${TIMESTAMP}"

TARGET="${TARGET:-all}"

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
    pkill -f "vllm"     2>/dev/null; sleep 2
    pkill -f "SUT_VLLM" 2>/dev/null; sleep 2
    pkill -f "main.py"  2>/dev/null; sleep 2
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
    nvidia-smi -q              > "${run_dir}/system/nvidia-smi-full.txt"  2>&1
    nvidia-smi mig -lgip       > "${run_dir}/system/mig_topology.txt"     2>&1
    nvidia-smi mig -lgi       >> "${run_dir}/system/mig_topology.txt"     2>&1
    nvidia-smi mig -lci       >> "${run_dir}/system/mig_topology.txt"     2>&1
    lscpu                      > "${run_dir}/system/lscpu.txt"             2>&1
    cat /proc/meminfo          > "${run_dir}/system/meminfo.txt"           2>&1
    python --version           > "${run_dir}/system/packages.txt"          2>&1
    pip show vllm torch transformers >> "${run_dir}/system/packages.txt"   2>&1
    env | grep -E "CUDA|VLLM|PYTORCH|NCCL|GPU" | sort \
                               > "${run_dir}/system/env_vars.txt"          2>&1
    set -e
}

classify_error() {
    local logfile="$1"
    set +e
    if   grep -q "out of memory\|CUBLAS_STATUS_ALLOC_FAILED\|CUDA out of memory" \
              "${logfile}" 2>/dev/null; then echo "OOM"
    elif grep -q "illegal memory access"       "${logfile}" 2>/dev/null; then echo "ILLEGAL_MEMORY_ACCESS"
    elif grep -q "Segmentation fault\|SIGSEGV" "${logfile}" 2>/dev/null; then echo "SEGFAULT"
    elif grep -q "EngineCore\|EngineDeadError" "${logfile}" 2>/dev/null; then echo "ENGINE_CORE_FATAL"
    elif grep -q "Timeout\|timed out"          "${logfile}" 2>/dev/null; then echo "TIMEOUT"
    elif grep -q "compilation\|inductor"       "${logfile}" 2>/dev/null; then echo "COMPILE_FAILED"
    elif grep -q "RuntimeError"                "${logfile}" 2>/dev/null; then echo "RUNTIME_ERROR"
    else echo "UNKNOWN"
    fi
    set -e
}

# ---------------------------------------------------------------------------
# Core server benchmark runner — one individual run
# ---------------------------------------------------------------------------
run_single() {
    local exp_id="$1"
    local run_idx="$2"
    local batch_size="$3"
    local dtype="$4"
    local desc="$5"
    local run_dir="${RESULTS_DIR}/${exp_id}/run_${run_idx}"

    mkdir -p "${run_dir}/system" "${run_dir}/mlperf_logs"
    info "  Run ${run_idx}/${RUNS_PER_EXP}: ${exp_id} (BS=${batch_size}, dtype=${dtype})"

    set +e
    nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu,power.draw \
        --format=csv,noheader > "${run_dir}/system/gpu_pre_run.txt" 2>/dev/null
    set -e

    capture_system_info "${run_dir}"

    local start_time; start_time=$(date +%s)
    cd "${INFERENCE_DIR}"

    set +e
    python -u - 2>&1 <<PYCODE | tee "${run_dir}/run.log"
import mlflow, subprocess, os, sys, json, re, time
sys.path.insert(0, '.')

# ── Full MLPerf Server summary parser ─────────────────────────────────────────
def parse_server_summary(path):
    """
    Parses ALL metrics from mlperf_log_summary.txt for the Server scenario.
    Returns a flat dict. All latency values are in nanoseconds (raw from log).
    """
    metrics = {}

    KEY_MAP = {
        # Throughput
        'Completed samples per second'              : 'completed_samples_per_sec',
        'Completed tokens per second'               : 'completed_tokens_per_sec',
        'Result is'                                 : 'result_valid',
        'Scheduled samples per second'              : 'scheduled_samples_per_sec',
        # E2E latency
        'Min latency (ns)'                          : 'e2e_min_ns',
        'Max latency (ns)'                          : 'e2e_max_ns',
        'Mean latency (ns)'                         : 'e2e_mean_ns',
        '50.00 percentile latency (ns)'             : 'e2e_p50_ns',
        '90.00 percentile latency (ns)'             : 'e2e_p90_ns',
        '95.00 percentile latency (ns)'             : 'e2e_p95_ns',
        '97.00 percentile latency (ns)'             : 'e2e_p97_ns',
        '99.00 percentile latency (ns)'             : 'e2e_p99_ns',
        '99.90 percentile latency (ns)'             : 'e2e_p99_9_ns',
        # TTFT — Time to First Token
        'Min First Token latency (ns)'              : 'ttft_min_ns',
        'Max First Token latency (ns)'              : 'ttft_max_ns',
        'Mean First Token latency (ns)'             : 'ttft_mean_ns',
        '50.00 percentile first token latency (ns)' : 'ttft_p50_ns',
        '90.00 percentile first token latency (ns)' : 'ttft_p90_ns',
        '95.00 percentile first token latency (ns)' : 'ttft_p95_ns',
        '97.00 percentile first token latency (ns)' : 'ttft_p97_ns',
        '99.00 percentile first token latency (ns)' : 'ttft_p99_ns',
        '99.90 percentile first token latency (ns)' : 'ttft_p99_9_ns',
        # TPOT — Time per Output Token
        'Min Time per Output Token (ns)'                 : 'tpot_min_ns',
        'Max Time per Output Token (ns)'                 : 'tpot_max_ns',
        'Mean Time per Output Token (ns)'                : 'tpot_mean_ns',
        '50.00 percentile time to output token (ns)'     : 'tpot_p50_ns',
        '90.00 percentile time to output token (ns)'     : 'tpot_p90_ns',
        '95.00 percentile time to output token (ns)'     : 'tpot_p95_ns',
        '97.00 percentile time to output token (ns)'     : 'tpot_p97_ns',
        '99.00 percentile time to output token (ns)'     : 'tpot_p99_ns',
        '99.90 percentile time to output token (ns)'     : 'tpot_p99_9_ns',
        # Test parameters
        'target_qps'                                : 'target_qps',
        'ttft_latency (ns)'                         : 'ttft_constraint_ns',
        'tpot_latency (ns)'                         : 'tpot_constraint_ns',
        'min_duration (ms)'                         : 'min_duration_ms',
        'min_query_count'                           : 'min_query_count',
        'samples_per_query'                         : 'samples_per_query',
    }

    try:
        with open(path) as f:
            content = f.read()

        # TTFT / TPOT early stopping pass/fail lines
        for label, key in [('TTFT Early Stopping', 'ttft_early_stopping'),
                            ('TPOT Early Stopping', 'tpot_early_stopping')]:
            m = re.search(rf'{label} Result:\s*\n\s*\*\s*(.*)', content)
            if m:
                metrics[key] = m.group(1).strip()

        for line in content.split('\n'):
            line = line.strip()
            for prefix, key in KEY_MAP.items():
                if line.startswith(prefix):
                    raw = line.split(':', 1)[-1].strip()
                    try:
                        metrics[key] = float(raw)
                    except ValueError:
                        metrics[key] = raw
                    break
    except Exception as e:
        metrics['parse_error'] = str(e)
    return metrics


def ns_to_ms(v):
    try:
        f = float(v)
        return f"{f/1e6:.2f}" if f != 0 else "0.00 ⚠ (FirstTokenComplete not called)"
    except (TypeError, ValueError):
        return "N/A"


run_name = '${exp_id}_run${run_idx}'
run_dir  = '${run_dir}'
exp_id   = '${exp_id}'
run_idx  = ${run_idx}

mlflow.set_tracking_uri('${MLFLOW_TRACKING_URI}')
mlflow.set_experiment('${MLFLOW_EXPERIMENT_NAME}')

start_ts = time.time()
mlflow.start_run(run_name=run_name)
try:
    # ── Parameters ────────────────────────────────────────────────────────
    mlflow.log_param('exp_id',               exp_id)
    mlflow.log_param('run_index',            run_idx)
    mlflow.log_param('batch_size',           ${batch_size})
    mlflow.log_param('dtype',                '${dtype}')
    mlflow.log_param('description',          '${desc}')
    mlflow.log_param('scenario',             '${SCENARIO}')
    mlflow.log_param('total_sample_count',   ${TOTAL_SAMPLE_COUNT})
    mlflow.log_param('gpu_count',            ${GPU_COUNT})
    mlflow.log_param('tensor_parallel_size', '${GPU_COUNT}')
    mlflow.log_param('model_path',           '${CHECKPOINT_PATH}')
    mlflow.log_param('dataset_path',         '${DATASET_PATH}')
    mlflow.log_param('runs_per_exp',         ${RUNS_PER_EXP})
    mlflow.log_param('mode',                 'server_performance')

    # ── Version / hardware tags ──────────────────────────────────────────
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
        mlflow.set_tag('gpu_model', gpu); mlflow.set_tag('cpu_model', cpu)
        mlflow.set_tag('nvidia_driver', drv); mlflow.set_tag('host', host)
    except Exception as e:
        mlflow.set_tag('hw_tag_error', str(e))

    mlflow.set_tag('exp_id', exp_id); mlflow.set_tag('run_index', str(run_idx))
    mlflow.set_tag('dtype', '${dtype}'); mlflow.set_tag('scenario', '${SCENARIO}')

    # ── Pre-run GPU snapshot ─────────────────────────────────────────────
    try:
        pg = subprocess.check_output([
            'nvidia-smi','--query-gpu=memory.used,memory.free,temperature.gpu,'
            'power.draw,utilization.gpu','--format=csv,noheader,nounits'
        ]).decode().strip().split('\n')[0].split(',')
        for k, v in zip(['mem_used_mb','mem_free_mb','temp_c','power_w','util_gpu_pct'], pg):
            try: mlflow.log_metric(f'gpu_pre_{k}', float(v.strip()))
            except: pass
    except: pass

    # ── Run benchmark ────────────────────────────────────────────────────
    bench_cmd = [
        'python', '-u', 'main.py',
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
    bench_proc = subprocess.run(bench_cmd, check=False)
    mlflow.log_param('bench_exit_code', bench_proc.returncode)

    # ── Post-run GPU snapshot ────────────────────────────────────────────
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
                try:
                    mlflow.log_metric(f'gpu_post_{h}',
                                      float(''.join(c for c in v if c in '0123456789.')))
                except: pass
    except Exception as e:
        mlflow.set_tag('gpu_post_run_error', str(e))

    # ── Parse MLPerf server summary ──────────────────────────────────────
    summary_candidates = [
        os.path.join(run_dir, 'mlperf_logs', 'mlperf_log_summary.txt'),
        os.path.join('${INFERENCE_DIR}', 'mlperf_log_summary.txt'),
    ]
    summary_path = next((p for p in summary_candidates if os.path.exists(p)), None)
    sm = {}
    if summary_path:
        sm = parse_server_summary(summary_path)
        for k, v in sm.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v)
            else:
                mlflow.set_tag(k, str(v))
        print("\n" + "="*62, flush=True)
        print("SERVER METRICS SUMMARY", flush=True)
        print("="*62, flush=True)
        print(f"  Tokens/sec          : {sm.get('completed_tokens_per_sec','N/A')}", flush=True)
        print(f"  Samples/sec         : {sm.get('completed_samples_per_sec','N/A')}", flush=True)
        print(f"  Result              : {sm.get('result_valid','N/A')}", flush=True)
        print(f"  TTFT early stop     : {sm.get('ttft_early_stopping','N/A')}", flush=True)
        print(f"  TPOT early stop     : {sm.get('tpot_early_stopping','N/A')}", flush=True)
        print(f"", flush=True)
        print(f"  TTFT mean           : {ns_to_ms(sm.get('ttft_mean_ns'))} ms", flush=True)
        print(f"  TTFT p90            : {ns_to_ms(sm.get('ttft_p90_ns'))} ms", flush=True)
        print(f"  TTFT p99            : {ns_to_ms(sm.get('ttft_p99_ns'))} ms", flush=True)
        print(f"  TTFT constraint     : {ns_to_ms(sm.get('ttft_constraint_ns'))} ms", flush=True)
        print(f"", flush=True)
        print(f"  TPOT mean           : {ns_to_ms(sm.get('tpot_mean_ns'))} ms", flush=True)
        print(f"  TPOT p90            : {ns_to_ms(sm.get('tpot_p90_ns'))} ms", flush=True)
        print(f"  TPOT p99            : {ns_to_ms(sm.get('tpot_p99_ns'))} ms", flush=True)
        print(f"  TPOT constraint     : {ns_to_ms(sm.get('tpot_constraint_ns'))} ms", flush=True)
        print(f"", flush=True)
        print(f"  E2E mean            : {ns_to_ms(sm.get('e2e_mean_ns'))} ms", flush=True)
        print(f"  E2E p99             : {ns_to_ms(sm.get('e2e_p99_ns'))} ms", flush=True)
        print("="*62, flush=True)
        if sm.get('ttft_mean_ns', 0) == 0:
            print("[WARNING] TTFT=0 → lg.FirstTokenComplete() was never called. "
                  "SUT must use AsyncLLMEngine + stream_output().", flush=True)
            mlflow.set_tag('ttft_zero_warning', 'true')
    else:
        mlflow.set_tag('summary_missing', 'true')
        print("[WARNING] mlperf_log_summary.txt not found", flush=True)

    mlflow.log_metric('duration_sec', time.time() - start_ts)

    # ── Artifacts ────────────────────────────────────────────────────────
    for subdir, artifact_path in [
        (os.path.join(run_dir, 'system'),      'system'),
        (os.path.join(run_dir, 'mlperf_logs'), 'mlperf_logs'),
    ]:
        if os.path.isdir(subdir):
            mlflow.log_artifacts(subdir, artifact_path=artifact_path)
    for fname in ('run.log', 'SUT_VLLM.py', 'result.txt', 'experiment_meta.txt'):
        fpath = os.path.join(run_dir, fname)
        if os.path.exists(fpath):
            mlflow.log_artifact(fpath)
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

    # ── Parse key metrics from summary for result.txt ─────────────────────
    local summary_file="${run_dir}/mlperf_logs/mlperf_log_summary.txt"
    [[ -f "${summary_file}" ]] || summary_file="${INFERENCE_DIR}/mlperf_log_summary.txt"

    ns_to_ms() { echo "${1:-0}" | awk '{if($1==0) printf "0.00"; else printf "%.2f", $1/1e6}'; }

    local tok_s samp_s valid target_qps
    local ttft_mean ttft_p90 ttft_p99 ttft_constraint
    local tpot_mean tpot_p90 tpot_p99 tpot_constraint
    local e2e_mean e2e_p99
    local ttft_es tpot_es
    set +e
    if [[ -f "${summary_file}" ]]; then
        tok_s=$(         grep "Completed tokens per second"       "${summary_file}" | awk -F: '{print $NF}' | xargs)
        samp_s=$(        grep "Completed samples per second"      "${summary_file}" | awk -F: '{print $NF}' | xargs)
        valid=$(         grep "Result is"                         "${summary_file}" | awk '{print $NF}' | head -1)
        target_qps=$(    grep "target_qps"                        "${summary_file}" | awk -F: '{print $NF}' | xargs)
        ttft_mean=$(     grep "Mean First Token latency (ns)"     "${summary_file}" | awk -F: '{print $NF}' | xargs)
        ttft_p90=$(      grep "90.00 percentile first token"      "${summary_file}" | awk -F: '{print $NF}' | xargs)
        ttft_p99=$(      grep "99.00 percentile first token"      "${summary_file}" | awk -F: '{print $NF}' | xargs)
        ttft_constraint=$(grep "ttft_latency (ns)"                "${summary_file}" | awk -F: '{print $NF}' | xargs)
        tpot_mean=$(     grep "Mean Time per Output Token (ns)"   "${summary_file}" | awk -F: '{print $NF}' | xargs)
        tpot_p90=$(      grep "90.00 percentile time to output"   "${summary_file}" | awk -F: '{print $NF}' | xargs)
        tpot_p99=$(      grep "99.00 percentile time to output"   "${summary_file}" | awk -F: '{print $NF}' | xargs)
        tpot_constraint=$(grep "tpot_latency (ns)"                "${summary_file}" | awk -F: '{print $NF}' | xargs)
        e2e_mean=$(      grep "Mean latency (ns)"                 "${summary_file}" | awk -F: '{print $NF}' | xargs)
        e2e_p99=$(       grep "99.00 percentile latency"          "${summary_file}" | awk -F: '{print $NF}' | xargs)
        ttft_es=$(       grep -A1 "TTFT Early Stopping"           "${summary_file}" | grep "\*" | sed 's/.*\* //')
        tpot_es=$(       grep -A1 "TPOT Early Stopping"           "${summary_file}" | grep "\*" | sed 's/.*\* //')
    fi
    set -e

    cat > "${run_dir}/result.txt" <<RESEOF
Experiment:           ${exp_id}
Run:                  ${run_idx}/${RUNS_PER_EXP}
Description:          ${desc}
Batch Size:           ${batch_size}
Dtype:                ${dtype}
Scenario:             ${SCENARIO}
Target QPS:           ${target_qps:-N/A}
Duration:             ${duration}s
Exit Code:            ${exit_code}

── Throughput ─────────────────────────────────────────────────────
Completed tokens/sec   : ${tok_s:-N/A}
Completed samples/sec  : ${samp_s:-N/A}
Result                 : ${valid:-N/A}

── TTFT (Time to First Token) ─────────────────────────────────────
TTFT constraint (ms)   : $(ns_to_ms "${ttft_constraint:-0}")
TTFT mean (ms)         : $(ns_to_ms "${ttft_mean:-0}")
TTFT p90  (ms)         : $(ns_to_ms "${ttft_p90:-0}")
TTFT p99  (ms)         : $(ns_to_ms "${ttft_p99:-0}")
TTFT early stopping    : ${ttft_es:-N/A}

── TPOT (Time per Output Token) ───────────────────────────────────
TPOT constraint (ms)   : $(ns_to_ms "${tpot_constraint:-0}")
TPOT mean (ms)         : $(ns_to_ms "${tpot_mean:-0}")
TPOT p90  (ms)         : $(ns_to_ms "${tpot_p90:-0}")
TPOT p99  (ms)         : $(ns_to_ms "${tpot_p99:-0}")
TPOT early stopping    : ${tpot_es:-N/A}

── End-to-End Latency ─────────────────────────────────────────────
E2E mean (ms)          : $(ns_to_ms "${e2e_mean:-0}")
E2E p99  (ms)          : $(ns_to_ms "${e2e_p99:-0}")
RESEOF

    if [[ -n "${tok_s:-}" && "${tok_s}" != "N/A" ]]; then
        success "    run_${run_idx}: ${tok_s} tok/s | TTFT_p99=$(ns_to_ms "${ttft_p99:-0}")ms | TPOT_p99=$(ns_to_ms "${tpot_p99:-0}")ms | [${valid:-N/A}]  (${duration}s)"
    else
        local err_type; err_type=$(classify_error "${run_dir}/run.log")
        error "    run_${run_idx}: FAILED (${err_type})  (${duration}s)"
        case "${err_type}" in
            OOM)                   warn "    → OOM: reduce max_num_seqs or max_model_len" ;;
            SEGFAULT)              warn "    → Segfault: check PYTORCH_CUDA_ALLOC_CONF" ;;
            ENGINE_CORE_FATAL)     warn "    → Engine fatal: vLLM version incompatibility" ;;
            ILLEGAL_MEMORY_ACCESS) warn "    → max_num_batched_tokens too high" ;;
            COMPILE_FAILED)        warn "    → torch.compile error" ;;
        esac
    fi

    log "    Cooldown 30s..."; sleep 30
    cleanup_gpu
}

# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
run_experiment() {
    local exp_id="$1"
    local batch_size="$2"
    local dtype="$3"
    local sut_func="$4"
    local desc="$5"

    case "${TARGET}" in
        all) ;;
        *) [[ "${TARGET}" == "${exp_id}" ]] || return 0 ;;
    esac

    local exp_dir="${RESULTS_DIR}/${exp_id}"
    mkdir -p "${exp_dir}"

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
scenario:     ${SCENARIO}
mode:         server_performance
runs:         ${RUNS_PER_EXP}
timestamp:    $(date -u +%Y-%m-%dT%H:%M:%SZ)
sut_func:     ${sut_func}
METAEOF

    for run_idx in $(seq 1 "${RUNS_PER_EXP}"); do
        run_single "${exp_id}" "${run_idx}" "${batch_size}" "${dtype}" "${desc}"
    done

    aggregate_experiment "${exp_id}" "${desc}"
}

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
aggregate_experiment() {
    local exp_id="$1"
    local desc="$2"
    local exp_dir="${RESULTS_DIR}/${exp_id}"

    python3 - <<PYAGG
import os, statistics

exp_dir = '${exp_dir}'
exp_id  = '${exp_id}'
desc    = '${desc}'

fields = {
    'Completed tokens/sec':   [],
    'Completed samples/sec':  [],
    'TTFT mean (ms)':         [],
    'TTFT p90  (ms)':         [],
    'TTFT p99  (ms)':         [],
    'TPOT mean (ms)':         [],
    'TPOT p90  (ms)':         [],
    'TPOT p99  (ms)':         [],
    'E2E mean (ms)':          [],
    'E2E p99  (ms)':          [],
}

def parse_result(path):
    d = {}
    with open(path) as f:
        for line in f:
            if ':' in line:
                k, _, v = line.partition(':')
                d[k.strip()] = v.strip()
    return d

for run_idx in range(1, ${RUNS_PER_EXP} + 1):
    rfile = os.path.join(exp_dir, f'run_{run_idx}', 'result.txt')
    if not os.path.exists(rfile):
        continue
    d = parse_result(rfile)
    for key in fields:
        val = d.get(key, '')
        # Strip the warning suffix if present
        val = val.split(' ')[0] if val else ''
        if val and val not in ('N/A', 'FAILED', '0.00'):
            try:
                fields[key].append(float(val))
            except ValueError:
                pass

def stat(vals, label):
    if not vals:
        return f"  {label:28s}: N/A"
    mean  = statistics.mean(vals)
    stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
    cv    = (stdev / mean * 100) if mean != 0 else 0.0
    return (f"  {label:28s}: mean={mean:8.2f}  stdev={stdev:6.2f}  "
            f"CV={cv:4.1f}%  values={[f'{v:.2f}' for v in vals]}")

sections = [
    ("Throughput",           ['Completed tokens/sec', 'Completed samples/sec']),
    ("TTFT (ms)",            ['TTFT mean (ms)', 'TTFT p90  (ms)', 'TTFT p99  (ms)']),
    ("TPOT (ms)",            ['TPOT mean (ms)', 'TPOT p90  (ms)', 'TPOT p99  (ms)']),
    ("E2E Latency (ms)",     ['E2E mean (ms)', 'E2E p99  (ms)']),
]

lines = [
    f"Experiment:  {exp_id}",
    f"Description: {desc}",
    f"Runs:        ${RUNS_PER_EXP}",
    "",
]
for section_name, keys in sections:
    lines.append(f"── {section_name}")
    for k in keys:
        lines.append(stat(fields[k], k))
    lines.append("")

text = '\n'.join(lines)
print(text)
with open(os.path.join(exp_dir, 'aggregate.txt'), 'w') as f:
    f.write(text + '\n')
PYAGG
}

# =============================================================================
# SUT GENERATOR FUNCTIONS
#
# CANONICAL SUTSERVER (AsyncLLMEngine)
# ─────────────────────────────────────
# Every function outputs a complete SUT_VLLM.py containing:
#   • The full canonical SUTServer with AsyncLLMEngine + stream_output() +
#     lg.FirstTokenComplete() — required for TTFT measurement in Server scenario
#   • A thin _ExpXX subclass that overrides ONLY load_model()
#   • SUTServer.load_model = _ExpXX.load_model (monkey-patch for main.py)
#
# ISOLATION GUARANTEE
# ────────────────────
# The baseline AsyncEngineArgs with ALL fields pinned explicitly:
#
#   dtype                  = self.dtype        (bfloat16 from CLI)
#   quantization           = None              ← explicit
#   gpu_memory_utilization = 0.90              ← explicit
#   max_model_len          = 131072            ← explicit (not None/auto)
#   enable_prefix_caching  = False             ← explicit
#   enable_chunked_prefill = False             ← explicit (vLLM may auto-enable)
#   enforce_eager          = False             ← explicit (CUDA Graphs ON)
#   max_num_seqs           = 256               ← explicit (vLLM default, pinned)
#   scheduler_delay_factor = 0.0               ← explicit
#   preemption_mode        = "recompute"       ← explicit (vLLM default, pinned)
#   num_scheduler_steps    = 1                 ← explicit
#   async_scheduling       = False             ← explicit
#   max_num_batched_tokens : NOT SET           ← only set when chunked_prefill=True
#   env PYTORCH_CUDA_ALLOC_CONF: UNSET         ← only set in exp_17
#
# Each experiment changes EXACTLY ONE argument (or one inseparable pair).
# =============================================================================

# ---------------------------------------------------------------------------
# Shared SUT body written by every sut_expXX function
# ---------------------------------------------------------------------------
_sut_common_body() {
cat << 'SUTEOF'
import asyncio
import os
import numpy as np
import array
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.inputs import TokensPrompt
import queue
import threading
import logging
import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-SUT")


class SUT:
    """Base class — Offline scenario. Preserved for compatibility with main.py."""
    def __init__(self, model_path=None, dtype="bfloat16", batch_size=None,
                 total_sample_count=13368, dataset_path=None,
                 use_cached_outputs=False, workers=1, tensor_parallel_size=1):
        self.model_path = model_path or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.batch_size  = batch_size if batch_size else 1
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count, self.data_object.perf_count,
            self.data_object.LoadSamplesToRam, self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42, max_tokens=128, min_tokens=1)
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        raise NotImplementedError

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self):
        return self.qsl

    def predict(self, **kwargs):
        raise NotImplementedError

    def flush_queries(self):
        pass

    def __del__(self):
        pass


class SUTServer(SUT):
    """
    Server scenario SUT using AsyncLLMEngine.

    WHY AsyncLLMEngine IS MANDATORY:
    The Server scenario sends queries one at a time and requires
    lg.FirstTokenComplete() to be called as soon as the first output token
    is produced. This enables LoadGen to measure TTFT accurately. The sync
    LLM.generate() path cannot do this because it blocks until generation
    is fully complete. Without FirstTokenComplete(), all TTFT/TPOT values
    in the MLPerf log will be zero and LoadGen will mark every sample as
    an error.
    """

    def __init__(self, model_path=None, dtype="bfloat16", batch_size=None,
                 total_sample_count=13368, dataset_path=None,
                 use_cached_outputs=False, workers=1, tensor_parallel_size=1):
        self.model_path = model_path or "meta-llama/Meta-Llama-3.1-8B-Instruct"
        self.batch_size  = batch_size if batch_size else 1
        self.dtype = dtype
        self.tensor_parallel_size = tensor_parallel_size
        if not torch.cuda.is_available():
            assert False, "torch gpu is not available, exiting..."
        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count, self.data_object.perf_count,
            self.data_object.LoadSamplesToRam, self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42, max_tokens=128, min_tokens=1)
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()
        self.request_id = 0

    # ── load_model is overridden per experiment ──────────────────────────
    def load_model(self):
        """
        Baseline AsyncEngineArgs — ALL fields explicitly set so no vLLM
        auto-setting can change experiment behaviour.
        """
        log.info("Loading model (exp_00 baseline)...")
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype                  = self.dtype,     # bfloat16 from CLI
            quantization           = None,           # explicit: no quantization
            tensor_parallel_size   = self.tensor_parallel_size,
            gpu_memory_utilization = 0.90,           # tune if OOM or for max throughput
            max_model_len          = 2668,           # dataset max input (2540) + max output (128)
            enable_prefix_caching  = False,          # explicit
            enable_chunked_prefill = False,          # explicit (vLLM may auto-enable)
            enforce_eager          = False,          # explicit: CUDA Graphs ON
            max_num_seqs           = 256,            # explicit: vLLM default pinned
            scheduler_delay_factor = 0.0,            # explicit
            preemption_mode        = "recompute",    # explicit: vLLM default pinned
            num_scheduler_steps    = 1,              # explicit
            async_scheduling       = False,          # explicit
            max_num_batched_tokens = 2668,           # match max_model_len when chunked_prefill=False
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")

    # ── Server scenario infrastructure — DO NOT CHANGE IN EXPERIMENTS ───

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self):
        return self.qsl

    def start(self):
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries, daemon=True)
            worker.start()
            self.worker_threads[j] = worker

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        for worker in self.worker_threads:
            worker.join()

    async def stream_output(self, qitem, results_generator):
        """
        Stream tokens from the async engine.
        lg.FirstTokenComplete() is called on the very first token (TTFT).
        lg.QuerySamplesComplete() is called when generation is fully done (E2E).
        Both calls are required for Server scenario metrics to be non-zero.
        """
        first_token_sent = False
        async for request_output in results_generator:
            if not first_token_sent and request_output.outputs:
                first_tokens = list(request_output.outputs[0].token_ids)
                response_data = array.array(
                    "B", np.array(first_tokens, np.int32).tobytes())
                bi = response_data.buffer_info()
                lg.FirstTokenComplete(
                    [lg.QuerySampleResponse(qitem.id, bi[0], bi[1])])
                first_token_sent = True

        # Full response complete
        pred_output_tokens = list(request_output.outputs[0].token_ids)
        n_tokens = len(pred_output_tokens)
        response_array = array.array(
            "B", np.array(pred_output_tokens, np.int32).tobytes())
        bi = response_array.buffer_info()
        lg.QuerySamplesComplete(
            [lg.QuerySampleResponse(qitem.id, bi[0], bi[1], n_tokens)])

        with self.sample_counter_lock:
            self.sample_counter += 1
            log.info(f"Samples run: {self.sample_counter}")

    def process_queries(self):
        """One query at a time — streams through async engine."""
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break
            input_ids_tensor = TokensPrompt(
                prompt_token_ids=self.data_object.input_ids[qitem.index])
            results_generator = self.model.generate(
                prompt=input_ids_tensor,
                sampling_params=self.sampling_params,
                request_id=str(self.request_id),
            )
            self.request_id += 1
            asyncio.run(self.stream_output(qitem, results_generator))

    def issue_queries(self, query_samples):
        """LoadGen sends one sample at a time in Server scenario."""
        self.query_queue.put(query_samples[0])

    def flush_queries(self):
        pass

    def predict(self, **kwargs):
        raise NotImplementedError

    def __del__(self):
        pass
SUTEOF
}

# ── exp_00: BASELINE ─────────────────────────────────────────────────────────
# All AsyncEngineArgs explicitly set to vLLM defaults.
# No env var overrides. This is the reference point for all experiments.
sut_exp00() {
    _sut_common_body
    # load_model() in the common body IS the baseline — no override needed
}

# ── exp_01: enforce_eager=True (CUDA Graphs OFF) ──────────────────────────────
# CHANGE: enforce_eager=False → True
# CUDA Graphs record the sequence of GPU operations on the first batch and replay
# them for subsequent batches without re-issuing each kernel from the CPU. Turning
# this OFF should show a regression, confirming CUDA Graphs are beneficial. All
# other flags identical to baseline.
sut_exp01() {
    _sut_common_body
    cat << 'SUTEOF'

class _Exp01(SUTServer):
    def load_model(self):
        log.info("Loading model [exp_01: enforce_eager=True — CUDA Graphs OFF]...")
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype                  = self.dtype,
            quantization           = None,
            tensor_parallel_size   = self.tensor_parallel_size,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens  = 131072,         # explicit: = max_model_len when chunked_prefill=False
            enforce_eager          = True,            # ← CHANGE (baseline: False)
            max_num_seqs           = 256,
            scheduler_delay_factor = 0.0,
            preemption_mode        = "recompute",
            num_scheduler_steps    = 1,
            async_scheduling       = False,
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")

SUTServer.load_model = _Exp01.load_model
SUTEOF
}

# ── exp_02: chunked_prefill=True + max_num_batched_tokens=8192 ───────────────
# CHANGE: enable_chunked_prefill=False → True; max_num_batched_tokens added
# These two are an inseparable pair: chunked prefill is meaningless without the
# token cap, and setting max_num_batched_tokens without chunked prefill is a
# no-op (or causes issues in vLLM 0.10.0).
# Fine granularity (8192 tokens/chunk) → maximum TTFT reduction, lowest throughput.
# All other flags identical to baseline.
sut_exp02() {
    _sut_common_body
    cat << 'SUTEOF'

class _Exp02(SUTServer):
    def load_model(self):
        log.info("Loading model [exp_02: chunked_prefill=True + batched_tokens=8192]...")
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype                  = self.dtype,
            quantization           = None,
            tensor_parallel_size   = self.tensor_parallel_size,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enable_prefix_caching  = False,
            enable_chunked_prefill = True,            # ← CHANGE (baseline: False)
            max_num_batched_tokens = 8192,            # ← CHANGE (required pair)
            enforce_eager          = False,
            max_num_seqs           = 256,
            scheduler_delay_factor = 0.0,
            preemption_mode        = "recompute",
            num_scheduler_steps    = 1,
            async_scheduling       = False,
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")

SUTServer.load_model = _Exp02.load_model
SUTEOF
}

# ── exp_03: chunked_prefill=True + max_num_batched_tokens=16384 ──────────────
sut_exp03() {
    _sut_common_body
    cat << 'SUTEOF'

class _Exp03(SUTServer):
    def load_model(self):
        log.info("Loading model [exp_03: chunked_prefill=True + batched_tokens=16384]...")
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype                  = self.dtype,
            quantization           = None,
            tensor_parallel_size   = self.tensor_parallel_size,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enable_prefix_caching  = False,
            enable_chunked_prefill = True,            # ← CHANGE (baseline: False)
            max_num_batched_tokens = 16384,           # ← CHANGE (required pair)
            enforce_eager          = False,
            max_num_seqs           = 256,
            scheduler_delay_factor = 0.0,
            preemption_mode        = "recompute",
            num_scheduler_steps    = 1,
            async_scheduling       = False,
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")

SUTServer.load_model = _Exp03.load_model
SUTEOF
}

# ── exp_04: chunked_prefill=True + max_num_batched_tokens=32768 ──────────────
sut_exp04() {
    _sut_common_body
    cat << 'SUTEOF'

class _Exp04(SUTServer):
    def load_model(self):
        log.info("Loading model [exp_04: chunked_prefill=True + batched_tokens=32768]...")
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype                  = self.dtype,
            quantization           = None,
            tensor_parallel_size   = self.tensor_parallel_size,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enable_prefix_caching  = False,
            enable_chunked_prefill = True,            # ← CHANGE (baseline: False)
            max_num_batched_tokens = 32768,           # ← CHANGE (required pair)
            enforce_eager          = False,
            max_num_seqs           = 256,
            scheduler_delay_factor = 0.0,
            preemption_mode        = "recompute",
            num_scheduler_steps    = 1,
            async_scheduling       = False,
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")

SUTServer.load_model = _Exp04.load_model
SUTEOF
}

# ── exp_05: max_model_len=2668 ────────────────────────────────────────────────
# CHANGE: max_model_len=131072 → 2668
# Right-sizes the KV cache to the CNN/DailyMail dataset maximum (2540 + 128).
# Makes each KV slot 49× smaller, allowing more concurrent sequences in the KV
# cache. Key effect in Server scenario: reduces KV eviction under sustained QPS,
# which improves TPOT consistency. All other flags identical to baseline.
sut_exp05() {
    _sut_common_body
    cat << 'SUTEOF'

class _Exp05(SUTServer):
    def load_model(self):
        log.info("Loading model [exp_05: max_model_len=2668]...")
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype                  = self.dtype,
            quantization           = None,
            tensor_parallel_size   = self.tensor_parallel_size,
            gpu_memory_utilization = 0.90,
            max_model_len          = 2668,            # ← CHANGE (baseline: 131072)
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens  = 131072,         # explicit: = max_model_len when chunked_prefill=False
            enforce_eager          = False,
            max_num_seqs           = 256,
            scheduler_delay_factor = 0.0,
            preemption_mode        = "recompute",
            num_scheduler_steps    = 1,
            async_scheduling       = False,
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")

SUTServer.load_model = _Exp05.load_model
SUTEOF
}

# ── exp_06: scheduler_delay_factor=0.05 ──────────────────────────────────────
# CHANGE: scheduler_delay_factor=0.0 → 0.05
# Tells the scheduler to wait up to (0.05 × mean_batch_time) before forming the
# next batch. This accumulates more queries in the queue before dispatching,
# improving TPOT by creating larger decode batches. The cost is added queueing
# delay before the first token, worsening TTFT. Small value = conservative trade.
sut_exp06() {
    _sut_common_body
    cat << 'SUTEOF'

class _Exp06(SUTServer):
    def load_model(self):
        log.info("Loading model [exp_06: scheduler_delay_factor=0.05]...")
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype                  = self.dtype,
            quantization           = None,
            tensor_parallel_size   = self.tensor_parallel_size,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens  = 131072,         # explicit: = max_model_len when chunked_prefill=False
            enforce_eager          = False,
            max_num_seqs           = 256,
            scheduler_delay_factor = 0.05,            # ← CHANGE (baseline: 0.0)
            preemption_mode        = "recompute",
            num_scheduler_steps    = 1,
            async_scheduling       = False,
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")

SUTServer.load_model = _Exp06.load_model
SUTEOF
}

# ── exp_07: scheduler_delay_factor=0.10 ──────────────────────────────────────
sut_exp07() {
    _sut_common_body
    cat << 'SUTEOF'

class _Exp07(SUTServer):
    def load_model(self):
        log.info("Loading model [exp_07: scheduler_delay_factor=0.10]...")
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype                  = self.dtype,
            quantization           = None,
            tensor_parallel_size   = self.tensor_parallel_size,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens  = 131072,         # explicit: = max_model_len when chunked_prefill=False
            enforce_eager          = False,
            max_num_seqs           = 256,
            scheduler_delay_factor = 0.10,            # ← CHANGE (baseline: 0.0)
            preemption_mode        = "recompute",
            num_scheduler_steps    = 1,
            async_scheduling       = False,
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")

SUTServer.load_model = _Exp07.load_model
SUTEOF
}

# ── exp_08: scheduler_delay_factor=0.20 ──────────────────────────────────────
# Most aggressive delay — expected to breach the TTFT SLO. Confirms the upper
# bound of the delay sweep. All other flags identical to baseline.
sut_exp08() {
    _sut_common_body
    cat << 'SUTEOF'

class _Exp08(SUTServer):
    def load_model(self):
        log.info("Loading model [exp_08: scheduler_delay_factor=0.20]...")
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype                  = self.dtype,
            quantization           = None,
            tensor_parallel_size   = self.tensor_parallel_size,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens  = 131072,         # explicit: = max_model_len when chunked_prefill=False
            enforce_eager          = False,
            max_num_seqs           = 256,
            scheduler_delay_factor = 0.20,            # ← CHANGE (baseline: 0.0)
            preemption_mode        = "recompute",
            num_scheduler_steps    = 1,
            async_scheduling       = False,
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")

SUTServer.load_model = _Exp08.load_model
SUTEOF
}

# ── exp_09: max_num_seqs=128 ──────────────────────────────────────────────────
# CHANGE: max_num_seqs=256 → 128
# Reduces concurrent sequences. Less KV cache pressure under load. Each sequence
# gets more KV cache headroom — may improve TPOT tail at the cost of throughput.
# Lower concurrency also reduces preemption events.
sut_exp09() {
    _sut_common_body
    cat << 'SUTEOF'

class _Exp09(SUTServer):
    def load_model(self):
        log.info("Loading model [exp_09: max_num_seqs=128]...")
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype                  = self.dtype,
            quantization           = None,
            tensor_parallel_size   = self.tensor_parallel_size,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens  = 131072,         # explicit: = max_model_len when chunked_prefill=False
            enforce_eager          = False,
            max_num_seqs           = 128,             # ← CHANGE (baseline: 256)
            scheduler_delay_factor = 0.0,
            preemption_mode        = "recompute",
            num_scheduler_steps    = 1,
            async_scheduling       = False,
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")

SUTServer.load_model = _Exp09.load_model
SUTEOF
}

# ── exp_10: max_num_seqs=512 ──────────────────────────────────────────────────
sut_exp10() {
    _sut_common_body
    cat << 'SUTEOF'

class _Exp10(SUTServer):
    def load_model(self):
        log.info("Loading model [exp_10: max_num_seqs=512]...")
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype                  = self.dtype,
            quantization           = None,
            tensor_parallel_size   = self.tensor_parallel_size,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens  = 131072,         # explicit: = max_model_len when chunked_prefill=False
            enforce_eager          = False,
            max_num_seqs           = 512,             # ← CHANGE (baseline: 256)
            scheduler_delay_factor = 0.0,
            preemption_mode        = "recompute",
            num_scheduler_steps    = 1,
            async_scheduling       = False,
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")

SUTServer.load_model = _Exp10.load_model
SUTEOF
}

# ── exp_11: max_num_seqs=1024 ─────────────────────────────────────────────────
# High concurrency. Risk of KV cache thrashing at target QPS. Monitor TPOT tail.
sut_exp11() {
    _sut_common_body
    cat << 'SUTEOF'

class _Exp11(SUTServer):
    def load_model(self):
        log.info("Loading model [exp_11: max_num_seqs=1024]...")
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype                  = self.dtype,
            quantization           = None,
            tensor_parallel_size   = self.tensor_parallel_size,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens  = 131072,         # explicit: = max_model_len when chunked_prefill=False
            enforce_eager          = False,
            max_num_seqs           = 1024,            # ← CHANGE (baseline: 256)
            scheduler_delay_factor = 0.0,
            preemption_mode        = "recompute",
            num_scheduler_steps    = 1,
            async_scheduling       = False,
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")

SUTServer.load_model = _Exp11.load_model
SUTEOF
}

# ── exp_12: preemption_mode=recompute (CONTROL) ───────────────────────────────
# CHANGE: none — preemption_mode="recompute" is already the baseline.
# This is an explicit CONTROL experiment that should produce identical results
# to exp_00. Confirms that our explicit pinning of preemption_mode is working
# correctly. Any difference from exp_00 indicates a measurement artefact.
sut_exp12() {
    _sut_common_body
    cat << 'SUTEOF'

# exp_12: preemption_mode=recompute (CONTROL — identical to baseline)
# Should produce the same results as exp_00. If different, something is wrong
# with baseline reproducibility or our pinning of this flag.
SUTEOF
}

# ── exp_13: preemption_mode=swap ──────────────────────────────────────────────
# CHANGE: preemption_mode="recompute" → "swap"
# When the KV cache fills and a higher-priority sequence arrives, the evicted
# sequence's KV entries are copied to CPU RAM and swapped back in when it
# resumes. This adds PCIe round-trip latency. For CNN/DailyMail (mean 870 tokens)
# recomputing is usually faster than the PCIe transfer — expected regression.
sut_exp13() {
    _sut_common_body
    cat << 'SUTEOF'

class _Exp13(SUTServer):
    def load_model(self):
        log.info("Loading model [exp_13: preemption_mode=swap]...")
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype                  = self.dtype,
            quantization           = None,
            tensor_parallel_size   = self.tensor_parallel_size,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens  = 131072,         # explicit: = max_model_len when chunked_prefill=False
            enforce_eager          = False,
            max_num_seqs           = 256,
            scheduler_delay_factor = 0.0,
            preemption_mode        = "swap",          # ← CHANGE (baseline: "recompute")
            num_scheduler_steps    = 1,
            async_scheduling       = False,
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")

SUTServer.load_model = _Exp13.load_model
SUTEOF
}

# ── exp_14: num_scheduler_steps=2 ────────────────────────────────────────────
# CHANGE: num_scheduler_steps=1 → 2
# Runs 2 decode steps before checking EOS and rescheduling. Reduces per-step
# scheduling overhead. In Server scenario with many short requests, this may
# improve throughput but delays EOS detection by 1 step — sequences that
# naturally end at step 1 waste one extra decode step.
sut_exp14() {
    _sut_common_body
    cat << 'SUTEOF'

class _Exp14(SUTServer):
    def load_model(self):
        log.info("Loading model [exp_14: num_scheduler_steps=2]...")
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype                  = self.dtype,
            quantization           = None,
            tensor_parallel_size   = self.tensor_parallel_size,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens  = 131072,         # explicit: = max_model_len when chunked_prefill=False
            enforce_eager          = False,
            max_num_seqs           = 256,
            scheduler_delay_factor = 0.0,
            preemption_mode        = "recompute",
            num_scheduler_steps    = 2,               # ← CHANGE (baseline: 1)
            async_scheduling       = False,
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")

SUTServer.load_model = _Exp14.load_model
SUTEOF
}

# ── exp_15: num_scheduler_steps=4 ────────────────────────────────────────────
sut_exp15() {
    _sut_common_body
    cat << 'SUTEOF'

class _Exp15(SUTServer):
    def load_model(self):
        log.info("Loading model [exp_15: num_scheduler_steps=4]...")
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype                  = self.dtype,
            quantization           = None,
            tensor_parallel_size   = self.tensor_parallel_size,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens  = 131072,         # explicit: = max_model_len when chunked_prefill=False
            enforce_eager          = False,
            max_num_seqs           = 256,
            scheduler_delay_factor = 0.0,
            preemption_mode        = "recompute",
            num_scheduler_steps    = 4,               # ← CHANGE (baseline: 1)
            async_scheduling       = False,
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")

SUTServer.load_model = _Exp15.load_model
SUTEOF
}

# ── exp_16: async_scheduling=True ────────────────────────────────────────────
# CHANGE: async_scheduling=False → True
# CPU scheduling work for batch N+1 overlaps with GPU execution of batch N.
# Hides scheduling latency behind compute. The benefit depends on the ratio of
# scheduling overhead to batch GPU time — larger batches amortise this better.
sut_exp16() {
    _sut_common_body
    cat << 'SUTEOF'

class _Exp16(SUTServer):
    def load_model(self):
        log.info("Loading model [exp_16: async_scheduling=True]...")
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype                  = self.dtype,
            quantization           = None,
            tensor_parallel_size   = self.tensor_parallel_size,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens  = 131072,         # explicit: = max_model_len when chunked_prefill=False
            enforce_eager          = False,
            max_num_seqs           = 256,
            scheduler_delay_factor = 0.0,
            preemption_mode        = "recompute",
            num_scheduler_steps    = 1,
            async_scheduling       = True,            # ← CHANGE (baseline: False)
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")

SUTServer.load_model = _Exp16.load_model
SUTEOF
}

# ── exp_17: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ─────────────────
# CHANGE: env var PYTORCH_CUDA_ALLOC_CONF unset → expandable_segments:True
# This is a PyTorch allocator setting, not a vLLM arg. In Server scenario with
# many concurrent short-lived KV allocations, the default fixed-block allocator
# can fragment GPU memory over time. Expandable segments allow growing existing
# blocks instead of always requesting new ones, reducing fragmentation.
# Engine args are identical to baseline — only the env var changes.
sut_exp17() {
    _sut_common_body
    cat << 'SUTEOF'

class _Exp17(SUTServer):
    def load_model(self):
        log.info("Loading model [exp_17: expandable_segments=True]...")
        # Set env var before engine init — must happen before CUDA context creation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype                  = self.dtype,       # all other args IDENTICAL to baseline
            quantization           = None,
            tensor_parallel_size   = self.tensor_parallel_size,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens  = 131072,         # explicit: = max_model_len when chunked_prefill=False
            enforce_eager          = False,
            max_num_seqs           = 256,
            scheduler_delay_factor = 0.0,
            preemption_mode        = "recompute",
            num_scheduler_steps    = 1,
            async_scheduling       = False,
            # max_num_batched_tokens: NOT SET — only env var changes
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")

SUTServer.load_model = _Exp17.load_model
SUTEOF
}

# ── exp_18: FP8 quantization ──────────────────────────────────────────────────
# CHANGE: dtype=bfloat16 → float16, quantization=None → "fp8"
# These two are an inseparable pair in vLLM 0.10.0: FP8 checkpoint requires
# float16 activations (not bfloat16). Cannot be separated.
# Effect: 2× weight memory reduction → larger effective KV cache → more
# concurrent sequences at the same gpu_memory_utilization.
# Requires the FP8-quantized checkpoint at CHECKPOINT_PATH.
sut_exp18() {
    _sut_common_body
    cat << 'SUTEOF'

class _Exp18(SUTServer):
    def load_model(self):
        log.info("Loading model [exp_18: FP8 quantization — dtype=float16, quantization=fp8]...")
        # NOTE: dtype=float16 is required with FP8 quantization in vLLM 0.10.0.
        # FP8 checkpoints force float16 activations. Cannot use bfloat16 here.
        # This is an inseparable pair — not a violation of isolation.
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype                  = "float16",        # ← CHANGE (baseline: bfloat16)
            quantization           = "fp8",            # ← CHANGE (baseline: None)
            tensor_parallel_size   = self.tensor_parallel_size,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens  = 131072,         # explicit: = max_model_len when chunked_prefill=False
            enforce_eager          = False,
            max_num_seqs           = 256,
            scheduler_delay_factor = 0.0,
            preemption_mode        = "recompute",
            num_scheduler_steps    = 1,
            async_scheduling       = False,
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")

SUTServer.load_model = _Exp18.load_model
SUTEOF
}

# ── exp_C1: COMBO — chunked_prefill + max_model_len=2668 ─────────────────────
# Run after exp_02-04 and exp_05 results are known.
# Combines the primary TTFT lever (chunked prefill) with KV cache density.
# max_num_batched_tokens=16384 chosen as the medium point from the sweep.
sut_exp_C1() {
    _sut_common_body
    cat << 'SUTEOF'

class _ExpC1(SUTServer):
    def load_model(self):
        log.info("Loading model [exp_C1: chunked_16384 + max_model_len=2668]...")
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype                  = self.dtype,
            quantization           = None,
            tensor_parallel_size   = self.tensor_parallel_size,
            gpu_memory_utilization = 0.90,
            max_model_len          = 2668,            # from exp_05
            enable_prefix_caching  = False,
            enable_chunked_prefill = True,            # from exp_03
            max_num_batched_tokens = 16384,           # from exp_03
            enforce_eager          = False,
            max_num_seqs           = 256,
            scheduler_delay_factor = 0.0,
            preemption_mode        = "recompute",
            num_scheduler_steps    = 1,
            async_scheduling       = False,
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")

SUTServer.load_model = _ExpC1.load_model
SUTEOF
}

# ── exp_C2: COMBO — chunked_prefill + async_scheduling ───────────────────────
# Run after exp_02-04 and exp_16 results are known.
sut_exp_C2() {
    _sut_common_body
    cat << 'SUTEOF'

class _ExpC2(SUTServer):
    def load_model(self):
        log.info("Loading model [exp_C2: chunked_16384 + async_scheduling=True]...")
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype                  = self.dtype,
            quantization           = None,
            tensor_parallel_size   = self.tensor_parallel_size,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,
            enable_prefix_caching  = False,
            enable_chunked_prefill = True,            # from exp_03
            max_num_batched_tokens = 16384,           # from exp_03
            enforce_eager          = False,
            max_num_seqs           = 256,
            scheduler_delay_factor = 0.0,
            preemption_mode        = "recompute",
            num_scheduler_steps    = 1,
            async_scheduling       = True,            # from exp_16
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")

SUTServer.load_model = _ExpC2.load_model
SUTEOF
}

# ── exp_C3: COMBO — max_model_len=2668 + max_num_seqs=512 ────────────────────
# Run after exp_05 and exp_10 results are known.
# Dense KV cache + higher concurrency.
sut_exp_C3() {
    _sut_common_body
    cat << 'SUTEOF'

class _ExpC3(SUTServer):
    def load_model(self):
        log.info("Loading model [exp_C3: max_model_len=2668 + max_num_seqs=512]...")
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype                  = self.dtype,
            quantization           = None,
            tensor_parallel_size   = self.tensor_parallel_size,
            gpu_memory_utilization = 0.90,
            max_model_len          = 2668,            # from exp_05
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,
            max_num_batched_tokens  = 131072,         # explicit: = max_model_len when chunked_prefill=False
            enforce_eager          = False,
            max_num_seqs           = 512,             # from exp_10
            scheduler_delay_factor = 0.0,
            preemption_mode        = "recompute",
            num_scheduler_steps    = 1,
            async_scheduling       = False,
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")

SUTServer.load_model = _ExpC3.load_model
SUTEOF
}

# ── exp_C4: COMBO — EDIT BEFORE RUNNING ──────────────────────────────────────
# Template. Replace the engine args below with the winners from exp_01..exp_18.
# Add a comment for each non-baseline value indicating which experiment it came from.
sut_exp_C4() {
    _sut_common_body
    cat << 'SUTEOF'

class _ExpC4(SUTServer):
    def load_model(self):
        log.info("Loading model [exp_C4: stacked winners — EDIT BEFORE RUNNING]...")
        # ── EDIT THIS BLOCK based on exp_01..exp_18 results ──────────────
        # Replace each placeholder comment with the winning value.
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype                  = self.dtype,             # or "float16" if exp_18 won
            quantization           = None,                   # or "fp8" if exp_18 won
            tensor_parallel_size   = self.tensor_parallel_size,
            gpu_memory_utilization = 0.90,
            max_model_len          = 131072,                 # or 2668 if exp_05 won
            enable_prefix_caching  = False,
            enable_chunked_prefill = False,                  # True + token cap if exp_02-04 won
            # max_num_batched_tokens = ...,                  # only if chunked_prefill=True
            enforce_eager          = False,                  # keep False unless exp_01 showed gain
            max_num_seqs           = 256,                    # or 512/1024 if exp_10/11 won
            scheduler_delay_factor = 0.0,                    # or 0.05/0.10 if exp_06/07 won
            preemption_mode        = "recompute",            # or "swap" if exp_13 won
            num_scheduler_steps    = 1,                      # or 2/4 if exp_14/15 won
            async_scheduling       = False,                  # or True if exp_16 won
        )
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")

SUTServer.load_model = _ExpC4.load_model
SUTEOF
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
main() {
    mkdir -p "${RESULTS_DIR}"

    log "========================================================"
    log "vLLM MLPerf Server Scenario — Isolated Ablation"
    log "$(date)"
    log "========================================================"
    log "SUT:      SUTServer (AsyncLLMEngine + FirstTokenComplete)"
    log "Baseline: ALL AsyncEngineArgs pinned explicitly"
    log "Policy:   ONE change per experiment; combinations after individual results"
    log "Metrics:  tok/s | TTFT (mean/p90/p99) | TPOT (mean/p90/p99) | E2E p99"
    log "Runs/exp: ${RUNS_PER_EXP}"
    log "Results:  ${RESULTS_DIR}"
    log "MLflow:   ${MLFLOW_EXPERIMENT_NAME}"
    log "========================================================"

    # --- Experiment list: (exp_id, batch_size, dtype, sut_func, desc) ---
    local experiments=(
        "exp_00|16|bfloat16|sut_exp00|BASELINE: all AsyncEngineArgs explicit; no knobs changed"
        "exp_01|16|bfloat16|sut_exp01|enforce_eager=True [CUDA Graphs OFF]: regression control"
        "exp_02|16|bfloat16|sut_exp02|chunked_prefill=True + batched_tokens=8192 [finest interleaving]"
        "exp_03|16|bfloat16|sut_exp03|chunked_prefill=True + batched_tokens=16384 [medium interleaving]"
        "exp_04|16|bfloat16|sut_exp04|chunked_prefill=True + batched_tokens=32768 [coarser interleaving]"
        "exp_05|16|bfloat16|sut_exp05|max_model_len=2668 [right-size KV cache to dataset max]"
        "exp_06|16|bfloat16|sut_exp06|scheduler_delay_factor=0.05 [conservative batch accumulation]"
        "exp_07|16|bfloat16|sut_exp07|scheduler_delay_factor=0.10 [moderate batch accumulation]"
        "exp_08|16|bfloat16|sut_exp08|scheduler_delay_factor=0.20 [aggressive — expect TTFT SLO breach]"
        "exp_09|16|bfloat16|sut_exp09|max_num_seqs=128 [restrict concurrency below vLLM default]"
        "exp_10|16|bfloat16|sut_exp10|max_num_seqs=512 [expand concurrency above vLLM default]"
        "exp_11|16|bfloat16|sut_exp11|max_num_seqs=1024 [high concurrency — risk of KV cache thrash]"
        "exp_12|16|bfloat16|sut_exp12|preemption_mode=recompute [CONTROL: identical to baseline; confirms pinning]"
        "exp_13|16|bfloat16|sut_exp13|preemption_mode=swap [PCIe cost — expected regression for short sequences]"
        "exp_14|16|bfloat16|sut_exp14|num_scheduler_steps=2 [amortise scheduling overhead, delay EOS]"
        "exp_15|16|bfloat16|sut_exp15|num_scheduler_steps=4 [more aggressive amortisation]"
        "exp_16|16|bfloat16|sut_exp16|async_scheduling=True [overlap CPU scheduling with GPU execution]"
        "exp_17|16|bfloat16|sut_exp17|PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True [fragmentation reduction]"
        "exp_18|16|float16|sut_exp18|FP8 quantization: dtype=float16 + quantization=fp8 [inseparable pair]"
        "exp_C1|16|bfloat16|sut_exp_C1|COMBO: chunked_prefill=16384 + max_model_len=2668"
        "exp_C2|16|bfloat16|sut_exp_C2|COMBO: chunked_prefill=16384 + async_scheduling=True"
        "exp_C3|16|bfloat16|sut_exp_C3|COMBO: max_model_len=2668 + max_num_seqs=512"
        "exp_C4|16|bfloat16|sut_exp_C4|COMBO: stacked winners — EDIT sut_exp_C4() before running"
    )

    # Find the start index
    local start_idx=0
    if [[ "${TARGET}" != "all" ]]; then
        for i in "${!experiments[@]}"; do
            IFS='|' read -r exp_id _ _ _ _ <<< "${experiments[$i]}"
            if [[ "${exp_id}" == "${TARGET}" ]]; then
                start_idx=$i
                break
            fi
        done
    fi

    # Run from start_idx to end
    for ((i=start_idx; i<${#experiments[@]}; i++)); do
        IFS='|' read -r exp_id batch_size dtype sut_func desc <<< "${experiments[$i]}"
        run_experiment "$exp_id" "$batch_size" "$dtype" "$sut_func" "$desc"
    done

    log "========================================================"
    log "All experiments complete. Results in: ${RESULTS_DIR}"
    log "========================================================"
}

main "$@"