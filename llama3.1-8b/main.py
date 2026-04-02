import subprocess
import mlperf_loadgen as lg
import argparse
import os
import logging
import sys
import requests
import json
import re

sys.path.insert(0, os.getcwd())

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-MAIN")

# function to check the model name in server matches the user specified one


def verify_model_name(user_specified_name, url):
    response = requests.get(url)
    if response.status_code == 200:
        response_dict = response.json()
        server_model_name = response_dict["data"][0]["id"]
        if user_specified_name == server_model_name:
            return {"matched": True, "error": False}
        else:
            return {
                "matched": False,
                "error": f"User specified {user_specified_name} and server model name {server_model_name} mismatch!",
            }
    else:
        return {
            "matched": False,
            "error": f"Failed to get a valid response. Status code: {response.status_code}",
        }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["Offline", "Server", "SingleStream"],
        default="Offline",
        help="Scenario",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model name",
    )
    parser.add_argument("--dataset-path", type=str, default=None, help="")
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="Run accuracy mode")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="data type of the model, choose from float16, bfloat16 and float32",
    )
    parser.add_argument(
        "--audit-conf",
        type=str,
        default="audit.conf",
        help="audit config for LoadGen settings during compliance runs",
    )
    parser.add_argument(
        "--user-conf",
        type=str,
        default="user.conf",
        help="user config for user LoadGen settings such as target QPS",
    )
    # TODO: This interpretation of 'total-sample-count' is a little
    # misleading. Fix it
    parser.add_argument(
        "--total-sample-count",
        type=int,
        default=13368,
        help="Number of samples to use in benchmark.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Model batch-size to use in benchmark.",
    )
    parser.add_argument(
        "--output-log-dir", type=str, default="output-logs", help="Where logs are saved"
    )
    parser.add_argument(
        "--enable-log-trace",
        action="store_true",
        help="Enable log tracing. This file can become quite large",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers to process queries",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=8,
        help="Tensor parallelism degree",
    )
    parser.add_argument("--vllm", action="store_true", help="vllm mode")
    parser.add_argument(
        "--api-model-name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Model name(specified in llm server)",
    )
    parser.add_argument(
        "--api-server",
        type=str,
        default=None,
        help="Specify an api endpoint call to use api mode",
    )
    parser.add_argument(
        "--lg-model-name",
        type=str,
        default="llama3_1-8b",
        choices=["llama3_1-8b", "llama3_1-8b-edge"],
        help="Model name(specified in llm server)",
    )

    args = parser.parse_args()
    return args


def parse_summary_file(summary_path):
    """Parse MLPerf summary.txt file and extract metrics"""
    metrics = {}
    
    try:
        with open(summary_path, 'r') as f:
            content = f.read()
        
        # Parse key metrics
        samples_pattern = r"Samples per second:\s+([\d.]+)"
        tokens_pattern = r"Tokens per second:\s+([\d.e+\-]+)"
        valid_pattern = r"Result is\s+:\s+(\w+)"
        
        samples_match = re.search(samples_pattern, content)
        if samples_match:
            metrics["samples_per_second"] = float(samples_match.group(1))
        
        tokens_match = re.search(tokens_pattern, content)
        if tokens_match:
            metrics["tokens_per_second"] = float(tokens_match.group(1))
        
        valid_match = re.search(valid_pattern, content)
        if valid_match:
            metrics["is_valid"] = valid_match.group(1) == "VALID"
        
        # Parse latency statistics (converting from ns to ms)
        latency_patterns = {
            "min_latency_ms": r"Min latency \(ns\)\s*:\s+([\d.e+\-]+)",
            "max_latency_ms": r"Max latency \(ns\)\s*:\s+([\d.e+\-]+)",
            "mean_latency_ms": r"Mean latency \(ns\)\s*:\s+([\d.e+\-]+)",
            "p50_latency_ms": r"50.00 percentile latency \(ns\)\s*:\s+([\d.e+\-]+)",
            "p90_latency_ms": r"90.00 percentile latency \(ns\)\s*:\s+([\d.e+\-]+)",
            "p95_latency_ms": r"95.00 percentile latency \(ns\)\s*:\s+([\d.e+\-]+)",
            "p97_latency_ms": r"97.00 percentile latency \(ns\)\s*:\s+([\d.e+\-]+)",
            "p99_latency_ms": r"99.00 percentile latency \(ns\)\s*:\s+([\d.e+\-]+)",
            "p99.9_latency_ms": r"99.90 percentile latency \(ns\)\s*:\s+([\d.e+\-]+)",
        }
        
        for metric_name, pattern in latency_patterns.items():
            match = re.search(pattern, content)
            if match:
                # Convert nanoseconds to milliseconds
                ns_value = float(match.group(1))
                metrics[metric_name] = ns_value / 1_000_000
        
        log.info(f"Parsed summary metrics: {metrics}")
        return metrics
    
    except Exception as e:
        log.warning(f"Failed to parse summary file: {e}")
        return {}


scenario_map = {
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
    "singlestream": lg.TestScenario.SingleStream,
}


scenario_map = {
    "offline": lg.TestScenario.Offline,
    "server": lg.TestScenario.Server,
    "singlestream": lg.TestScenario.SingleStream,
}


def main():
    args = get_args()

    settings = lg.TestSettings()
    settings.scenario = scenario_map[args.scenario.lower()]
    # mlperf.conf is automatically loaded by the loadgen
    # settings.FromConfig(args.mlperf_conf, "llama3_1-8b", args.scenario)
    settings.FromConfig(args.user_conf, args.lg_model_name, args.scenario)

    if args.accuracy:
        settings.mode = lg.TestMode.AccuracyOnly
    else:
        settings.mode = lg.TestMode.PerformanceOnly

    os.makedirs(args.output_log_dir, exist_ok=True)
    log_output_settings = lg.LogOutputSettings()
    log_output_settings.outdir = args.output_log_dir
    log_output_settings.copy_summary_to_stdout = True
    log_settings = lg.LogSettings()
    log_settings.log_output = log_output_settings
    log_settings.enable_trace = args.enable_log_trace

    if args.vllm:
        from SUT_VLLM import SUT, SUTServer
    else:
        raise NotImplementedError

    sut_map = {"offline": SUT, "server": SUTServer, "singlestream": SUTServer}

    sut_cls = sut_map[args.scenario.lower()]

    if args.vllm:
        sut = sut_cls(
            model_path=args.model_path,
            dtype=args.dtype,
            batch_size=args.batch_size,
            dataset_path=args.dataset_path,
            total_sample_count=args.total_sample_count,
            workers=args.num_workers,
            tensor_parallel_size=args.tensor_parallel_size
        )
    else:
        sut = sut_cls(
            model_path=args.model_path,
            dtype=args.dtype,
            batch_size=args.batch_size,
            dataset_path=args.dataset_path,
            total_sample_count=args.total_sample_count,
            workers=args.num_workers,
        )

    # Start sut before loadgen starts
    sut.start()
    lgSUT = lg.ConstructSUT(sut.issue_queries, sut.flush_queries)
    log.info("Starting Benchmark run")
    lg.StartTestWithLogSettings(
        lgSUT,
        sut.qsl,
        settings,
        log_settings,
        args.audit_conf)

    log.info("Run Completed!")
    
    # Parse and log final results to MLflow before stopping SUT
    summary_path = os.path.join(args.output_log_dir, "mlperf_log_summary.txt")
    log.info(f"Looking for summary file at: {summary_path}")
    
    if os.path.exists(summary_path):
        log.info(f"Summary file size: {os.path.getsize(summary_path)} bytes")
        summary_metrics = parse_summary_file(summary_path)
        if summary_metrics:
            log.info(f"Successfully parsed metrics: {summary_metrics}")
            sut.log_final_results(summary_metrics)
        else:
            log.warning("Summary file exists but no metrics were parsed")
    else:
        log.error(f"Summary file not found at {summary_path}")
        log.info(f"Output directory contents: {os.listdir(args.output_log_dir) if os.path.exists(args.output_log_dir) else 'directory does not exist'}")

    # Stop sut after completion
    sut.stop()

    log.info("Destroying SUT...")
    lg.DestroySUT(lgSUT)

    log.info("Destroying QSL...")
    lg.DestroyQSL(sut.qsl)


if __name__ == "__main__":
    main()