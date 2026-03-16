import asyncio
import os
import time
import numpy as np
import array
import torch
from torch.nn.functional import pad
from vllm import LLM, AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.inputs import TokensPrompt

import pickle
import time
import threading
import tqdm
import queue

import logging
from typing import TYPE_CHECKING, Optional, List
from pathlib import Path

import mlflow
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
        # Set this to True *only for test accuracy runs* in case your prior
        # session was killed partway through
        workers=1,
        tensor_parallel_size=8
    ):

        self.model_path = model_path or f"meta-llama/Meta-Llama-3.1-8B-Instruct"

        if not batch_size:
            batch_size = 1
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
            dtype=dtype
        )
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam,
        )

        self.load_model()
        gen_kwargs = {
            "temperature": 0.0,
            "top_p": 1,
            "top_k": 1,
            "seed": 42,
            "max_tokens": 128,
            "min_tokens": 1
        }
        self.sampling_params = SamplingParams(**gen_kwargs)
        # self.sampling_params.all_stop_token_ids.add(self.model.get_tokenizer().eos_token_id)

        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()

        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()
        
        # MLflow tracking for latencies
        self.latencies = []
        self.latencies_lock = threading.Lock()
        self.setup_mlflow()

    def setup_mlflow(self):
        """Initialize MLflow tracking"""
        self.mlflow_enabled = False
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        
        try:
            mlflow.set_tracking_uri(mlflow_uri)
            
            # Test connection by getting tracking URI
            actual_uri = mlflow.get_tracking_uri()
            log.info(f"MLflow tracking URI: {actual_uri}")
            
            experiment_name = "llama3.1-8b-inference"
            mlflow.set_experiment(experiment_name)
            
            # Create a meaningful run name with timestamp
            from datetime import datetime
            run_name = f"llama3.1-8b-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            mlflow.start_run(run_name=run_name)
            
            # Log parameters
            mlflow.log_param("model_path", self.model_path)
            mlflow.log_param("batch_size", self.batch_size)
            mlflow.log_param("dtype", self.dtype)
            mlflow.log_param("tensor_parallel_size", self.tensor_parallel_size)
            mlflow.log_param("num_workers", self.num_workers)
            
            self.mlflow_enabled = True
            log.info(f"MLflow tracking started - Experiment: {experiment_name}")
            log.info(f"Run name: {run_name}")
            log.info(f"Access MLflow at: {mlflow_uri}")
        except Exception as e:
            log.error(f"MLflow setup failed: {e}")
            log.error("Make sure MLflow server is running: mlflow server --host localhost --port 5000")
            self.mlflow_enabled = False

    def calculate_percentile(self, percentile):
        """Calculate latency percentile from collected latencies"""
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        idx = int((percentile / 100.0) * len(sorted_latencies))
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    def log_final_results(self, summary_dict):
        """Log final MLPerf summary metrics to MLflow
        
        Args:
            summary_dict: Dictionary containing final metrics from MLPerf summary
                Expected keys: samples_per_second, tokens_per_second, min_latency_ms, 
                max_latency_ms, mean_latency_ms, and percentile keys like p50_latency_ms, etc.
        """
        try:
            log.info("Logging final MLPerf results to MLflow...")
            
            # Log key metrics
            if "samples_per_second" in summary_dict:
                mlflow.log_metric("final_samples_per_sec", summary_dict["samples_per_second"])
            if "tokens_per_second" in summary_dict:
                mlflow.log_metric("final_tokens_per_sec", summary_dict["tokens_per_second"])
            
            # Log latency statistics
            if "min_latency_ms" in summary_dict:
                mlflow.log_metric("final_min_latency_ms", summary_dict["min_latency_ms"])
            if "max_latency_ms" in summary_dict:
                mlflow.log_metric("final_max_latency_ms", summary_dict["max_latency_ms"])
            if "mean_latency_ms" in summary_dict:
                mlflow.log_metric("final_mean_latency_ms", summary_dict["mean_latency_ms"])
            
            # Log percentiles
            percentiles = [50, 90, 95, 97, 99, 99.9]
            for p in percentiles:
                key = f"p{p}_latency_ms"
                if key in summary_dict:
                    mlflow.log_metric(f"final_{key}", summary_dict[key])
            
            # Log validation results
            if "is_valid" in summary_dict:
                mlflow.log_param("final_result_valid", summary_dict["is_valid"])
            
            log.info("Final results logged to MLflow successfully!")
        except Exception as e:
            log.warning(f"Failed to log final results: {e}")

    def start(self):
        # Create worker threads
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        for worker in self.worker_threads:
            worker.join()
        
        # End MLflow run
        if self.mlflow_enabled:
            try:
                mlflow.end_run()
                log.info("MLflow run ended")
            except Exception as e:
                log.error(f"Failed to end MLflow run: {e}")

    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic"""
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break

            query_ids = [q.index for q in qitem]

            tik1 = time.time()

            input_ids_tensor = [
                self.data_object.input_ids[q.index] for q in qitem]
            # input_text_tensor = [
            #     self.data_object.input[q.index] for q in qitem]
            # for in_text in input_text_tensor:
            #     log.info(f"Input: {in_text}")

            tik2 = time.time()
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor, sampling_params=self.sampling_params
            )
            pred_output_tokens = []
            for output in outputs:
                pred_output_tokens.append(list(output.outputs[0].token_ids))
                # log.info(f"Output: {output.outputs[0].text}")
            tik3 = time.time()

            processed_output = self.data_object.postProcess(
                pred_output_tokens,
                query_id_list=query_ids,
            )
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                response = [
                    lg.QuerySampleResponse(
                        qitem[i].id,
                        bi[0],
                        bi[1],
                        n_tokens)]
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
                    
                    # Track latency for percentile calculations
                    latency_ms = (tik3 - tik2) * 1000
                    with self.latencies_lock:
                        self.latencies.append(latency_ms)
                    
                    # Log metrics to MLflow (only if enabled)
                    if self.mlflow_enabled:
                        try:
                            mlflow.log_metric("latency_ms", latency_ms, step=self.sample_counter)
                            mlflow.log_metric("p50_latency_ms", self.calculate_percentile(50), step=self.sample_counter)
                            mlflow.log_metric("p90_latency_ms", self.calculate_percentile(90), step=self.sample_counter)
                            mlflow.log_metric("p95_latency_ms", self.calculate_percentile(95), step=self.sample_counter)
                            mlflow.log_metric("p99_latency_ms", self.calculate_percentile(99), step=self.sample_counter)
                            mlflow.log_metric("p99.9_latency_ms", self.calculate_percentile(99.9), step=self.sample_counter)
                        except Exception as e:
                            log.error(f"MLflow logging failed: {e}")
                            self.mlflow_enabled = False

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
        )
        log.info("Loaded model")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self):
        return self.qsl

    def predict(self, **kwargs):
        raise NotImplementedError

    def issue_queries(self, query_samples):
        """Receives samples from loadgen and adds them to queue. Users may choose to batch here"""

        list_prompts_tokens = []
        list_prompts_attn_masks = []

        log.info(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[: self.batch_size])
            query_samples = query_samples[self.batch_size:]
        log.info(f"IssueQuery done")

    def flush_queries(self):
        pass

    def __del__(self):
        pass


class SUTServer(SUT):
    def __init__(
        self,
        model_path=None,
        dtype="bfloat16",
        total_sample_count=13368,
        dataset_path=None,
        batch_size=None,
        workers=1,
        tensor_parallel_size=8
    ):

        super().__init__(
            model_path=model_path,
            dtype=dtype,
            total_sample_count=total_sample_count,
            dataset_path=dataset_path,
            workers=workers,
            tensor_parallel_size=tensor_parallel_size,
        )
        self.request_id = 0

        self.first_token_queue = queue.Queue()

    def start(self):
        # Create worker threads
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

    async def stream_output(self, qitem, results_generator):
        first = True
        async for request_output in results_generator:
            output_response = request_output
            if first:
                first_tokens = list(output_response.outputs[0].token_ids)
                response_data = array.array(
                    "B", np.array(first_tokens, np.int32).tobytes())
                bi = response_data.buffer_info()
                response = [lg.QuerySampleResponse(qitem.id, bi[0], bi[1])]
                lg.FirstTokenComplete(response)
                first = False

        outputs = output_response
        pred_output_tokens = list(output_response.outputs[0].token_ids)
        n_tokens = len(pred_output_tokens)
        response_array = array.array(
            "B", np.array(pred_output_tokens, np.int32).tobytes()
        )
        bi = response_array.buffer_info()
        response = [
            lg.QuerySampleResponse(
                qitem.id,
                bi[0],
                bi[1],
                n_tokens)]
        lg.QuerySamplesComplete(response)

    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic"""
        while True:

            qitem = self.query_queue.get()
            if qitem is None:
                break

            input_ids_tensor = TokensPrompt(
                prompt_token_ids=self.data_object.input_ids[qitem.index])

            # TODO: This PoC is super slow with significant overhead. Best to
            # create a patch to `generate`
            results_generator = self.model.generate(
                prompt=input_ids_tensor, sampling_params=self.sampling_params, request_id=str(
                    self.request_id)
            )
            self.request_id += 1
            asyncio.run(self.stream_output(qitem, results_generator))

    def issue_queries(self, query_samples):
        self.query_queue.put(query_samples[0])

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        for worker in self.worker_threads:
            worker.join()

        self.first_token_queue.put(None)
        self.ft_response_thread.join()

    def load_model(self):
        log.info("Loading model")
        self.engine_args = AsyncEngineArgs(
            self.model_path,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size)
        self.model = AsyncLLMEngine.from_engine_args(self.engine_args)
        log.info("Loaded model")