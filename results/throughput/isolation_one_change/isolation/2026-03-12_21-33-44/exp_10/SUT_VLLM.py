# EXP_10: async_scheduling=True [vLLM 0.10.0]
# ONLY CHANGE from baseline: async_scheduling not set -> True
# Overlaps Python scheduler with GPU execution. At BS=16 (~836 batches),
# there is meaningful scheduler work between batches, unlike BS=1024 (~13 batches).
# Previous result at BS=1024 was -3.4%, but that had stacked configs.
# This isolates the scheduler overhead at OG batch size.
import os
import array
import threading
import queue
import logging
import torch
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
        self.data_object = Dataset(
            self.model_path, dataset_path=self.dataset_path,
            total_sample_count=total_sample_count, dtype=dtype)
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
        )
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
            quantization=None,                 # baseline: NO quant
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,       # baseline value
            max_model_len=131072,              # baseline value
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enforce_eager=False,
            cpu_offload_gb=0,
            scheduler_delay_factor=0.0,
            async_scheduling=True,             # ← CHANGE (from baseline: not set)
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start()
            self.worker_threads[j] = w

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)
        for w in self.worker_threads:
            w.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break
            query_ids = [q.index for q in qitem]
            input_ids_tensor = [
                self.data_object.input_ids[q.index] for q in qitem]
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [
                list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(
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
