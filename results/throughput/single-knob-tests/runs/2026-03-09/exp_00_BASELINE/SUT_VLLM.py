# EXP_00: BASELINE
# Fixed config. Every other experiment changes exactly ONE thing from this.
#
# Pinned flags (OFF unless the experiment under test changes them):
#   enable_chunked_prefill=False     — tested in exp_A1, exp_A3
#   max_num_batched_tokens not set   — tested in exp_A2, exp_A3
#   gpu_memory_utilization=0.95      — tested in exp_A4
#   enable_prefix_caching=False      — tested in exp_A5
#   scheduler_delay_factor=0.0       — tested in exp_A6
#   async_scheduling not set         — tested in exp_B1
#   num_scheduler_steps not set      — tested in exp_B2, exp_B3
#   compilation_config not set       — tested in exp_B4
#   VLLM_ATTENTION_BACKEND default   — tested in exp_B5
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
        self.sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, top_k=1, seed=42,
            max_tokens=128, min_tokens=1,
            ignore_eos=False, skip_special_tokens=True,
        )
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
        # VLLM_ATTENTION_BACKEND not set — default Flash Attention 3

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="float16",               # fp8 quant forces float16 regardless
            quantization="fp8",            # always on
            tensor_parallel_size=1,        # always on (8B fits on 1 MIG slice)
            gpu_memory_utilization=0.95,   # BASELINE (exp_A4 tests 0.98)
            max_model_len=2668,            # always on (dataset max 2540 + output 128)
            max_num_seqs=512,              # always on
            enable_prefix_caching=False,   # BASELINE OFF (exp_A5 tests True)
            enable_chunked_prefill=False,  # BASELINE OFF (exp_A1/A3 tests True)
            enforce_eager=False,           # always on (CUDA graphs)
            cpu_offload_gb=0,              # always on
            scheduler_delay_factor=0.0,    # BASELINE (exp_A6 tests 0.1)
            # max_num_batched_tokens not set  — BASELINE (exp_A2/A3 tests 65536)
            # async_scheduling not set        — BASELINE (exp_B1 tests True)
            # num_scheduler_steps not set     — BASELINE (exp_B2/B3 tests 2/4)
            # compilation_config not set      — BASELINE (exp_B4 tests 3)
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
            outputs = self.model.generate(
                prompt_token_ids=input_ids_tensor,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            processed_output = self.data_object.postProcess(
                pred_output_tokens, query_id_list=query_ids
            )
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)
                ])
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

