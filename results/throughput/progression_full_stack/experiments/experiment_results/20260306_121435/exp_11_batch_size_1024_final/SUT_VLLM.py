# EXP 10 / FINAL CONFIG: +enable_prefix_caching=False
# CHANGE: enable_prefix_caching True → False
# CNN DailyMail articles are all unique — zero cache hit rate on this dataset.
# Prefix caching adds a hash lookup per sequence on every prefill with no benefit.
# Confirmed optimal batch size: 1024 (BS=512 gives 2881, BS=2048 regresses to 2832)
import asyncio, os, array, threading, queue, logging
import torch, numpy as np
from vllm import LLM, AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.inputs import TokensPrompt
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
        self.sampling_params = SamplingParams(temperature=0.0, top_p=1, top_k=1,
                                              seed=42, max_tokens=128, min_tokens=1,
                                              ignore_eos=False, skip_special_tokens=True)
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
        torch.set_float32_matmul_precision('high')
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

    def load_model(self):
        log.info("Loading model...")
        self.model = LLM(
            self.model_path,
            dtype="float16",
            quantization="fp8",
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=0.95,
            max_model_len=2668,
            max_num_batched_tokens=65536,
            max_num_seqs=512,
            enable_prefix_caching=False,  # CHANGE: zero hit rate on CNN DailyMail
            enable_chunked_prefill=True,
            enforce_eager=False,
            scheduler_delay_factor=0.0,
            cpu_offload_gb=0,
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
            processed_output = self.data_object.postProcess(pred_output_tokens,
                                                             query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)])
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self): return self.qsl
    def predict(self, **kwargs): raise NotImplementedError

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

    def flush_queries(self): pass
    def __del__(self): pass

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items(): log.info(f"  {k}: {v}")

class SUTServer(SUT):
    def __init__(self, model_path=None, dtype="bfloat16", total_sample_count=13368,
                 dataset_path=None, batch_size=None, workers=1, tensor_parallel_size=1):
        super().__init__(model_path=model_path, dtype=dtype,
                         total_sample_count=total_sample_count,
                         dataset_path=dataset_path, workers=workers,
                         tensor_parallel_size=tensor_parallel_size)
        self.request_id = 0
        self.first_token_queue = queue.Queue()

    def start(self):
        for j in range(self.num_workers):
            w = threading.Thread(target=self.process_queries, daemon=True)
            w.start(); self.worker_threads[j] = w

    async def stream_output(self, qitem, results_generator):
        first = True
        async for request_output in results_generator:
            output_response = request_output
            if first:
                first_token = [output_response.outputs[0].token_ids[0]]
                response_data = array.array("B", np.array(first_token, np.int32).tobytes())
                bi = response_data.buffer_info()
                lg.FirstTokenComplete([lg.QuerySampleResponse(qitem.id, bi[0], bi[1])])
                first = False
        pred_output_tokens = list(output_response.outputs[0].token_ids)
        n_tokens = len(pred_output_tokens)
        response_array = array.array("B", np.array(pred_output_tokens, np.int32).tobytes())
        bi = response_array.buffer_info()
        lg.QuerySamplesComplete([lg.QuerySampleResponse(qitem.id, bi[0], bi[1], n_tokens)])

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None: break
            input_ids_tensor = TokensPrompt(
                prompt_token_ids=self.data_object.input_ids[qitem.index])
            results_generator = self.model.generate(
                prompt=input_ids_tensor, sampling_params=self.sampling_params,
                request_id=str(self.request_id), use_tqdm=False)
            self.request_id += 1
            asyncio.run(self.stream_output(qitem, results_generator))

    def issue_queries(self, query_samples):
        self.query_queue.put(query_samples[0])

    def stop(self):
        for _ in range(self.num_workers): self.query_queue.put(None)
        for w in self.worker_threads: w.join()
        if hasattr(self, "ft_response_thread"):
            self.first_token_queue.put(None)
            self.ft_response_thread.join()
