# EXP_B3: +disable_async_output_proc=True (serialise output processing)
# CHANGE: False->True. V1 default runs output processing async with next GPU step.
# Measures the async output processing benefit for this workload.
import array, logging, os, queue, threading, time
import torch
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
from vllm.inputs import TokensPrompt
import mlperf_loadgen as lg
from dataset import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-8B-SUT")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision("high")


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
                                   self.data_object.perf_count,
                                   self.data_object.LoadSamplesToRam,
                                   self.data_object.UnloadSamplesFromRam)
        self.load_model()
        self.sampling_params = SamplingParams(temperature=0.0, top_p=1, top_k=1,
                                              seed=42, max_tokens=128, min_tokens=1)
        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        self.query_queue = queue.Queue()
        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def load_model(self):
        log.info("Loading model (vLLM 0.17.1 V1)...")
        self.model = LLM(
            self.model_path,
            dtype="float16", quantization="fp8", tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2668, max_num_seqs=512, max_num_batched_tokens=16384,
            enable_prefix_caching=True,
            # (serial output processing enabled in this experiment)
            compilation_config=2, enforce_eager=False, cpu_offload_gb=0,
            attention_backend="FLASH_ATTN",
        )
        log.info("Loaded model")

    def start(self):
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries, daemon=True)
            worker.start(); self.worker_threads[j] = worker

    def stop(self):
        for _ in range(self.num_workers): self.query_queue.put(None)
        for worker in self.worker_threads: worker.join()

    def process_queries(self):
        while True:
            qitem = self.query_queue.get()
            if qitem is None: break
            query_ids = [q.index for q in qitem]
            tik1 = time.time()
            # vLLM V1 API: TokensPrompt wrapper required; raw list of lists removed
            prompts = [TokensPrompt(prompt_token_ids=self.data_object.input_ids[q.index])
                       for q in qitem]
            tik2 = time.time()
            outputs = self.model.generate(prompts, sampling_params=self.sampling_params)
            pred_output_tokens = [list(o.outputs[0].token_ids) for o in outputs]
            tik3 = time.time()
            processed_output = self.data_object.postProcess(pred_output_tokens,
                                                            query_id_list=query_ids)
            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array("B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                lg.QuerySamplesComplete([lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)])
            tok = time.time()
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")
                log.info(f"\tBatchMaker: {tik2-tik1:.4f}s | Inference: {tik3-tik2:.4f}s | Post: {tok-tik3:.4f}s | Total: {tok-tik1:.4f}s")

    def issue_queries(self, query_samples):
        log.info(f"IssueQuery: {len(query_samples)} samples")
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

    def log_final_results(self, summary_dict):
        log.info("=== Final MLPerf Results ===")
        for k, v in summary_dict.items(): log.info(f"  {k}: {v}")


class SUTServer(SUT): pass
