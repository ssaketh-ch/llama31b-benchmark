# STK_03_KV2668: +max_model_len=2668, max_num_batched_tokens=2668.
# Added on top of stk_02: right-size KV cache to dataset max.
# max_num_batched_tokens=2668 matches max_model_len to stabilise chunked prefill.
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
            batch_size = 1024
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
        self.model = LLM(
            self.model_path,
            dtype="float16",                # from stk_01
            tensor_parallel_size=self.tensor_parallel_size,
            quantization="fp8",             # from stk_01
            max_model_len=2668,             # ADDED stk_03
            max_num_batched_tokens=2668,    # match max_model_len
        )
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
            # Build TokensPrompt list (vLLM V1 API; prompt_token_ids kwarg removed)
            prompts = [
                TokensPrompt(prompt_token_ids=self.data_object.input_ids[q.index])
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
                lg.QuerySamplesComplete([
                    lg.QuerySampleResponse(qitem[i].id, bi[0], bi[1], n_tokens)
                ])

            tok = time.time()
            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                log.info(f"Samples run: {self.sample_counter}")
                log.info(f"\tBatchMaker time: {tik2 - tik1:.4f}s")
                log.info(f"\tInference time:  {tik3 - tik2:.4f}s")
                log.info(f"\tPostprocess time:{tok  - tik3:.4f}s")
                log.info(f"\t==== Total time: {tok  - tik1:.4f}s")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self):
        return self.qsl

    def predict(self, **kwargs):
        raise NotImplementedError

    def issue_queries(self, query_samples):
        log.info(f"IssueQuery started with {len(query_samples)} samples")
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
