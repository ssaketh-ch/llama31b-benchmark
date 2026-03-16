# Llama-3.1-8B SUT Guide

Quick reference for the System Under Test (SUT) variants under `llama3.1-8b/`. Use this to choose the right harness for your run.

## SUT variants

### SUT_VLLM_OG.py
- Purpose: Original reference SUT using `AsyncLLMEngine`; minimal tuning; defaults mimic upstream MLPerf reference.
- Key settings: BF16 weights, TP=8 (upstream default), batch_size=1 if not provided, no quantization, no sorting, max_model_len default (131072), gpu_memory_utilization default (~0.90). Async engine, no chunked prefill, no prefix cache toggles. Good for functional parity with upstream, not for performance on single MIG.

### SUT_VLLM.py
- Purpose: OG baseline but with sync `LLM` engine for consistent benchmarking; otherwise matches OG defaults.
- Key settings: BF16, TP=1, batch_size=16 (if unset), no quantization, no sorting, max_model_len=131072, gpu_memory_utilization=0.90. No hardware env tweaks. Use when you want the clean reference baseline for comparisons.

### SUT_VLLM_TEST.py
- Purpose: One-knob test harness (exp_A/B/C) with a fixed tuned baseline and single-change experiments.
- Baseline defaults: FP8 weights, float16 compute, TP=1, batch_size=1024, max_model_len=2668, max_num_seqs=512, gpu_memory_utilization=0.95, sorted batching, allocator `expandable_segments`, TF32 enabled.
- Flags toggled by experiments: chunked prefill, max_num_batched_tokens, gpu_memory_utilization, prefix cache, scheduler_delay_factor, async_scheduling, num_scheduler_steps, compilation_config, attention backend (FlashInfer).
- Use when running the single-knob scripts to isolate impact of each flag.

### SUT_VLLM_OPTIMAL_NVTX.py
- Purpose: Tuned high-throughput path with NVTX ranges for profiling.
- Key settings: FP8 weights, float16 compute, TP=1, batch_size=1024, max_model_len=2668, max_num_batched_tokens=65536, max_num_seqs=512, gpu_memory_utilization=0.95, sorted batching, chunked prefill on, prefix cache off, allocator `expandable_segments`, TF32 enabled, NVTX ranges around load/generate/postprocess.
- Use when you want peak throughput plus profiling markers (e.g., Nsight Systems).

## Choosing a SUT
- Need upstream parity / functional baseline: SUT_VLLM_OG.py.
- Need reference baseline with sync engine for comparisons: SUT_VLLM.py.
- Running single-knob experiments: SUT_VLLM_TEST.py.
- Profiling tuned stack with NVTX: SUT_VLLM_OPTIMAL_NVTX.py.

## Notes
- All SUTs use Offline scenario by default; Server scenario exists in upstream but not exercised here.
- Ensure `llama3.1-8b/dataset` is populated (or use download scripts) before running; the dataset is git-ignored.
