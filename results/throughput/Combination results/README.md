# Combination Results
## Throughput Sweep for Stacked and Combined vLLM Optimizations

This directory contains a focused throughput study for **LLaMA 3.1-8B on vLLM v0.17.1 (V1 engine)**.
Unlike the broader top-level report, this folder is a self-contained experiment pack centered on
one strong serving baseline and a set of targeted follow-up combinations.

Each experiment folder includes:

- `SUT_VLLM.py`: exact serving configuration used for the run
- `experiment_meta.txt`: experiment ID, timestamp, batch size, dtype, vLLM version, and engine
- `result.txt`: final throughput, validity flag, duration, and exit code
- `run.log`: run-time console output
- `mlperf_logs/` and `system/`: benchmark and system artifacts captured during execution

The file `stacked_summary_v17.csv` is the easiest place to compare runs side by side.

---

## Baseline

The reference configuration is:

- FP8 weights
- batch size `1024`
- `gpu_memory_utilization=0.90`
- `max_model_len=2668`
- prefix caching enabled
- async output enabled
- compilation level `O2`
- `FLASH_ATTN`

Baseline result:

| Experiment | Tokens/sec | Samples/sec | Valid |
|---|---:|---:|---|
| `exp_00_BASELINE` | `3023.34` | `23.6199` | `INVALID` |

The `INVALID` status here does not mean the run failed technically: exit code is `0`, and the
throughput number is still useful for directional comparison inside this folder.

---

## Experiment Groups

### Phase A: Memory and batching knobs

These runs tweak one scheduling or memory control at a time against the stacked baseline.

| Experiment | Change | Tokens/sec | Delta vs baseline |
|---|---|---:|---:|
| `exp_A1_gmu_0.95` | raise `gpu_memory_utilization` to `0.95` | `3117.49` | `+3.11%` |
| `exp_A2_batched_tokens_8192` | halve `max_num_batched_tokens` to `8192` | `3094.45` | `+2.35%` |
| `exp_A3_batched_tokens_32768` | raise `max_num_batched_tokens` to `32768` | `3148.96` | `+4.16%` |
| `exp_A4_max_seqs_1024` | raise `max_num_seqs` to `1024` | `3041.02` | `+0.58%` |
| `exp_A5_block_size_32` | set `block_size=32` | `2993.40` | `-0.99%` |
| `exp_A6_prefix_cache_off` | disable prefix caching | `3072.75` | `+1.63%` |
| `exp_A7_skip_tokenizer` | enable `skip_tokenizer_init` | `3141.87` | `+3.92%` |

Takeaway: the strongest Phase A gains come from larger token budget (`A3`), higher GPU memory
utilization (`A1`), and skipping tokenizer init (`A7`). None of these alone changes the throughput
regime dramatically, but several are modest wins.

### Phase B: Runtime/backend toggles

These runs probe compilation and backend choices on top of the same baseline family.

| Experiment | Change | Tokens/sec | Delta vs baseline |
|---|---|---:|---:|
| `exp_B1_compile_O3` | `compilation_config=3` | `3180.24` | `+5.19%` |
| `exp_B2_compile_O0` | `compilation_config=0` | `3149.65` | `+4.18%` |
| `exp_B3_no_async_output` | serial output processing | `3195.83` | `+5.71%` |
| `exp_B4_flashinfer` | switch to `FLASHINFER` | `3261.60` | `+7.88%` |

Takeaway: `FLASHINFER` is the best non-speculative result in this directory and the clearest
single upgrade over the baseline stack.

### Phase C: Speculative decoding

These runs add n-gram speculative decoding and lower GPU memory utilization to make room for it.

| Experiment | Change | Tokens/sec | Valid |
|---|---|---:|---|
| `exp_C1_ngram_3tok_lmax3` | 3 speculative tokens, lookup max 3, `gmu=0.78` | `2936.85` | `INVALID` |
| `exp_C2_ngram_5tok_lmax4` | 5 speculative tokens, lookup max 4, `gmu=0.78` | `2766.57` | `VALID` |
| `exp_C3_ngram_5tok_gmu82` | 5 speculative tokens, `gmu=0.82` | `2781.44` | `VALID` |
| `exp_C4_ngram_3tok_O3` | speculative decode plus `O3` | `2972.58` | `INVALID` |

Takeaway: speculative decoding did **not** beat the stacked baseline here. The only `VALID` runs in
this group (`C2` and `C3`) are both materially slower than `exp_00`.

### Phase D: Combination templates

These folders attempt to combine winners from earlier phases.

| Experiment | Description | Tokens/sec | Valid |
|---|---|---:|---|
| `exp_D1_best_A_plus_best_B` | template for best A + best B | `3102.12` | `INVALID` |
| `exp_D2_best_A_plus_best_C` | template for best A + best C | `2696.25` | `VALID` |

Important note: both `D1` and `D2` are marked in the metadata as **template-style follow-ups**
(`EDIT SUT FIRST`). They are useful as combination probes, but should not be treated as final,
fully locked production configs without inspecting the corresponding `SUT_VLLM.py`.

---

## Full Results Table

This table mirrors `stacked_summary_v17.csv` in README form so all experiments can be scanned
without opening the CSV directly.

| Experiment | Group | Tokens/sec | Samples/sec | Delta vs baseline | Valid | Duration (s) | Description |
|---|---|---:|---:|---:|---|---:|---|
| `exp_00` | Baseline | `3023.34` | `23.6199` | `+0.00%` | `INVALID` | `649` | BASELINE: FP8+BS1024+gmu0.90+len2668+APC+asyncOut+O2+FLASH_ATTN |
| `exp_A1` | Phase A | `3117.49` | `24.3554` | `+3.11%` | `INVALID` | `591` | `gpu_memory_utilization=0.95` |
| `exp_A2` | Phase A | `3094.45` | `24.1754` | `+2.35%` | `INVALID` | `594` | `max_num_batched_tokens=8192` |
| `exp_A3` | Phase A | `3148.96` | `24.6012` | `+4.16%` | `INVALID` | `584` | `max_num_batched_tokens=32768` |
| `exp_A4` | Phase A | `3041.02` | `23.7580` | `+0.58%` | `INVALID` | `600` | `max_num_seqs=1024` |
| `exp_A5` | Phase A | `2993.40` | `23.3860` | `-0.99%` | `INVALID` | `608` | `block_size=32` |
| `exp_A6` | Phase A | `3072.75` | `24.0059` | `+1.63%` | `INVALID` | `594` | prefix caching disabled |
| `exp_A7` | Phase A | `3141.87` | `24.5459` | `+3.92%` | `INVALID` | `584` | `skip_tokenizer_init=True` |
| `exp_B1` | Phase B | `3180.24` | `24.8456` | `+5.19%` | `INVALID` | `588` | `compilation_config=3` (`O3`) |
| `exp_B2` | Phase B | `3149.65` | `24.6067` | `+4.18%` | `INVALID` | `573` | `compilation_config=0` (`O0`) |
| `exp_B3` | Phase B | `3195.83` | `24.9675` | `+5.71%` | `INVALID` | `594` | serial output processing |
| `exp_B4` | Phase B | `3261.60` | `25.4813` | `+7.88%` | `INVALID` | `566` | `attention_backend=FLASHINFER` |
| `exp_C1` | Phase C | `2936.85` | `22.9441` | `N/A` | `INVALID` | `626` | n-gram spec, 3 tokens, lookup max 3, `gmu=0.78` |
| `exp_C2` | Phase C | `2766.57` | `21.6138` | `N/A` | `VALID` | `661` | n-gram spec, 5 tokens, lookup max 4, `gmu=0.78` |
| `exp_C3` | Phase C | `2781.44` | `21.7300` | `N/A` | `VALID` | `655` | n-gram spec, 5 tokens, `gmu=0.82` |
| `exp_C4` | Phase C | `2972.58` | `23.2232` | `N/A` | `INVALID` | `618` | n-gram spec + `O3` |
| `exp_D1` | Phase D | `3102.12` | `24.2353` | `N/A` | `INVALID` | `607` | best A + best B template |
| `exp_D2` | Phase D | `2696.25` | `21.0645` | `N/A` | `VALID` | `673` | best A + best C template |

---

## Quick Ranking

Best throughput runs in this folder:

| Rank | Experiment | Tokens/sec | Notes |
|---|---|---:|---|
| 1 | `exp_B4_flashinfer` | `3261.60` | best overall result, `INVALID` |
| 2 | `exp_B3_no_async_output` | `3195.83` | strong backend/runtime result, `INVALID` |
| 3 | `exp_B1_compile_O3` | `3180.24` | strong compile-level gain, `INVALID` |
| 4 | `exp_B2_compile_O0` | `3149.65` | surprisingly above baseline, `INVALID` |
| 5 | `exp_A3_batched_tokens_32768` | `3148.96` | best Phase A result, `INVALID` |

Best `VALID` runs:

| Experiment | Tokens/sec | Comment |
|---|---:|---|
| `exp_C3_ngram_5tok_gmu82` | `2781.44` | fastest valid run in this folder |
| `exp_C2_ngram_5tok_lmax4` | `2766.57` | slightly slower, also valid |
| `exp_D2_best_A_plus_best_C` | `2696.25` | valid but weakest of the valid set |

---

## How to Read This Folder

If you are reviewing the data for decisions:

1. Start with `stacked_summary_v17.csv` for a quick leaderboard.
2. Open the matching `result.txt` to confirm throughput, duration, and validity.
3. Inspect `SUT_VLLM.py` to see the exact config diff from the baseline.
4. Treat `VALID` status as the stronger signal for benchmark acceptability, and treat `INVALID`
   runs as directional performance measurements unless separately validated.

---

## Main Takeaways

- The stacked baseline already starts from an aggressive configuration rather than a plain default.
- Among non-speculative tweaks, `FLASHINFER` (`B4`) is the strongest improvement in this dataset.
- Several Phase A and Phase B toggles produce modest positive gains, suggesting the baseline still
  has some headroom for runtime/backend tuning.
- Speculative decoding under these settings underperforms the baseline, even in the runs that are
  marked `VALID`.
- The Phase D folders are useful combination experiments, but they should be read as follow-up
  templates rather than final blessed configs.
