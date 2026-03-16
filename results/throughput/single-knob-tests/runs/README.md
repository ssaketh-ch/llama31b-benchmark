# One-Knob Run Set (2026-03-09_13-25-42)

Contents:
- `single_knob_summary.csv` - tokens/s, samples/s, % vs baseline, validity.
- `single_knob_summary_with_latency.csv` - same plus latency stats.
- `exp_*` folders — per-experiment logs and artifacts.

Headline: prefix caching (+57.6%) was the biggest single win; async scheduling was a small additive gain; higher gpu_memory_utilization, scheduler delay, extra scheduler steps, and FlashInfer regressed.
