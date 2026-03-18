# One-Knob Run Set (2026-03-09_13-25-42)

Contents:
- `variant_summary.csv` — tokens/s, samples/s, % vs baseline, validity.
- `variant_summary_with_latency.csv` — same plus latency stats.
- `exp_*` folders — per-experiment logs and artifacts.

## Highlights
- **Prefix caching** (+57.6%) was the most impactful single change.
- **Async scheduling** provided a small additive gain.
- Higher GPU memory utilization, scheduler delay, extra scheduler steps, and FlashInfer regressed throughput.

See the summary CSVs for full results and latency breakdowns.
