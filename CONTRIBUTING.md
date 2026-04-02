# Contributing

This project is a reproducible MLPerf benchmarking harness for Llama-3.1-8B.

## Development setup

```bash
make setup
```

This creates a virtual environment and installs dependencies from `env/requirements.txt`.

## Adding experiments

- **Isolation**: add an experiment to `scripts/run_isolation.sh` — change exactly one vLLM knob vs the baseline, pin all others explicitly.
- **Combinations**: add to `scripts/run_stacked.sh` — stack multiple knobs once individual effects are understood.
- Every experiment should have a clear hypothesis and a regression comparison.

## Code style

- Python: 4-space indent, explicit imports, logging over print.
- Shell: `set -uo pipefail`, quote all variable expansions, 2-space indent.
- See `.editorconfig` for editor-level formatting rules.

## Reproducibility notes

- Always run `nvidia-smi` before and after benchmark experiments.
- GPU memory state can affect results. A `nvidia-smi --gpu-reset` is run between experiments in the isolation/stacked scripts.
- Record the vLLM version, PyTorch version, and CUDA version with every result.
