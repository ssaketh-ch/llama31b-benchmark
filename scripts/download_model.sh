#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <huggingface_repo> <local_dir>" >&2
  echo "Example: $0 meta-llama/Meta-Llama-3.1-8B-Instruct ./llama3.1-8b/model" >&2
  exit 1
fi

REPO="$1"
DEST="$2"
mkdir -p "$DEST"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli is required; install via: pip install huggingface_hub" >&2
  exit 1
fi

echo "Downloading $REPO to $DEST ..."
huggingface-cli download "$REPO" --local-dir "$DEST" --resume-download

echo "Done. Files stored at: $DEST"
