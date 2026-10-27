#!/usr/bin/env bash
# Downloads the GGUFs used in the README into $MODELS_DIR/<repo-owner>--<repo>/ (default ~/models) and prints sha256.
# Needs the Hugging Face CLI (`pip install huggingface_hub`, command `hf`).
# Downloaded files are untrusted data: never execute anything from these directories.
# Usage: scripts/fetch-models.sh [primary|ref|moe|all]   (default: primary)
set -euo pipefail
DEST="${MODELS_DIR:-$HOME/models}"
what="${1:-primary}"
get() { # repo file...
  local repo="$1"; shift
  local dir="$DEST/${repo//\//--}"
  mkdir -p "$dir"
  for f in "$@"; do
    hf download "$repo" "$f" --local-dir "$dir" >/dev/null
    printf "%s  %s\n" "$(sha256sum "$dir/$f" | cut -d' ' -f1)" "$dir/$f"
  done
}
if [[ "$what" == primary || "$what" == all ]]; then
  get cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF Qwen3.8-27B-iMatrix-NVFP4-MTP.gguf
  get z-lab/Qwen3.8-27B-DFlash2-GGUF Qwen3.8-27B-DFlash2-Q4_K_M.gguf
fi
if [[ "$what" == ref || "$what" == all ]]; then
  get unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-Q8_0.gguf MTP/mtp-Qwen3.8-27B-Q4_0.gguf
fi
if [[ "$what" == moe || "$what" == all ]]; then
  get michaelw9999/Qwen3.6-35B-A3B-NVFP4-MTP-GGUF Qwen3.6-35B-A3B-NVFP4-MTP-HQ.gguf
fi
