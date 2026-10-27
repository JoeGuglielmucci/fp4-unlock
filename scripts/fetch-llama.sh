#!/usr/bin/env bash
# Creates third_party/llama.cpp (gitignored, never pushed) from upstream ggml-org/llama.cpp at the commit in
# docs/PINNED_COMMIT and applies patches/*.patch on top. Idempotent: re-running checks the tree is already patched.
# This repository carries only the patches; it does not vendor or submodule llama.cpp.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/third_party/llama.cpp"
PINNED="$(cat "$ROOT/docs/PINNED_COMMIT")"
UPSTREAM="${LLAMA_CPP_UPSTREAM:-https://github.com/ggml-org/llama.cpp.git}"
if [[ ! -d "$SRC/.git" ]]; then
  mkdir -p "$ROOT/third_party"
  git clone --no-checkout "$UPSTREAM" "$SRC"
fi
if ! git -C "$SRC" cat-file -e "$PINNED^{commit}" 2>/dev/null; then git -C "$SRC" fetch origin "$PINNED"; fi
if [[ ! -e "$SRC/ggml/src/ggml-cuda/fp4-smallm.cu" ]]; then
  git -C "$SRC" checkout -q --detach "$PINNED"
  git -C "$SRC" apply --check "$ROOT"/patches/*.patch
  git -C "$SRC" apply "$ROOT"/patches/*.patch
  echo "third_party/llama.cpp: upstream $(git -C "$SRC" rev-parse --short=9 HEAD) + $(ls "$ROOT"/patches/*.patch | wc -l) patches applied"
else
  echo "third_party/llama.cpp: already patched (upstream $(git -C "$SRC" rev-parse --short=9 HEAD))"
fi
