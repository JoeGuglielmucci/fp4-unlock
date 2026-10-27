#!/usr/bin/env bash
# Regenerates patches/*.patch from the working tree of third_party/llama.cpp against docs/PINNED_COMMIT.
# Four patches, one topic each (see patches/README.md). New files are included via intent-to-add.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/third_party/llama.cpp"
PINNED="$(cat "$ROOT/docs/PINNED_COMMIT")"
cd "$SRC"
git add -N ggml/src/ggml-cuda/fp4-smallm.cu ggml/src/ggml-cuda/fp4-smallm.cuh ggml/src/ggml-cuda/fp4-unlock.cuh 2>/dev/null || true
gen() { local out="$1"; shift; git diff "$PINNED" -- "$@" > "$ROOT/patches/$out"; echo "$out: $(grep -c '^diff --git' "$ROOT/patches/$out") files"; }
gen 0001-llama-model-skip-nextn-scale-tensors-when-mtp-is-off.patch src/llama-model.cpp
gen 0002-tests-batch-invariance-and-nvfp4-perf-shapes.patch tests/test-backend-ops.cpp
gen 0003-cuda-invariant-mode-fa-pins.patch ggml/src/ggml-cuda/fp4-unlock.cuh ggml/src/ggml-cuda/fattn-common.cuh ggml/src/ggml-cuda/fattn-mma-f16.cuh ggml/src/ggml-cuda/fattn.cu
gen 0004-cuda-fp4-smallm-kernel-and-mmvq-dispatch.patch ggml/src/ggml-cuda/fp4-smallm.cu ggml/src/ggml-cuda/fp4-smallm.cuh ggml/src/ggml-cuda/mmvq.cu
