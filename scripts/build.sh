#!/usr/bin/env bash
# Builds llama.cpp trees from third_party/llama.cpp (created on demand by scripts/fetch-llama.sh: upstream
# ggml-org/llama.cpp at docs/PINNED_COMMIT plus patches/*.patch; gitignored, this repo never vendors llama.cpp):
#   build-before : pinned upstream commit, unpatched, CUDA sm_120a   (frozen "before" column)
#   build        : current fork branch (patched), CUDA sm_120a
#   build-cpu    : current fork branch, CPU only (reference backend for test-backend-ops)
#   build-dev    : Phase 2 build (invariant MMVQ split);  build-p3 : Phase 3 build (dev + fp4-smallm kernel)
# Usage: scripts/build.sh [before|after|cpu|dev|p3|all]   (default: all; the shipped tree is p3)
set -euo pipefail
export PATH="/usr/local/cuda-13.2/bin:$PATH"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/third_party/llama.cpp"
PINNED="$(cat "$ROOT/docs/PINNED_COMMIT")"
"$ROOT/scripts/fetch-llama.sh"
JOBS="${JOBS:-$(nproc)}"
# bench/perf-shapes.patch adds the 27B NVFP4 shapes to test-backend-ops perf behind FP4_UNLOCK_PERF_SHAPES=1.
# It is benchmark-only (tests/test-backend-ops.cpp), so it goes on both trees: committed on the fork branch, and
# applied idempotently to the throwaway before-worktree here.
PERF_PATCH="$ROOT/bench/perf-shapes.patch"
apply_perf_patch() { # srcdir
  # The fork tree carries the shapes plus later test edits, so the reverse-apply check can fail even though the
  # shapes are present: the FP4_UNLOCK_PERF_SHAPES marker is the source of truth (chain6 rebuild failure, 2026-10-26).
  if grep -q FP4_UNLOCK_PERF_SHAPES "$1/tests/test-backend-ops.cpp"; then echo "perf-shapes patch already in $1"
  elif git -C "$1" apply "$PERF_PATCH"; then echo "perf-shapes patch applied to $1"
  else echo "perf-shapes patch does not apply to $1 (regenerate it for this tree)"; exit 1; fi
}
# cuBLAS 13 dev files are not installed system-wide (only libcublas12 runtime). Until the owner runs
# `sudo apt install libcublas-dev-13-2`, use the pip nvidia-cublas (cu13) wheel via .cuda-overlay/.
OVERLAY="$ROOT/.cuda-overlay"
CUDA_FLAGS=(-DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120a -DGGML_NATIVE=ON)
if [[ ! -e /usr/local/cuda-13.2/lib64/libcublas.so && -e "$OVERLAY/lib/libcublas.so" ]]; then
  CUDA_FLAGS+=(-DCUDA_cublas_LIBRARY="$OVERLAY/lib/libcublas.so" -DCUDA_cublasLt_LIBRARY="$OVERLAY/lib/libcublasLt.so"
               -DCMAKE_CUDA_FLAGS="-I$OVERLAY/include" -DCMAKE_CXX_FLAGS="-I$OVERLAY/include" -DCMAKE_C_FLAGS="-I$OVERLAY/include"
               -DCMAKE_BUILD_RPATH="$OVERLAY/lib" -DGGML_CUDA_NO_VMM=OFF)
fi
COMMON=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_CURL=ON)
what="${1:-all}"

build_tree() { # dir, srcdir, extra cmake flags...
  local dir="$1"; shift; local src="$1"; shift
  cmake -S "$src" -B "$dir" "${COMMON[@]}" "$@" > "$dir.cmake.log" 2>&1 || { tail -30 "$dir.cmake.log"; exit 1; }
  cmake --build "$dir" -j "$JOBS" > "$dir.build.log" 2>&1 || { tail -40 "$dir.build.log"; exit 1; }
  echo "built $dir ($(git -C "$src" rev-parse --short=9 HEAD))"
}

if [[ "$what" == before || "$what" == all ]]; then
  # worktree at the pinned commit so build-before never sees patches
  WT="$ROOT/third_party/llama.cpp-before"
  if [[ ! -d "$WT" ]]; then git -C "$SRC" worktree add --detach "$WT" "$PINNED"; fi
  git -C "$WT" checkout --detach "$PINNED" -q
  apply_perf_patch "$WT"
  build_tree "$ROOT/build-before" "$WT" "${CUDA_FLAGS[@]}"
fi
if [[ "$what" == after || "$what" == all ]]; then
  apply_perf_patch "$SRC"
  build_tree "$ROOT/build" "$SRC" "${CUDA_FLAGS[@]}"
fi
if [[ "$what" == dev ]]; then
  # Phase 2 development build of the fork tree: invariant MMVQ k split compiled in, FA invariant mode via FP4_UNLOCK_INVARIANT=1
  apply_perf_patch "$SRC"
  DEVF="${CUDA_DEV_FLAGS:-} -DFP4_UNLOCK_INVARIANT_MMVQ"
  if [[ ! -e /usr/local/cuda-13.2/lib64/libcublas.so && -e "$OVERLAY/lib/libcublas.so" ]]; then DEVF="-I$OVERLAY/include $DEVF"; fi
  build_tree "$ROOT/build-dev" "$SRC" "${CUDA_FLAGS[@]}" -DCMAKE_CUDA_FLAGS="$DEVF"
fi
if [[ "$what" == p3 ]]; then
  # Phase 3 development build: Phase 2 invariant MMVQ split plus the fp4-smallm kernel on by default (FP4_UNLOCK_KERNEL=0 disables)
  apply_perf_patch "$SRC"
  P3F="${CUDA_DEV_FLAGS:-} -DFP4_UNLOCK_INVARIANT_MMVQ -DGGML_CUDA_FP4_SMALLM"
  if [[ ! -e /usr/local/cuda-13.2/lib64/libcublas.so && -e "$OVERLAY/lib/libcublas.so" ]]; then P3F="-I$OVERLAY/include $P3F"; fi
  build_tree "$ROOT/build-p3" "$SRC" "${CUDA_FLAGS[@]}" -DCMAKE_CUDA_FLAGS="$P3F"
fi
if [[ "$what" == apoze ]]; then
  # PRD 1.8: scratch worktree with Apoze's 6263445 "cuda: stabilize quantized FA for speculative batches" staged on the pin
  WT="$ROOT/third_party/llama.cpp-apoze"
  [[ -d "$WT" ]] || { echo "no $WT (create with: git -C $SRC worktree add --detach $WT $PINNED && git -C $WT cherry-pick --no-commit 6263445513416ef93fd5d05391299f33c64702fa)"; exit 1; }
  build_tree "$ROOT/build-apoze" "$WT" "${CUDA_FLAGS[@]}"
fi
if [[ "$what" == cpu || "$what" == all ]]; then
  build_tree "$ROOT/build-cpu" "$SRC" -DGGML_CUDA=OFF -DGGML_NATIVE=ON
fi
