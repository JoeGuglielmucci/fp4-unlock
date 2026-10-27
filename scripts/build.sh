#!/usr/bin/env bash
# Builds llama.cpp from third_party/llama.cpp (created on demand by scripts/fetch-llama.sh: upstream ggml-org/llama.cpp
# at docs/PINNED_COMMIT plus patches/*.patch; gitignored, this repo never vendors llama.cpp):
#   build           : the patched build, FP4 kernel on by default (FP4_UNLOCK_KERNEL=0 turns it off)  -> scripts/build.sh after
#   build-invariant : patched build plus the pinned MMVQ k split that FP4_UNLOCK_INVARIANT=1 needs for
#                     bitwise batch invariance; 7-8 % slower on the verify rows                        -> scripts/build.sh invariant
#   build-before    : pinned upstream commit, unpatched, the A/B control                              -> scripts/build.sh before
#   build-cpu       : CPU-only build (reference backend for test-backend-ops)                         -> scripts/build.sh cpu
# Usage: scripts/build.sh [after|invariant|before|cpu|all]   (default: after). JOBS=N limits parallelism.
# Needs: CUDA 13.x toolkit (CUDA_HOME, default /usr/local/cuda-13.* or /usr/local/cuda), cmake, ninja, a C++17 compiler.
set -euo pipefail
CUDA_HOME="${CUDA_HOME:-$(ls -d /usr/local/cuda-13.* /usr/local/cuda 2>/dev/null | head -1)}"
[[ -x "$CUDA_HOME/bin/nvcc" ]] || { echo "nvcc not found under CUDA_HOME=$CUDA_HOME (install the CUDA 13 toolkit or set CUDA_HOME)"; exit 1; }
export PATH="$CUDA_HOME/bin:$PATH"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/third_party/llama.cpp"
PINNED="$(cat "$ROOT/docs/PINNED_COMMIT")"
"$ROOT/scripts/fetch-llama.sh"
JOBS="${JOBS:-$(nproc)}"
# bench/perf-shapes.patch adds the 27B NVFP4 shapes to test-backend-ops perf behind FP4_UNLOCK_PERF_SHAPES=1.
# It is benchmark-only (tests/test-backend-ops.cpp), so it is applied idempotently to both trees.
PERF_PATCH="$ROOT/bench/perf-shapes.patch"
apply_perf_patch() { # srcdir
  # The patched tree carries the shapes plus later test edits, so a reverse-apply check can fail even though the
  # shapes are present: the FP4_UNLOCK_PERF_SHAPES marker is the source of truth.
  if grep -q FP4_UNLOCK_PERF_SHAPES "$1/tests/test-backend-ops.cpp"; then echo "perf-shapes patch already in $1"
  elif git -C "$1" apply "$PERF_PATCH"; then echo "perf-shapes patch applied to $1"
  else echo "perf-shapes patch does not apply to $1 (regenerate it for this tree)"; exit 1; fi
}
# If the toolkit lacks the cuBLAS 13 development files (libcublas.so missing under $CUDA_HOME/lib64), either install
# them (e.g. `sudo apt install libcublas-dev-13-2`) or drop the pip nvidia-cublas (cu13) wheel contents into
# .cuda-overlay/{include,lib} (gitignored); the overlay is used only when present.
OVERLAY="$ROOT/.cuda-overlay"
CUDA_FLAGS=(-DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES=120a -DGGML_NATIVE=ON)
if [[ ! -e $CUDA_HOME/lib64/libcublas.so && -e "$OVERLAY/lib/libcublas.so" ]]; then
  CUDA_FLAGS+=(-DCUDA_cublas_LIBRARY="$OVERLAY/lib/libcublas.so" -DCUDA_cublasLt_LIBRARY="$OVERLAY/lib/libcublasLt.so"
               -DCMAKE_CUDA_FLAGS="-I$OVERLAY/include" -DCMAKE_CXX_FLAGS="-I$OVERLAY/include" -DCMAKE_C_FLAGS="-I$OVERLAY/include"
               -DCMAKE_BUILD_RPATH="$OVERLAY/lib" -DGGML_CUDA_NO_VMM=OFF)
fi
COMMON=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_CURL=ON)
what="${1:-after}"

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
  # the default build: patches applied, small-M FP4 kernel on by default
  apply_perf_patch "$SRC"
  AF="${CUDA_DEV_FLAGS:-} -DGGML_CUDA_FP4_SMALLM"
  if [[ ! -e $CUDA_HOME/lib64/libcublas.so && -e "$OVERLAY/lib/libcublas.so" ]]; then AF="-I$OVERLAY/include $AF"; fi
  build_tree "$ROOT/build" "$SRC" "${CUDA_FLAGS[@]}" -DCMAKE_CUDA_FLAGS="$AF"
fi
if [[ "$what" == invariant || "$what" == all ]]; then
  # same plus the compile-time MMVQ k-split pin: required for full bitwise invariance under FP4_UNLOCK_INVARIANT=1
  apply_perf_patch "$SRC"
  IF="${CUDA_DEV_FLAGS:-} -DGGML_CUDA_FP4_SMALLM -DFP4_UNLOCK_INVARIANT_MMVQ"
  if [[ ! -e $CUDA_HOME/lib64/libcublas.so && -e "$OVERLAY/lib/libcublas.so" ]]; then IF="-I$OVERLAY/include $IF"; fi
  build_tree "$ROOT/build-invariant" "$SRC" "${CUDA_FLAGS[@]}" -DCMAKE_CUDA_FLAGS="$IF"
fi
if [[ "$what" == cpu || "$what" == all ]]; then
  build_tree "$ROOT/build-cpu" "$SRC" -DGGML_CUDA=OFF -DGGML_NATIVE=ON
fi
