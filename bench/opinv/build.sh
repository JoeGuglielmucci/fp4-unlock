#!/usr/bin/env bash
# Build the op-level invariance probe against one tree's ggml libraries: bench/opinv/build.sh [before|after|invariant] [opinv|mmstress]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
t=${1:-after}; prog=${2:-opinv}
case $t in
  before) B=$ROOT/build-before; S=$ROOT/third_party/llama.cpp ;;
  after)  B=$ROOT/build;        S=$ROOT/third_party/llama.cpp ;;
  invariant) B=$ROOT/build-invariant; S=$ROOT/third_party/llama.cpp ;;
  *) echo "unknown tree $t" >&2; exit 2 ;;
esac
g++ -O2 -std=c++17 -I"$S/ggml/include" "$ROOT/bench/opinv/$prog.cpp" \
  -L"$B/bin" -lggml -lggml-base -lggml-cuda -lggml-cpu -Wl,-rpath,"$B/bin" -o "$B/bin/$prog"
echo "built $B/bin/$prog"
