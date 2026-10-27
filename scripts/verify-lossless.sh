#!/usr/bin/env bash
# The headline check, in one command: with FP4_UNLOCK_INVARIANT=1 on the invariant build, greedy output with a
# drafter on must be byte-identical to greedy output with the drafter off. Runs the spec-off reference twice
# (run-to-run control), then MTP n_max 7 and DFlash2 n_max 7 on f16 KV, 6 prompts x 512 tokens each, and prints
# the divergence rate per drafter (0.0 = lossless). About 6-8 minutes on an RTX 5090.
# Needs: scripts/build.sh invariant, scripts/fetch-models.sh (primary). Full matrix (f16 + q8_0 KV, n_max 3 and 7):
#   FP4_UNLOCK_INVARIANT=1 bench/invariance.sh invariant 27b-nvfp4
# Usage: scripts/verify-lossless.sh [after|invariant]   (default invariant; 'after' shows the default build diverging)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
tree="${1:-invariant}"
case "$tree" in
  invariant) [[ -x "$ROOT/build-invariant/bin/llama-server" ]] || { echo "build-invariant/ missing: run scripts/build.sh invariant" >&2; exit 1; } ;;
  after)     [[ -x "$ROOT/build/bin/llama-server" ]] || { echo "build/ missing: run scripts/build.sh" >&2; exit 1; } ;;
  *) echo "usage: $0 [after|invariant]" >&2; exit 2 ;;
esac
export FP4_UNLOCK_INVARIANT=1
OUT="${OUT:-$ROOT/bench/results/$(date +%F)-$tree-27b-nvfp4-verify-lossless}"
KVS="${KVS:-f16}" DRAFTERS="${DRAFTERS:-mtp-n7 dflash-n7}" OUT="$OUT" "$ROOT/bench/invariance.sh" "$tree" 27b-nvfp4
echo
echo "A divergence rate of 0.0 for every drafter means the drafter never changed a token. Results: $OUT/summary.txt"
