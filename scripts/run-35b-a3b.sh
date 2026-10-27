#!/usr/bin/env bash
# llama-server on the MoE model (demonstrates invariance; little kernel gain expected).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# tree: after (default; the patched build/), invariant (patched build with the pinned MMVQ split, build-invariant/),
# before (pinned upstream, unpatched). Runtime switches (FP4_UNLOCK_KERNEL/T/MIN_M/INVARIANT) pass through the
# environment, e.g. FP4_UNLOCK_INVARIANT=1 scripts/run-27b.sh invariant mtp 7
tree="${1:-after}"; spec="${2:-mtp}"; nmax="${3:-3}"
case "$tree" in
  after)     BIN="$ROOT/build/bin/llama-server" ;;
  invariant) BIN="$ROOT/build-invariant/bin/llama-server" ;;
  before)    BIN="$ROOT/build-before/bin/llama-server" ;;
  *) echo "unknown tree: $tree (after|invariant|before)" >&2; exit 2 ;;
esac
M="${MODELS_DIR:-$HOME/models}/michaelw9999--Qwen3.6-35B-A3B-NVFP4-MTP-GGUF/Qwen3.6-35B-A3B-NVFP4-MTP-HQ.gguf"
COMMON=(-m "$M" -ngl 99 --fit off -c "${CTX:-8192}" -fa on -ctk "${KV:-f16}" -ctv "${KV:-f16}" --port "${PORT:-8080}" --temp 0 --top-k 1 --seed 7)
case "$spec" in
  mtp)  exec "$BIN" "${COMMON[@]}" --spec-type draft-mtp --spec-draft-n-max "$nmax" --spec-draft-backend-sampling ;;
  none) exec "$BIN" "${COMMON[@]}" ;;
esac
