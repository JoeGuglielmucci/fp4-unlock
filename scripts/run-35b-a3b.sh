#!/usr/bin/env bash
# llama-server on the MoE model (demonstrates invariance; little kernel gain expected).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# tree: p3 (default; shipped build: invariant mode + fp4-smallm kernel), dev (invariant mode only), before (pinned
# upstream, unpatched), after (plain patched build/). Runtime switches (FP4_UNLOCK_KERNEL/T/MIN_M/INVARIANT) pass
# through the environment, e.g. FP4_UNLOCK_INVARIANT=1 scripts/run-27b.sh p3 mtp 7
tree="${1:-p3}"; spec="${2:-mtp}"; nmax="${3:-3}"
case "$tree" in
  p3)     BIN="$ROOT/build-p3/bin/llama-server" ;;
  dev)    BIN="$ROOT/build-dev/bin/llama-server" ;;
  before) BIN="$ROOT/build-before/bin/llama-server" ;;
  after)  BIN="$ROOT/build/bin/llama-server" ;;
  *) echo "unknown tree: $tree (p3|dev|before|after)" >&2; exit 2 ;;
esac
M="$HOME/models/michaelw9999--Qwen3.6-35B-A3B-NVFP4-MTP-GGUF/Qwen3.6-35B-A3B-NVFP4-MTP-HQ.gguf"
COMMON=(-m "$M" -ngl 99 --fit off -c "${CTX:-8192}" -fa on -ctk "${KV:-f16}" -ctv "${KV:-f16}" --port "${PORT:-8080}" --temp 0 --top-k 1 --seed 7)
case "$spec" in
  mtp)  exec "$BIN" "${COMMON[@]}" --spec-type draft-mtp --spec-draft-n-max "$nmax" --spec-draft-backend-sampling ;;
  none) exec "$BIN" "${COMMON[@]}" ;;
esac
