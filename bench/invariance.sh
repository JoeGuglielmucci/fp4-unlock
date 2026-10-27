#!/usr/bin/env bash
# Row 8: speculative decoding must not change greedy output. For each KV type: generate the spec-off reference
# twice (run-to-run control), then each drafter x n_max once, and diff tokens against the reference.
# Usage: bench/invariance.sh <before|after|invariant> <model-key>
# Env: KVS="f16 q8_0"  DRAFTERS="mtp-n3 mtp-n7 dflash-n3 dflash-n7"  CTX=8192  NPRED=512  OUT=<dir>  DRY=1
source "$(dirname "$0")/lib.sh"
tree="${1:?tree}"; mk="${2:?model key}"
KVS="${KVS:-f16 q8_0}"; DRAFTERS="${DRAFTERS:-mtp-n3 mtp-n7 dflash-n3 dflash-n7}"; CTX="${CTX:-8192}"; NPRED="${NPRED:-512}"
OUT="${OUT:-$ROOT/bench/results/$(date +%F)-$tree-$mk-invariance}"; mkdir -p "$OUT"
PROMPTS=("$ROOT"/bench/prompts/code-{1,2,3}.txt "$ROOT"/bench/prompts/prose-{1,2,3}.txt)
if [[ "${DRY:-0}" == 1 ]]; then NPRED=32; PROMPTS=("${PROMPTS[0]}"); KVS="${KVS_DRY:-f16}"; DRAFTERS="${DRAFTERS_DRY:-mtp-n3}"; fi
gpu_exclusive || exit 3
write_meta "$OUT" "$tree" "$mk" "kvs=$KVS" "drafters=$DRAFTERS" "ctx=$CTX" "n_predict=$NPRED" "dflash2_sha256=$(sha256_cached "$DRAFT_DFLASH2")"
REPORT="$OUT/report.csv"; SUMMARY="$OUT/summary.txt"; : > "$SUMMARY"
gen() { "$PY" "$HERE/invariance.py" gen --out "$1" --prompts "${PROMPTS[@]}" --system "$ROOT/bench/prompts/system.txt" --n-predict "$NPRED"; }
cmp() { "$PY" "$HERE/invariance.py" compare --ref "$1" --new "$2" --report "$REPORT" | tee -a "$SUMMARY"; }
for kv in $KVS; do
  echo "== KV $kv: reference (spec off), two runs" | tee -a "$SUMMARY"
  # shellcheck disable=SC2046
  server_start "$tree" "$OUT/server-none-$kv.log" $(target_args "$mk" "$CTX" "$kv")
  gen "$OUT/none-$kv-A"; gen "$OUT/none-$kv-B"; server_stop
  cmp "$OUT/none-$kv-A" "$OUT/none-$kv-B" || true     # run-to-run control; must be identical or nothing below means anything
  for d in $DRAFTERS; do
    [[ "$mk" != 27b-* && "$d" == dflash-* ]] && continue
    echo "== KV $kv: $d vs spec off" | tee -a "$SUMMARY"
    # shellcheck disable=SC2046
    server_start "$tree" "$OUT/server-$d-$kv.log" $(target_args "$mk" "$CTX" "$kv") $(spec_args "$d") || { echo "FAILED to start $d" | tee -a "$SUMMARY"; continue; }
    gen "$OUT/$d-$kv"; server_stop
    cmp "$OUT/none-$kv-A" "$OUT/$d-$kv" || true
  done
done
echo; echo "=== invariance summary ($tree, $mk) ==="; grep -E "^== |divergence rate" "$SUMMARY"
