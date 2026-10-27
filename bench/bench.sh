#!/usr/bin/env bash
# The single table (PRD Section 7): rows 1-7 and 10 for one tree x one model.
# Usage: bench/bench.sh <before|after> <model-key> [rows]      rows default "1,2,3,4,5,6,7"
# Env:   DRY=1 (n_predict 16, 1 run, first prompt only, short scaling list)   KV=f16|q8_0 (default f16)
#        CTX=8192   SPECS="none mtp-n3 mtp-n7 mtp-n8 dflash-n3 dflash-n7 synth4 synth8"   SLOTS="8 16"
#        OUT=<dir> to override the results directory.
# Output: bench/results/<date>-<tree>-<model>/{meta.txt,rows.csv,summary.csv,llama-bench.jsonl,scaling.csv,
#         telemetry-<label>.csv,server-<label>.log,text/}. Render with bench/render.py <dir>.
source "$(dirname "$0")/lib.sh"
tree="${1:?tree}"; mk="${2:?model key}"; rows="${3:-1,2,3,4,5,6,7}"
KV="${KV:-f16}"; CTX="${CTX:-8192}"; DRY="${DRY:-0}"
OUT="${OUT:-$ROOT/bench/results/$(date +%F)-$tree-$mk${DRY:+}}"; [[ "$DRY" == 1 ]] && OUT="${OUT}-dry"
mkdir -p "$OUT/text"
M="$(model_path "$mk")"; BIN="$(tree_bin "$tree")"
PROMPTS=("$ROOT"/bench/prompts/code-{1,2,3}.txt "$ROOT"/bench/prompts/prose-{1,2,3}.txt)
SPECS7="${SPECS7:-none mtp-n3}"; NPRED=512; RUNS=3; SPECS="${SPECS:-none mtp-n3 mtp-n7 mtp-n8 dflash-n3 dflash-n7 synth4 synth8}"; SLOTS="${SLOTS:-8 16}"
if [[ "$DRY" == 1 ]]; then NPRED=16; RUNS=1; PROMPTS=("${PROMPTS[0]}"); SPECS="${SPECS_DRY:-none mtp-n3}"; SLOTS="${SLOTS_DRY:-8}"; fi
has_row() { [[ ",$rows," == *",$1,"* ]]; }

write_meta "$OUT" "$tree" "$mk" "kv=$KV" "ctx=$CTX" "n_predict=$NPRED" "runs=$RUNS" "dry=$DRY" "rows=$rows" \
  "dflash2_sha256=$(sha256_cached "$DRAFT_DFLASH2")" "cmdline=$0 $*"
echo "results -> $OUT"; cat "$OUT/meta.txt"
gpu_exclusive || exit 3

# rows 1-2: llama-bench, batch-1 decode tg128 + prefill pp512 / pp8192 (llama-bench does its own warm-up; -r 3 keeps all samples in jsonl)
if has_row 1 || has_row 2; then
  gpu_cool
  pp="512,8192"; [[ "$DRY" == 1 ]] && pp="512"
  tp=$(telemetry_start "$OUT/telemetry-llama-bench.csv")
  "$BIN/llama-bench" -m "$M" -ngl 99 -fa 1 -ctk "$KV" -ctv "$KV" -p "$pp" -n 128 -r 3 -o jsonl > "$OUT/llama-bench.jsonl" 2> "$OUT/llama-bench.err" || { tail -5 "$OUT/llama-bench.err"; }
  telemetry_stop "$tp"
  "$PY" - "$OUT/llama-bench.jsonl" <<'PY'
import json,sys
for l in open(sys.argv[1]):
    if not l.strip(): continue
    d=json.loads(l); t=(f'pp{d["n_prompt"]}' if d.get("n_gen",0)==0 else f'tg{d["n_gen"]}')+(f'@d{d["n_depth"]}' if d.get("n_depth",0) else ""); print(f'  llama-bench {t:>8}: {d["avg_ts"]:.2f} +- {d["stddev_ts"]:.2f} t/s  (samples {len(d.get("samples_ts",[]))})')
PY
fi

# row 3: scaling curve (llama-batched-bench + test-backend-ops perf)
if has_row 3; then
  gpu_cool; DRY="$DRY" OUT="$OUT" KV="$KV" "$HERE/scaling.sh" "$tree" "$mk" || echo "row 3 (scaling) failed; continuing with the other rows"
fi

# rows 4-6 (+10): spec decode via llama-server, one server per drafter config
if has_row 4 || has_row 5 || has_row 6; then
  for label in $SPECS; do
    case "$label" in none|mtp-*|extmtp-*) has_row 4 || continue;; dflash-*) has_row 5 || continue;; synth*) has_row 6 || continue;; esac
    # drafters only exist for the 27B family; MoE/gptoss get none + mtp only
    if [[ "$mk" != 27b-* && "$label" == dflash-* ]]; then echo "skip $label for $mk (no DFlash2 drafter)"; continue; fi
    gpu_cool
    # shellcheck disable=SC2046
    server_start "$tree" "$OUT/server-$label.log" $(target_args "$mk" "$CTX" "$KV") $(spec_args "$label") || { echo "FAILED to start $label"; continue; }
    tp=$(telemetry_start "$OUT/telemetry-$label.csv")
    "$PY" "$HERE/server_rows.py" --label "$label" --prompts "${PROMPTS[@]}" --system "$ROOT/bench/prompts/system.txt" \
      --n-predict "$NPRED" --runs "$RUNS" --rows "$OUT/rows.csv" --summary "$OUT/summary.csv" \
      --server-log "$OUT/server-$label.log" --telemetry "$OUT/telemetry-$label.csv" --save-text-dir "$OUT/text" || echo "client failed for $label"
    telemetry_stop "$tp"; server_stop
  done
fi

# row 7 (+10): 8- and 16-slot aggregate, 300-token greedy answers, ignore_eos (imp's method), spec off and MTP n3
if has_row 7; then
  for np in $SLOTS; do
    for spec in ${SPECS7:-none mtp-n3}; do
      label="np$np-$spec"; gpu_cool
      ctx=$(( 4096 * np ))   # 4096 per slot; --fit off, so this must fit or the server refuses to start (recorded)
      # shellcheck disable=SC2046
      server_start "$tree" "$OUT/server-$label.log" $(target_args "$mk" "$ctx" "$KV") -np "$np" $(spec_args "$spec") || { echo "FAILED to start $label"; continue; }
      tp=$(telemetry_start "$OUT/telemetry-$label.csv")
      "$PY" "$HERE/server_rows.py" --label "$label" --prompts "${PROMPTS[@]}" --system "$ROOT/bench/prompts/system.txt" \
        --n-predict $([[ "$DRY" == 1 ]] && echo 16 || echo 300) --ignore-eos --concurrency "$np" --runs "$RUNS" \
        --rows "$OUT/rows.csv" --summary "$OUT/summary.csv" --server-log "$OUT/server-$label.log" --telemetry "$OUT/telemetry-$label.csv" || echo "client failed for $label"
      telemetry_stop "$tp"; server_stop
    done
  done
fi

"$PY" "$HERE/render.py" "$OUT" && echo "rendered $OUT/rows.md"
