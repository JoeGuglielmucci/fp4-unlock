#!/usr/bin/env bash
# Row 3: decode step time vs batch size M (the verify-step shape), two ways.
#  (a) end-to-end: llama-batched-bench, shared 512-token prompt, 128 decode steps at M parallel sequences.
#      step_ms(M) = T_TG * 1000 / 128; the ratio step(M)/step(1) is the K1 number.
#  (b) kernel-only: test-backend-ops perf MUL_MAT for nvfp4 (stock shape 4096x14336 n=1..512, and the 27B
#      shapes 17408x5120 / 5120x17408 at M = 1..64 when the binary was built with the perf-shapes patch).
# Usage: bench/scaling.sh <tree> <model-key>     env: OUT (results dir), KV, DRY=1, NPL="1,2,3,4,6,8,12,16,32,64"
source "$(dirname "$0")/lib.sh"
tree="${1:?tree}"; mk="${2:?model key}"; KV="${KV:-f16}"
OUT="${OUT:-$ROOT/bench/results/$(date +%F)-$tree-$mk-scaling}"; mkdir -p "$OUT"
M="$(model_path "$mk")"; BIN="$(tree_bin "$tree")"
NPL="${NPL:-1,2,3,4,6,8,12,16,32,64}"; [[ "${DRY:-0}" == 1 ]] && NPL="${NPL_DRY:-1,2,8}"
[[ -f "$OUT/meta.txt" ]] || write_meta "$OUT" "$tree" "$mk" "kv=$KV" "npl=$NPL"

# (a) batched-bench. Prompts are NOT shared (-pps): on the M-RoPE models (qwen35) the shared-prompt path fails the
# position check ("X < Y" in init: last stored position 512, batch starts at 512). -c holds max(M)*(pp+tg).
maxnpl=$(echo "$NPL" | tr ',' '\n' | sort -n | tail -1); CTXB=$(( maxnpl * (512 + 128) )); (( CTXB < 16384 )) && CTXB=16384
tp=$(telemetry_start "$OUT/telemetry-scaling.csv")
"$BIN/llama-batched-bench" -m "$M" -ngl 99 --fit off -fa on -ctk "$KV" -ctv "$KV" -c "$CTXB" -b 2048 -ub 512 \
  -npp 512 -ntg 128 -npl "$NPL" > "$OUT/batched-bench.log" 2>&1 || { echo "batched-bench failed:"; tail -20 "$OUT/batched-bench.log"; }
telemetry_stop "$tp"
"$PY" - "$OUT/batched-bench.log" "$OUT/scaling.csv" <<'PY'
import sys,re,csv
rows=[]
for l in open(sys.argv[1]):
    m=re.match(r'\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|',l)
    if m: rows.append([int(m.group(i)) for i in (1,2,3,4)]+[float(m.group(i)) for i in range(5,11)])
if not rows: sys.exit("no batched-bench rows parsed")
base=next((r[6]/r[1] for r in rows if r[2]==1),None)
w=csv.writer(open(sys.argv[2],"w")); w.writerow(["pp","tg","M","n_kv","t_pp_s","pp_tps","t_tg_s","tg_tps_aggregate","t_total_s","total_tps","step_ms","step_ratio_vs_M1"])
print("  M   step_ms  ratio  aggregate t/s")
for r in rows:
    step=r[6]/r[1]*1000; ratio=step/(base*1000) if base else float('nan')
    w.writerow(r+[f"{step:.3f}",f"{ratio:.3f}"]); print(f"  {r[2]:>3} {step:8.3f} {ratio:6.3f} {r[7]:10.2f}")
PY

# (b) kernel-only perf. Stock shapes always; 27B shapes if the binary honours FP4_UNLOCK_PERF_SHAPES.
"$BIN/test-backend-ops" perf -b CUDA0 -o MUL_MAT -p "type_a=nvfp4" > "$OUT/tbo-perf-stock.log" 2>&1 || true
FP4_UNLOCK_PERF_SHAPES=1 "$BIN/test-backend-ops" perf -b CUDA0 -o MUL_MAT -p "type_a=nvfp4.*(m=17408|m=5120)" > "$OUT/tbo-perf-27b.log" 2>&1 || true
"$PY" - "$OUT/tbo-perf-stock.log" "$OUT/tbo-perf-27b.log" "$OUT/tbo-perf.csv" <<'PYEOF'
import sys, re, csv, os
w = csv.writer(open(sys.argv[3], "w")); w.writerow(["m", "n", "k", "us", "tflops", "source"])
for f in sys.argv[1:3]:
    if not os.path.exists(f): continue
    for l in open(f, errors="replace"):
        m = re.search(r"m=(\d+),n=(\d+),k=(\d+).*?([\d.]+) us/run.*?([\d.]+) TFLOPS", l)
        if m:
            w.writerow([*m.groups(), f.split("tbo-perf-")[1].split(".")[0]])
            print(f"  m={m.group(1)} n={m.group(2)} k={m.group(3)}: {m.group(4)} us  {m.group(5)} TFLOPS")
PYEOF
