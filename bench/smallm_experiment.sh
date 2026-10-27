#!/usr/bin/env bash
# PRD 1.5: the one-line experiment. On the fork build, FP4_UNLOCK_SMALLM=1 routes NVFP4/MXFP4 mul_mat with 2 <= M <= 8
# to MMQ (FP4 tensor-core MMA, J=8 tile) instead of MMVQ (int8 dp4a). Kernel-only (test-backend-ops perf on the 27B
# shapes) under MMQ precision auto/q8/q4, then the speculative rows 4-6 end to end. Usage: bench/smallm_experiment.sh [rows]
source "$(dirname "$0")/lib.sh"
rows="${1:-3,4,5,6}"; OUT="${OUT:-$ROOT/bench/results/$(date +%F)-after-27b-nvfp4-smallm}"; mkdir -p "$OUT"
BIN="$(tree_bin after)"
echo "== kernel-only: 27B shapes, n=1..8, stock routing vs FP4_UNLOCK_SMALLM=1 under GGML_CUDA_MMQ_PREC auto/q8/q4"
for cfg in "stock:0:auto" "smallm:1:auto" "smallm-q8:1:q8" "smallm-q4:1:q4"; do
  IFS=: read -r label sm prec <<< "$cfg"
  FP4_UNLOCK_PERF_SHAPES=1 FP4_UNLOCK_SMALLM=$sm GGML_CUDA_MMQ_PREC=$prec "$BIN/test-backend-ops" perf -b CUDA0 -o MUL_MAT -p "type_a=nvfp4.*(m=17408|m=5120)" > "$OUT/tbo-$label.log" 2>&1 || true
  echo "-- $label"; grep -oE "m=[0-9]+,n=[0-9]+,k=[0-9]+.*?[0-9.]+ us/run" "$OUT/tbo-$label.log" | sed -E 's/\): .*? - /: /' | awk -F'[=,: ]' '$4<=8 || $4==512' | head -24
done
"$PY" - "$OUT" <<'PY'
import re,sys,os,csv
out=sys.argv[1]; rx=re.compile(r"m=(\d+),n=(\d+),k=(\d+).*?([\d.]+) us/run"); data={}
for label in ["stock","smallm","smallm-q8","smallm-q4"]:
    p=os.path.join(out,f"tbo-{label}.log")
    if os.path.exists(p):
        for l in open(p):
            m=rx.search(l)
            if m: data[(label,int(m.group(1)),int(m.group(2)),int(m.group(3)))]=float(m.group(4))
w=csv.writer(open(os.path.join(out,"kernel.csv"),"w")); w.writerow(["m","n","k","stock_us","smallm_us","smallm_q8_us","smallm_q4_us","speedup_auto"])
for (label,m,n,k),us in sorted(data.items()):
    if label!="stock": continue
    s=data.get(("smallm",m,n,k)); w.writerow([m,n,k,us,s,data.get(("smallm-q8",m,n,k)),data.get(("smallm-q4",m,n,k)),f"{us/s:.2f}" if s else ""])
print("wrote", os.path.join(out,"kernel.csv"))
PY
if [[ "$rows" != "kernel" ]]; then
  echo "== end to end: rows $rows on the fork build, FP4_UNLOCK_SMALLM=1 GGML_CUDA_MMQ_PREC=q8 (lossless column, W4A8)"
  FP4_UNLOCK_SMALLM=1 GGML_CUDA_MMQ_PREC=q8 OUT="$OUT/rows-q8" "$HERE/bench.sh" after 27b-nvfp4 "$rows"
  echo "== end to end with GGML_CUDA_MMQ_PREC=q4 (fast column, native W4A4 FP4 MMA, +43% KL in Phase 1.3)"
  FP4_UNLOCK_SMALLM=1 GGML_CUDA_MMQ_PREC=q4 OUT="$OUT/rows-q4" "$HERE/bench.sh" after 27b-nvfp4 "$rows"
fi
