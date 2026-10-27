#!/usr/bin/env bash
# PRD 1.7: reproduce issue #27329 (Qwen3.8-27B decode hang, CPU spin, no GPU work) on this card and the pinned commit.
# The issue's config: llama-server, ctx 262144, K/V cache iq4_nl, batch/ubatch 512, 12 threads, -fa auto, no spec.
# Usage: bench/stall_repro.sh [before|after] [model-key=27b-nvfp4] [reps=3]   env: KVT=iq4_nl CTX=262144 NPRED=2000
source "$(dirname "$0")/lib.sh"
tree="${1:-before}"; mk="${2:-27b-nvfp4}"; reps="${3:-3}"; KVT="${KVT:-iq4_nl}"; CTX="${CTX:-262144}"; NPRED="${NPRED:-2000}"
OUT="${OUT:-$ROOT/bench/results/$(date +%F)-$tree-$mk-stall-$KVT}"; mkdir -p "$OUT"
M="$(model_path "$mk")"; BIN="$(tree_bin "$tree")"
write_meta "$OUT" "$tree" "$mk" "issue=27329" "kv=$KVT" "ctx=$CTX" "n_predict=$NPRED" "reps=$reps"
tp=$(telemetry_start "$OUT/telemetry.csv")
server_start "$tree" "$OUT/server.log" -m "$M" -ngl 99 --fit off -c "$CTX" -ctk "$KVT" -ctv "$KVT" -b 512 -ub 512 -t 12 -tb 12 -fa auto --spec-type none --temp 0 --top-k 1 --seed 7 \
  || { echo "server failed to start (see $OUT/server.log)"; tail -20 "$OUT/server.log"; telemetry_stop "$tp"; exit 1; }
prompt="$(cat "$ROOT/bench/prompts/code-1.txt")"
for i in $(seq 1 "$reps"); do
  echo "== rep $i: $NPRED tokens, stall watchdog 180 s without progress"
  "$PY" - "$prompt" "$NPRED" "$OUT/rep-$i.json" <<'PY' &
import sys, json, time, urllib.request
prompt, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
body = {"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1, "seed": 7, "cache_prompt": False, "ignore_eos": True, "stream": True}
req = urllib.request.Request("http://127.0.0.1:8080/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
t0 = time.time(); last = t0; k = 0; timings = None
with urllib.request.urlopen(req, timeout=600) as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:"): continue
        d = json.loads(line[5:]); k += 1; last = time.time()
        if d.get("stop"): timings = d.get("timings"); break
json.dump({"tokens": k, "wall_s": time.time() - t0, "timings": timings}, open(out, "w"))
print(f"   {k} tokens in {time.time()-t0:.1f} s" + (f", {timings['predicted_per_second']:.1f} tok/s" if timings else ""))
PY
  cpid=$!; stalled=0; lastsize=0; lastchange=$(date +%s)
  while kill -0 "$cpid" 2>/dev/null; do
    sleep 5; sz=$(stat -c %s "$OUT/server.log"); now=$(date +%s)
    if (( sz != lastsize )); then lastsize=$sz; lastchange=$now; fi
    if (( now - lastchange > 180 )); then stalled=1; break; fi
  done
  if (( stalled )); then
    echo "   STALL: no server log progress for 180 s. Capturing gdb backtrace of all threads and nvidia-smi."
    nvidia-smi --query-gpu=utilization.gpu,clocks.sm,power.draw --format=csv > "$OUT/stall-$i-gpu.txt"
    top -b -n 1 -H -p "$SERVER_PID" | head -20 > "$OUT/stall-$i-threads.txt"
    gdb -p "$SERVER_PID" -batch -ex "thread apply all bt 12" > "$OUT/stall-$i-gdb.txt" 2>&1 || echo "   gdb failed (ptrace scope? owner: sudo sysctl kernel.yama.ptrace_scope=0)"
    kill "$cpid" 2>/dev/null; break
  fi
  wait "$cpid"
done
telemetry_stop "$tp"; server_stop
echo "result: stalled=$stalled; files in $OUT"
