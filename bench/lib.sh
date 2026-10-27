#!/usr/bin/env bash
# Shared helpers for the fp4-unlock benchmark harness. Source this file; do not run it.
# Conventions: tree = before|after|invariant (build-before/, build/, build-invariant/), model keys below, results under bench/results/.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # bench/ directory (scripts)
ROOT="${FP4_ROOT:-$(cd "$HERE/.." && pwd)}"             # repo root (builds, prompts, results); FP4_ROOT overrides
export PATH="/usr/local/cuda-13.2/bin:$PATH"
PY="$ROOT/.venv/bin/python"; [[ -x "$PY" ]] || PY=python3

# ---- models (MODELS_DIR, default ~/models; see scripts/fetch-models.sh) -------------------------------------------------------------------------
declare -A MODEL=(
  [27b-nvfp4]="${MODELS_DIR:-$HOME/models}/cdiamond--Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF/Qwen3.8-27B-iMatrix-NVFP4-MTP.gguf"
  [27b-q8]="${MODELS_DIR:-$HOME/models}/unsloth--Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf"
  [35b-a3b-nvfp4]="${MODELS_DIR:-$HOME/models}/michaelw9999--Qwen3.6-35B-A3B-NVFP4-MTP-GGUF/Qwen3.6-35B-A3B-NVFP4-MTP-HQ.gguf"
  [gptoss-mxfp4]="${MODELS_DIR:-$HOME/models}/gpt-oss-20b-MXFP4.gguf"
)
DRAFT_DFLASH2="${MODELS_DIR:-$HOME/models}/z-lab--Qwen3.8-27B-DFlash2-GGUF/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
MTP_HEAD_UNSLOTH="${MODELS_DIR:-$HOME/models}/unsloth--Qwen3.8-27B-GGUF/MTP/mtp-Qwen3.8-27B-Q4_0.gguf"

model_path() { local p="${MODEL[$1]:-}"; [[ -n "$p" ]] || { echo "unknown model key: $1 (keys: ${!MODEL[*]})" >&2; exit 2; }; echo "$p"; }
tree_bin()   { case "$1" in before) echo "$ROOT/build-before/bin";; after) echo "$ROOT/build/bin";; invariant) echo "$ROOT/build-invariant/bin";; cpu) echo "$ROOT/build-cpu/bin";; *) echo "unknown tree: $1" >&2; exit 2;; esac; }
tree_src()   { case "$1" in before) echo "$ROOT/third_party/llama.cpp-before";; *) echo "$ROOT/third_party/llama.cpp";; esac; }
tree_commit(){ git -C "$(tree_src "$1")" rev-parse --short=9 HEAD; }

# ---- hygiene -------------------------------------------------------------------------
gpu_query() { nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits | head -1; }
gpu_cool() { # wait until the GPU is below $1 C (default 60), at most $2 s (default 300)
  local max="${1:-60}" limit="${2:-300}" t0=$SECONDS t
  while :; do t=$(gpu_query temperature.gpu); (( t < max )) && break
    (( SECONDS - t0 > limit )) && { echo "gpu_cool: still ${t} C after ${limit}s, continuing" >&2; break; }; sleep 5; done
  echo "gpu temp at start: ${t} C"
}
gpu_exclusive() { # refuse to measure while anything else holds the GPU (a stray server contaminated one run on 2026-10-26)
  # memory and the compute-app list are authoritative; utilization lags a few seconds after a process exits, so it only triggers a short wait
  local used apps util i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    used=$(gpu_query memory.used); util=$(gpu_query utilization.gpu)
    apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
    if (( used > 2000 )) || [[ -n "$apps" ]]; then
      echo "gpu_exclusive: GPU already in use (${used} MiB, ${util}% util). Other GPU processes:" >&2
      echo "$apps" >&2
      [[ "${FP4_UNLOCK_SHARED_GPU:-0}" == 1 ]] && return 0
      return 1
    fi
    (( util > 5 )) || return 0
    sleep 1
  done
  echo "gpu_exclusive: utilization still ${util}% with no process holding memory after 10 s; continuing" >&2
  return 0
}
telemetry_start() { # $1 = csv path; 1 Hz power/clock/temp/util/mem log; echoes pid
  nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,memory.used --format=csv -l 1 > "$1" 2>/dev/null &
  echo $!
}
telemetry_stop() { kill "$1" 2>/dev/null || true; wait "$1" 2>/dev/null || true; }

sha256_cached() { # cache keyed on path+size+mtime so 17-29 GB files hash once
  local f="$1" cache="$ROOT/bench/results/sha256.cache" key
  key="$f $(stat -c '%s %Y' "$f")"
  mkdir -p "$(dirname "$cache")"; touch "$cache"
  local hit; hit=$(grep -F -- "$key" "$cache" | tail -1 | awk '{print $1}')
  [[ -n "$hit" ]] && { echo "$hit"; return; }
  local sum; sum=$(sha256sum "$f" | awk '{print $1}'); echo "$sum $key" >> "$cache"; echo "$sum"
}

write_meta() { # $1 = out dir, $2 = tree, $3 = model key, rest = extra "k=v" lines
  local dir="$1" tree="$2" mk="$3"; shift 3; mkdir -p "$dir"
  {
    echo "date=$(date -Is)"; echo "tree=$tree"; echo "llama_cpp_commit=$(tree_commit "$tree")"
    echo "fork_branch=$(git -C "$ROOT/third_party/llama.cpp" rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
    echo "fork_describe=$(git -C "$ROOT/third_party/llama.cpp" describe --always --dirty 2>/dev/null || echo n/a)"
    echo "repo_commit=$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo uncommitted)"
    echo "driver=$(gpu_query driver_version)"; echo "gpu=$(gpu_query name)"
    echo "cuda=$(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' | head -1)"
    echo "clocks_sm_mhz=$(gpu_query clocks.sm) clocks_mem_mhz=$(gpu_query clocks.mem) max_sm_mhz=$(gpu_query clocks.max.sm) power_limit_w=$(gpu_query power.limit) temp_c=$(gpu_query temperature.gpu)"
    echo "clocks_locked=${CLOCKS_LOCKED:-no}   # to lock: sudo nvidia-smi -lgc 3090,3090 -lmc 14001, then set CLOCKS_LOCKED=yes"
    p="$(model_path "$mk")"; echo "model=$mk path=${p/#$HOME/\~}"; echo "model_sha256=$(sha256_cached "$(model_path "$mk")")"
    echo "env GGML_CUDA_MMQ_PREC=${GGML_CUDA_MMQ_PREC:-} LLAMA_SPEC_CHAIN=${LLAMA_SPEC_CHAIN:-} GGML_CUDA_GRAPH_OPT=${GGML_CUDA_GRAPH_OPT:-} FP4_UNLOCK_SMALLM=${FP4_UNLOCK_SMALLM:-}"
    for kv in "$@"; do echo "$kv"; done
  } > "$dir/meta.txt"
}

# ---- llama-server lifecycle --------------------------------------------------------------------
SERVER_PID=""; SERVER_LOG=""; PORT="${PORT:-8080}"
port_wait_free() { # a just-killed server can hold the port for a few seconds; the next bind would fail with "HTTP server error"
  local i; for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$PORT " || return 0; sleep 0.5; done
  echo "port $PORT still busy after 30s" >&2; ss -ltnp 2>/dev/null | grep ":$PORT " >&2; return 1
}
server_start() { # $1 = tree, $2 = log file, rest = llama-server args. Waits for /health.
  local tree="$1" log="$2"; shift 2
  SERVER_LOG="$log"
  port_wait_free || return 1
  echo "cmd: $(tree_bin "$tree")/llama-server $*" > "$log"
  "$(tree_bin "$tree")/llama-server" "$@" ${SERVER_EXTRA:-} --port "$PORT" --host 127.0.0.1 >> "$log" 2>&1 &
  SERVER_PID=$!
  local i; for i in $(seq 1 600); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then return 0; fi
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died, see $log" >&2; tail -20 "$log" >&2; return 1; }
    sleep 0.5
  done
  echo "server did not become healthy in 300s" >&2; return 1
}
server_stop() {
  [[ -n "$SERVER_PID" ]] || return 0
  kill "$SERVER_PID" 2>/dev/null || true
  local i; for i in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 0.5; done
  kill -9 "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""
  port_wait_free || true
  # paranoia: nothing of ours left on the GPU
  local used; used=$(gpu_query memory.used); (( used > 2000 )) && echo "warning: ${used} MiB still used on the GPU after server_stop" >&2 || true
}
trap 'server_stop' EXIT

# Common target flags: greedy, fit off, FA on, explicit context, explicit KV type.
target_args() { # $1 = model key, $2 = ctx, $3 = kv type
  echo -m "$(model_path "$1")" -ngl "${NGL:-99}" --fit off -c "$2" -fa on -ctk "$3" -ctv "$3" --temp 0 --top-k 1 --seed 7
}
# Drafter flags by label: none | mtp-nN | dflash-nN | synthL (MTP head drafts, random accept at mean len L) | extmtp-nN (external unsloth head)
spec_args() {
  local label="$1"
  case "$label" in
    none)        echo "" ;;
    mtp-n*)      echo --spec-type draft-mtp --spec-draft-n-max "${label#mtp-n}" --spec-draft-backend-sampling ;;
    extmtp-n*)   echo -md "$MTP_HEAD_UNSLOTH" --spec-type draft-mtp --spec-draft-n-max "${label#extmtp-n}" --spec-draft-backend-sampling ;;
    dflash-n*)   echo -md "$DRAFT_DFLASH2" --spec-type draft-dflash --spec-draft-n-max "${label#dflash-n}" ;;
    synth*)      echo --spec-type draft-mtp --spec-draft-n-max 8 --spec-draft-backend-sampling --spec-synth-len "${label#synth}" ;;
    *) echo "unknown spec label: $label" >&2; return 2 ;;
  esac
}
