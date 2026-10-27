#!/usr/bin/env bash
# PRD 1.3: the vLLM column. vllm/vllm-openai:v0.31.0 serving unsloth/Qwen3.8-27B-NVFP4 (NVFP4 safetensors; a different
# quantization recipe from cdiamond's iMatrix GGUF, footnoted in the table). JIT caches are mounted so the first request
# is not a FlashInfer build; one warm-up per config anyway.
# Usage: bench/vllm_compare.sh [configs]   configs default "none mtp dflash"; env DRY=1, SLOTS="8 16", OUT=<dir>
#   none  : plain decode, batch 1 then concurrency 8 / 16 (ignore_eos, 300 tokens)
#   mtp   : --speculative-config "$SPEC_MTP"   (default: Qwen3.8 MTP, 3 tokens)
#   dflash: --speculative-config "$SPEC_DFLASH" (default: z-lab DFlash2 drafter, 7 tokens)
# The exact speculative-config JSON vLLM 0.31 accepts for this model family is confirmed by the DRY run; override via env.
source "$(dirname "$0")/lib.sh"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.31.0}"; VMODEL="${VMODEL:-unsloth/Qwen3.8-27B-NVFP4}"; VPORT="${VPORT:-8000}"
configs="${1:-none mtp dflash}"; SLOTS="${SLOTS:-8 16}"; NPRED=512; RUNS=3
OUT="${OUT:-$ROOT/bench/results/$(date +%F)-vllm-$(echo "$IMAGE" | tr '/:' '--')}"; [[ "${DRY:-0}" == 1 ]] && { OUT="$OUT-dry"; NPRED=16; RUNS=1; SLOTS="8"; }
mkdir -p "$OUT"
SPEC_MTP="${SPEC_MTP:-{\"method\":\"mtp\",\"num_speculative_tokens\":3}}"
SPEC_DFLASH="${SPEC_DFLASH:-{\"method\":\"dflash\",\"model\":\"z-lab/Qwen3.8-27B-DFlash2\",\"num_speculative_tokens\":7}}"
PROMPTS=("$ROOT"/bench/prompts/code-{1,2,3}.txt "$ROOT"/bench/prompts/prose-{1,2,3}.txt); [[ "${DRY:-0}" == 1 ]] && PROMPTS=("${PROMPTS[0]}")
CID=""
vstart() { # $1 = label, rest = extra vllm args
  local label="$1"; shift
  CID=$(docker run -d --gpus all --ipc=host -p "127.0.0.1:$VPORT:8000" \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$HOME/.cache/vllm:/root/.cache/vllm" -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$IMAGE" --model "$VMODEL" --max-model-len "${VCTX:-8192}" --gpu-memory-utilization "${VUTIL:-0.90}" --max-num-seqs "${VSEQS:-16}" \
      --max-num-batched-tokens 2048 --seed 7 --disable-log-requests "$@")
  echo "docker: $CID  args: --model $VMODEL --max-model-len ${VCTX:-8192} --gpu-memory-utilization ${VUTIL:-0.90} --max-num-seqs ${VSEQS:-16} $*" > "$OUT/vllm-$label.log"
  local i; for i in $(seq 1 1200); do
    curl -sf "http://127.0.0.1:$VPORT/health" >/dev/null 2>&1 && return 0
    docker ps -q --no-trunc | grep -q "$CID" || { echo "vllm container exited; log:"; docker logs "$CID" 2>&1 | tail -40; docker logs "$CID" > "$OUT/vllm-$label-crash.log" 2>&1; docker rm "$CID" >/dev/null 2>&1; CID=""; return 1; }
    sleep 1
  done; echo "vllm not healthy after 20 min"; return 1
}
vstop() { [[ -n "$CID" ]] && { docker logs "$CID" >> "$OUT/vllm-${1:-last}.log" 2>&1; docker stop -t 20 "$CID" >/dev/null 2>&1 || true; docker rm "$CID" >/dev/null 2>&1 || true; CID=""; }; sleep 3; }
trap 'vstop' EXIT
{
  echo "date=$(date -Is)"; echo "image=$IMAGE"; echo "vllm_version=$(docker run --rm --entrypoint python3 "$IMAGE" -c "import vllm; print(vllm.__version__)" 2>/dev/null | tail -1)"; echo "model=$VMODEL"
  echo "driver=$(gpu_query driver_version) gpu=$(gpu_query name) clocks_locked=${CLOCKS_LOCKED:-no}"; echo "spec_mtp=$SPEC_MTP"; echo "spec_dflash=$SPEC_DFLASH"
} > "$OUT/meta.txt"; cat "$OUT/meta.txt"
run_client() { # label, extra client args
  local label="$1"; shift
  local tp; tp=$(telemetry_start "$OUT/telemetry-$label.csv")
  "$PY" "$HERE/vllm_client.py" --url "http://127.0.0.1:$VPORT" --model "$VMODEL" --label "$label" --prompts "${PROMPTS[@]}" --system "$ROOT/bench/prompts/system.txt" \
     --runs "$RUNS" --rows "$OUT/rows.csv" --summary "$OUT/summary.csv" --telemetry "$OUT/telemetry-$label.csv" "$@" || echo "client failed: $label"
  telemetry_stop "$tp"
}
for cfg in $configs; do
  gpu_cool
  case "$cfg" in
    none)   vstart none || continue;   run_client none --n-predict "$NPRED"
            for np in $SLOTS; do run_client "np$np-none" --concurrency "$np" --ignore-eos --n-predict $([[ "${DRY:-0}" == 1 ]] && echo 16 || echo 300); done; vstop none ;;
    mtp)    vstart mtp --speculative-config "$SPEC_MTP" || continue;  run_client mtp --n-predict "$NPRED"
            for np in $SLOTS; do run_client "np$np-mtp" --concurrency "$np" --ignore-eos --n-predict $([[ "${DRY:-0}" == 1 ]] && echo 16 || echo 300); done; vstop mtp ;;
    dflash) vstart dflash --speculative-config "$SPEC_DFLASH" || continue; run_client dflash --n-predict "$NPRED"; vstop dflash ;;
  esac
done
echo "== vLLM summary ($OUT/summary.csv)"; cat "$OUT/summary.csv" 2>/dev/null
