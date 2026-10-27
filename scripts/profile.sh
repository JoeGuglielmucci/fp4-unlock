#!/usr/bin/env bash
# PRD 1.4: nsys capture of a fixed 200-token synthetic-drafter run (MTP head drafts n_max 8, random accept at mean
# len L), one-shot via llama-cli -st so the process exits and nsys finalises. Then a kernel-time summary CSV.
# Usage: scripts/profile.sh [before|after|dev|p3] [synth_len] [out_prefix]      needs nsys (cuda-nsight-systems-13-2)
set -euo pipefail
export PATH="/usr/local/cuda-13.2/bin:$PATH"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
tree="${1:-after}"; len="${2:-8}"; out="${3:-$ROOT/bench/results/$(date +%F)-$tree-profile-synth$len}"
case "$tree" in before) BIN="$ROOT/build-before/bin";; dev) BIN="$ROOT/build-dev/bin";; p3) BIN="$ROOT/build-p3/bin";; *) BIN="$ROOT/build/bin";; esac
M="$HOME/models/cdiamond--Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF/Qwen3.8-27B-iMatrix-NVFP4-MTP.gguf"
CMD=("$BIN/llama-cli" -m "$M" -ngl 99 --fit off -c 8192 -fa on --temp 0 --top-k 1 --seed 7 -n 200 -st
     --spec-type draft-mtp --spec-draft-n-max 8 --spec-draft-backend-sampling --spec-synth-len "$len"
     -sys "$(cat "$ROOT/bench/prompts/system.txt")" -p "$(cat "$ROOT/bench/prompts/code-1.txt")")
command -v nsys >/dev/null || { echo "nsys not installed (sudo apt install cuda-nsight-systems-13-2)"; exit 1; }
nsys profile -t cuda,nvtx --cuda-graph-trace=node -o "$out" --force-overwrite true "${CMD[@]}" > "$out.stdout" 2>&1 < /dev/null
nsys stats --report cuda_gpu_kern_sum --format csv -o "$out" "$out.nsys-rep" > /dev/null
echo "wrote $out.nsys-rep and ${out}_cuda_gpu_kern_sum.csv"
head -25 "${out}_cuda_gpu_kern_sum.csv" | cut -c1-160
