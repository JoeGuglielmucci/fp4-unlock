#!/usr/bin/env python3
"""Row 9: KL divergence of the NVFP4 model vs the Q8_0 reference, per CUDA matmul path.

The reference logits are produced once (llama-perplexity --save-all-logits on the Q8_0 GGUF). Then the NVFP4 GGUF is
scored with --kl-divergence-base under several configurations that force a given matmul path for the NVFP4 tensors:
  mmvq      : -b 8 -ub 8               every ubatch has 8 tokens -> MMVQ (int8 dp4a), the path a verify step of n_max<=7 uses
  mmq-auto  : -b 512 -ub 512           MMQ with GGML_CUDA_MMQ_PREC unset (auto = W4A4 FP4 MMA on sm_120 for nvfp4)
  mmq-q8    : -b 512, GGML_CUDA_MMQ_PREC=q8   MMQ W4A8
  mmq-q4    : -b 512, GGML_CUDA_MMQ_PREC=q4   MMQ W4A4
  kernel/kernel-off : -b 8 -ub 8, FP4_UNLOCK_KERNEL=1 / =0 (patched tree: small-M kernel vs MMVQ control)
  smallm    : -b 8 -ub 8, FP4_UNLOCK_SMALLM=1 GGML_CUDA_MMQ_PREC=q8 (after tree only); smallm-q4 same with q4
Note: llama-perplexity does not run speculative decoding; the verify step uses exactly these matmul paths, so this is the
per-path quality number ("spec on" is implied by the batch size, not a flag).
Usage: bench/kld.py <before|after|invariant> [--paths mmvq,mmq-auto,mmq-q8,mmq-q4] [--ctx 512] [--chunks N] [--out DIR]
"""
import argparse, os, re, subprocess, sys, csv, datetime
ROOT = os.environ.get("FP4_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {"27b-nvfp4": os.path.join(os.environ.get("MODELS_DIR", os.path.expanduser("~/models")), "cdiamond--Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF/Qwen3.8-27B-iMatrix-NVFP4-MTP.gguf"),
          "27b-q8": os.path.join(os.environ.get("MODELS_DIR", os.path.expanduser("~/models")), "unsloth--Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf")}
PATHS = {"mmvq": (["-b", "8", "-ub", "8"], {}), "mmq-auto": (["-b", "512", "-ub", "512"], {}),
         "mmq-q8": (["-b", "512", "-ub", "512"], {"GGML_CUDA_MMQ_PREC": "q8"}), "mmq-q4": (["-b", "512", "-ub", "512"], {"GGML_CUDA_MMQ_PREC": "q4"}),
         "smallm": (["-b", "8", "-ub", "8"], {"FP4_UNLOCK_SMALLM": "1", "GGML_CUDA_MMQ_PREC": "q8"}),
         "smallm-q4": (["-b", "8", "-ub", "8"], {"FP4_UNLOCK_SMALLM": "1", "GGML_CUDA_MMQ_PREC": "q4"}),
         "kernel": (["-b", "8", "-ub", "8"], {"FP4_UNLOCK_KERNEL": "1"}),
         "kernel-off": (["-b", "8", "-ub", "8"], {"FP4_UNLOCK_KERNEL": "0"})}
KEYS = ["Mean    KLD", "Median  KLD", "99.9%   KLD", "99.0%   KLD", "Maximum KLD", "Mean ln(PPL(Q)/PPL(base))", "Same top p"]

def run(cmd, env, log):
    e = dict(os.environ, PATH="/usr/local/cuda-13.2/bin:" + os.environ.get("PATH", ""), **env)
    with open(log, "w") as f:
        f.write("cmd: " + " ".join(cmd) + "\nenv: " + str(env) + "\n"); f.flush()
        p = subprocess.run(cmd, env=e, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode, open(log, errors="replace").read()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tree"); ap.add_argument("--model", default="27b-nvfp4"); ap.add_argument("--ref", default="27b-q8")
    ap.add_argument("--paths", default="mmvq,mmq-auto,mmq-q8,mmq-q4"); ap.add_argument("--ctx", type=int, default=512); ap.add_argument("--chunks", type=int, default=0)
    ap.add_argument("--text", default=os.path.join(ROOT, "bench/prompts/kl-set-5k.txt")); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    binp = os.path.join(ROOT, {"before": "build-before", "cpu": "build-cpu", "invariant": "build-invariant"}.get(a.tree, "build"), "bin", "llama-perplexity")
    out = a.out or os.path.join(ROOT, "bench/results", f"{datetime.date.today()}-{a.tree}-{a.model}-kld"); os.makedirs(out, exist_ok=True)
    ref_logits = os.path.join(ROOT, "bench/results", f"ref-logits-{a.ref}-c{a.ctx}-chunks{a.chunks or 'all'}.bin")
    common = ["-ngl", "99", "--fit", "off", "-fa", "on", "-c", str(a.ctx), "-f", a.text] + (["--chunks", str(a.chunks)] if a.chunks else [])
    if not os.path.exists(ref_logits):
        print(f"producing reference logits -> {ref_logits}", flush=True)
        rc, log = run([binp, "-m", MODELS[a.ref], *common, "-b", "512", "-ub", "512", "--save-all-logits", ref_logits], {}, os.path.join(out, "ref.log"))
        if rc != 0 or not os.path.exists(ref_logits): sys.exit("reference run failed:\n" + log[-2000:])
    w = csv.writer(open(os.path.join(out, "kld.csv"), "a", newline="")); w.writerow(["tree", "model", "path", "batch", "env"] + KEYS)
    for path in a.paths.split(","):
        args, env = PATHS[path]
        print(f"[{path}] {' '.join(args)} {env}", flush=True)
        rc, log = run([binp, "-m", MODELS[a.model], *common, *args, "--kl-divergence", "--kl-divergence-base", ref_logits], env, os.path.join(out, f"{path}.log"))
        vals = []
        for k in KEYS:
            m = re.search(re.escape(k) + r"\s*:\s*([-\d.]+)", log); vals.append(m.group(1) if m else "")
        print("   " + "  ".join(f"{k.split()[0]}={v}" for k, v in zip(KEYS, vals)) + ("" if rc == 0 else f"   (exit {rc})"), flush=True)
        w.writerow([a.tree, a.model, path, " ".join(args), " ".join(f"{k}={v}" for k, v in env.items())] + vals)
    print(f"wrote {out}/kld.csv")

if __name__ == "__main__": main()
