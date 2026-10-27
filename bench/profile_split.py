#!/usr/bin/env python3
"""Bucket an `nsys stats --report cuda_gpu_kern_sum --format csv` report into the PRD 1.4 categories and print
percentages of GPU kernel time. Usage: profile_split.py kern_sum.csv [--label L] [--out split.csv]
Buckets: mmvq (int8 dp4a GEMV, the kernel the project replaces), mmq (FP4/int8 tensor-core GEMM), quantize (src1 -> q8_1
for both), fa (flash attention), gdn (gated delta net / ssm), norm-rope-other (elementwise, norm, rope, copies, embeddings),
sampler/other (argmax, softmax, top-k on device), unknown."""
import argparse, csv, re, sys, collections
BUCKETS = [
    ("fp4-mma",         r"fp4_smallm"),  # this project's kernels (v1..v5): must precede mmvq/mmq, the names overlap
    ("mmvq",            r"mul_mat_vec_q|mul_mat_vec_f|fused_mul_mat_vec"),
    ("mmq",             r"mul_mat_q\b|mul_mat_q<|mmq|mul_mat_f<|gemm|cutlass|cublas|sm120|nvjet|xmma"),
    ("quantize",        r"quantize_q8_1|quantize_mmq|quantize"),
    ("fa",              r"flash_attn|fattn"),
    ("gdn",             r"gated_delta|delta_net|gdn|ssm_|ssm_conv|ssm_scan|rwkv|solve_tri|cumsum|tri_"),
    ("norm-rope-other", r"rms_norm|norm|rope|silu|glu|swiglu|add|mul\b|k_bin_bcast|bin_bcast|cpy|get_rows|scale|concat|set_rows|unary|op_|pad|clamp|sigmoid|l2"),
    ("sampler",         r"argmax|soft_max|softmax|top_k|topk|sampl|argsort|cumsum|dist"),
]
def bucket(name):
    for b, rx in BUCKETS:
        if re.search(rx, name, re.I): return b
    return "unknown"
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--label", default=""); ap.add_argument("--out", default=""); ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv)))
    if not rows: sys.exit("no rows in " + a.csv)
    tcol = next((c for c in rows[0] if c.lower().startswith("total time")), None)
    ncol = next((c for c in rows[0] if c.lower().startswith("name")), None)
    icol = next((c for c in rows[0] if c.lower().startswith("instances")), None)
    if not tcol or not ncol: sys.exit(f"unexpected columns: {list(rows[0])}")
    tot = collections.Counter(); inst = collections.Counter(); per = []
    for r in rows:
        t = float(r[tcol]); b = bucket(r[ncol]); tot[b] += t; inst[b] += int(float(r[icol])) if icol else 0; per.append((t, r[ncol], b))
    T = sum(tot.values())
    print(f"{a.label} GPU kernel time total {T/1e6:.1f} ms")
    print(f"{'bucket':18}{'ms':>10}{'%':>7}{'launches':>10}")
    for b, _ in BUCKETS + [("unknown", "")]:
        if tot[b]: print(f"{b:18}{tot[b]/1e6:10.1f}{100*tot[b]/T:7.1f}{inst[b]:10d}")
    print(f"\ntop {a.top} kernels:")
    for t, n, b in sorted(per, reverse=True)[:a.top]: print(f"  {100*t/T:5.1f}%  [{b}] {n[:110]}")
    if a.out:
        w = csv.writer(open(a.out, "a", newline="")); w.writerow(["label", "bucket", "ms", "pct", "launches"])
        for b, _ in BUCKETS + [("unknown", "")]: w.writerow([a.label, b, f"{tot[b]/1e6:.3f}", f"{100*tot[b]/T:.2f}", inst[b]])
if __name__ == "__main__": main()
