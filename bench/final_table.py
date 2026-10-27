#!/usr/bin/env python3
"""Render the README results tables from bench/results summary.csv files.

Usage: bench/final_table.py [results_dir]   (default: bench/results)
Columns are result directories (one benchmark run each); a row is a server config label (bench.sh SPECS).
Missing cells print as "-". Nothing here is measured, only rendered.
"""
import csv, os, sys
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results")
DAY = "2026-10-27"

def load(*dirs):
    out = {}
    for d in dirs:
        p = os.path.join(R, d, "summary.csv")
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            out.setdefault(r["label"], r)   # first directory listed wins
    return out

def q8(d):   # q8_0 KV runs get their own row labels
    return {k + ", q8_0 KV": v for k, v in d.items()}

COLS = [
    ("before (upstream)", {**load(f"{DAY}-final-before-f16", f"{DAY}-final-before-np8"), **q8(load(f"{DAY}-final-before-q8"))}),
    ("fp4-unlock (default build)", {**load(f"{DAY}-final-after-f16", f"{DAY}-final-after-np8"), **q8(load(f"{DAY}-final-after-q8"))}),
    ("kernel off (FP4_UNLOCK_KERNEL=0)", {**load(f"{DAY}-final-after-off-f16", f"{DAY}-final-after-off-np8"), **q8(load(f"{DAY}-final-after-off-q8"))}),
    ("invariant build + FP4_UNLOCK_INVARIANT=1", load(f"{DAY}-final-invariant-f16")),
]
ROWS = ["none", "mtp-n3", "mtp-n5", "mtp-n7", "dflash-n3", "dflash-n5", "dflash-n7",
        "mtp-n7, q8_0 KV", "dflash-n7, q8_0 KV", "np8-none", "np8-mtp-n3", "np8-mtp-n7"]
NAMES = {"none": "no drafter (batch 1)", "np8-none": "8 slots, no drafter (aggregate)",
         "np8-mtp-n3": "8 slots, mtp-n3 (aggregate)", "np8-mtp-n7": "8 slots, mtp-n7 (aggregate)"}

def cell(r, k, fmt):
    if r is None or r.get(k) in (None, "", "nan"):
        return "-"
    try:
        return fmt % float(r[k])
    except ValueError:
        return r[k]

def table(key, fmt, title, deltas=True):
    print(f"\n{title}\n")
    extra = " default vs before | default vs kernel off |" if deltas else ""
    print("| config | " + " | ".join(c for c, _ in COLS) + " |" + extra)
    print("|---|" + "---|" * (len(COLS) + (2 if deltas else 0)))
    for lab in ROWS:
        cells = [cell(d.get(lab), key, fmt) for _, d in COLS]
        if all(c == "-" for c in cells):
            continue
        line = f"| {NAMES.get(lab, lab)} | " + " | ".join(cells) + " |"
        if deltas:
            b, a, c = COLS[0][1].get(lab), COLS[1][1].get(lab), COLS[2][1].get(lab)
            d_b = d_c = "-"
            if a:
                if b: d_b = "%+.1f %%" % (100 * (float(a[key]) / float(b[key]) - 1))
                if c: d_c = "%+.1f %%" % (100 * (float(a[key]) / float(c[key]) - 1))
            line += f" {d_b} | {d_c} |"
        print(line)

if __name__ == "__main__":
    table("tok_s_median", "%.1f", "Decode tok/s (median of 3 timed runs, greedy, 8K ctx, 512-token answers; 8-slot rows: aggregate, 300 tokens)")
    table("acc_rate", "%.3f", "Draft acceptance rate", deltas=False)
    table("mean_len", "%.2f", "Mean accepted length per verify step", deltas=False)
    table("power_w", "%.0f", "Average board power (W) during the timed runs", deltas=False)
