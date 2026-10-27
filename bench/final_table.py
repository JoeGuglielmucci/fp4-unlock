#!/usr/bin/env python3
"""Render the README headline tables from bench/results summary.csv files.

Usage: bench/final_table.py [results_dir]   (default: bench/results)
Columns are result directories; a row is a server config label (bench.sh SPECS). Missing cells print as "-".
Change the COLS/ROWS lists below when the final directories are renamed; nothing here is measured, only rendered.
"""
import csv, os, sys
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results")

def load(*dirs):
    out = {}
    for d in dirs:
        p = os.path.join(R, d, "summary.csv")
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            out.setdefault(r["label"], r)   # first directory listed wins
    return out

COLS = [
    ("before (upstream 63b4b303)", load("2026-10-26-before-27b-nvfp4", "2026-10-26-before-27b-nvfp4-dflash-n3")),
    ("kernel off (FP4_UNLOCK_KERNEL=0)", load("2026-10-27-p3-27b-nvfp4-final-off", "2026-10-27-p3-27b-nvfp4-final-np-off", "2026-10-27-p3-27b-nvfp4-final-q8kv-off")),
    ("fp4-unlock default", load("2026-10-27-p3-27b-nvfp4-final-default", "2026-10-27-p3-27b-nvfp4-final-np", "2026-10-27-p3-27b-nvfp4-final-q8kv")),
    ("invariant mode (FP4_UNLOCK_INVARIANT=1)", load("2026-10-27-p3-27b-nvfp4-final-inv")),
]
ROWS = ["none", "mtp-n3", "mtp-n5", "mtp-n7", "mtp-n8", "dflash-n3", "dflash-n5", "dflash-n7",
        "synth4", "synth6", "synth8", "np8-none", "np8-mtp-n3", "np8-mtp-n7", "np16-none", "np16-mtp-n3", "np16-mtp-n7"]

def cell(r, k, fmt):
    if r is None or r.get(k) in (None, "", "nan"):
        return "-"
    try:
        return fmt % float(r[k])
    except ValueError:
        return r[k]

def table(key, fmt, title):
    print(f"\n{title}\n")
    print("| config | " + " | ".join(c for c, _ in COLS) + " | default vs before | default vs kernel off |")
    print("|---|" + "---|" * (len(COLS) + 2))
    for lab in ROWS:
        cells = [cell(d.get(lab), key, fmt) for _, d in COLS]
        if all(c == "-" for c in cells):
            continue
        b, c, a = COLS[0][1].get(lab), COLS[1][1].get(lab), COLS[2][1].get(lab)
        d_b = d_c = "-"
        if key == "tok_s_median" and a:
            if b: d_b = "%+.1f %%" % (100 * (float(a[key]) / float(b[key]) - 1))
            if c: d_c = "%+.1f %%" % (100 * (float(a[key]) / float(c[key]) - 1))
        print(f"| {lab} | " + " | ".join(cells) + f" | {d_b} | {d_c} |")

table("tok_s_median", "%.1f", "Decode tok/s (median of 3 timed runs, greedy, 8K ctx, 512-token answers; np rows: aggregate over 8/16 slots, 300 tokens)")
table("acc_rate", "%.3f", "Draft acceptance rate")
table("mean_len", "%.2f", "Mean accepted length per verify step")
table("power_w", "%.0f", "Average board power (W) during the timed runs")
