#!/usr/bin/env python3
"""Render one results directory (from bench.sh) into rows.md, the results table for that tree x model."""
import csv, json, os, sys, statistics
d = sys.argv[1]; L = []
meta = dict(l.split("=", 1) for l in open(os.path.join(d, "meta.txt")) if "=" in l and not l.startswith("env")) if os.path.exists(os.path.join(d, "meta.txt")) else {}
L.append(f"# {os.path.basename(d)}\n")
L.append("| key | value |\n|---|---|")
for k in ["date", "tree", "llama_cpp_commit", "fork_describe", "driver", "cuda", "gpu", "clocks_sm_mhz", "clocks_locked", "model", "model_sha256", "kv", "ctx", "n_predict", "runs"]:
    if k in meta: L.append(f"| {k} | {meta[k].strip()} |")
lb = os.path.join(d, "llama-bench.jsonl")
if os.path.exists(lb):
    L.append("\n## Rows 1-2: llama-bench (batch 1)\n\n| test | t/s | stddev | samples |\n|---|---|---|---|")
    for l in open(lb):
        if l.strip():
            j = json.loads(l); t = (f'pp{j["n_prompt"]}' if j.get("n_gen", 0) == 0 else f'tg{j["n_gen"]}') + (f'@d{j["n_depth"]}' if j.get("n_depth", 0) else ""); L.append(f'| {t} | {j["avg_ts"]:.2f} | {j["stddev_ts"]:.2f} | {" ".join(f"{s:.1f}" for s in j.get("samples_ts", []))} |')
sc = os.path.join(d, "scaling.csv")
if os.path.exists(sc):
    L.append("\n## Row 3: decode step time vs batch M (llama-batched-bench, pp512 per sequence, tg128; M separate sequences, not one verify batch)\n\n| M | step ms | step(M)/step(1) | aggregate t/s |\n|---|---|---|---|")
    for r in csv.DictReader(open(sc)): L.append(f'| {r["M"]} | {r["step_ms"]} | {r["step_ratio_vs_M1"]} | {r["tg_tps_aggregate"]} |')
tp = os.path.join(d, "tbo-perf.csv")
if os.path.exists(tp):
    L.append("\n## Row 3b: test-backend-ops perf, MUL_MAT nvfp4 x f32\n\n| m | n | k | us/run | TFLOPS |\n|---|---|---|---|---|")
    for r in csv.DictReader(open(tp)): L.append(f'| {r["m"]} | {r["n"]} | {r["k"]} | {r["us"]} | {r["tflops"]} |')
sm = os.path.join(d, "summary.csv")
if os.path.exists(sm):
    L.append("\n## Rows 4-7, 10: llama-server (greedy, cache_prompt off, median of 3 timed runs after 1 warm-up)\n")
    L.append("| config | slots | n_predict | tok/s | acceptance | mean accepted len | avg power W |\n|---|---|---|---|---|---|---|")
    for r in csv.DictReader(open(sm)):
        L.append(f'| {r["label"]} | {r["concurrency"] or 1} | {r["n_predict"]} | {r["tok_s_median"]} | {r["acc_rate"]} | {r["mean_len"]} | {r["power_w"]} |')
    rows = os.path.join(d, "rows.csv")
    if os.path.exists(rows):
        per = {}
        for r in csv.DictReader(open(rows)):
            if "#" in r["prompt"]: continue
            per.setdefault(r["label"], {}).setdefault(r["prompt"], []).append(float(r["tok_s"]))
        if per:
            prompts = sorted({p for v in per.values() for p in v})
            L.append("\n### Per-prompt median tok/s\n\n| config | " + " | ".join(prompts) + " |\n|---|" + "---|" * len(prompts))
            for lab, v in per.items(): L.append(f"| {lab} | " + " | ".join(f"{statistics.median(v[p]):.1f}" if p in v else "" for p in prompts) + " |")
open(os.path.join(d, "rows.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
