#!/usr/bin/env python3
"""Drive a running llama-server through the server rows (4-7 and 10) of the results table.

Per prompt: one untimed warm-up, then N timed runs (default 3). Greedy, cache_prompt off, so every run
re-prefills and decodes from scratch. Timings come from the server's own `timings` object; acceptance from
draft_n / draft_n_accepted; mean accepted length from the server log lines
"draft acceptance = ... mean len = X" written after the log offset at which this script started.

Concurrency mode (--concurrency K): K requests in flight (prompts cycle through the set), ignore_eos,
aggregate tok/s = sum(predicted_n) / wall seconds of the whole batch. Warm-up + N timed batches.

Writes one CSV line per (prompt, run) to --rows and one summary line per label to --summary.
Average board power over the timed window is read from a telemetry CSV (nvidia-smi -l 1) if given.
"""
import argparse, csv, hashlib, json, os, statistics, sys, time, threading, urllib.request
from datetime import datetime

def post(url, path, body, timeout=3600):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def render_prompt(url, system, user):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return post(url, "/apply-template", {"messages": msgs})["prompt"]

def completion(url, prompt, n_predict, ignore_eos=False, n_probs=0):
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "top_k": 1, "seed": 7,
            "cache_prompt": False, "ignore_eos": ignore_eos, "return_tokens": True, "samplers": ["top_k"]}
    if n_probs: body["n_probs"] = n_probs
    t0 = time.perf_counter(); r = post(url, "/completion", body); wall = time.perf_counter() - t0
    r["_wall_s"] = wall
    return r

def mean_len_from_log(path, offset):
    """Average of 'mean len = X' over log lines written after byte offset. Returns (mean_len, new_offset)."""
    if not path or not os.path.exists(path): return (float("nan"), offset)
    vals = []
    with open(path, "rb") as f:
        f.seek(offset); data = f.read(); new = offset + len(data)
    for line in data.decode(errors="replace").splitlines():
        if "mean len =" in line:
            try: vals.append(float(line.split("mean len =")[1].split()[0]))
            except ValueError: pass
    return (statistics.fmean(vals) if vals else float("nan"), new)

def power_avg(csv_path, t_start, t_end):
    """Mean power.draw [W] for telemetry samples whose timestamp lies in [t_start, t_end] (epoch seconds)."""
    if not csv_path or not os.path.exists(csv_path): return float("nan")
    vals = []; near = []
    with open(csv_path) as f:
        for row in csv.reader(f):
            if len(row) < 2 or row[0].strip() == "timestamp": continue
            try:
                ts = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
                if t_start <= ts <= t_end: vals.append(float(row[1].strip().split()[0]))
                elif t_start - 1.5 <= ts <= t_end + 1.5: near.append(float(row[1].strip().split()[0]))
            except ValueError: continue
    # a sub-second timed window (dry runs) can fall between two 1 Hz samples: use the neighbouring samples
    return statistics.fmean(vals) if vals else (statistics.fmean(near) if near else float("nan"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--system", default=None)
    ap.add_argument("--n-predict", type=int, default=512)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=0, help="K in-flight requests (row 7); 0 = sequential rows 4-6")
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--rows", required=True, help="per-run CSV (appended)")
    ap.add_argument("--summary", required=True, help="per-label CSV (appended)")
    ap.add_argument("--server-log", default=None)
    ap.add_argument("--telemetry", default=None)
    ap.add_argument("--save-text-dir", default=None, help="write generated text per prompt/run here")
    a = ap.parse_args()

    system = open(a.system).read() if a.system else None
    prompts = [(os.path.splitext(os.path.basename(p))[0], render_prompt(a.url, system, open(p).read())) for p in a.prompts]
    log_off = os.path.getsize(a.server_log) if a.server_log and os.path.exists(a.server_log) else 0

    rows_new = not os.path.exists(a.rows); sum_new = not os.path.exists(a.summary)
    rows_f = open(a.rows, "a", newline=""); rows_w = csv.writer(rows_f)
    if rows_new: rows_w.writerow(["label", "prompt", "run", "prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "tok_s", "draft_n", "draft_acc", "acc_rate", "wall_s", "text_sha8"])

    t_timed_start = None; t_timed_end = None
    per_prompt_medians = []; acc_n = 0; acc_d = 0; agg = []

    if a.concurrency == 0:
        for name, prompt in prompts:
            for _ in range(a.warmup): completion(a.url, prompt, min(a.n_predict, 64), a.ignore_eos)
            if a.server_log: _, log_off = mean_len_from_log(a.server_log, log_off)   # drop warm-up lines
            t_timed_start = t_timed_start or time.time()
            tps = []
            for run in range(1, a.runs + 1):
                r = completion(a.url, prompt, a.n_predict, a.ignore_eos); t = r["timings"]
                dn = int(t.get("draft_n", 0) or 0); da = int(t.get("draft_n_accepted", 0) or 0)
                sha8 = hashlib.sha256(r["content"].encode()).hexdigest()[:8]
                rows_w.writerow([a.label, name, run, t["prompt_n"], f'{t["prompt_ms"]:.1f}', t["predicted_n"], f'{t["predicted_ms"]:.1f}',
                                 f'{t["predicted_per_second"]:.2f}', dn, da, f"{(da/dn if dn else 0):.4f}", f'{r["_wall_s"]:.3f}', sha8])
                rows_f.flush(); tps.append(t["predicted_per_second"]); acc_n += da; acc_d += dn
                if a.save_text_dir:
                    os.makedirs(a.save_text_dir, exist_ok=True)
                    open(os.path.join(a.save_text_dir, f"{a.label}.{name}.run{run}.txt"), "w").write(r["content"])
            per_prompt_medians.append(statistics.median(tps))
            print(f"[{a.label}] {name}: median {statistics.median(tps):.2f} tok/s over {a.runs} runs, acc {acc_n}/{acc_d}", flush=True)
        t_timed_end = time.time()
        tps_summary = statistics.median(per_prompt_medians)
    else:
        K = a.concurrency
        def batch(n_predict):
            out = [None] * K
            def worker(i):
                name, prompt = prompts[i % len(prompts)]
                out[i] = (name, completion(a.url, prompt, n_predict, a.ignore_eos))
            th = [threading.Thread(target=worker, args=(i,)) for i in range(K)]
            t0 = time.perf_counter(); [t.start() for t in th]; [t.join() for t in th]
            return out, time.perf_counter() - t0
        for _ in range(a.warmup): batch(min(a.n_predict, 64))
        if a.server_log: _, log_off = mean_len_from_log(a.server_log, log_off)
        t_timed_start = time.time()
        for run in range(1, a.runs + 1):
            out, wall = batch(a.n_predict)
            gen = sum(r["timings"]["predicted_n"] for _, r in out); agg.append(gen / wall)
            for i, (name, r) in enumerate(out):
                t = r["timings"]; dn = int(t.get("draft_n", 0) or 0); da = int(t.get("draft_n_accepted", 0) or 0)
                rows_w.writerow([a.label, f"{name}#{i}", run, t["prompt_n"], f'{t["prompt_ms"]:.1f}', t["predicted_n"], f'{t["predicted_ms"]:.1f}',
                                 f'{t["predicted_per_second"]:.2f}', dn, da, f"{(da/dn if dn else 0):.4f}", f"{wall:.3f}", hashlib.sha256(r["content"].encode()).hexdigest()[:8]])
                acc_n += da; acc_d += dn
            rows_f.flush()
            print(f"[{a.label}] batch {run}: {gen} tokens in {wall:.2f}s = {gen/wall:.1f} tok/s aggregate", flush=True)
        t_timed_end = time.time()
        tps_summary = statistics.median(agg)

    mean_len, _ = mean_len_from_log(a.server_log, log_off)
    pw = power_avg(a.telemetry, t_timed_start, t_timed_end)
    sum_f = open(a.summary, "a", newline=""); sum_w = csv.writer(sum_f)
    if sum_new: sum_w.writerow(["label", "concurrency", "n_predict", "runs", "tok_s_median", "acc_rate", "mean_len", "power_w", "timed_s"])
    sum_w.writerow([a.label, a.concurrency, a.n_predict, a.runs, f"{tps_summary:.2f}", f"{(acc_n/acc_d if acc_d else 0):.4f}",
                    f"{mean_len:.3f}", f"{pw:.1f}", f"{t_timed_end - t_timed_start:.1f}"])
    print(f"[{a.label}] SUMMARY tok/s={tps_summary:.2f} acc={acc_n}/{acc_d} mean_len={mean_len:.2f} power={pw:.0f}W", flush=True)

if __name__ == "__main__":
    main()
