#!/usr/bin/env python3
"""vLLM comparison client. Streams /v1/chat/completions so decode tok/s excludes prefill:
decode_tps = (completion_tokens - 1) / (t_last_token - t_first_token). Concurrency K sends K prompts at once;
aggregate = sum(completion_tokens) / wall. Acceptance comes from /metrics deltas (spec_decode counters).
Usage: vllm_client.py --url http://127.0.0.1:8000 --model NAME --label L --prompts ... [--system S] [--n-predict 512]
       [--runs 3] [--warmup 1] [--concurrency K] [--ignore-eos] --rows rows.csv --summary summary.csv [--telemetry csv]
"""
import argparse, csv, json, os, statistics, sys, threading, time, urllib.request, re
from datetime import datetime

def metrics(url):
    time.sleep(2)  # Prometheus counters are updated by the engine's stat logger, not per request
    try:
        with urllib.request.urlopen(url + "/metrics", timeout=30) as r: txt = r.read().decode()
    except Exception: return {}
    out = {}
    for l in txt.splitlines():
        if l.startswith("#") or "spec_decode" not in l: continue
        m = re.match(r"(\S+?)(\{[^}]*\})?\s+([\d.e+-]+)", l)
        if m: out[m.group(1) + (m.group(2) or "")] = float(m.group(3))
    return out

def stream_completion(url, model, prompt, n, ignore_eos):
    # prompt = (system, user). /v1/chat/completions applies the model's chat template server-side, the same thing
    # server_rows.py does for llama.cpp via /apply-template. A raw /v1/completions prompt made the instruct model stop
    # after one token (dry run 2026-10-27).
    system, user = prompt
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    body = {"model": model, "messages": msgs, "max_tokens": n, "temperature": 0, "seed": 7, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": ignore_eos}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); t_first = None; t_last = None; n_tok = None; text = []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"): continue
            payload = line[5:].strip()
            if payload == "[DONE]": break
            d = json.loads(payload)
            if d.get("usage"): n_tok = d["usage"].get("completion_tokens")
            ch = d.get("choices") or []
            piece = (ch[0].get("delta") or {}).get("content") if ch else None
            if piece:
                now = time.perf_counter(); text.append(piece)
                if t_first is None: t_first = now
                t_last = now
    wall = time.perf_counter() - t0
    if n_tok is None: n_tok = len(text)
    decode_tps = (n_tok - 1) / (t_last - t_first) if (t_first and t_last and t_last > t_first and n_tok > 1) else float("nan")
    return {"n": n_tok, "ttft": (t_first - t0) if t_first else float("nan"), "decode_tps": decode_tps, "wall": wall, "text": "".join(text)}

def power_avg(csv_path, t0, t1):
    if not csv_path or not os.path.exists(csv_path): return float("nan")
    vals = []
    for row in csv.reader(open(csv_path)):
        if len(row) < 2 or row[0].strip() == "timestamp": continue
        try:
            ts = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
            if t0 <= ts <= t1: vals.append(float(row[1].split()[0]))
        except ValueError: pass
    return statistics.fmean(vals) if vals else float("nan")

def main():
    ap = argparse.ArgumentParser()
    for k, v in [("--url", "http://127.0.0.1:8000"), ("--model", None), ("--label", None), ("--system", None), ("--rows", None), ("--summary", None), ("--telemetry", None)]:
        ap.add_argument(k, default=v, required=v is None and k in ("--model", "--label", "--rows", "--summary"))
    ap.add_argument("--prompts", nargs="+", required=True); ap.add_argument("--n-predict", type=int, default=512)
    ap.add_argument("--runs", type=int, default=3); ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=0); ap.add_argument("--ignore-eos", action="store_true")
    a = ap.parse_args()
    system = open(a.system).read().strip() if a.system else ""
    prompts = [(os.path.splitext(os.path.basename(p))[0], (system, open(p).read())) for p in a.prompts]
    rows_new = not os.path.exists(a.rows); rw = csv.writer(open(a.rows, "a", newline=""))
    if rows_new: rw.writerow(["label", "prompt", "run", "completion_tokens", "ttft_s", "decode_tok_s", "wall_s"])
    m0 = metrics(a.url); t_start = None; agg = []; per = []
    if a.concurrency == 0:
        for name, prompt in prompts:
            for _ in range(a.warmup): stream_completion(a.url, a.model, prompt, min(a.n_predict, 64), a.ignore_eos)
            if t_start is None: t_start = time.time(); m0 = metrics(a.url)
            tps = []
            for run in range(1, a.runs + 1):
                r = stream_completion(a.url, a.model, prompt, a.n_predict, a.ignore_eos)
                rw.writerow([a.label, name, run, r["n"], f'{r["ttft"]:.3f}', f'{r["decode_tps"]:.2f}', f'{r["wall"]:.3f}']); tps.append(r["decode_tps"])
            per.append(statistics.median(tps)); print(f"[{a.label}] {name}: median decode {per[-1]:.2f} tok/s", flush=True)
        summary_tps = statistics.median(per)
    else:
        K = a.concurrency
        def batch(n):
            out = [None] * K
            def w(i):
                name, prompt = prompts[i % len(prompts)]; out[i] = (name, stream_completion(a.url, a.model, prompt, n, a.ignore_eos))
            th = [threading.Thread(target=w, args=(i,)) for i in range(K)]; t0 = time.perf_counter(); [t.start() for t in th]; [t.join() for t in th]
            return out, time.perf_counter() - t0
        for _ in range(a.warmup): batch(min(a.n_predict, 64))
        t_start = time.time(); m0 = metrics(a.url)
        for run in range(1, a.runs + 1):
            out, wall = batch(a.n_predict); gen = sum(r["n"] for _, r in out); agg.append(gen / wall)
            for i, (name, r) in enumerate(out): rw.writerow([a.label, f"{name}#{i}", run, r["n"], f'{r["ttft"]:.3f}', f'{r["decode_tps"]:.2f}', f"{wall:.3f}"])
            print(f"[{a.label}] batch {run}: {gen} tokens / {wall:.2f}s = {gen/wall:.1f} tok/s aggregate", flush=True)
        summary_tps = statistics.median(agg)
    t_end = time.time(); m1 = metrics(a.url)
    acc = drafts = float("nan")
    for k, v in m1.items():
        d = v - m0.get(k, 0)
        if "accepted_tokens" in k and "per_pos" not in k: acc = d
        if "num_draft_tokens" in k: drafts = d
    rate = acc / drafts if drafts and drafts == drafts and drafts > 0 else float("nan")
    sn = not os.path.exists(a.summary); sw = csv.writer(open(a.summary, "a", newline=""))
    if sn: sw.writerow(["label", "concurrency", "n_predict", "runs", "tok_s_median", "acc_rate", "accepted", "drafted", "power_w", "timed_s"])
    sw.writerow([a.label, a.concurrency, a.n_predict, a.runs, f"{summary_tps:.2f}", f"{rate:.4f}", acc, drafts, f"{power_avg(a.telemetry, t_start, t_end):.1f}", f"{t_end - t_start:.1f}"])
    print(f"[{a.label}] SUMMARY tok/s={summary_tps:.2f} acc_rate={rate:.4f} ({acc}/{drafts})", flush=True)

if __name__ == "__main__": main()
