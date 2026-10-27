#!/usr/bin/env python3
"""Row 8 client: generate greedy completions with top-2 logprobs and compare against a reference set.

gen:     invariance.py gen --out DIR --prompts ... [--system S] [--n-predict 512]
compare: invariance.py compare --ref DIR_A --new DIR_B [--report FILE]
Each prompt is saved as DIR/<name>.json = {"tokens":[...], "content": str, "top2":[[(tok,logprob),(tok,logprob)],...],
"timings": {...}}. compare prints, per prompt: identical | diverged at index i (ref top-2, new top-2 at i), and the
overall divergence rate, and appends a CSV line to --report.
"""
import argparse, csv, json, os, sys, urllib.request

def post(url, path, body):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r: return json.loads(r.read().decode())

def gen(a):
    os.makedirs(a.out, exist_ok=True)
    system = open(a.system).read() if a.system else None
    for p in a.prompts:
        name = os.path.splitext(os.path.basename(p))[0]
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": open(p).read()}]
        prompt = post(a.url, "/apply-template", {"messages": msgs})["prompt"]
        r = post(a.url, "/completion", {"prompt": prompt, "n_predict": a.n_predict, "temperature": 0, "top_k": 1, "seed": 7,
                                        "cache_prompt": False, "return_tokens": True, "n_probs": 2, "samplers": ["top_k"]})
        top2 = []
        for tok in r.get("completion_probabilities", []):
            cands = tok.get("top_logprobs") or tok.get("top_probs") or []
            top2.append([(c.get("id"), c.get("logprob", c.get("prob"))) for c in cands[:2]])
        json.dump({"tokens": r["tokens"], "content": r["content"], "top2": top2, "timings": r.get("timings", {})},
                  open(os.path.join(a.out, name + ".json"), "w"))
        t = r.get("timings", {}); dn = t.get("draft_n", 0) or 0; da = t.get("draft_n_accepted", 0) or 0
        print(f"  gen {name}: {len(r['tokens'])} tokens, {t.get('predicted_per_second', 0):.1f} tok/s, acc {da}/{dn}", flush=True)

def compare(a):
    names = sorted(f[:-5] for f in os.listdir(a.ref) if f.endswith(".json"))
    n_div = 0; w = None
    if a.report:
        new_file = not os.path.exists(a.report); f = open(a.report, "a", newline=""); w = csv.writer(f)
        if new_file: w.writerow(["ref", "new", "prompt", "identical", "first_diverging_index", "n_ref", "n_new", "ref_top2", "new_top2", "ref_gap", "new_gap"])
    for n in names:
        A = json.load(open(os.path.join(a.ref, n + ".json"))); B = json.load(open(os.path.join(a.new, n + ".json")))
        ta, tb = A["tokens"], B["tokens"]
        i = next((k for k in range(min(len(ta), len(tb))) if ta[k] != tb[k]), None)
        if i is None and len(ta) == len(tb):
            print(f"  {n}: identical ({len(ta)} tokens)")
            if w: w.writerow([os.path.basename(a.ref), os.path.basename(a.new), n, 1, "", len(ta), len(tb), "", "", "", ""])
            continue
        n_div += 1
        if i is None: i = min(len(ta), len(tb))
        ra = A["top2"][i] if i < len(A["top2"]) else []; rb = B["top2"][i] if i < len(B["top2"]) else []
        gap = lambda t: (t[0][1] - t[1][1]) if len(t) == 2 and t[0][1] is not None and t[1][1] is not None else float("nan")
        print(f"  {n}: DIVERGED at token {i} (ref {len(ta)} / new {len(tb)} tokens); ref top-2 {ra} gap {gap(ra):.4f}; new top-2 {rb} gap {gap(rb):.4f}")
        if w: w.writerow([os.path.basename(a.ref), os.path.basename(a.new), n, 0, i, len(ta), len(tb), json.dumps(ra), json.dumps(rb), f"{gap(ra):.5f}", f"{gap(rb):.5f}"])
    print(f"  divergence rate {n_div}/{len(names)} ({os.path.basename(a.new)} vs {os.path.basename(a.ref)})")
    return n_div

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen"); g.add_argument("--url", default="http://127.0.0.1:8080"); g.add_argument("--out", required=True)
    g.add_argument("--prompts", nargs="+", required=True); g.add_argument("--system"); g.add_argument("--n-predict", type=int, default=512)
    c = sub.add_parser("compare"); c.add_argument("--ref", required=True); c.add_argument("--new", required=True); c.add_argument("--report")
    a = ap.parse_args()
    gen(a) if a.cmd == "gen" else sys.exit(1 if compare(a) else 0)
