#!/usr/bin/env python3
"""Concurrent decode throughput: aggregate and per-stream.

Sends N simultaneous completions and reports both the sum of per-request decode
rates (aggregate server throughput) and the per-stream rate each client sees.
These are different numbers and both matter: aggregate is what a multi-user
server delivers, per-stream is what one user experiences.
"""
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPTS = [
    "Explain in 200 words how a B-tree index speeds up database lookups.",
    "Write a Python function that merges overlapping intervals, with a docstring.",
    "Explain in 200 words why the sieve of Eratosthenes beats trial division.",
    "Describe in 200 words how NUMA affects memory bandwidth on multi-socket servers.",
    "Explain in 200 words what a Mixture-of-Experts layer does in a transformer.",
    "Write a short Python class implementing a bounded LRU cache with comments.",
    "Explain in 200 words how speculative decoding accelerates LLM inference.",
    "Describe in 200 words the difference between latency and throughput.",
]

def one(args, idx):
    port, model, maxtok = args
    pl = {"model": model,
          "messages": [{"role": "user", "content": PROMPTS[idx % len(PROMPTS)]}],
          "max_tokens": maxtok, "temperature": 0.0, "seed": 42 + idx}
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                               data=json.dumps(pl).encode(),
                               headers={"Content-Type": "application/json"})
    try:
        d = json.loads(urllib.request.urlopen(r, timeout=3600).read())
        t = d.get("timings", {}) or {}
        return t.get("predicted_per_second"), t.get("predicted_n")
    except Exception as e:
        print("  req failed:", e, file=sys.stderr)
        return None, None

if __name__ == "__main__":
    port, model, n, maxtok = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        res = list(ex.map(lambda i: one((port, model, maxtok), i), range(n)))
    wall = time.time() - t0
    rates = [r for r, _ in res if r]
    toks = sum(k for _, k in res if k)
    if not rates:
        print("NO SUCCESSFUL RUNS"); sys.exit(1)
    print(f"  concurrency {n}: per-stream mean {sum(rates)/len(rates):.3f} tok/s | "
          f"AGGREGATE {sum(rates):.3f} tok/s | wall-clock {toks/wall:.3f} tok/s "
          f"({toks} tokens in {wall:.1f}s)")
