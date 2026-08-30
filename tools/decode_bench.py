#!/usr/bin/env python3
"""Controlled decode-throughput harness for llama-server.

Reports emitted-token decode rate from the server's own timings block, so the
number is independent of client-side latency. Speculative arms additionally
report draft acceptance.
"""
import argparse, json, statistics, sys, time, urllib.request, urllib.error

WORKLOADS = {
    "prose": "Write a clear 250-word explanation of how a B-tree index speeds up "
             "database lookups compared to a full table scan. Plain prose, no lists.",
    "code":  "Write a Python function `merge_intervals(intervals)` that merges "
             "overlapping closed intervals and returns them sorted. Include a "
             "docstring and three doctest examples.",
    "recall": "List the first 40 prime numbers, then explain in one paragraph why "
              "the sieve of Eratosthenes is more efficient than trial division.",
}


def post(url, payload, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def run(args):
    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    rows = []
    for name in args.workloads.split(","):
        prompt = WORKLOADS[name]
        for rep in range(args.reps):
            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "seed": 42,
                "cache_prompt": False,
            }
            if args.reasoning_effort:
                payload["reasoning_effort"] = args.reasoning_effort
            if args.spec_n_max is not None:
                payload["speculative"] = {"n_max": args.spec_n_max}
                if args.spec_p_min is not None:
                    payload["speculative"]["p_min"] = args.spec_p_min
            t0 = time.time()
            try:
                d = post(url, payload, args.timeout)
            except urllib.error.HTTPError as e:
                print(f"  {name}#{rep} HTTP {e.code}: {e.read()[:300].decode()}", file=sys.stderr)
                continue
            except Exception as e:
                print(f"  {name}#{rep} FAILED: {e}", file=sys.stderr)
                continue
            wall = time.time() - t0
            tm = d.get("timings", {}) or {}
            ch = d.get("choices", [{}])[0].get("message", {}) or {}
            content = ch.get("content") or ""
            reasoning = ch.get("reasoning_content") or ""
            row = {
                "workload": name, "rep": rep,
                "decode_tps": tm.get("predicted_per_second"),
                "prompt_tps": tm.get("prompt_per_second"),
                "n_predict": tm.get("predicted_n"),
                "draft_acc": tm.get("draft_acceptance_rate"),
                "wall_s": round(wall, 2),
                "content_chars": len(content),
                "reasoning_chars": len(reasoning),
                "finish": d.get("choices", [{}])[0].get("finish_reason"),
            }
            rows.append(row)
            print(f"  {name}#{rep}: {row['decode_tps']:.3f} tok/s  "
                  f"n={row['n_predict']}  acc={row['draft_acc']}  "
                  f"chars={row['content_chars']}  {row['finish']}", flush=True)

    vals = [r["decode_tps"] for r in rows if r["decode_tps"]]
    summary = {}
    if vals:
        summary = {
            "n": len(vals),
            "mean_tps": round(statistics.mean(vals), 3),
            "median_tps": round(statistics.median(vals), 3),
            "min_tps": round(min(vals), 3),
            "max_tps": round(max(vals), 3),
            "stdev": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
        }
        accs = [r["draft_acc"] for r in rows if r.get("draft_acc")]
        if accs:
            summary["mean_draft_acc"] = round(statistics.mean(accs), 4)
        print(f"\nSUMMARY {args.label}: mean {summary['mean_tps']} tok/s  "
              f"median {summary['median_tps']}  range {summary['min_tps']}-{summary['max_tps']}"
              + (f"  acc {summary['mean_draft_acc']}" if accs else ""))
    else:
        print(f"\nSUMMARY {args.label}: NO SUCCESSFUL RUNS", file=sys.stderr)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"label": args.label, "summary": summary, "rows": rows}, f, indent=2)
    return 0 if vals else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--model", default="default")
    p.add_argument("--label", default="run")
    p.add_argument("--workloads", default="prose,code")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--spec-n-max", type=int, default=None)
    p.add_argument("--spec-p-min", type=float, default=None)
    p.add_argument("--out", default=None)
    sys.exit(run(p.parse_args()))
