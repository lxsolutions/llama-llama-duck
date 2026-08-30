#!/usr/bin/env python3
"""Four deterministic prompts with known answers, against a running llama-server.

Run this ONCE per configuration before trusting any throughput number. A tok/s
figure computed from a token stream nobody read is not a measurement -- see
benchmarks/qwen4exp-tensor-split-corruption.md, where a configuration that was
45% "faster" was emitting pure garbage, silently, for an entire campaign.

Usage: quality_probe.py <port> [model_alias]
Expected: 391 / Canberra / yes.
"""
import json,sys,urllib.request
Q=[("math","What is 17 * 23?"),
   ("fact","What is the capital of Australia?"),
   ("logic","If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops Lazzies?")]
port=sys.argv[1]
for name,q in Q:
    pl={"model":"bench","messages":[{"role":"user","content":q}],"max_tokens":600,
        "temperature":0.0,"seed":42}
    r=urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(pl).encode(),headers={"Content-Type":"application/json"})
    try:
        d=json.loads(urllib.request.urlopen(r,timeout=1200).read())
        m=d["choices"][0]["message"]
        c=(m.get("content") or "").strip().replace("\n"," ")
        rc=(m.get("reasoning_content") or "").strip().replace("\n"," ")
        out = c if c else "[reasoning] "+rc
        print(f"  {name:6}: {out[:110]!r}")
    except Exception as e: print(f"  {name:6}: ERR {e}")
