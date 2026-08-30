# GLM-5.3 full: a composite `--spec-type` that fails every request

Host: 4 x Xeon Gold 6242, 64 physical cores, 755 GiB, 4 NUMA nodes, no GPU.
Measured 2026-08-29.

Model: GLM-5.3 (non-Flash) `UD-Q4_K_XL`, 11 shards, ~467 GB, architecture
`glm-dsa`, on the four-device CPU-NUMA tensor-sharded path with a detached MTP
draft head.

## The failure

A long-running server was healthy by every liveness signal and returned an
error on **100% of completions**:

    GET  /health              -> {"status":"ok"}
    GET  /v1/models           -> 200, full metadata
    POST /v1/chat/completions -> 500
      {"code":500,"message":"decode() failed: failed to process speculative batch"}

The error comes from `common_speculative_process()` in
`tools/server/server-context.cpp`, which throws unconditionally when the
speculative batch fails.

Two properties made this hard to spot:

- **`/health` returns `ok`.** The model is loaded and the HTTP layer is fine;
  only decode is broken. Any monitor polling `/health` reports green.
- **Disabling speculation per request does not avoid it.** Sending
  `{"speculative": {"n_max": 0}}` still 500s, because the speculative context is
  constructed at startup from `--spec-type` and is already in a bad state. This
  strongly suggests a request-level bug and sends you looking in the wrong place.

## The cause

The server had been started with a **composite** speculative type:

    --spec-type ngram-mod,draft-mtp

Restarting with the single validated type and nothing else changed:

    --spec-type draft-mtp

restored correct operation immediately — HTTP 200, exact expected content.

The composite form is accepted by argument parsing and by startup, and the
server then runs indefinitely while failing every decode. There is no
startup-time validation that catches it.

**Do not combine `ngram-mod` with `draft-mtp` on this engine.** If you want
both behaviours, choose per request, and verify a completion actually returns
200 after any `--spec-type` change. A server that starts is not a server that
works.

## Throughput after the fix

Restored on the four-device path, 16 threads/node, 200-token generations,
`temperature=0`, `seed=42`, `reasoning_effort=low`:

| arm | decode tok/s |
| --- | ---: |
| raw, `speculative.n_max=0` | 3.78 |
| **default `n_max=2`, `p_min=0`** | **5.11** |
| final restored service, prose+code mix, n=4 | **5.32** (range 4.59–6.21) |

Note the spread on that last row. Individual 200-token generations on this
model vary about ±15% around the mean, which is larger than several of the
configuration differences people try to tune. Any comparison here that turns on
less than ~15% needs more repetitions than two, and several published
single-repeat numbers — including some in this repository's earlier files —
should be read with that in mind.

Correctness was re-verified after restore (`17 * 23` -> `391`, and exact
passage reproduction on the replay workload).

Speculative depth swept live (request-scoped, no restart needed), novel prose:

| `n_max` / `p_min` | decode tok/s |
| --- | ---: |
| **2 / 0** | **4.885** |
| 4 / 0 | 4.412 |
| 32 / 0.8 | 4.450 |
| 18 / 0.75 | 3.883 |
| 8 / 0.5 | 3.781 |

Shallow-and-ungated wins, reproducing the earlier finding on this model that
deep drafts and confidence gating both cost throughput. On a short replay
workload (copy-a-passage-then-summarize), which is the traffic the deep profile
was tuned for, `n=2/p=0` still led — 6.230 vs 5.877 tok/s — with both arms
reproducing the source text exactly. The deep agentic profile's advantage
requires longer, warmer repetitive context than this test provides; it should
not be treated as a general default.

**Speculative parameters are request-scoped on this server.** Sweeping them
costs nothing but request time — no reload. Sweep them before touching anything
that requires a 20-minute model reload.

## Speculative sweep, re-run on a verified-quiet host

The earlier depth sweep was taken on a contended box. Re-run at 99% idle,
request-scoped so no reload was needed:

| `n_max` / `p_min` | decode tok/s |
| --- | ---: |
| 0 (no speculation) | 4.691 |
| **2 / 0** | **6.349** |
| 32 / 0.85 | 5.630 |
| 18 / 0.75 | 5.239 |
| 8 / 0.5 | 4.340 |

`n=2, p=0` remains the winner and the ordering is unchanged, so the original
conclusion survives a clean re-measurement. What the quiet run adds is a
trustworthy **speculative multiplier of 1.35x** (4.691 -> 6.349), against 1.18x
measured under load.

That number closes the question for this model. Raw decode at 4.691 tok/s is 44%
of the 10.6 tok/s ceiling its 33.97 GB/token allows. Reaching 10 tok/s while
keeping the same 1.35x multiplier needs raw decode at 7.4, i.e. **70%
extraction** — a 1.6x kernel improvement, not a configuration change. Deeper
drafts do not substitute: acceptance rises to 72–77% at `n=18` but throughput
falls, because draft passes cost more than the extra accepted tokens return.

## Threads per node: no effect here, which is itself the finding

Qwen3.8-Flash-Next gains 14% on this host by dropping from 16 to 12 threads per
NUMA node. That does not transfer:

| threads/node | raw tok/s | `n2/p0` tok/s (matched prose) |
| ---: | ---: | ---: |
| 12 | 3.98 | 4.99 |
| 16 | 3.78 | 5.11 |

The two configurations are within the run-to-run spread of each other — single
prose repeats ranged 4.34–5.27 at 12 threads, so a 2% gap between means is not
a result. Production was returned to 16, the configuration the rest of this
model's tuning was derived at.

Worth stating plainly because the temptation after the Flash-Next result is to
apply 12 everywhere. Three models, three answers: Flash-Next wants 12
(decisively), Qwen3.8-27B wants 16, GLM-5.3 full does not care. **Each of these
cost a measurement to establish and none of them predicted the others.**

Note the sweep is expensive here in a way it is not elsewhere — every arm costs
a ~20 minute reload, against seconds for a request-scoped speculative
parameter. Order your sweeps by reload cost, not by expected effect size.

## Operational notes for a ~470 GB model

**Load takes 21 minutes and peaks well above its steady state.** RSS climbed to
**628 GB** during the repack pass before settling at **491 GB**. Size the host
for the peak, not the resident set. A first attempt with 119 GB of unrelated
files in `/dev/shm` never converged: it reached the same 628 GB, exhausted swap,
and sat with all worker threads parked in `futex_wait` while one thread churned.
That is not a slow load, it is a thrash, and it does not recover — kill it,
free the memory, and restart rather than waiting.

**`tmpfs` is not free space.** Files staged in `/dev/shm` are unreclaimable RAM
competing directly with the model. Staging a 101 GB download there was enough to
turn a working configuration into a non-converging one. Move large staged
artifacts onto disk before loading a model sized near host memory.

**RSS during load is misleading.** Much of it is mapped file pages that the
kernel can drop; `free -h` showing 3 GB free next to 250 GB of `buff/cache` is
normal here and not itself a warning sign. The signal to watch is `available`
together with swap: swap fully consumed *and* `available` collapsing is the
thrash signature.
