# What it takes to reach 10 tok/s on a 4-socket CPU

Target: **>10 decode tok/s** on four models, one host — 4 x Xeon Gold 6242,
64 physical cores, 755 GiB DDR4-2400 across 24 populated channels, 4 NUMA nodes,
no accelerator. Theoretical peak 460.8 GB/s; **measured achievable 360 GB/s**
NUMA-local, 138.9 GB/s interleaved.

This file records how close each model got, and — more usefully — the arithmetic
that says which of them *can* get there and what would have to change.

## Where the four models landed

Decode tok/s, quiet host, `temperature=0`, server-reported `predicted_per_second`.
"Active GB/tok" is the corrected figure — routed experts scaled by
`n_used / n_expert`, gathered embeddings excluded (see the correction below):

| model | quant | active GB/tok | start | best | raw ceiling @360 GB/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B | Q4_0, 16.1 GB file | ~16 | 6.12 | **8.33** prose / **10.90** replay | 22.5 |
| Qwen3.8-Flash-Next | Q2_K_XL, 78.8 GB file | **4.48** | 6.68 | **9.21** | 80.3 |
| GLM-5.3 full | Q4_K_XL, 467 GB file | **33.97** | 0 (broken) | **5.32** prose / **6.25** replay | **10.6** |
| GLM-5.3-Flash | IQ2_XXS, 102 GB file | **9.16** | — | **2.02** | 39.3 (locked to 1 socket) |

## The arithmetic nobody can argue with

Decode at batch 1 is bandwidth-bound: `tok/s = bandwidth / bytes-per-token`. So
the target sets a hard requirement before any tuning:

Using **corrected active bytes/token** (see the correction section below — an
earlier revision of this table used file bytes and was badly wrong for MoE):

| model | active GB/tok | GB/s for 10 tok/s | GB/s for 13 tok/s | as % of 360 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8-Flash-Next | 4.48 | 45 | 58 | **16%** |
| GLM-5.3-Flash | 9.16 | 92 | 119 | 33% |
| Qwen3.8-27B | ~16 | 160 | 208 | 58% |
| **GLM-5.3 full** | **33.97** | **340** | **442** | **123%** |

**GLM-5.3 full cannot reach 10 tok/s non-speculatively at Q4_K_XL, and cannot
reach 13 at all.** 13 tok/s would need 442 GB/s against a measured 360 GB/s
ceiling — more bandwidth than the machine has, before any inefficiency. Not a
kernel problem, not a placement problem: arithmetic. The routes past it are
speculative decoding, a smaller quant (measured worse — see the quant curve), or
a different model.

The other three have plenty of bandwidth headroom and are limited by something
else entirely.

Speculation is the only legitimate way past that wall, since K accepted tokens
are emitted per single target forward pass. It does help — 5.32 -> 6.25 moving
from novel prose to replay — but the measured multiplier on this host is 1.18x,
not the ~2.4x that clearing 10 would require. (Earlier notes here record 12.92;
that figure did not reproduce — see below.)

**State the workload with the number, or the number means nothing.**

## The hardware really does deliver — measure it before blaming it

Host DIMM configuration read from SMBIOS: **24 DIMMs at 2400 MT/s**, i.e. all 24
channels populated (6 per socket). Theoretical peak is
`24 x 2400 x 8 = 460.8 GB/s`.

Measured with `tools/membw.c`, four sockets concurrently, NUMA-local:

| threads/socket | per-socket GB/s | aggregate |
| ---: | --- | ---: |
| 16 | 94.8 / 90.1 / 90.8 / 84.1 | **359.8 GB/s** |
| 24 | 93.3 / 88.3 / 88.3 / 89.3 | 359.2 GB/s |
| 32 | 95.4 / 90.5 / 81.0 / 83.4 | 350.3 GB/s |

**360 GB/s achievable, 78% of theoretical peak** — a normal DDR4 STREAM result,
and flat in thread count above 16. Use 360, not 460.8, as the ceiling.

## Extraction efficiency — and a correction that changes the conclusion

An earlier revision of this file reported that all models land near "one third"
of bandwidth, and inferred one shared structural limit. **That was wrong**, and
wrong in a way worth documenting: it divided by *file* bytes, which is only
valid for a dense model. For MoE, the weights read per token are the routed
experts actually selected, plus the dense parts — a small fraction of the file.

There is a second trap on top of that. Computing active bytes by classifying
tensor names put **28.80 GB** of Qwen3.8-Flash-Next into "global" — because
`per_layer_token_embd.weight` is a 28.8 GB *embedding table*. Embeddings are
**gathered**, not streamed: a token reads one row, a few hundred bytes. Counting
that table as active bytes inflated the figure more than sevenfold.

Corrected, using per-tensor classification with experts scaled by
`n_expert_used / n_expert` and embeddings excluded:

| model | active GB/token | tok/s | achieved GB/s | **% of 360** | raw ceiling |
| --- | ---: | ---: | ---: | ---: | ---: |
| GLM-5.3 full (8/256 experts) | 33.97 | 5.32 | 181 | **50%** | 10.6 |
| Qwen3.8-27B (dense) | ~16 | 7.42 | 119 | **33%** | 22.5 |
| Qwen3.8-Flash-Next (10/512) | **4.48** | 9.21 | 41 | **11%** | 80.3 |
| GLM-5.3-Flash (8/288) | **9.16** | 2.02 | 18.5 | **5%** | 39.3 |

The models are not all alike, and the spread is 10x:

- **GLM-5.3 full is nearly bandwidth-bound** at 50% of achievable. Its 10.6 tok/s
  ceiling is real, and no kernel work gets it to 13 — that needs 442 GB/s, which
  is 123% of what the hardware does. Speculation is the only route.
- **The MoE models are nowhere near bandwidth-bound.** Flash-Next uses 11% of
  available bandwidth and GLM-5.3-Flash 5%. They are not short of memory
  throughput; they are spending their time somewhere else.

For Flash-Next the arithmetic is now encouraging rather than damning:
**13 tok/s needs only 58 GB/s, 16% of the machine**, against 11% today — a 1.45x
kernel improvement, not a hardware limit.

### Where MoE decode actually goes

Flash-Next runs **512 experts, 10 used, 48 layers**. Each routed expert matmul is
`[2560 x 640]`, and there are `10 x 48 = 480` of them per token, then sharded
four ways so each device handles roughly `[640 x 640]`. That is hundreds of tiny
matmuls per token, each with dispatch and collective overhead, through
`MUL_MAT_ID`'s single-row `gemv` path.

This is consistent with this repository's earlier MoE kernel measurements (repack
buys MoE ~1.0–1.26x against dense's 2.5–6.7x) and with its observation that MoE
`gemv` only becomes `gemm` above three rows. **The MoE expert-gather path, not
memory bandwidth, is the binding constraint on three of these four models.**

## `GGML_CPU_NUMA_POLL` is worth 7.7% on MoE, and the default is far too low

Spin-wait length before a worker sleeps. With hundreds of tiny expert matmuls per
token, wake-up latency is a large fraction of the work. Qwen3.8-Flash-Next,
12 threads/node, everything else fixed:

| `GGML_CPU_NUMA_POLL` | decode tok/s | run range |
| ---: | ---: | --- |
| 0 | 8.775 | 8.692–8.857 |
| 50 | 8.980 | — |
| 500 | 9.205 | 9.187–9.224 |
| 2000 | 9.368 | 9.353–9.384 |
| 10000 | 9.418 | 9.377–9.458 |
| **100000** | **9.452** | 9.241–9.662 |

Monotonic and tight, saturating around 10⁴. **+7.7% from one environment
variable**, on a model where every configuration knob had already been swept.
The previously published profiles used 50–100.

Re-sweeping threads at the new poll did not move the optimum (t12 still best;
t14 8.27, t16 8.25), so the two knobs are independent here.

### The same sweep on Qwen3.8-27B did *not* reproduce — a variance lesson

A first pass at poll `10000` on Qwen3.8-27B measured **10.379 tok/s** on novel
prose (range 10.357–10.401), which would have cleared the 10 tok/s bar outright
and been the headline of this file. It did not survive re-measurement:

| run | poll | reps | mean | range |
| --- | ---: | ---: | ---: | --- |
| first | 10000 | 2 | **10.379** | 10.357–10.401 |
| bracket | 3000 | 3 | 9.155 | 8.974–9.400 |
| bracket | 30000 | 3 | 9.104 | 8.896–9.254 |
| **verify** | **10000** | **6** | **8.616** | **7.084–9.400** |

Six repetitions at the identical configuration span **7.08 to 9.40** — a 33%
spread that swallows the entire apparent effect. The first result was a pair of
lucky draws whose *internal* range (0.4%) looked reassuringly tight.

**A narrow range within one short run says nothing about reproducibility.** This
model carries an MTP draft, and speculative arms are far noisier than raw ones
because acceptance varies per prompt. Flash-Next's poll curve above is credible
precisely because it is monotonic across six settings; a single point is not.

Treat any single-configuration claim on this host that rests on fewer than ~6
repetitions as unproven, and be especially suspicious when it lands just past a
target you were aiming at.

## Knobs that are exhausted (swept, no further gain)

Qwen3.8-27B, four `CPU-NUMA` devices:

| knob | values tried | best |
| --- | --- | --- |
| threads/node | 8, 12, 16, 20, 24 | **16** (7.42; others 6.8–6.96) |
| draft depth `n_max` | 1, 2, 3 | **2** (8.33 vs 7.46 / 7.04) |
| draft threads | 4, 8, default | **default** (8.33 vs 7.91 / 7.71) |
| draft device | 1 node, 4 nodes | **4 nodes** (8.33 vs 7.05) |
| hugepages | on, off | **off** (8.33 vs 7.75) |
| NUMA poll | 0, 100 | **100** (8.33 vs 7.88) |
| quant | Q8_K_XL 31.4 GB, Q4_0 16.1 GB | **Q4_0** (7.42 vs 6.12 raw) |

Halving the bytes per token (Q8 -> Q4_0) bought only +21%, not +100%, because
Q4_0 extraction is *lower* (119 GB/s) than Q8's (192 GB/s) — the dequant cost
partly cancels the byte saving. **Byte reduction and kernel efficiency trade
against each other; neither alone predicts throughput.**

### The quant curve has an interior optimum, and Q4_0 is it

Full sweep on Qwen3.8-27B, same host, same engine, same MTP head:

| quant | file bytes | raw tok/s | +MTP tok/s | achieved GB/s (raw) |
| --- | ---: | ---: | ---: | ---: |
| UD-Q3_K_XL | 13.1 GB | 5.25 | 6.08 | 69 |
| **Q4_0** | **16.1 GB** | **7.42** | **8.33** | **119** |
| UD-Q4_K_XL | 17.6 GB | — | 7.86 | — |
| UD-Q8_K_XL | 31.4 GB | 6.12 | — | 192 |

Going *down* from Q4_0 to Q3_K_XL costs 29% throughput while saving 18% of the
bytes — Q3_K extraction collapses to 69 GB/s. Going *up* to UD-Q4_K_XL costs 6%
despite similar size, consistent with plain Q4_0's much higher repack coverage
(~86% of bytes anonymous vs the ~25% previously measured for a UD Q4 variant).

So the quant curve is not monotonic in either direction and the optimum is
interior. **Do not assume a smaller quant is faster.** Three separate models in
this repository now show the same thing, with the optimum in a different place
each time: Q4_0 for 27B, Q2_K_XL for Flash-Next, Q4_K_XL for GLM-5.3 full.

Note also that UD-Q3_K_XL had *higher* draft acceptance (61.8%) than Q4_0
(56.5%) and was still far slower — one more case where acceptance moved opposite
to throughput.

## `GGML_CPU_REPACK` is a build option, not a runtime switch

Worth stating because it invalidates an obvious experiment. Running the same
model with `GGML_CPU_REPACK=0` and `=1`:

| arm | tok/s | RssAnon |
| --- | ---: | ---: |
| `GGML_CPU_REPACK=0` | 3.725 | 13,812,652 kB |
| `GGML_CPU_REPACK=1` | 3.800 | 13,812,652 kB |

**Byte-identical RssAnon.** The environment variable does nothing; repack is
compiled in via `-DGGML_CPU_REPACK=ON` and is always active. The near-identical
throughput is not evidence that "repack does not help" — it is evidence that
both arms were repacked. (13.8 GB of a 16.1 GB model is anonymous, i.e. ~86%
repacked, better than the ~75% previously measured for plain Q4_0.)

The NUMA-device path has its own separate `GGML_CPU_NUMA_REPACK` runtime knob,
which does work. Do not confuse them.

## Speculation only pays if the draft is cheap, and here it is not

Qwen3.8-27B with its published MTP head (1.37 GB, Q4_0):

    draft acceptance = 0.56452 (105 accepted / 186 generated), mean len = 2.13

Healthy acceptance — yet throughput rose only 7.42 -> 8.33, a 1.12x multiplier
where mean length 2.13 suggests 2x should be available. Decomposing the cycle:

    target forward pass          ~135 ms
    2 x draft forward pass       ~122 ms   <- 47% of the cycle
    emitted per cycle             2.13 tokens  -> 8.33 tok/s

**The draft costs 61 ms per token for a 1.37 GB model** — about 22 GB/s, some
16x worse than the same host's streaming rate. A free draft would put this model
at ~15.8 tok/s. Draft *efficiency*, not draft acceptance and not draft depth, is
the binding constraint on speculative gain here.

This reframes the earlier finding that "optimizing for acceptance is a trap".
Acceptance was never the problem. Nor is placement: draft-on-one-node vs
draft-on-four was 7.05 vs 8.33, and cutting draft threads to 4 or 8 made it
worse. The draft pass is simply far from bandwidth on a model this small.

## Two independent MTP heads, both exactly zero

For Qwen3.8-Flash-Next, speculation is not merely inefficient — it is broken:

| draft head | size | acceptance | tok/s |
| --- | ---: | ---: | ---: |
| none | — | — | **8.98** |
| `drluoto/...-MTP-GGUF` Q8_0 | 4.14 GB | **0.00000** (0/198) | 3.13 |
| `dzannotti/...-MTP-GGUF` Q4_K_M | 2.62 GB | **0.00000** (0/198) | 3.14 |

Two independently published heads, different quants, different authors, both
scoring *exactly* zero accepted tokens out of 198. That rules out the draft
artifacts and points at the engine's MTP path for this architecture. Both runs
also logged `backend offload failed for seq_id=N; using CPU sampler`.

A single 0% result looks like a bad download. **Two is a bug report.**

## On the traffic these models actually serve, two of four clear 10

Every number above is novel prose — the worst case for speculation. Agentic and
coding traffic is repetitive: the model re-emits text it was given. Re-measuring
on a file-reproduce-then-extend workload with `ngram-mod` (`n_match=8`):

| model | novel prose | replay | replay acceptance |
| --- | ---: | ---: | ---: |
| **Qwen3.8-27B** (`ngram-mod`) | 8.33 | **10.90 mean, 17.13 warm** | 43.6% -> 74.3% |
| GLM-5.3 full (`draft-mtp` n18/p0.75) | 5.32 | 6.25 | 72.8–77.5% |
| Qwen3.8-Flash-Next (`ngram-mod`) | 8.98 | 4.14 | **0.3%** |

Qwen3.8-27B goes from 8.33 to 10.90 and **rises across the run** — 6.61, 8.97,
17.13 on successive repetitions as context accumulates repetition to draft from.
Mean accepted draft length reached **43.62 tokens per target forward pass**.
That is the mechanism working exactly as intended, and it is why a single cold
measurement understates `ngram-mod` badly.

**One model, not two, cleared 10 tok/s under measurement here.**

### A 12.92 figure for GLM-5.3 full that did not reproduce

Earlier notes in this repository record GLM-5.3 full at **12.92 tok/s** on an
agentic replay suite at 93.07% acceptance. Re-measured on the restored service
with the same `n_max=18 / p_min=0.75` profile, a file-reproduce replay workload
gave **6.25 tok/s** (range 5.56–6.83) at 72.8–77.5% acceptance and 5.84–7.50
accepted tokens per target pass.

The mechanism is clearly working — acceptance and draft length are both healthy —
but the throughput is less than half the recorded figure. The two runs used
different replay suites (theirs ~1,400 tokens with warm reasoning context, this
one ~500 tokens), and the 27B result above shows how strongly warm-up matters,
so this is not necessarily a contradiction. It is, however, **unreproduced**, and
the 12.92 number should not be quoted as this host's replay throughput without
re-deriving it from a published prompt set.

Treat GLM-5.3 full as **5.32 novel / 6.25 replay** until someone reproduces
better with a suite they can publish.

### Flash-Next's speculation is broken in all three mechanisms

| mechanism | acceptance |
| --- | ---: |
| `drluoto` MTP head (Q8_0, 4.14 GB) | 0.00000 |
| `dzannotti` MTP head (Q4_K_M, 2.62 GB) | 0.00000 |
| `ngram-mod`, replay workload | 0.003–0.005 |

`ngram-mod` uses **no draft model at all** — it drafts from repetition already in
the context. Scoring ~0.3% on the identical prompt where Qwen3.8-27B scores
74.3% therefore cannot be blamed on a draft artifact. Combined with two
independent MTP heads at exactly zero, this localizes the fault to the engine's
speculative path for this architecture.

Practical consequence: Flash-Next must run with `--spec-type none`. Every
speculative arm tested made it 30–55% *slower*, and it is the one model here with
no route past its bandwidth ceiling.

## Summary: what each model would need

| model | vs 10 tok/s | what it would still require |
| --- | --- | --- |
| **Qwen3.8-27B** | **met on replay** (10.90 mean / 17.13 warm); 1.20x short on novel prose | efficiency 33% -> 40%, or a draft pass that runs at bandwidth |
| GLM-5.3 full | 1.6x short on replay (6.25), 1.9x on novel prose | impossible raw at Q4 (needs 99% of peak) — speculation only, and its measured multiplier is 1.18x not the 2.4x once recorded |
| Qwen3.8-Flash-Next | 1.09x short of 10, 1.41x of 13 | at 11% of bandwidth — MoE gather path, or fix the engine's speculative path for this arch. Not bandwidth. |
| GLM-5.3-Flash | 5.0x short | per-tensor sharding rules for `glm5next` (4x), then the same MoE gather work |

**One of the four cleared the bar under measurement here** — Qwen3.8-27B, on
replay traffic. GLM-5.3 full has a recorded 12.92 that did not reproduce (see
above). The other two are each blocked on a *specific, identified* engine defect
rather than on tuning.

### Attempted: porting the GLM engine's IQ repack kernels (failed)

The GLM integration patch adds ~4,800 lines of x86 kernel work the Qwen engine
lacks, including **IQ2_XS and IQ3_XXS repack paths** — directly relevant, since
Qwen3.8-Flash-Next is 22.8 GB of IQ2_XS, a format with no repack in stock
llama.cpp and a measured 7.2 GB/s.

Extracting the five kernel files (`repack.cpp`, `repack.h`,
`arch/x86/repack.cpp`, `traits.*`) as a patch did not apply — the two trees have
different bases and the files have diverged by ~1,700 lines. Copying them
wholesale from the GLM tree **compiled cleanly**, which is misleading, and then
aborted at model load:

    repack.cpp:6515: GGML_ASSERT(size == ggml_nbytes(tensor)) failed

and it aborted with the IQ flags *off* as well, i.e. the copy broke the working
baseline rather than only the new path. The repacked-buffer size calculation
depends on surrounding definitions that also differ between the trees. Reverted;
the engine is back to 9.21 tok/s.

**A clean compile is not evidence of a successful kernel port.** Doing this
properly means porting the size/traits plumbing together with the kernels, or
rebasing the GLM work onto the Qwen engine's base commit — not copying files.
Flash-Next's speculation returns ~0% acceptance in all three mechanisms, and
`glm5next` has no tensor-parallel sharding rules so it runs on one socket of four.

There is no single shared limit — that earlier claim was an artifact of dividing
by file bytes. Corrected, the four models sit at **50% / 33% / 11% / 5%** of
achievable bandwidth, and only GLM-5.3 full is anywhere near memory-bound. The
highest-value target on this class of machine is the **MoE expert-gather path**:
Flash-Next and GLM-5.3-Flash are spending 89% and 95% of their time somewhere
other than reading weights.

### If you take one thing from this file

Quote the workload with the number. The same binary, same flags, same host, same
day gives Qwen3.8-27B **8.33 or 17.13 tok/s** depending only on whether the
prompt is novel or repetitive. Benchmarks that do not say which are unfalsifiable.
