# What it takes to reach 10 tok/s on a 4-socket CPU

Target: **>10 decode tok/s** on four models, one host — 4 x Xeon Gold 6242,
64 physical cores, 755 GiB DDR4-2933, 4 NUMA nodes, no accelerator. Measured
bandwidth ceiling: **365 GB/s** NUMA-local, 138.9 GB/s interleaved.

This file records how close each model got, and — more usefully — the arithmetic
that says which of them *can* get there and what would have to change.

## Where the four models landed

Decode tok/s, quiet host, `temperature=0`, server-reported `predicted_per_second`:

| model | quant | bytes/token | start | best | ceiling at 365 GB/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B | Q4_0, 16.1 GB | ~16 GB | 6.12 | **8.33** | 22.8 |
| Qwen3.8-Flash-Next | Q2_K_XL, 78.8 GB | ~14 GB | 6.68 | **8.98** | ~26 |
| GLM-5.3 full | Q4_K_XL, 467 GB | ~36 GB | 0 (broken) | **5.32** general / **12.92** replay | **10.1** |
| GLM-5.3-Flash | IQ2_XXS, 102 GB | ~14 GB | — | **2.02** | ~26 (locked) |

## The arithmetic nobody can argue with

Decode at batch 1 is bandwidth-bound: `tok/s = bandwidth / bytes-per-token`. So
the target sets a hard requirement before any tuning:

| model | GB/s needed for 10 tok/s | as % of 365 GB/s |
| --- | ---: | ---: |
| Qwen3.8-Flash-Next | 140 | 38% |
| Qwen3.8-27B | 160 | 44% |
| GLM-5.3-Flash | 140 | 38% (but see below) |
| **GLM-5.3 full** | **360** | **99%** |

**GLM-5.3 full cannot reach 10 tok/s non-speculatively at Q4_K_XL.** Not with
better kernels, not with better placement — 10 tok/s needs 99% of a bandwidth
figure that a *pure streaming benchmark* only just achieves. Any claim of >10
tok/s on this model is either speculative decoding, a smaller quant, or a
measurement error.

Speculation is the legitimate way past it, and it works: 12.92 tok/s on agentic
replay at 93% draft acceptance, because K accepted tokens are emitted per single
target forward pass. That is a real >10 result, on the traffic it was tuned for.
It does not transfer to novel prose, where the same model runs 5.32.

**State the workload with the number, or the number means nothing.**

## Measured extraction efficiency, and the gap that is actually available

| model | tok/s | achieved GB/s | % of 365 |
| --- | ---: | ---: | ---: |
| Qwen3.8-27B Q4_0, raw | 7.42 | 119 | 33% |
| Qwen3.8-Flash-Next, raw | 8.98 | ~126 | 35% |
| GLM-5.3 full, raw | 3.78 | 136 | 37% |

All three land near **one third** of streaming bandwidth. That consistency,
across three models, two engines, and three quants, says the limit is structural
to the decode path rather than specific to any model.

For reference, this repository's isolated kernel benchmark measured repacked
Q4_0 at 43.1 GB/s against 95.3 GB/s available on one socket — **45%**. So the
kernels alone reach 45% and whole-model decode reaches 33%; the missing ~12
points are the rest of the graph (attention, norms, KV, collective).

Closing 33% -> 45% would put Qwen3.8-27B at 10.2 tok/s raw and Flash-Next near
12. **That is where the remaining headroom is, and it is kernel and graph work,
not configuration.** Every configuration knob below was swept and is exhausted.

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

## Summary: what each model would need

| model | gap to 10 tok/s | what it actually requires |
| --- | --- | --- |
| Qwen3.8-Flash-Next | 1.11x | graph/kernel efficiency 35% -> 39%; or a working MTP path |
| Qwen3.8-27B | 1.20x | efficiency 33% -> 40%, or a draft pass that runs at bandwidth |
| GLM-5.3 full | met on replay (12.92); 1.9x on novel prose | impossible raw at Q4 (needs 99% of peak) — speculation only |
| GLM-5.3-Flash | 5.0x | per-tensor sharding rules for `glm5next` (4x), then quant work |

Three of the four are gated on the same thing: **whole-model decode extracts
~33% of streaming bandwidth where the kernels alone reach 45%.** That is one
problem, not four, and it is the highest-value target on this class of machine.
