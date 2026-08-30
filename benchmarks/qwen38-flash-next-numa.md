# Qwen3.8-Flash-Next on a 4-socket CPU: +34% from tensor-sharding, and three dead ends

Host: Lenovo SR950, 4 x Xeon Gold 6242 (Cascade Lake), 64 physical cores,
755 GiB DDR4-2933, 4 NUMA nodes, no accelerator. Measured 2026-08-29.

Model: `unsloth/Qwen3.8-Flash-Next-GGUF`, `UD-Q2_K_XL`, 3 shards, 78.8 GB,
176.9B parameters (MoE). Engine is the Qwen3.8-Flash-Next support branch with
the `CPU-NUMA` device backend compiled in.

All numbers are the server's own `predicted_per_second` (emitted generation
tokens/second) at `temperature=0`, `seed=42`, `cache_prompt=false`, 200 tokens,
2 repetitions per workload. Run-to-run spread inside a configuration was under
1% except where noted.

## Headline

| configuration | decode tok/s |
| --- | ---: |
| one NUMA node, 32 threads, `--load-mode none` (prior published profile) | 6.68 |
| four `CPU-NUMA` devices, 16 threads/node, no repack | 7.51 |
| four `CPU-NUMA` devices, 16 threads/node, repack | 7.71 |
| four `CPU-NUMA` devices, **12 threads/node**, repack | **8.98** |

**+34.4% over the single-node profile.** The published profile for this model
pinned it to one socket, which caps it at that socket's ~95.3 GB/s of the
host's ~365 GB/s aggregate. Tensor-sharding across four devices lifts the cap.

## Thread count per node is the dominant knob, and 12 is the optimum

Four devices, repack on, everything else fixed:

| threads/node | total threads | decode tok/s |
| ---: | ---: | ---: |
| 8 | 32 | 8.48 |
| 10 | 40 | 8.10 |
| **12** | **48** | **8.98** |
| 14 | 56 | 7.76 |
| 16 | 64 | 7.71 |

Going from 12 to 16 threads per node costs **14%**, despite using 33% more
cores. The plateau is wide and flat on the low side (8 threads still gives
8.48) and falls off sharply on the high side. **When in doubt, undersubscribe** —
the cost of guessing low is small, the cost of guessing high is not.

Do not port the number. A 12-thread optimum was also reported for
Qwen3.8-27B, which invites treating 12 as a host constant, but the 27B
measurement below peaks at 16 on this quant. The reproducible finding is the
*shape* — a broad low plateau with a sharp high-side falloff — not the location
of the peak.

Note the non-monotonicity at 10 threads. It was reproducible across repeats and
is not explained here.

## Sharding is working — placement is not the story

Page placement measured from `/proc/<pid>/numa_maps` during generation was
even in every four-device arm:

    N0=13.7GB  N1=14.3GB  N2=13.8GB  N3=13.7GB

`RssAnon` was ~54.6 GB against `RssFile` ~0.2 GB, confirming the weights are
copied into node-bound anonymous memory rather than left on the shared mmap.
There was no placement skew to fix; the gain came from parallel execution.

Even so, at 8.98 tok/s the four-device arm extracts only about 30% of the
365 GB/s the host can sustain. The single-node arm was closer to its own
ceiling. **The bottleneck moved from bandwidth to something else when the model
was sharded** — the remaining headroom is in the collective and the kernels,
not in memory.

## Dead end 1: the MTP draft head gives exactly zero accepted tokens

`drluoto/Qwen3.8-Flash-Next-MTP-GGUF` (`mtp-Qwen3.8-Flash-Next-Q8_0.gguf`,
4,142,897,248 bytes, SHA-256
`b9880220df29fc224bbce408c867cd5d9c021263b754033ea624b669e374f4ec`) was listed
as an unbenchmarked candidate. It is now benchmarked:

| arm | decode tok/s |
| --- | ---: |
| no speculation | **8.98** |
| `draft-mtp`, `n_max=2`, draft on all four devices | 3.13 |
| `draft-mtp`, `n_max=2`, draft pinned to one node | 3.17 |
| `draft-mtp`, `n_max=4`, draft on all four devices | 2.14 |

The server's own accounting explains it:

    draft acceptance = 0.00000 (0 accepted / 198 generated), mean len = 1.00

**Zero accepted drafts out of 198.** Every draft token is pure overhead, which
is why deeper drafts are monotonically worse. The engine also logged, once per
sequence:

    spec common_specu: backend offload failed for seq_id=0; using CPU sampler

This head is either mismatched to this model revision or unsupported by this
engine path. Do not pair them without re-checking acceptance first.

**This also closes an open question.** Pinning the draft to a single node was
listed as an untried idea on the theory that a small draft model pays
cross-socket collective cost it cannot amortize. Tried: 3.17 vs 3.13 tok/s,
i.e. nothing. When acceptance is zero, draft placement cannot matter — check
acceptance before optimizing draft placement.

## Dead end 2: `ngram-mod` is catastrophic on novel generation

| arm | decode tok/s |
| --- | ---: |
| no speculation | **8.98** |
| `ngram-mod`, `n_match=8`, `n_min=48`, `n_max=64` | 4.88 |

**−46%.** `ngram-mod` drafts from repetition already present in the context. On
novel prose there is nothing to draft from, so it pays verification cost on
drafts that are then rejected.

This does not contradict the +40% previously measured for `ngram-mod` on a
26B model — that gain was on agentic/file-edit replay traffic, where the model
re-emits text it was given. It does mean **`ngram-mod` must not be enabled as a
global default.** It is a workload-scoped option, and the workload has to
actually be repetitive.

## Dead end 3: Q4 is *not* faster here — the quant rule does not generalize

On this same host, GLM-5.3 measured **faster** at `UD-Q4_K_XL` than at the
compact Q2/Q3 tiers, because its low tiers were dominated by IQ2_XS/IQ3_XXS,
whose LUT dequant is compute-bound at ~7.2 GB/s and cannot repack. That result
tempts a general rule: "on this box, go up to Q4."

It does not hold:

| quant | file bytes | threads/node | decode tok/s |
| --- | ---: | ---: | ---: |
| `UD-Q2_K_XL` | 78.8 GB | 12 | **8.98** |
| `UD-Q4_K_XL` | 111.3 GB | 12 | 8.07 |
| `UD-Q4_K_XL` | 111.3 GB | 16 | 7.36 |

Q4 is **10% slower**. The type census explains the difference. Flash-Next's
`UD-Q2_K_XL` is not an IQ2-dominated file:

| type | params | share | GB | repack? |
| --- | ---: | ---: | ---: | --- |
| IQ4_NL | 91,465,564,160 | 51.69% | 51.4 | YES |
| IQ2_XS | 78,852,915,200 | 44.56% | 22.8 | no |
| Q5_K | 2,823,946,240 | 1.60% | 1.9 | YES |
| others | | 1.15% | 2.7 | mixed |

The majority of the bytes are already IQ4_NL, which repacks. So this quant does
not carry the compute-bound penalty that made GLM's Q3 slow, and moving to Q4
buys no kernel efficiency while costing 41% more bytes per token.

**Parse the tensor-type census before predicting what a quant change will do.**
`tools/gguf_types.py` does this. Quant tier names do not predict throughput;
the type mix does.

Q4 remains the better *quality* artifact for a 176.9B model, and 10% is a
defensible price for it. That is a quality decision, not a speed one.

## Recommended configuration

    GGML_CPU_NUMA_DEVICES=1
    GGML_CPU_NUMA_THREADS=12
    GGML_CPU_NUMA_POLL=50
    GGML_CPU_NUMA_DIRECT_ALLREDUCE=1
    GGML_CPU_NUMA_REPACK=1

    --device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3
    --split-mode tensor --tensor-split 1,1,1,1
    --gpu-layers 999 --load-mode mmap --fit off
    --threads 12 --threads-batch 12
    --cache-type-k f16 --cache-type-v f16
    --spec-type none

See [`profiles/launch-qwen38-flash-next.sh`](../profiles/launch-qwen38-flash-next.sh).

## Qwen3.8-27B, same host, for comparison

`Qwen3.8-27B-UD-Q8_K_XL`, 31.4 GB, four devices, `q8_0` KV, flash attention on:

| configuration | decode tok/s |
| --- | ---: |
| 12 threads/node, no repack | 6.03 |
| 12 threads/node, repack | 5.94 |
| **16 threads/node, repack** | **6.12** |
| 16 threads/node, `f16` KV, `--flash-attn off` | fails to load |

Repack is within noise on this model and this quant, and unlike Flash-Next the
optimum sits at 16 rather than 12 threads per node. **The thread optimum is not
a host constant after all — re-sweep it per model.** The single safe
generalization is the shape of the curve, not the location of its peak.
