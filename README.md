# llama-llama-duck

Measurement tools and findings for running large language models on
**multi-socket CPU servers** with `llama.cpp`.

Everything here came out of tuning a 4-socket Intel Xeon Gold 6242 box
(64 physical cores, 755 GiB DDR4-2933, no GPU). The headline result is that
several "obvious" optimizations do nothing, one undocumented behaviour costs
~9x on Mixture-of-Experts models, and the single variable that actually
predicts throughput is **active bytes per token**.

## TL;DR — one variable predicts everything

Measured on one 4-socket box, same engine, same day:

| model | active bytes/token | decode tok/s | achieved GB/s |
| --- | ---: | ---: | ---: |
| 26B MoE, tiny experts, Q4_0 | ~1.0 GB | **16.6** | ~17 |
| 35B MoE, 3B active, Q4_K_M | ~1.7 GB | **13.7** | ~23 |
| 27B **dense**, Q4_0 | ~16 GB | 4.3 | **69** |
| 754B MoE, ~2.7 bpw | ~18.5 GB | 4.3 | **79** |
| 284B MoE, MXFP4 | ~5.7 GB | 1.9 | 11 |

> **Reading the GB/s column.** These are *non-speculative* runs, where
> `tok/s x active bytes/token` genuinely measures DRAM traffic. Do **not** apply
> that formula to a speculative run: accepted drafts emit K tokens per single
> target forward pass, so weights are not re-read per emitted token. The product
> then measures *effective output throughput*, not bandwidth, and will exceed the
> real DRAM rate by the acceptance multiple. Quote raw bandwidth only from
> speculation-off arms.

**Throughput is set by active bytes per token, not parameter count, not
dense-vs-MoE, not quant name.** The two "slow" models above have the *highest*
achieved bandwidth of anything measured — they are not inefficient, they simply
move 10-18x more bytes per token than the fast ones.

Memory ceilings on this host: **138.9 GB/s interleaved, 365 GB/s NUMA-local**
(4 x 95.3 GB/s, measured concurrently with `tools/membw.c`). So a model at
16 GB/token cannot reach 12 tok/s on the interleaved path at all — that needs
192 GB/s. Budget accordingly before choosing a model.

A useful planning rule for this class of machine: **to clear ~12 tok/s you need
roughly <=2 GB of active weights per token.**

Adding speculative decoding gave a further +40% to +90% on the models tested.
Note this raises *emitted* tokens/second without raising DRAM traffic
proportionally — it is a reduction in target forward passes per emitted token,
which is why it can carry a model past a rate its raw bandwidth alone would not
support.

## Repack helps dense matmuls far more than MoE expert matmuls

`llama.cpp`'s repack path (`GGML_CPU_REPACK`) re-interleaves quantized weights
into a blocked layout for AVX-512/VNNI. Measured with `tools/kbench_id.cpp`,
comparing **repacked against non-repacked within each op type** (the only valid
comparison — see the caveat below):

| type | op | default | repacked | speedup |
| --- | --- | ---: | ---: | ---: |
| MXFP4 | `MUL_MAT` (2D, dense) | 0.525 ms | **0.078 ms** | **6.73x** |
| MXFP4 | `MUL_MAT_ID` (3D, MoE) | 0.415 ms | 0.445 ms | **0.93x** |
| Q4_K | `MUL_MAT` (2D, dense) | 0.363 ms | **0.146 ms** | 2.48x |
| Q4_K | `MUL_MAT_ID` (3D, MoE) | 0.533 ms | 0.424 ms | 1.26x |

The tensors are repacked in both cases. Dense gains 2.5-6.7x; MoE gains
0.93-1.26x, i.e. nothing, and for MXFP4 it is marginally negative.

Sweeping batch size, MoE repack gain rises slowly and never approaches dense:

| tokens | rows/expert | default | repacked | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.19 | 0.491 ms | 0.337 ms | 1.46x |
| 8 | 1.50 | 3.583 ms | 2.588 ms | 1.38x |
| 32 | 6.00 | 13.652 ms | 8.670 ms | 1.57x |
| 64 | 12.00 | 27.430 ms | 16.593 ms | 1.65x |

Absolute time scales nearly linearly with batch, so per-token cost improves only
~1.3x across a 64x batch — more tokens simply touch more distinct experts.
**Batching concurrent requests does not rescue MoE decode.**

A plausible mechanism is the row threshold in `forward_mul_mat_id`
(`repack.cpp`): *"If there are more than three rows in src1, use gemm; otherwise,
use gemv."* The blocked layout amortizes across rows, and autoregressive decode
gives each expert one row. Note the MoE path does still call the **repacked**
`gemv` — it is not falling back to generic `vec_dot` — so this is a question of
how much the layout buys for a single row, not of the kernel being bypassed.

### Caveat: do not compare across op types with this harness

An earlier version of this document used these numbers to argue MoE is
structurally slower than dense in absolute terms. **That inference was wrong and
has been withdrawn.** The two arms do unequal work: the 2D arm computes ONE
`[4096x2048]` matmul while the 3D arm computes SIX expert matmuls
(`n_expert_used=6`). Per matmul the MoE arm is ~0.069 ms against the dense arm's
0.525 ms, so the cross-op comparison says nothing useful.

Only the **within-op** default-vs-repacked ratios above are sound. If you want an
absolute dense-vs-MoE comparison, normalize by `n_expert_used` and by the actual
FLOPs, which this harness does not currently do.

## Optimizing for draft acceptance is a trap

Sweeping `p_min` on a 27B dense model with a matched MTP draft head:

| p_min | draft acceptance | decode tok/s |
| ---: | ---: | ---: |
| **0** | 63% | **11.04** |
| 0.5 | 79% | 9.35 |
| 0.7 | 86% | 8.46 |
| 0.85 | 89% | 7.72 |

Acceptance rises monotonically while throughput falls 30%. Confidence gating
suppresses drafts that would have been accepted anyway, and the drafts it does
emit are shorter. **Aggregate tokens/second is the only valid objective**;
acceptance rate was anti-correlated with it across this entire sweep.

Draft depth wants to be shallow too: `n_max=2` beat both 1 and 3
(11.04 / 10.17 / 10.37). And for `ngram-mod`, `n_match=8` beat the default 24 by
38% on replay traffic (10.40 -> 14.32 tok/s) -- the engine warns that 8 is "too
small", but speculation is verified against the target model, so accepted tokens
are exactly what the target would have produced. The warning concerns draft
hit-rate, not output correctness.

These optima are **per model**. A different model on the same host preferred
`p_min=0.8` and much deeper drafts. Re-derive them; do not port them.

## Speculative decoding pays twice on CPU

Measured on the 26B dense model, `--spec-type ngram-mod` at 64/48/24:

| arm | novel prose | file-edit replay |
| --- | ---: | ---: |
| baseline | 16.70 / 16.74 tok/s | 16.12 tok/s |
| n-gram speculation | **21.24 / 23.73 tok/s** | **22.72 tok/s** |

Roughly +40%, with 100% draft acceptance. The usual explanation is "fewer forward
passes", but on CPU there is a second effect: verifying K draft tokens puts K
rows through each matmul. For MoE that can cross the 3-row `gemv`/`gemm`
threshold described above. For dense models it improves arithmetic intensity in
the same direction. Either way, speculation is worth more on a CPU-bound server
than its acceptance rate alone suggests.

`ngram-mod` drafts from repetition in the context, so it needs no draft model and
helps most on agentic/file-editing traffic where the model re-emits text it was
given.

## Things that turned out not to matter

Four plausible explanations were tested and eliminated. Publishing the dead ends
because they cost real time:

**NUMA page placement.** Fixing badly skewed placement (286 GB split
11.8/116/25.4/132.6 GB across four nodes) to near-perfect interleaving moved
decode from 1.70 to only 1.89 tok/s.

**Cross-socket synchronization.** Barrier cost does scale badly with socket span
(`tools/barrier.c`): 1.06 / 2.10 / 4.78 us for 1 / 2 / 4 sockets. But at roughly
1000 barriers per token that is ~4.8 ms against a measured 529 ms/token — about
1% of decode time.

**Socket count.** Pinning a model to a single socket (perfect locality, 16
threads) versus spreading over four (64 threads) changed nothing: 1.80 vs 1.88
tok/s, with the single socket fully saturated at 16.2/16 cores busy.

**Memory bandwidth.** The box sustains 365 GB/s NUMA-local aggregate. Decode was
using 22% of it.

## Per-socket tensor sharding is different from page placement

The page-placement result above applies to one ordinary CPU backend executing a
whole graph. Exposing each NUMA node as a separate llama.cpp device changes the
execution model: weights are tensor-sharded across sockets, every shard uses a
local persistent thread pool, and the Meta backend runs the shards concurrently.

An experimental Linux patch against llama.cpp build 249 produced this raw
decode result on a 31.45 GB Qwen3.8-27B GGUF:

| configuration | decode tok/s |
| --- | ---: |
| one NUMA device, 16 cores | 2.524 |
| four devices, generic Meta collective | 5.239 |
| four devices, direct F32 collective | **7.169** |

The direct path was 36.8% faster than the corrected generic four-device path
and 2.84x the one-device result. After tuning the model MTP head, a full-context
exact-output request reached 16.49 tok/s at 93.3% draft acceptance. A separate
agentic replay profile reached 18.37 tok/s on first pass and 23.58 tok/s after a
reasoning pattern repeated.

This is an opt-in experimental backend, not a claim that every model benefits.
The same measurements found a 12-core-per-socket optimum and a small regression
from huge-page advice. The patch, runtime contract, validation notes, and exact
benchmark commands are in [`patches/README.md`](patches/README.md) and
[`benchmarks/qwen38-27b-cpu-numa.md`](benchmarks/qwen38-27b-cpu-numa.md).

## A server that starts is not a server that works

A GLM-5.3 server ran for hours returning **500 on every completion** while
`/health` returned `ok` and `/v1/models` returned full metadata:

    {"code":500,"message":"decode() failed: failed to process speculative batch"}

The cause was a **composite** speculative type, `--spec-type ngram-mod,draft-mtp`,
which argument parsing accepts and startup accepts. Restarting with
`--spec-type draft-mtp` fixed it outright.

Two things made this misleading:

- **`/health` is green.** It reports that the model loaded, not that decode
  works. Health-check any serving change with an actual completion.
- **Disabling speculation per request does not dodge it.** `{"speculative":
  {"n_max": 0}}` still 500s, because the speculative context was built at
  startup. That symptom argues for a request-level bug and costs you time.

Validate `--spec-type` changes with one real request before walking away.

## Gotchas worth knowing

**`--numa distribute` overrides `numactl`.** Passing both `numactl --interleave=all`
and `--numa distribute` is *worse* than either alone — llama.cpp's NUMA init
resets the memory policy `numactl` installed. In one test 55 GB of 60 GB landed
on a single node. The correct pairing is `numactl --interleave=all` for page
placement plus `--numa numactl` ("use the CPU map provided by numactl").

**Default readahead starves mmap model loading.** With `read_ahead_kb=128` (the
kernel default) a 147 GB model streamed at 46 MB/s while 64 cores sat 97.4% idle
— purely disk-bound. Sequential parallel prefetch into page cache
(`tools/prefetch-model.sh`) took load time from ~55 minutes to under 3.

**IQ2_XS / IQ3_XXS cannot repack at all.** They are absent from
`ggml_repack_get_optimal_repack_type()`. Placing such a tensor in the repack
buffer **segfaults**: the traits lookup returns null, `extra` stays null, and
`get_tensor_traits` dereferences it.

**Quant names do not tell you the tensor types.** One "Q2_K_XL" model was 94.4%
IQ2_XS/IQ3_XXS by parameter count. `IQ4_XS` is *not* repack-eligible despite the
Q4-ish name. Parse the GGUF tensor types before committing to a download.

**Measuring memory bandwidth with a naive sum is wrong.** `s += a[i]` is a serial
floating-point dependency chain that the compiler may not vectorize without
`-ffast-math`. On this box that reports ~93 GB/s, which is exactly
`2.9 GHz / 4-cycle latency * 8 B * 16 threads` — close enough to the real figure
to be believed. Use independent accumulators or `_mm512_stream_load_si512`
(`tools/membw.c`).

## Kernel throughput by quant format

Measured with `tools/kbench.cpp` at a realistic expert shape, 16 threads pinned
to one socket (95.3 GB/s available):

| type | bpw | default | repacked | speedup | effective GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| IQ2_XS | 2.31 | 0.504 ms | n/a | — | 7.2 |
| IQ3_XXS | 3.06 | 0.377 ms | n/a | — | 12.8 |
| Q2_K | 2.62 | 0.326 ms | **0.178 ms** | 1.84x | 23.2 |
| Q4_K | 4.50 | 0.278 ms | **0.188 ms** | 1.48x | 37.7 |
| Q4_0 | 4.50 | 0.335 ms | **0.164 ms** | 2.04x | 43.1 |
| MXFP4 | 4.25 | 0.321 ms | **0.133 ms** | 2.42x | 33.5 |

IQ2_XS sustains 7.6% of available bandwidth. Q2_K repacked is 2.83x faster at
essentially the same footprint (2.62 vs 2.31 bpw).

Socket scaling is shape-dependent and can be negative — always measure:

| shape / format | 16 thr / 1 socket | 64 thr / 4 sockets |
| --- | ---: | ---: |
| MXFP4 `[4096x2048]` | **0.069 ms** | 0.201 ms (2.9x worse) |
| Q2_K `[6144x2048]` | 0.124 ms | **0.101 ms** (1.23x better) |

## Repack eligibility by type is necessary, not sufficient

`tools/gguf_types.py` reports which tensor types can use the repack path. That is
a useful screen, but it **overestimates** what actually happens, because the
`_8x8` traits also require shape constraints (`ne[1] % 8 == 0`). Two builds of the
same 27B model:

| build | type census says | actually repacked (RssAnon/total) | tok/s |
| --- | ---: | ---: | ---: |
| dynamic `UD-Q4_K_M` | 68% eligible | **~25%** (4.9 of 16.5 GB) | 2.7 |
| plain `Q4_0` | 97% eligible | **~75%** (12.1 of 16.0 GB) | 4.3 |

The dynamic Q4_K_M build was *slower than the Q8 build of the same model* despite
moving half the bytes. Uniform `Q4_0` is the format that repacks most reliably.

**Ground truth is `RssAnon` vs `RssFile` in `/proc/<pid>/status` during load**, not
a type census. With mmap, repacked tensors become anonymous pages while
un-repacked ones stay file-backed.

## Thread count is per-model and can collapse

Sweeping a 27B dense model, same build, same host:

| threads | decode tok/s |
| ---: | ---: |
| 16 | 3.40 |
| 32 | 3.98 |
| 48 | **4.33** |
| 64 | **1.44** |

A 3x cliff between 48 and 64 threads on a 64-core machine. Matrix ops below a
certain size parallelize negatively — dispatch and cross-socket cache traffic
cost more than the arithmetic. Always sweep; never assume all cores is best.
If you are A/B-ing anything else, pin thread count across arms or this will
swamp your result.

## The most important result here is a wrong-answer bug, not a speedup

`--split-mode tensor` **silently corrupts** output on the `qwen4exp`
architecture (Qwen3.8-Flash-Next). Same binary, same model file, same prompt,
`temperature=0`:

| configuration | "What is 17 * 23?" |
| --- | --- |
| one NUMA node | `17 × 23 = 391` |
| four devices, tensor split | `ggio **.** (!(( (!(( (!(( (!((...` |

No error, no warning, no crash — and the corrupt path is **~45% faster**, so a
throughput-driven tuning campaign selects for it. This one did, across a dozen
sweeps, and every Flash-Next number previously published here was void as a
result.

`llm_arch_supports_sm_tensor()` is an **exclusion list**: `glm5next` is named and
correctly refused, `qwen4exp` is not named and is therefore silently permitted.
Default-allow is the wrong polarity when the failure mode is fluent nonsense.
Qwen3.8-27B (`qwen35`) and GLM-5.3 full (`glm-dsa`) were re-verified and are
correct.

Full evidence:
[`benchmarks/qwen4exp-tensor-split-corruption.md`](benchmarks/qwen4exp-tensor-split-corruption.md).
Run [`tools/quality_probe.py`](tools/quality_probe.py) against any configuration
before trusting its tok/s.

## Four models on one host, and what actually moved each one

All four measured on the same 4-socket box the same day, same harness, decode
tokens/second at `temperature=0`:

| model | before | after | what changed |
| --- | ---: | ---: | --- |
| Qwen3.8-Flash-Next (176.9B MoE, Q2_K_XL) | 6.68 | **6.68** | no change — the faster 4-device path **silently corrupts output** |
| Qwen3.8-27B (dense) | 6.12 | **8.62** prose / ~10.9 replay | Q8_K_XL -> Q4_0; MTP at depth 2; ngram-mod on replay |
| GLM-5.3 full (Q4_K_XL, ~467 GB) | **0** | **5.32** prose / **6.25** replay | removed a composite `--spec-type` that failed every request |
| GLM-5.3-Flash (IQ2_XXS, ~102 GB) | — | **2.02** | blocked: no tensor-parallel for `glm5next` |

Chasing a **10 tok/s** bar across all four produced two results that matter more
than any tok/s figure.

**A fixed ~92 ms per token, independent of model size.** Subtracting the
bandwidth-limited portion from measured decode time leaves 94 / 91 / 92 ms for
GLM-5.3 full, Qwen3.8-27B and Qwen3.8-Flash-Next — three models spanning **7.6x
in active bytes per token**. Flash-Next spends **88% of every token not reading
weights**, which is why a dozen configuration knobs each moved it 2–3%. Details
and caveats: [`fixed-per-token-overhead.md`](benchmarks/fixed-per-token-overhead.md).

**And the gap is software, not silicon.** On the same host, same
model, same quant, `ik_llama.cpp` extracts **78%** of the bandwidth available to
it while mainline's CPU-NUMA path extracts **33%**. That rules out a hardware
explanation for the gap: **2.4x is demonstrably reachable on this silicon.**

Applied to the 360 GB/s NUMA-local ceiling, that would put Flash-Next at ~62,
GLM-5.3-Flash at ~30, and Qwen3.8-27B at ~17.6 tok/s. GLM-5.3 full stays the
exception — at 33.97 GB per token its absolute ceiling is 10.6 tok/s no matter
how good the kernels get, so it needs a working speculative multiplier rather
than faster matmuls.

Full arithmetic, the exhausted-knob table, and what each model would still
need: [`benchmarks/ten-tokens-per-second.md`](benchmarks/ten-tokens-per-second.md).

The largest single win was not a kernel — it was noticing a model was pinned to
one of four sockets. The second largest was noticing a server that answered
`/health` with `ok` was failing 100% of completions.

Neither would have been found by tuning. Both were found by checking what the
running system was actually doing.

## Check what fraction of the machine your model is actually on

The single largest win measured in this repository was not a kernel or a patch.
It was noticing that a model's launch profile pinned it to **one** of four
sockets, capping it at ~95.3 GB/s of the host's ~365 GB/s.

Qwen3.8-Flash-Next (176.9B MoE, `UD-Q2_K_XL`, 78.8 GB):

| configuration | decode tok/s |
| --- | ---: |
| one NUMA node, 32 threads | 6.68 |
| four `CPU-NUMA` devices, 16 threads/node | 7.71 |
| four `CPU-NUMA` devices, **12 threads/node** | **8.98** |

**+34%**, from a launch-flag change. Single-node pinning is often introduced as
a *correctness* workaround during bring-up — to dodge an allocation failure or
an unvalidated code path — and then never revisited once it works. Audit for it
before optimizing anything else. `--list-devices` with `GGML_CPU_NUMA_DEVICES=1`
tells you whether the sharded path is even available in your build.

Note the ceiling did not move: at 8.98 tok/s the sharded arm still extracts only
~30% of 365 GB/s, where the single-node arm was near its own limit. Sharding
moved the bottleneck off memory and onto the collective and the kernels. That is
where the next win is, and it is a harder one.

Details: [`benchmarks/qwen38-flash-next-numa.md`](benchmarks/qwen38-flash-next-numa.md).

## Check draft acceptance before optimizing anything about the draft

A published MTP head for Qwen3.8-Flash-Next scored **0.00000 acceptance
(0 accepted / 198 generated)**. Speculation was therefore pure overhead:

| arm | decode tok/s |
| --- | ---: |
| no speculation | **8.98** |
| `draft-mtp` `n_max=2`, draft across four devices | 3.13 |
| `draft-mtp` `n_max=2`, draft pinned to one node | 3.17 |
| `draft-mtp` `n_max=4` | 2.14 |

This retires an idea previously listed here as promising-but-untried: pinning a
small draft model to a single socket so it does not pay cross-socket collective
cost. Measured, it changed nothing (3.17 vs 3.13) — because when acceptance is
zero, no amount of draft placement matters.

`acceptance == 0` and `acceptance` merely *low* look identical in a throughput
number but have completely different fixes. The server prints it; read it first:

    draft acceptance = 0.00000 (0 accepted / 198 generated), mean len = 1.00

Related but distinct from the earlier finding that *maximizing* acceptance is a
trap. Both hold: do not tune for acceptance, but do check it is nonzero.

## `ngram-mod` is workload-scoped, not a default

| model / workload | baseline | `ngram-mod` |
| --- | ---: | ---: |
| 26B dense, file-edit replay | 16.12 | **22.72** |
| 176.9B MoE, novel prose | **8.98** | 4.88 |

Same flag, +41% and −46%. `ngram-mod` drafts from repetition already in the
context; novel generation has none, so it pays verification on drafts that are
then rejected. Ship it as a per-request option, never as a server default.

## The "go up to Q4" rule does not generalize

GLM-5.3 measured *faster* at `UD-Q4_K_XL` than at its compact Q2/Q3 tiers,
because those tiers were dominated by IQ2_XS/IQ3_XXS — compute-bound at
~7.2 GB/s and not repack-eligible. It is tempting to generalize.

Qwen3.8-Flash-Next, same host, same engine family:

| quant | file bytes | decode tok/s |
| --- | ---: | ---: |
| `UD-Q2_K_XL` | 78.8 GB | **8.98** |
| `UD-Q4_K_XL` | 111.3 GB | 8.07 |

Q4 is 10% **slower** here. The difference is the type census, not the tier name:
Flash-Next's "Q2" file is 51.7% IQ4_NL, which repacks fine, so it never paid the
compute-bound penalty that made GLM's Q3 slow — leaving Q4 with 41% more bytes
per token and no kernel efficiency to recover it.

Run `tools/gguf_types.py` before predicting what a quant change will do. The
tier name in the filename does not tell you which kernels will run.

## Published engineering bundles

The repository now includes the source deltas and measured launch contracts,
not just the standalone diagnostic tools:

| Area | Artifact |
| --- | --- |
| llama.cpp source work | [`three pinned, checksummed patch bundles`](patches/README.md) |
| upstream-derived code provenance | [`patch attribution`](patches/ATTRIBUTION.md) |
| ready-to-edit server launchers | [`SR950 profiles`](profiles/README.md) |
| Qwen3.8-27B raw and speculative results | [`focused CPU-NUMA benchmark`](benchmarks/qwen38-27b-cpu-numa.md) |
| GLM-5.3 full/Flash and Qwen3.8 Flash Next | [`model tuning results`](benchmarks/sr950-model-profiles.md) |
| Qwen3.8-Flash-Next sharding sweep + 3 dead ends | [`Flash-Next NUMA benchmark`](benchmarks/qwen38-flash-next-numa.md) |
| GLM-5.3 full: composite-spec failure and recovery | [`composite spec failure`](benchmarks/glm53-full-composite-spec-failure.md) |
| GLM-5.3-Flash: backend ports, tensor-parallel does not | [`Flash NUMA port`](benchmarks/glm53-flash-numa-port.md) |
| Moving the CPU-NUMA backend to another branch | [`porting guide`](patches/PORTING.md) |
| What a 10 tok/s target actually requires | [`ten tok/s analysis`](benchmarks/ten-tokens-per-second.md) |
| Why `glm5next` cannot tensor-parallel (exact tensor) | [`glm5next TP blocker`](benchmarks/glm5next-tensor-parallel-blocker.md) |
| 78% extraction is achievable; mainline NUMA gets 33% | [`kernel efficiency ceiling`](benchmarks/kernel-efficiency-ceiling.md) |
| A fixed ~92 ms/token, invariant across a 7.6x byte range | [`fixed per-token overhead`](benchmarks/fixed-per-token-overhead.md) |
| **Tensor split silently corrupts `qwen4exp`** | [`qwen4exp corruption`](benchmarks/qwen4exp-tensor-split-corruption.md) |

Patch bundles target exact upstream commits and are intentionally separate
where their source bases differ. Start with [`patches/README.md`](patches/README.md)
before applying or rebasing them.

## Tools

| file | what it does |
| --- | --- |
| `tools/membw.c` | AVX-512 memory bandwidth, independent accumulators, per-node |
| `tools/barrier.c` | OpenMP barrier cost vs thread count / socket span |
| `tools/kbench.cpp` | quantized `MUL_MAT` throughput, with and without repack |
| `tools/kbench_id.cpp` | `MUL_MAT` (dense) vs `MUL_MAT_ID` (MoE) repack comparison |
| `tools/gguf_types.py` | parse GGUF tensor types and repack-eligibility |
| `tools/prefetch-model.sh` | warm page cache before mmap load |
| `tools/decode_bench.py` | decode tok/s from the server's own timings, plus draft acceptance |
| `tools/sweep_config.sh` | one arm = one fresh server; prints NUMA placement and RssAnon/RssFile |
| `tools/quality_probe.py` | four deterministic prompts with known answers — run before trusting any tok/s |
| `tools/active_bytes.py` | active bytes/token from the GGUF tensor table — scales experts by n_used/n_expert and excludes gathered embeddings |

`decode_bench.py` reads `predicted_per_second` out of the server's `timings`
block rather than timing the HTTP round trip, so client latency and prompt
processing do not contaminate the decode number. It also surfaces
`draft_acceptance_rate` — check it is nonzero before believing any speculative
result.

`sweep_config.sh` gives each arm a fresh process. A server that has already
generated tokens beats a cold one, so reusing a process across arms flatters
whichever ran second. It also dumps page placement and `RssAnon` vs `RssFile`,
which is how you confirm a sharded run actually copied weights into node-bound
memory instead of leaving them on the shared mmap.

Build instructions are at the top of each file. The C++ tools link against a
`llama.cpp` build's `libggml*.so`.

## Method notes

Verify repack is actually engaging: the `repack tensor ... _8x8` log line is
`GGML_LOG_DEBUG` and does **not** appear at normal server verbosity, so grepping
logs proves nothing. Instead compare `RssAnon` and `RssFile` in
`/proc/<pid>/status` during load — with mmap they grow in lockstep, because the
source is read from the mapping while the repacked destination is written to an
anonymous buffer.

Verify placement with `/proc/<pid>/numa_maps`, separating anonymous from
file-backed pages. Aggregate `numastat -p` conflates them and is misleading: in
one run it showed a 1.93x spread while the anonymous destination that actually
matters was even within 4%.

Throughput on this class of machine is non-deterministic at `temperature=0`.
Identical requests produce identical answer content but different reasoning
lengths, because parallel floating-point reduction order varies with thread
scheduling. Benchmark by aggregate tokens/second, never by comparing output
length, and treat single samples as noise.

## License

MIT
