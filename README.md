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

Adding n-gram speculative decoding gave a further +40% on the models tested.

## The big finding: CPU repack does not help MoE at batch 1

`llama.cpp` has a "repack" path (`GGML_CPU_REPACK`) that re-interleaves quantized
weights into a blocked layout for AVX-512/VNNI GEMM. It is a large win — but only
for dense 2D `MUL_MAT`. Measured with `tools/kbench_id.cpp`, same tensor, same
threads, only the op differs:

| type | op | default | repacked | speedup |
| --- | --- | ---: | ---: | ---: |
| MXFP4 | `MUL_MAT` (2D, dense) | 0.525 ms | **0.078 ms** | **6.73x** |
| MXFP4 | `MUL_MAT_ID` (3D, MoE) | 0.415 ms | 0.445 ms | **0.93x** |
| Q4_K | `MUL_MAT` (2D, dense) | 0.363 ms | **0.146 ms** | 2.48x |
| Q4_K | `MUL_MAT_ID` (3D, MoE) | 0.533 ms | 0.424 ms | 1.26x |

The tensors *are* repacked in both cases. The MoE path simply does not benefit,
and for MXFP4 it is slightly slower.

The cause is in `ggml/src/ggml-cpu/repack.cpp`, inside `forward_mul_mat_id`:

```
// If there are more than three rows in src1, use gemm; otherwise, use gemv.
```

The blocked layout exists to amortize work **across rows**. During autoregressive
decode each expert receives exactly one token — one row — so it takes the `gemv`
path and the interleaving buys nothing. During prefill each expert receives many
rows, takes `gemm`, and gets the full speedup.

This is directly observable end to end. On one MoE model, identical weights and
placement:

```
decode  (1 row/expert)    :  1.88 tok/s
prefill (512-token batch) : 41.97 tok/s   -> 22x higher per-token rate
```

**Practical consequences**

- For MoE models, "is this quant repack-eligible?" is nearly irrelevant. One model
  tested was 99.8% repack-eligible by bytes and still decoded at 1.9 tok/s,
  because 97.4% of those bytes were routed experts.
- Dense models take the 2D path for everything and get the full benefit. But
  note this does **not** mean "dense is faster" — a MoE model with small experts
  can be far faster than a dense one, because active bytes dominate. Density
  determines which kernel path you get; active bytes determine how much work
  there is.
- Speculative decoding is worth more than its acceptance rate implies: verifying
  K draft tokens puts K rows through each expert, which can cross the 3-row
  threshold and flip `gemv` into `gemm`.

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

## Tools

| file | what it does |
| --- | --- |
| `tools/membw.c` | AVX-512 memory bandwidth, independent accumulators, per-node |
| `tools/barrier.c` | OpenMP barrier cost vs thread count / socket span |
| `tools/kbench.cpp` | quantized `MUL_MAT` throughput, with and without repack |
| `tools/kbench_id.cpp` | `MUL_MAT` (dense) vs `MUL_MAT_ID` (MoE) repack comparison |
| `tools/gguf_types.py` | parse GGUF tensor types and repack-eligibility |
| `tools/prefetch-model.sh` | warm page cache before mmap load |

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
