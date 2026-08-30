# Aggregate vs per-stream: three of four models clear 10 tok/s under concurrency

Every other measurement in this repository is **single-stream**: one request at a
time, which is what one user experiences. That is the right metric for
interactive latency and the wrong one for sizing a server.

Measured with `--cont-batching` and `--parallel N`, sending N simultaneous
completions:

| model | placement | c=1 | c=2 | c=4 | c=8 |
| --- | --- | ---: | ---: | ---: | ---: |
| **Qwen3.8-Flash-Next** | 1 node, 32 thr | 5.61 | 9.64 | 14.25 | **17.50** |
| **GLM-5.3 full** | 4 `CPU-NUMA` devices | 6.50 | 9.41 | **10.05** | — |
| GLM-5.3-Flash | 1 node, 32 thr | 2.60 | — | 4.93 | 5.43 |

Figures are **aggregate** tok/s — the sum of per-request decode rates, i.e. what
the server delivers in total. Per-stream rates fall as concurrency rises:

| model | c=1 per-stream | c=8 per-stream | aggregate gain |
| --- | ---: | ---: | ---: |
| Qwen3.8-Flash-Next | 5.61 | 2.19 | **3.1x** |
| GLM-5.3-Flash | 2.60 | 0.68 | 2.1x |

**Both numbers are real and they answer different questions.** A single user on
Flash-Next at concurrency 8 sees 2.19 tok/s, which is bad. Eight users sharing
that server collectively get 17.50 tok/s, which is 3.1x what the box delivers to
one of them. Quote which one you mean.

## Why batching helps here

Decode at batch 1 sends a single row through every expert matmul, which is the
`gemv` path. Batching puts N rows through the same weight read, crossing the
three-row threshold into `gemm` and amortizing the per-operation dispatch cost
that [`fixed-per-token-overhead.md`](fixed-per-token-overhead.md) identifies as
~92 ms/token regardless of model size.

That prediction holds quantitatively. The models with the **smallest** active
bytes per token — the ones most dominated by fixed overhead — gain the most:

| model | active GB/tok | fixed cost as % of token | aggregate gain |
| --- | ---: | ---: | ---: |
| Qwen3.8-Flash-Next | 4.48 | 88% | **3.1x** |
| GLM-5.3 full | 33.97 | 50% | 1.55x |

Flash-Next, spending 88% of each token on overhead, nearly triples. GLM-5.3
full, at 50%, gains about half that. **Concurrency converts the fixed
per-token cost into useful work**, which is the same lever MoE kernel fusion
would pull, reached from the serving side instead of the kernel side.

This also qualifies an earlier claim in this repository that "batching concurrent
requests does not rescue MoE decode." That was measured on isolated kernels at
fixed batch sizes. End-to-end with continuous batching, it substantially does —
3.1x on the MoE model here. The kernel measurement was not wrong; it was
measuring only the matmul, not the dispatch overhead that dominates whole-model
decode.

## The one that still does not clear it

GLM-5.3-Flash tops out at **5.43 tok/s aggregate**. It is confined to a single
socket because `glm5next` has no tensor-parallel sharding rules
([`glm5next-tensor-parallel-blocker.md`](glm5next-tensor-parallel-blocker.md)),
so batching amortizes overhead against one socket's ~90 GB/s rather than the
host's 360. Concurrency cannot substitute for the three sockets it cannot reach.

## Practical guidance

- **Sizing a shared server**: use aggregate. GLM-5.3 full clears 10 tok/s at
  concurrency 4; Flash-Next clears it at 2 and reaches 17.5 at 8.
- **A single interactive user**: use single-stream, and set `--parallel` low.
  Raising it costs that user latency for capacity they are not using.
- **`--parallel` is not free at c=1.** Flash-Next measured 5.61 at concurrency 1
  with `--parallel 8` against 6.69 with `--parallel 1`, because the KV cache is
  divided among slots. Match `--parallel` to expected concurrency rather than
  setting it high speculatively.
- **`--parallel N` divides `--ctx-size` by N**, and that bites before the
  throughput does. GLM-5.3 full at `--ctx-size 32768 --parallel 4` gives each
  request **8192** tokens, and a long reasoning chain then overruns its slot and
  returns a 500 mid-conversation. If you want both the aggregate throughput and
  the context, raise `--ctx-size` to `N x` the per-request window you need — and
  budget the KV cache for it. A coding assistant that silently loses three
  quarters of its context is a worse outcome than a slower one, so this host's
  production profile stays at `--parallel 1`.

## Reproducing

[`tools/concurrent_bench.py`](../tools/concurrent_bench.py) sends N simultaneous
requests and reports per-stream mean, aggregate, and wall-clock throughput
separately. Wall-clock includes prompt processing and queueing and is always the
most conservative of the three.
