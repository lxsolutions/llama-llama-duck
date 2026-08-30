# A fixed ~92 ms per token, independent of model size

Working backwards from measured decode times on one 4-socket host, using
**corrected active bytes/token** and the **measured** 360 GB/s NUMA-local
ceiling:

| model | tok/s | ms/token | active GB/tok | ms if bandwidth-limited | **residual** |
| --- | ---: | ---: | ---: | ---: | ---: |
| GLM-5.3 full | 5.32 | 188 | 33.97 | 94 | **94** |
| Qwen3.8-27B | 7.42 | 135 | ~16 | 44 | **91** |
| Qwen3.8-Flash-Next | 9.65 | 104 | 4.48 | 12 | **92** |

Three models spanning **7.6x in active bytes per token** — 4.48 GB to 33.97 GB —
across two engine builds, three architectures (MoE 8/256, dense, MoE 10/512), and
three quant families, all leave the same **~92 ms** residual once the
bandwidth-limited portion is subtracted.

That is not a coincidence and it reframes everything else in this repository.

## What it means

Decode time decomposes roughly as `bytes / bandwidth + fixed_cost`, and on this
host `fixed_cost ≈ 92 ms`. The fraction of a token spent *not* moving weights:

| model | bandwidth-limited | fixed cost | fixed cost as % of token |
| --- | ---: | ---: | ---: |
| GLM-5.3 full | 94 ms | 94 ms | 50% |
| Qwen3.8-27B | 44 ms | 91 ms | 67% |
| Qwen3.8-Flash-Next | 12 ms | 92 ms | **88%** |

**Qwen3.8-Flash-Next spends 88% of every token not reading weights.** This is
why it sat at 11% of achievable bandwidth, and why every configuration knob —
threads, poll, hugepages, repack, load-mode, mlock, device count, quant, KV
type, flash-attention, batch geometry — moved it only a few percent each. They
were all competing for the 12% of the token that is actually memory traffic.

If the fixed cost went to zero:

| model | today | bandwidth-limited only |
| --- | ---: | ---: |
| Qwen3.8-Flash-Next | 9.65 | **83** |
| Qwen3.8-27B | 7.42 | **22.7** |
| GLM-5.3 full | 5.32 | **10.6** |

It will not go to zero. But even halving it clears 10 tok/s on three of the four
models, and it explains why GLM-5.3 full is the stubborn one twice over: it has
both the highest byte cost *and* no room underneath it.

## Honest caveats

**The residual is an upper bound on "overhead", not a pure measurement of it.**
The 360 GB/s figure comes from `tools/membw.c` doing large sequential
non-temporal reads. Real decode gathers scattered expert slices and reads
quantized blocks that must be dequantized, so some of the 92 ms is genuinely
memory traffic running below streaming rate, not dispatch cost. The decomposition
says "not explained by peak bandwidth", which is weaker than "pure overhead".

What makes it interesting is the **invariance**, not the absolute value. A
residual that were mostly sub-peak bandwidth behaviour should scale with bytes
moved; it does not. It is flat across a 7.6x range.

**It is not the server layer.** `llama-bench` on Flash-Next, single device,
12 threads, measured 5.22–5.72 tok/s against the HTTP server's 6.68 on the same
single-node configuration. The server is not adding the cost; if anything the
server path measured slightly faster.

**It does shrink with more devices, but sub-linearly.** Flash-Next single-device
is ~182 ms/token (132 ms residual); four devices is 104 ms/token (92 ms
residual). Four times the compute removed 30% of the fixed cost. That is the
signature of per-operation cost that partially parallelizes, not of a serial
section.

## Where to look

Flash-Next issues roughly `10 experts x 48 layers = 480` routed expert matmuls
per token, each `[2560 x 640]`, then sharded four ways to about `[640 x 640]`
per device, through `MUL_MAT_ID`'s single-row `gemv` path. GLM-5.3 full issues
`8 x 78 = 624`. Per-operation cost times hundreds of operations is the shape
that produces a size-independent constant.

This is consistent with, and sharper than, the earlier observation that MoE
repack buys ~1.0–1.26x against dense's 2.5–6.7x: the expert path is not limited
by how fast its arithmetic runs.

The two concrete targets, in order:

1. **Fuse the MoE expert path** so a token issues tens of operations rather than
   hundreds. The sibling GLM engine carries `GGML_CPU_MOE_GATE_UP_FUSION`,
   `GGML_CPU_MOE_WEIGHTED_SUM_FUSION` and
   `GGML_CPU_MOE_DOWN_WEIGHTED_SUM_FUSION`; the Qwen engine has none of them.
2. **Reduce per-op cross-device cost**, since sharding multiplies operation count
   by the device count while dividing each one's work.

Combined with the separately measured fact that `ik_llama.cpp` extracts 78% of
its available bandwidth where this path extracts 33%
([`kernel-efficiency-ceiling.md`](kernel-efficiency-ceiling.md)), the picture is
consistent: **the CPU MoE decode path on this class of machine is
operation-count-bound, not bandwidth-bound.**
