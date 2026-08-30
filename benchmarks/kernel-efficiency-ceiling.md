# 78% of bandwidth is achievable on this CPU — mainline's NUMA path gets 33%

The most useful number measured in this repository is not a tok/s figure. It is
that **two engines on the same host, same model, same quant, extract wildly
different fractions of the bandwidth available to them** — and the more
efficient one proves how much is being left on the table.

## The measurement

Qwen3.8-27B (`Q4_0`, 16.1 GB, dense, ~16 GB active bytes/token), same host,
same day, quiet machine:

| engine | placement | tok/s | achieved GB/s | bandwidth available to it | **extraction** |
| --- | --- | ---: | ---: | ---: | ---: |
| mainline + CPU-NUMA backend | 4 devices, tensor-split | 7.42 | 119 | 360 (NUMA-local) | **33%** |
| `ik_llama.cpp` b4691 | 64 threads, `numactl --interleave=all`, `-rtr` | 6.78 | **108** | 138.9 (interleaved) | **78%** |

`ik_llama.cpp` is *slower in absolute terms* — 6.78 against 7.42 — because it has
no per-socket device backend and is therefore stuck on the interleaved path with
its 138.9 GB/s ceiling. But it converts **78%** of what it can reach into useful
work, against the NUMA backend's **33%**.

Neither number is a criticism of either project. They are optimizing different
things: one has NUMA-local placement and tensor parallelism, the other has much
better CPU kernels. **Nothing on this host currently has both.**

## Why this matters more than the tok/s

Before this comparison, "the kernels could be better" was a hypothesis. The
33% figure could plausibly have been a hardware property — memory-controller
behaviour under gather-heavy access, prefetcher defeat, something unfixable.

`ik_llama` rules that out. 78% is achievable on this exact silicon, with this
exact quant, on this exact model. The gap is software.

Applying ik_llama's demonstrated 78% to the NUMA-local ceiling of 360 GB/s gives
**281 GB/s**, and that changes every projection in this repository:

| model | active GB/tok | today | at 281 GB/s | clears 10? |
| --- | ---: | ---: | ---: | --- |
| Qwen3.8-Flash-Next | 4.48 | 9.45 | **62** | yes, hugely |
| GLM-5.3-Flash | 9.16 | 2.02 | **30** | yes |
| Qwen3.8-27B | ~16 | 8.62 | **17.6** | yes |
| GLM-5.3 full | 33.97 | 5.32 | 8.3 | **no** — still needs speculation |

Three of the four models are not short of hardware by any margin. They are short
of roughly **2.4x in kernel efficiency**, and that 2.4x has been demonstrated to
exist on this machine by a different engine.

GLM-5.3 full remains the exception in both directions: it is already the best
extractor measured here at 50%, and even perfect kernels leave it at 8.3 tok/s,
because 33.97 GB per token against 360 GB/s is simply 10.6 tok/s at the absolute
limit. It needs a working speculative multiplier, not faster matmuls.

## The catch: `ik_llama` cannot run the models that would benefit most

    general.architecture = qwen4exp
    error loading model architecture: unknown model architecture: 'qwen4exp'

    general.architecture = glm5next
    error loading model architecture: unknown model architecture: 'glm5next'

Both Flash models are too new for `ik_llama` b4691. Qwen3.8-27B (`qwen35`) is
the only one of the four it loads — and that is the one model where the NUMA
backend's placement advantage already outweighs ik's kernel advantage.

So this is not a "switch engines" recommendation. It is a measurement of the
prize.

## What the work actually is

Two routes, both real:

1. **Bring ik-class kernels to the NUMA device backend.** A wholesale file copy
   of a sibling engine's kernels was tried and failed (see
   [`ten-tokens-per-second.md`](ten-tokens-per-second.md)) — it compiled and then
   aborted at load because the size/traits plumbing differs. Doing it properly
   means rebasing, not copying.
2. **Bring per-socket devices to `ik_llama`.** The CPU-NUMA patch in this
   repository is additive and has already been shown to port cleanly across
   lineages ([`patches/PORTING.md`](../patches/PORTING.md)). This may be the
   shorter path, and would also need `qwen4exp` / `glm5next` architecture
   support to cover all four models.

Either way the target is now quantified rather than aspirational: **2.4x, and it
is known to be reachable on this CPU.**

## Reproducing

    # mainline + NUMA devices
    GGML_CPU_NUMA_DEVICES=1 GGML_CPU_NUMA_THREADS=16 GGML_CPU_NUMA_POLL=100 \
    GGML_CPU_NUMA_DIRECT_ALLREDUCE=1 GGML_CPU_NUMA_REPACK=1 \
    llama-server --model Qwen3.8-27B-Q4_0.gguf --gpu-layers 999 \
      --device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 \
      --split-mode tensor --tensor-split 1,1,1,1 --threads 16

    # ik_llama.cpp
    numactl --interleave=all \
    llama-server --model Qwen3.8-27B-Q4_0.gguf -ngl 0 --threads 64 -rtr

Divide `tok/s x active_bytes_per_token` by the bandwidth each configuration can
actually reach — 360 GB/s NUMA-local, 138.9 GB/s interleaved, both measured with
[`tools/membw.c`](../tools/membw.c). Comparing raw tok/s alone hides the whole
result.
