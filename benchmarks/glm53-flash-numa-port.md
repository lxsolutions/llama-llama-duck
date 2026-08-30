# GLM-5.3-Flash: the NUMA backend ports, tensor-parallel does not

Host: 4 x Xeon Gold 6242, 64 physical cores, 755 GiB, 4 NUMA nodes, no GPU.
Measured 2026-08-29.

Model: `unsloth/GLM-5.3-Flash-GGUF`, `UD-IQ2_XXS`, revision
`2975ab414d30340466d8c51533c6e91f0cca64c1`, 4 shards, 101,844,951,808 bytes,
SHA-256 verified against the published manifest. Architecture `glm5next`.
Engine: Unsloth llama.cpp PR 27754 line, base commit `2e0e57f1`.

## Result

| configuration | decode tok/s |
| --- | ---: |
| one NUMA node, 16 threads | 1.919 |
| one NUMA node, **32 threads** | **2.015** |
| one NUMA node, 48 threads | 1.937 |
| four `CPU-NUMA` devices, tensor split | **does not load** |

This model is stuck near 2 tok/s on this host, and the reason is not a tuning
mistake. It is the slowest of the four models measured here by a wide margin.

## What worked: the backend ports cleanly

The stock `glm5next` engine reports no NUMA devices:

    GGML_CPU_NUMA_DEVICES=1 llama-server --list-devices
    Available devices:
      (none)

`patches/llama.cpp-b249-cpu-numa.patch` applies to this tree **with no fuzz**,
despite targeting a different upstream lineage, and after a rebuild the devices
appear:

    Available devices:
      CPU-NUMA0: ... / NUMA node 0 / 10 physical cores
      CPU-NUMA1: ...
      CPU-NUMA2: ...
      CPU-NUMA3: ...

So the device layer, buffer placement, worker pools, and collective are all
available. See [`../patches/PORTING.md`](../patches/PORTING.md).

## What did not: `--split-mode tensor` is gated per architecture

    error loading model: LLAMA_SPLIT_MODE_TENSOR not implemented
    for architecture 'glm5next'

The gate is `llm_arch_supports_sm_tensor()` in `src/llama-arch.cpp`. Read it
carefully — **it is an exclusion list, not an allow list**. The architectures
named in the switch `return false`; everything else `return true`. `GLM5NEXT`
is named, so it is excluded. This reads backwards on a first pass and is worth
flagging.

`GLM_DSA` is also named in the stock list, yet the GLM integration patch runs
GLM-5.3 full tensor-sharded — because that patch *implemented* the sharding for
`glm-dsa` and removed it from the exclusion list. The list is a record of work
done, not a capability table.

### The gate is load-bearing, not merely conservative

The obvious shortcut is to delete the `GLM5NEXT` case and see what happens. It
was tried, on the theory that the graph might already be shard-safe:

    llama_meta_device_get_split_state(...)
    ggml_backend_meta_alloc_ctx_tensors_from_buft(...)
    llama_model_base::load_tensors(...)
    -> ggml_abort

It aborts during `load_tensors`, before any token is produced. That is the good
failure: the split-state logic has no rule for how `glm5next`'s tensors divide,
so it refuses rather than silently sharding something incorrectly. **No
correctness risk was taken and none should be inferred as safe** — a build that
got past this point without per-architecture rules would be a build producing
quietly wrong output.

Enabling this model on the sharded path therefore requires real work:
classifying every `glm5next` tensor as split-by-row, split-by-column, or
replicated, matching what the GLM patch did for `glm-dsa`. That is not a flag.

## Why it matters more here than elsewhere

`UD-IQ2_XXS` is exactly the quant profile that this repository's kernel
measurements identify as worst-case: IQ2-family tensors are compute-bound near
7.2 GB/s and have no repack path at all. So GLM-5.3-Flash is simultaneously

- confined to one socket's ~95.3 GB/s, and
- running kernels that cannot use much of even that,

which is how a 100 GB model lands at 2 tok/s. The two fixes point in different
directions and both are open:

1. **Tensor-parallel for `glm5next`** — unlocks the other three sockets.
2. **A higher quant tier** — GLM-5.3 full measured *faster* at `UD-Q4_K_XL`
   than at compact IQ tiers on this host, roughly doubling extraction
   efficiency (22% -> 45% of available bandwidth). The same is plausible here,
   though it is not automatic: the identical hypothesis was tested on
   Qwen3.8-Flash-Next and **failed** (Q4 was 10% slower), because that model's
   Q2 file was already majority IQ4_NL. Check the type census first.

Of the two, the quant change is cheap to test and the tensor-parallel work is
not. Test the quant first.

## Reproduction notes

Thread count was swept on the single-node path and the peak is flat: 1.919 /
2.015 / 1.937 for 16 / 32 / 48 threads. There is no meaningful tuning left on
that path — 32 threads on one node is 16 physical cores plus SMT, and the model
is not thread-starved, it is bandwidth- and kernel-starved.

`--flash-attn off` is required for this architecture on this engine.
