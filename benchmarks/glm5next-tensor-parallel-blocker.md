# Why `glm5next` cannot tensor-parallel: located to the exact tensor

GLM-5.3-Flash is the slowest model measured in this repository — **2.02 tok/s** —
and the reason is not tuning. It runs on **one socket of four** because
`--split-mode tensor` refuses the `glm5next` architecture, and one socket's
~90 GB/s against its 9.16 GB active bytes/token caps it at **9.8 tok/s even at
100% efficiency**. It cannot clear 10 tok/s without this fix.

This file records exactly what blocks it, so the work is scoped rather than
guessed at.

## The gate, and why removing it is not the fix

    error loading model: LLAMA_SPLIT_MODE_TENSOR not implemented
    for architecture 'glm5next'

from `llm_arch_supports_sm_tensor()` in `src/llama-arch.cpp`. Note that function
is an **exclusion list** — the architectures named in the switch `return false`
and everything else returns `true`, which reads backwards on first encounter.

Deleting the `LLM_ARCH_GLM5NEXT` case lets the model start loading and then
abort:

    src/llama-model.cpp:463: GGML_ASSERT(!suffix_fallback.empty()) failed

## What that assert actually means

`llama_meta_device_get_split_state()` decides, per tensor, which axis to shard.
Many tensors are not split independently but *aligned to a reference tensor*,
looked up by name as `prefix + suffix`:

```c
const ggml_tensor * tensor_axis_0 = suffix.empty()
    ? tensor
    : ud->model->get_tensor((prefix + suffix).c_str());
if (tensor_axis_0 == nullptr) {
    GGML_ASSERT(!suffix_fallback.empty());   // <-- fires here
    tensor_axis_0 = ud->model->get_tensor((prefix + suffix_fallback).c_str());
}
```

The bare assert does not say which tensor failed. Adding one log line
([`glm5next-split-state-diagnostic.patch`](../patches/glm5next-split-state-diagnostic.patch))
answers it immediately:

    SPLIT-STATE MISS tensor='blk.0.ssm_beta.weight' prefix='blk.0.'
                     suffix='ssm_out.weight' (no fallback)

The SSM split rule aligns `ssm_beta` to `ssm_out.weight`. **`glm5next` has no
`ssm_out.weight`.**

## What `glm5next` actually looks like

Layer 0 of GLM-5.3-Flash (`UD-IQ2_XXS`), read from the GGUF tensor table:

| tensor | shape | group |
| --- | --- | --- |
| `attn_q/k/v.weight` | `[4096, 8192]` | standard attention |
| `attn_output.weight` | `[8192, 4096]` | standard attention |
| `ssm_a` | `[64]` | linear attention |
| `ssm_beta.weight` | `[4096, 64]` | linear attention |
| `ssm_conv1d_{q,k,v}.weight` | `[4, 1, 8192]` | linear attention |
| `ssm_dt.bias` | `[8192]` | linear attention |
| `ssm_f_a.weight` / `ssm_f_b.weight` | `[4096, 128]` / `[128, 8192]` | linear attention |
| `ssm_g_a.weight` / `ssm_g_b.weight` | `[4096, 128]` / `[128, 8192]` | linear attention |
| `ssm_norm.weight` | `[128]` | linear attention |
| `hc_attn_{base,fn,scale}.weight` | `[24]`, `[16384, 24]`, `[3]` | hyper-connections |
| `hc_ffn_{base,fn,scale}.weight` | `[24]`, `[16384, 24]`, `[3]` | hyper-connections |
| `ffn_{up,gate,down}.weight` | `[4096, 12288]` etc. | dense FFN (layer 0) |

Three things make this not a pattern-matching fix:

1. **The SSM output path is factored.** Where the existing rules expect a single
   `ssm_out.weight`, `glm5next` has low-rank `ssm_f_a`/`ssm_f_b` and
   `ssm_g_a`/`ssm_g_b` pairs. Sharding a factored projection requires splitting
   the two halves on *opposite* axes and getting the reduction right.
2. **SSM carries recurrent state.** Splitting a state-space layer across devices
   is not the same problem as splitting a matmul — the state must either be
   replicated or partitioned consistently with the convolution and decay terms
   (`ssm_a`, `ssm_dt`, `ssm_conv1d_*`).
3. **Hyper-connections mix across the residual stream.** `hc_attn_fn` is
   `[16384, 24]` and combines branches; a naive row/column split changes the
   arithmetic.

## Why this was not attempted blind

Every one of those three can be made to *run* by adding a `suffix_fallback` and
picking an axis. None of them will be *correct* without matching the sharding to
the architecture's data flow, and a wrong choice produces plausible-looking but
silently wrong output — the worst possible failure for a serving deployment, and
directly contrary to the "without sacrificing quality" constraint this work is
under.

The gate is therefore load-bearing and correctly placed. It is a record of work
not yet done, not conservatism to be switched off.

## Scope of the actual fix

1. Add `glm5next` sharding rules for the three tensor groups above, deriving each
   axis from the graph in `build_glm5next` rather than by analogy.
2. Provide `suffix_fallback` values so the alignment lookup resolves.
3. Remove `LLM_ARCH_GLM5NEXT` from the exclusion list.
4. Validate by **exact output comparison** against the single-socket arm at
   `temperature=0` — not by "it produces text".

Expected payoff, from the measured numbers: the model currently extracts 21% of
its single socket. Four-socket tensor parallelism at the same efficiency is
~8.1 tok/s, and with the MoE gather work discussed in
[`ten-tokens-per-second.md`](ten-tokens-per-second.md) it has ~39 tok/s of
headroom before memory bandwidth binds.

Meanwhile, the one-line diagnostic patch is worth applying on its own: it turns
an opaque assert into the name of the offending tensor.
