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

## Correction: it is NOT the routed-expert matmuls

An earlier revision of this file concluded that the fixed cost was
per-operation dispatch over the hundreds of routed-expert matmuls, and
recommended MoE expert fusion as the fix. **That was tested directly and is
wrong.**

`--override-kv qwen4exp.expert_used_count` changes how many experts each token
routes to, and therefore how many expert matmuls are issued per token. On
Qwen3.8-Flash-Next, single node, correct (non-sharded) path, all other settings
fixed:

| experts used | expert matmuls/token | tok/s | vs baseline |
| ---: | ---: | ---: | ---: |
| 10 (default) | 480 | 3.440 | — |
| 6 | 288 | 3.592 | +4.4% |
| 4 | 192 | 3.720 | **+8.1%** |

**Cutting expert matmuls by 60% buys 8%.** If the ~92 ms residual were dominated
by per-expert dispatch, removing three fifths of those operations would have
removed a large fraction of it. It did not.

(Absolute values here are depressed relative to the 6.69 tok/s measured
elsewhere for this configuration, because the page cache was cold after many
model loads. The arms were run back-to-back under identical conditions, so the
*comparison* holds even though the levels are low. Output remained correct at
all three expert counts.)

So the fixed cost lives somewhere other than the routed-expert path — candidates
are the dense attention block (1.80 GB/token, 40% of this model's active
bytes), per-token graph dispatch that does not scale with expert count, or
sampling and token bookkeeping. **Fusing the MoE expert path would not have
delivered the 2.4x**, and this repository recommended it before testing it.

Anyone attacking this should profile per-operation time inside a single decode
step before choosing a target. The byte accounting says where the *bandwidth*
goes; it does not say where the *time* goes, and on this host those are
different questions.

## Profiled: attention dominates, experts do not appear

Rather than infer further, the dormant `GGML_CPU_OP_PROFILE` machinery in the
GLM engine was enabled (`GGML_CPU_OP_PROFILE='*'` — note the filter is an
op-name list and `*` is the only wildcard; `all` silently matches nothing) and a
decode step captured on GLM-5.3 full.

By operation type within the captured graph:

| op | count | % of captured time |
| --- | ---: | ---: |
| `MUL_MAT` | 11 | **87.4%** |
| `CONCAT` | 1 | 3.4% |
| `LIGHTNING_INDEXER` | 1 | 3.3% |
| `RMS_NORM` | 2 | 2.3% |
| `FLASH_ATTN_EXT` | 1 | 2.2% |
| `TOP_K` | 1 | 1.4% |

By tensor:

| % | tensor |
| ---: | --- |
| 28.6% | `blk.N.attn_output.weight` |
| 17.8% | `blk.N.attn_q_a.weight` |
| 10.1% | `blk.N.attn_q_b.weight` |
| 10.1% | `blk.N.indexer.attn_q_b.weight` |
| 4.8% | `blk.N.attn_kv_a_mqa.weight` |
| 4.0% | `blk.N.attn_k_b.weight` |
| 3.8% | `blk.N.attn_v_b.weight` |

**Every entry in the top seven is an attention tensor — roughly 79% of captured
time — and no `ffn_*_exps` op appears at all.** That is independent
confirmation of the expert-count result above, reached by a completely different
method: the routed-expert path is not where decode time goes on this host.

`attn_output.weight` alone is 28.6%, consistent with the byte accounting that
put dense attention at 14.62 GB of GLM-5.3 full's 33.97 GB/token — the single
largest component, read in full every token regardless of routing.

**Limitation, stated plainly:** the capture contains 17 nodes from one graph, not
a whole 78-layer token, and the sampled layer is one of the dense early layers,
which biases *against* seeing expert ops. Treat this as strong corroboration of
the expert-count experiment rather than a complete decode breakdown. A full
per-layer capture across a MoE layer is the obvious next step, and the machinery
for it now has a documented invocation.

### And attention is already repacked, so there is no configuration win there

The obvious follow-up — "attention dominates, so turn on attention repacking" —
was checked and is already the case. In `repack.cpp`:

```c
const bool eligible_matrix = ggml_n_dims(cur) == 2 && (
    strstr(cur->name, ".attn_") != nullptr ||            // unconditional
    (x86_vnni_ffn_enabled    && strstr(cur->name, ".ffn_") != nullptr) ||
    (x86_vnni_output_enabled && strcmp(cur->name, "output.weight") == 0));
```

`.attn_` tensors are eligible unconditionally once `GGML_CPU_Q8_0_REPACK=1`,
which the production profile sets. `attn_output.weight` is `[4096, 6144]`, so it
satisfies both shape constraints (`ne[0] % 32 == 0`, `ne[1] % 8 == 0`) and is
repacked. The `_FFN` and `_OUTPUT` flags gate *additional* experiments, not
attention.

So the 28.6% attributed to `attn_output.weight` is its **already-optimized**
cost, on the repacked AVX-512/VNNI path, with `X_TILE=auto`. There is no unset
switch behind it. Improving it means a better attention kernel or fewer
attention bytes — a smaller quant for those tensors specifically, which the
Unsloth UD scheme deliberately keeps high.

The two remaining unset Q8_0 flags were then tested rather than assumed:
`GGML_CPU_Q8_0_REPACK_OUTPUT=1` together with `GGML_CPU_Q8_0_REPACK_FORCE=1`
(which repacks every eligible Q8_0 tensor, not just attention/FFN/output)
measured **6.267 tok/s against 6.35 baseline** — neutral to marginally negative,
inside run-to-run spread. Production keeps them off.

That is the end of the configuration search on this host: every knob identified
across this repository has now been swept and the remaining levers all require
new code.

## Where to look

Given the expert-count result above, the remaining candidates, in the order a
profiler should check them:

1. **The dense attention block.** On Flash-Next it is 1.80 GB/token — 40% of
   active bytes — and it is read in full every token regardless of routing. It
   is untouched by the expert-count experiment, which makes it the leading
   suspect for a cost that did not move when experts were cut by 60%.
2. **Per-token graph dispatch and synchronization that does not scale with
   expert count** — the fixed work of walking the graph, launching the thread
   pool, and reducing at layer boundaries, once per token.
3. **Sampling and token bookkeeping**, which is per-token by construction and
   invisible to every weight-side optimization.

What is *not* on this list any more is MoE expert fusion, which an earlier
revision recommended and the measurement above rules out as the dominant term.

The separately measured fact that `ik_llama.cpp` extracts 78% of its available
bandwidth where this path extracts 33%
([`kernel-efficiency-ceiling.md`](kernel-efficiency-ceiling.md)) still stands,
and still says a 2.4x is reachable in software on this silicon. It just does not
say *which* code is responsible, and this file's attempt to infer that from byte
accounting was wrong.

**Profile a single decode step per-operation before choosing a target.** Byte
accounting tells you where the bandwidth goes. It does not tell you where the
time goes, and on this host those turned out to be different questions.
