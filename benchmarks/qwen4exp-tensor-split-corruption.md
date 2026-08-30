# `--split-mode tensor` silently corrupts `qwen4exp` output

**Severity: silent wrong-answer bug.** No error, no warning, no crash, no
degraded-quality signal. The server is healthy, throughput looks *better* than
the correct configuration, and every token is nonsense.

## Reproduction

Qwen3.8-Flash-Next (`unsloth/Qwen3.8-Flash-Next-GGUF`, `UD-Q2_K_XL`,
architecture `qwen4exp`), same host, same binary, same model file, same prompts,
`temperature=0`:

**Four `CPU-NUMA` devices, `--split-mode tensor --tensor-split 1,1,1,1`:**

    "What is 17 * 23?"
      -> ggio **.** (!(( (!(( (!(( (!(( (!(( (!(( (!(( (!(( (!(( (!((...

    "What is the capital of Australia?"
      -> **.**3333333333333333333333333333333333333333333333333333...

    "If all Bloops are Razzies and all Razzies are Lazzies, are all
     Bloops Lazzies?"
      -> ├──333333333333333333333333333333333333333333333333333333...

**Same model, single NUMA node, plain CPU backend, no tensor split:**

    "What is 17 * 23?"                  -> 17 × 23 = 391
    "What is the capital of Australia?" -> The capital of Australia is **Canberra**.
    "...are all Bloops Lazzies?"        -> Yes. If all Bloops are Razzies, and all
                                           Razzies are Lazzies, then all Bloops are
                                           Lazzies.

The only difference is the split mode.

Not a chat-template problem: identical garbage with and without `--jinja`.
Not a quant problem: the same file is correct on the single-node path.

## `--split-mode layer` is correct — the fault is tensor sharding specifically

Same model, same four `CPU-NUMA` devices, same binary, changing only the split
mode:

| split mode | output | tok/s |
| --- | --- | ---: |
| `tensor` | `ggio **.** (!(( (!((...` | 9.65 |
| **`layer`** | **`17 × 23 = 391`** / Canberra / valid syllogism | 5.41 |
| none (single node) | `17 × 23 = 391` / Canberra / valid syllogism | 6.69 |

This narrows the bug usefully. The architecture loads, executes and produces
correct results on the multi-device backend when whole layers are assigned to
devices. It only breaks when individual tensors are sharded. The model, the
quant, the NUMA device backend, the collectives and the graph are all fine —
**the defect is in the per-tensor split rules for `qwen4exp`**, exactly as the
`glm5next` gate exists to prevent.

`layer` is also 19% *slower* than a single node here, since batch-1 decode
executes layers sequentially and gains no compute parallelism while paying
cross-device activation transfers. So it is a correctness datapoint, not a
workaround: the fastest correct configuration for this model remains one NUMA
node.

## Scope — this architecture only

Checked every model in this repository on its own tensor-split configuration:

| model | architecture | `--split-mode tensor` | output |
| --- | --- | --- | --- |
| Qwen3.8-27B | `qwen35` (dense) | allowed | **correct** — 391 / Canberra / valid syllogism |
| GLM-5.3 full | `glm-dsa` (MoE 8/256) | allowed | **correct** — 391, and exact passage reproduction |
| **Qwen3.8-Flash-Next** | **`qwen4exp` (MoE 10/512)** | **allowed** | **CORRUPT** |
| GLM-5.3-Flash | `glm5next` | **refused at load** | n/a — gate fires correctly |

`llm_arch_supports_sm_tensor()` in `src/llama-arch.cpp` is an **exclusion list**:
architectures named in the switch return `false`, everything else returns `true`.
`GLM5NEXT` is named, so it is refused. **`QWEN4EXP` is not named, so it is
permitted — and it has no correct sharding rules.**

Default-allow is the wrong polarity for a feature that fails this way. A new
architecture is silently opted in to tensor parallelism the moment it is added,
and the failure mode is fluent nonsense rather than a crash.

## What this cost, and why it is worth publishing

Every Qwen3.8-Flash-Next throughput number previously recorded in this
repository — 7.51, 7.71, 8.98, 9.21, 9.45, 9.65 tok/s, and a 10.067 that appeared
to clear a 10 tok/s target — was measured on the four-device path. **All of them
are void.** They are the speed of producing garbage. The retraction is in
[`ten-tokens-per-second.md`](ten-tokens-per-second.md).

The only valid figure for this model on this host is the single-node one:
**6.68 tok/s**.

Worse, the corrupt configuration is *faster*, so an optimization campaign
measuring only throughput will actively select for it. It did here, across a
dozen sweeps.

### The tell that was present the whole time

Every benchmark row on the corrupt path reported `content_chars: 0`. The model
was emitting only reasoning-channel tokens and no content. That field was
visible in every single run and read as a quirk of a reasoning model hitting its
token cap. It was the bug, in plain sight, for the entire campaign.

**Throughput harnesses must assert on output, not just count tokens.** A
tok/s number computed from a token stream nobody inspected is not a measurement
of anything. `tools/quality_probe.py` is four deterministic prompts with known
answers; running it once per configuration would have caught this immediately.

## Every correct configuration, exhaustively (quiet host)

With tensor-split ruled out, the remaining question is whether any *correct*
multi-socket arrangement recovers the bandwidth. None does:

| configuration | threads | tok/s | correct? |
| --- | ---: | ---: | :--: |
| **1 node, `--membind=0`** | **32** | **6.69** | yes |
| 1 node | 24 | 6.43 | yes |
| 1 node | 16 | 5.65 | yes |
| 4 devices, `--split-mode layer` | 16/dev | 5.41 | yes |
| 2 nodes, `--membind=0,1` | 48 | 3.47 | yes |
| 2 nodes, `--membind=0,1` | 64 | 3.34 | yes |
| 4 nodes, `--interleave=all` | 96 | 2.68 | yes |
| 4 devices, `--split-mode tensor` | 12/dev | ~9.6 | **NO — garbage** |

Handing the ordinary CPU backend more sockets makes it **worse**, roughly
halving throughput at two nodes and more at four, because a single thread pool
spanning sockets pays cross-socket memory latency and synchronization on every
operation. That is precisely the problem the per-socket device backend exists to
solve — and on this architecture that backend is the one that corrupts.

So this model is **capped at 6.69 tok/s until the sharding rules are written**.
Not a tuning ceiling: an exhausted search over every arrangement that produces
correct output.

## Root cause, localized

Qwen3.8-Flash-Next is a **hybrid attention/SSM** architecture. Layer 1 contains,
alongside ordinary MoE and attention tensors:

    blk.1.ssm_a            [48]           blk.1.ssm_alpha.weight  [2560, 48]
    blk.1.ssm_beta.weight  [2560, 48]     blk.1.ssm_conv1d.weight [4, 10240]
    blk.1.ssm_dt.bias      [48]           blk.1.ssm_norm.weight   [128]
    blk.1.ssm_out.weight   [6144, 2560]

The split-state function carries SSM patterns written for Mamba/Jamba-style
architectures — `pattern_ssm_dt`, `pattern_ssm_a`, `pattern_ssm_alpha`,
`pattern_ssm_beta`, `pattern_ssm_conv1d`, `pattern_ssm_out_weight`. These match
`qwen4exp`'s tensors **by name while not sharing their semantics**, so the wrong
split axes are applied and the output is silently wrong.

This also explains the asymmetry with `glm5next`, which is refused rather than
corrupted. `glm5next` factors its SSM output into `ssm_f_a`/`ssm_f_b` pairs and
has **no `ssm_out.weight`**, so the reference lookup fails and it aborts.
`qwen4exp` *does* have `ssm_out.weight`, so every lookup resolves, the rules
apply cleanly, and nothing complains.

**Name-based tensor classification across architectures is the underlying
hazard.** A pattern like `blk\.\d*\.ssm_out.weight` is an assertion about
semantics dressed up as a string match.

### Three attempts at a fix, and why it is not a patch

Replication is always arithmetically safe and these tensors are small, so
mirroring the SSM block looks like a clean fix. There is even a precedent in the
same function — LFM2's shortconv block is mirrored, conv state included, for
exactly this reason. It does not converge:

| attempt | mirrored set | result |
| ---: | --- | --- |
| 1 | `blk.N.ssm_*` | `GGML_ASSERT(ret.axis != ...UNKNOWN)` at load |
| 2 | + `cache_r_l*`, `cache_s_l*` | same assert |
| 3 | + `attn_gate.weight` | same assert |

Each attempt exposed another tensor anchored on the SSM block. The reason is
visible in the rules: **every** SSM config resolves its axis against
`ssm_out.weight` —

    ssm_dt, ssm_a          -> AXIS_0, ref "ssm_out.weight"
    ssm_alpha, ssm_beta    -> AXIS_1, ref "ssm_out.weight"
    ssm_conv1d             -> AXIS_1, ref "ssm_out.weight"
    cache_r_l*, cache_s_l* -> AXIS_0, ref "ssm_out.weight"
    attn_gate              -> AXIS_1, ref "attn_output.weight",
                                      fallback "ssm_out.weight"

and this architecture has **no `attn_output.weight`** — its attention output is
projected through `ssm_out.weight`. So attention and SSM are one entangled group
anchored on a single tensor, and mirroring that tensor invalidates every
dependent at once.

The assert names no tensor, so each round costs a rebuild to learn the next
dependent. More importantly, an edit that merely makes it *load* is worthless
here: this whole document exists because a configuration that loads, runs and
benchmarks well was emitting garbage. Any fix must be validated by **exact
output comparison against the single-node arm at `temperature=0`**, not by
absence of a crash.

### What the assert actually reports

Instrumenting the assert site shows it is **not** a missing tensor rule. The
failure is in `handle_generic()` in `ggml-backend-meta.cpp`, where an operation's
source split-states must agree:

    SPLIT-MISMATCH nsrc=10 scalar_only=0 axes=[1, 10, 99, 99, 99, 99, 99, 99, 99, 99]

A **ten-source** operation receiving one source split on axis 1, one **mirrored**
(the tensor just changed), and the rest unset. `split_states_equal` rejects the
combination and the axis resolves to `UNKNOWN`.

That is the whole difficulty in one line. Mirroring any tensor in this block
propagates a constraint into every fused operation it feeds, and satisfying that
constraint means mirroring its co-inputs, which propagates further. Because the
SSM path is interleaved with attention in every layer, the closure of "everything
that must be mirrored" grows until little remains split — **at which point
tensor parallelism buys nothing, since mirrored weights are read in full by every
device.**

So the choice is not "find the right rule". It is either design a split for this
architecture's graph that keeps fused operations consistent, or accept that this
block is not tensor-parallelizable in this framework and exclude the
architecture — which is exactly the treatment `glm5next` already receives, and
which now looks less like a gap and more like the correct answer.

Reverted; engine verified back to correct single-node output.

## Recommended handling

1. Do not use `--split-mode tensor` with `qwen4exp`. Use the single-node profile
   (`SINGLE_NODE=1` in
   [`launch-qwen38-flash-next.sh`](../profiles/launch-qwen38-flash-next.sh)).
2. Add `LLM_ARCH_QWEN4EXP` to the exclusion list until sharding rules exist —
   the same treatment `glm5next` already correctly receives.
3. Upstream, consider inverting the polarity so tensor parallelism is
   **opt-in per architecture**. The cost of wrongly excluding an architecture is
   a missed speedup; the cost of wrongly including one is undetected wrong
   answers in production.

## Wider lesson

This is the exact failure mode that argued against forcing `glm5next` past its
gate by adding a `suffix_fallback` and guessing an axis (see
[`glm5next-tensor-parallel-blocker.md`](glm5next-tensor-parallel-blocker.md)).
That was not a hypothetical caution. The same class of bug was already live in a
sibling architecture on the same host, and it took a deliberate correctness check
to find it — throughput measurement alone never would have.
