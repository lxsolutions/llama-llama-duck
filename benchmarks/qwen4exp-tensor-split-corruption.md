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
