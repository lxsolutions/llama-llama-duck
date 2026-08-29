# Qwen3.8-27B CPU-NUMA benchmark

This benchmark isolates the effect of per-socket tensor sharding and the
experimental direct all-reduce in
[`patches/llama.cpp-b249-cpu-numa.patch`](../patches/llama.cpp-b249-cpu-numa.patch).

## Test system

| Item | Value |
| --- | --- |
| CPU | 4 x Intel Xeon Gold 6242 |
| Topology | 4 NUMA nodes, 16 physical cores per node, SMT enabled |
| Memory | 755 GiB DDR4-2933 |
| Measured local bandwidth | about 365 GB/s aggregate across four nodes |
| Accelerator | none |
| llama.cpp base | build 249, `3173a56471c1753650cd806694145ffd6dcace67` |
| Model | Qwen3.8-27B GGUF, 31.45 GB |
| Decode setup | prompt 0, generation 32, 3 repetitions, Q8_0 KV, Flash Attention |

Each multi-node arm used equal tensor split across `CPU-NUMA0` through
`CPU-NUMA3`. The thread count is per node, not host-wide.

## Raw decode sweep

| Configuration | Tokens/s | Relative to generic four-node |
| --- | ---: | ---: |
| One NUMA device, 16 cores | 2.524 | 0.48x |
| Four devices, generic collective, 16 cores/node | 5.239 | 1.00x |
| Direct collective, 16 cores/node, poll 50 | 6.091 | 1.16x |
| Direct collective, 10 cores/node, poll 50 | 6.783 | 1.29x |
| Direct collective, 12 cores/node, poll 50 | 7.054 | 1.35x |
| Direct collective, 14 cores/node, poll 50 | 6.849 | 1.31x |
| Direct collective, 12 cores/node, poll 100 | **7.169** | **1.37x** |
| Previous row plus `MADV_HUGEPAGE` | 6.989 | 1.33x |

The selected raw profile is 36.8% faster than the corrected generic
four-device path and 2.84x the one-device result. Twelve physical cores per
socket beat both 10 and 14, and huge-page advice was a small regression.

## Speculative production smoke

The selected 12-core/poll-100 profile was also exercised through
`llama-server` at a 262,144-token context with the model's MTP head.

| Request | Output | Tokens/s | Draft acceptance |
| --- | --- | ---: | ---: |
| Count 1 through 50, deliberately small output cap | cap reached after 38 | 14.275 | 116/123, 94.3% |
| Count 1 through 30 | exact requested sequence and stop | **14.063** | 109/121, 90.1% |
| Final MTP profile, count 1 through 30 | exact requested sequence and stop | **16.486** | 140/150, 93.3% |

The server passed health, model-list, and metrics checks, then shut down cleanly.

## Speculative workload profiles

The original `n_max=3`, `p_min=0.4` MTP profile was not optimal. A matched
novel-prose sweep selected `n_max=3`, `n_min=0`, and `p_min=0.2`: its 12.299
tokens/s median was 12.1% above the old gate at 10.967. Depth two fell to 13.384
tokens/s on structured output, while depth four fell to 13.558; suppressing
singleton draft batches also did not help.

For copied-context traffic, MTP-only reached 17.357 tokens/s with exact output.
Combining `ngram-mod` with MTP and reducing `n_match` from 24 to 8 reached
18.369 tokens/s on first-pass replay and 23.581 tokens/s after the same
reasoning pattern repeated. The short-match profile regressed cold novel prose
to about 9.37 tokens/s, so it should be enabled only for iterative agentic or
file-replay workloads. Speculation remains target-verified in both profiles.

## Correctness check

A greedy 24-token generation with seed `424242` was run once with the direct
collective enabled and once with Meta's generic fallback. The output bytes were
identical; both had SHA-256
`c62e27b97eaf4cc96ffa69a95e2b94f7be92d8c17e699d3b35f612437c352a99`.

That check establishes equivalence for the tested graph and prompt. The direct
path still performs a deliberately narrow runtime validation and falls back for
unsupported types, layouts, shapes, ownership, or backend combinations.

## Reproduce the raw arm

```bash
export GGML_CPU_NUMA_DEVICES=1
export GGML_CPU_NUMA_THREADS=12
export GGML_CPU_NUMA_POLL=100
export GGML_CPU_NUMA_DIRECT_ALLREDUCE=1

llama-bench \
  --model "$MODEL" \
  --n-prompt 0 \
  --n-gen 32 \
  --repetitions 3 \
  --threads 12 \
  --n-gpu-layers 999 \
  --device CPU-NUMA0/CPU-NUMA1/CPU-NUMA2/CPU-NUMA3 \
  --split-mode tensor \
  --tensor-split 1/1/1/1
```

Unset `GGML_CPU_NUMA_DIRECT_ALLREDUCE` for the generic-collective control. For
the one-node control, select only `CPU-NUMA0` and use 16 threads.
