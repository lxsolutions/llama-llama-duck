# GLM-5.3 and Qwen3.8 tuning results

These results extend the focused
[`Qwen3.8-27B CPU-NUMA benchmark`](qwen38-27b-cpu-numa.md). Measurements used
one four-socket Lenovo SR950 with 4 x Xeon Gold 6242 CPUs, 64 physical cores,
755 GiB of memory, and no accelerator. Reported decode rates are emitted
generation tokens per second.

Results are model-, quant-, source-, and workload-specific. “Candidate” means
the artifact fit the memory/topology plan but was not benchmarked; it is not a
performance claim.

## GLM-5.3 full

The full UD-Q4_K_XL model used 11 verified GGUF shards, a detached MTP head,
32K context, and Q8_0 KV. The four-way CPU-NUMA backend ran 16 physical cores
per socket with polling at 50.

| Workload/profile | Tokens/s | Draft acceptance |
| --- | ---: | ---: |
| Initial non-speculative control | 4.62 | n/a |
| General MTP, depth 2, `p_min=0` | 6.27–6.69 | workload-dependent |
| 1,400-token replay, mean of 3 | **7.16** | 84.8% |
| Agentic replay, first pass | 13.308 | 93.79% |
| Agentic replay, second pass | 12.546 | 92.35% |
| Agentic replay, combined | **12.920** | 93.07% |

> **Not reproduced (2026-08-29).** A later re-measurement of the restored
> service, same `n_max=18 / p_min=0.75` profile, reached only **6.25 tok/s** on a
> ~500-token file-reproduce replay, at 72.8–77.5% acceptance. Acceptance and
> draft length were healthy, so the mechanism works; the suites differ and
> warm-up matters a great deal. Do not quote 12.92 as this host's replay
> throughput without re-deriving it from a published prompt set. See
> [`ten-tokens-per-second.md`](ten-tokens-per-second.md).

All six agentic correctness gates passed. The general server profile reserves a
maximum speculative depth of 32 but defaults each request to depth 2 and
`p_min=0`. The agentic result used a request-scoped override of depth 18 and
`p_min=0.75`:

```json
{
  "speculative": {
    "n_max": 18,
    "p_min": 0.75
  }
}
```

That override should remain workload-scoped. Higher acceptance is not itself
the objective; aggregate throughput and exact target verification are.

The integrated patch adds exact compact IQ2/IQ3 paths, expanded Q5_K and Q8_0
x86 paths, and fused MoE operations. Two apparently promising kernel changes
were rejected by full-service A/B:

- a Q5_K layout improved its microbenchmark but moved service decode from
  11.47 to 10.55 tokens/s;
- a Q8_0 tile-2 experiment did not retain a whole-model gain.

The detached MTP tensor payload measured 7,697,448,080 bytes. Pinning that
sidecar to a single NUMA node is available as an explicit profile option but
has not yet passed a controlled placement A/B.

## GLM-5.3 Flash

A prior node-local compatible quant measured about 3.44–3.47 tokens/s. The
current compact candidates were sized before download or service interruption:

| Candidate | Text model | Projector | Combined | Status |
| --- | ---: | ---: | ---: | --- |
| IQ2 | 101,844,951,808 B | 1,128,047,200 B | 102,972,999,008 B | unbenchmarked |
| IQ3 | 120,367,571,715 B | 1,128,047,200 B | 121,495,618,915 B | unbenchmarked |

The compatibility source line was clean at commit `2e0e57` (the support work
tracked upstream PR 27754). The established starting contract is one NUMA node,
32 threads, preferred local memory, mmap loading, F16 KV, and
`--flash-attn off`. Strict binding should be tested after model residency is
known; preferred placement avoids an avoidable hard failure during exploration.

No throughput number is claimed for the IQ2 or IQ3 candidates.

## Qwen3.8 Flash Next

> **Superseded 2026-08-29.** The single-node conclusion below was correct for
> the engine build available at the time, which had no `CPU-NUMA` device
> backend. A build that exposes `CPU-NUMA0..3` reaches **8.98 tok/s**, +34% over
> the best single-node arm. Full sweep, including three measured dead ends, in
> [`qwen38-flash-next-numa.md`](qwen38-flash-next-numa.md). The table below is
> retained as the single-node reference.

The measured UD-Q2_K_XL text-plus-projector footprint was about 79.77 GB, which
fits comfortably on one node in this host.

| Placement / threads | Tokens/s |
| --- | ---: |
| Untuned control | 2.58 |
| Four sockets, unbound | 3.52 |
| Four sockets, interleaved, 64 threads | 2.67 |
| Two sockets | 3.25 |
| One node, 8 threads | 5.54 |
| One node, 12 threads | 5.78 |
| One node, 16 threads | 5.84 |
| One node, 24 threads | 6.04 |
| One node, 32 threads | **6.47** |
| Full serving configuration | 6.32 |
| *Four `CPU-NUMA` devices, 12 threads/node, repack* | ***8.98*** |

On the single-node path this model wants strict CPU and memory binding, 32
threads, `--load-mode none`, and F16 KV. Q8_0 KV is not a lower-memory
substitute here; the tested path asserts. On the four-device path it wants
12 threads **per node** — 16 is 14% worse than 12, and worse than 8.

A detached MTP sidecar candidate was 4,142,897,248 bytes with SHA-256
`b9880220df29fc224bbce408c867cd5d9c021263b754033ea624b669e374f4ec`.
Its public source was
[`drluoto/Qwen3.8-Flash-Next-MTP-GGUF`](https://huggingface.co/drluoto/Qwen3.8-Flash-Next-MTP-GGUF)
at revision `67de7592b670ef454a903574d5e2aa6c8e1d6b46`.

**It is now benchmarked and it is a dead end**: 0.00000 draft acceptance
(0 accepted / 198 generated), taking decode from 8.98 to 3.13 tok/s. Pinning the
draft to a single node — previously listed here as a promising untried option —
changed nothing (3.17). The launcher continues to leave speculation off by
default.

## Reproduction

Use the generic launchers in [`profiles/`](../profiles/README.md). Preserve a
non-speculative arm, record cold and warm speculative runs separately, and
compare exact or deterministic output whenever changing collectives or kernels.
Do not promote a microbenchmark result without a complete model/service A/B.
