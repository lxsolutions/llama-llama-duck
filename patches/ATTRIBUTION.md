# Patch attribution

The repository separates locally developed engineering from changes derived
from upstream work. Patch filenames pin the source revision they target.

## `llama.cpp-b249-cpu-numa.patch`

This experimental Linux CPU-NUMA backend was developed for this repository
against llama.cpp build 249 (`3173a56471c1753650cd806694145ffd6dcace67`).

## `llama.cpp-b249-qwen4exp-mtp.patch`

The Qwen3.8 / Qwen4 experimental model and MTP support is derived from these
public llama.cpp contributions:

- [ggml-org/llama.cpp PR #27836](https://github.com/ggml-org/llama.cpp/pull/27836)
  by Ryan Monsurate, specifically commits `d303eec923f92ccab7109e97d95cb5c1ab83e0d2`,
  `d72620018f612bb05d16e585108e3440947a1f9f`, and
  `1d8de7c1b0c7d2febf8f983174d8e6a711e2b1af`.
- [detached MTP-sidecar fix `a82a58a`](https://github.com/crusaderky/llama.cpp/commit/a82a58a57fc307e5cec0dc68db64d143339be4f2)
  by crusaderky.
- [detached MTP-sidecar fix `57bb668`](https://github.com/crusaderky/llama.cpp/commit/57bb668674d9fb0d382885e5b04911c6437f8e83)
  by drluoto.

The published patch is a narrowly scoped, combined engineering snapshot for
reproducibility. It does not claim original authorship of those upstream
changes.

## `llama.cpp-a302733-glm-sr950.patch`

This integration patch combines llama.cpp compatibility work, CPU-NUMA
execution, compact-quant and x86 kernel work, GLM tensor-parallel execution,
speculative-server plumbing, and tests against
`a30273376ef669023334fc20ad02ae4ed8196a65`. Any upstream-derived code retains
the notices and history present in the patch. See the patch itself and
[`README.md`](README.md) for the exact audited delta and limitations.
