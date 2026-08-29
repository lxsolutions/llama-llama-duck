# Reproducible llama.cpp patch bundles

These patches publish the exact source deltas used for the measured CPU and
NUMA work in this repository. Each bundle is pinned to one upstream commit so
it can be audited, reproduced, and rebased deliberately.

| Patch | Pinned llama.cpp base | Scope |
| --- | --- | --- |
| [`llama.cpp-b249-cpu-numa.patch`](llama.cpp-b249-cpu-numa.patch) | `3173a56471c1753650cd806694145ffd6dcace67` (build 249) | Linux per-socket CPU devices, asynchronous execution, Meta fixes, direct F32 collective |
| [`llama.cpp-b249-qwen4exp-mtp.patch`](llama.cpp-b249-qwen4exp-mtp.patch) | `3173a56471c1753650cd806694145ffd6dcace67` (build 249) | Qwen3.8/Qwen4 experimental conversion, architecture, and detached MTP-sidecar support |
| [`llama.cpp-a302733-glm-sr950.patch`](llama.cpp-a302733-glm-sr950.patch) | `a30273376ef669023334fc20ad02ae4ed8196a65` | Integrated CPU-NUMA, GLM tensor-parallel/speculative support, compact-quant and x86 kernel work, tests |

The two build-249 patches may be applied together. The GLM integration is a
separate source line and must not be stacked on them. Verify downloads against
[`SHA256SUMS`](SHA256SUMS), and see [`ATTRIBUTION.md`](ATTRIBUTION.md) for
provenance.

## Build-249 CPU-NUMA backend

`llama.cpp-b249-cpu-numa.patch` adds an opt-in Linux CPU device for each NUMA
node. It:

- discovers one hardware thread per physical core while respecting affinity;
- exposes `CPU-NUMA0`, `CPU-NUMA1`, and so on as llama.cpp devices;
- allocates device buffers with strict `mbind()` placement;
- gives each device a persistent, pinned worker pool and asynchronous
  dispatcher;
- fixes Meta host-view mapping and graph-metadata lifetime issues;
- provides a narrow direct host-memory F32 all-reduce, with the corrected
  generic Meta collective as fallback;
- enables device selection in server, common tools, and benchmark binaries.

The feature is Linux-only and disabled unless `GGML_CPU_NUMA_DEVICES=1`.
Runtime documentation is added as `docs/backend/CPU-NUMA.md` in the patched
tree.

Apply and build:

```bash
git checkout 3173a56471c1753650cd806694145ffd6dcace67
git apply --check /path/to/llama.cpp-b249-cpu-numa.patch
git apply /path/to/llama.cpp-b249-cpu-numa.patch

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=ON
cmake --build build -j
```

The measured four-node runtime contract is:

```bash
export GGML_CPU_NUMA_DEVICES=1
export GGML_CPU_NUMA_THREADS=12
export GGML_CPU_NUMA_POLL=100
export GGML_CPU_NUMA_DIRECT_ALLREDUCE=1
```

Select `CPU-NUMA0` through `CPU-NUMA3`, use tensor split `1,1,1,1`, and treat
the thread count as *per node*. The ready-to-edit Qwen profile is
[`profiles/launch-qwen38-27b.sh`](../profiles/launch-qwen38-27b.sh).

## Build-249 Qwen experimental and MTP support

`llama.cpp-b249-qwen4exp-mtp.patch` is deliberately path-scoped to Qwen
conversion, GGUF mappings, architecture/model loading, and Qwen4 experimental
model code. Apply it to a clean base by itself, or after the CPU-NUMA patch:

```bash
git checkout 3173a56471c1753650cd806694145ffd6dcace67
git apply /path/to/llama.cpp-b249-cpu-numa.patch       # optional
git apply --check /path/to/llama.cpp-b249-qwen4exp-mtp.patch
git apply /path/to/llama.cpp-b249-qwen4exp-mtp.patch
```

Build with the same CMake command above. For MTP, pass a detached sidecar:

```bash
llama-server \
  --model "$MODEL" \
  --spec-type draft-mtp \
  --spec-draft-model "$MTP_MODEL" \
  --spec-draft-n-max 3 \
  --spec-draft-n-min 0 \
  --spec-draft-p-min 0.2
```

The source basis and authors are listed in
[`ATTRIBUTION.md`](ATTRIBUTION.md).

## GLM/SR950 integration

`llama.cpp-a302733-glm-sr950.patch` is the complete tracked delta from the
pinned base. It contains:

- CPU-NUMA devices, Meta correctness fixes, sharded execution, and collectives;
- GLM model, conversion, tensor-parallel, MTP, and server integration;
- exact compact IQ2/IQ3 kernels and expanded Q5_K/Q8_0 x86 paths;
- repack, multi-row speculative-verification, and MoE kernel work;
- focused backend, argument-parser, model-resolution, and server tests.

Apply it only to its own base:

```bash
git checkout a30273376ef669023334fc20ad02ae4ed8196a65
git apply --check /path/to/llama.cpp-a302733-glm-sr950.patch
git apply /path/to/llama.cpp-a302733-glm-sr950.patch

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_OPENMP=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=ON
cmake --build build -j
```

Use [`profiles/launch-glm53-full.sh`](../profiles/launch-glm53-full.sh) as the
four-node starting point. The benchmark-selected general profile uses MTP
depth 2 with no confidence gate. The deeper request-scoped profile is intended
only for agentic replay traffic.

## Validation and limits

- All three files pass `git apply --check` against their exact pinned bases.
- Both build-249 patches also apply sequentially to the same clean checkout.
- The build-249 combined source built in Release mode. Of 62 CTest entries, 61
  passed; the remaining tokenizer case could not run because six test fixtures
  were checked out as 132-byte Git LFS pointer files.
- The GLM source delta is whitespace-clean and applies cleanly. Its benchmarked
  executable exercised the production model, MTP, server endpoints, and focused
  kernel paths.
- Strict `mbind()` may fail in containers, restricted services, or when a node
  lacks local memory. It fails explicitly instead of silently losing locality.
- The direct collective accepts only compatible contiguous F32 tensors owned
  by matching CPU-NUMA devices. All other cases use the generic path.
- Every rebase needs a clean build, backend tests, deterministic output
  comparison, and a full-service A/B. Microbench wins have regressed the
  complete service before.

These patches are experimental research artifacts, not upstream-supported
interfaces.
