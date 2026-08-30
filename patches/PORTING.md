# Porting the CPU-NUMA backend to another llama.cpp branch

The single largest measured win in this repository comes from exposing each
NUMA node as a separate llama.cpp device and tensor-sharding across them. The
catch is that model-support work usually lives on a *different* branch than the
NUMA work, so the models that most need the backend are often built from a tree
that does not have it.

Two of the four models tuned on this host were in exactly that state. Their
launch profiles pinned them to one socket — not as a tuning choice, but because
the sharded path did not exist in their engine build.

## First: check whether your build even has it

    GGML_CPU_NUMA_DEVICES=1 llama-server --list-devices

Expected on a 4-socket host:

    Available devices:
      CPU-NUMA0: ... / NUMA node 0 / N physical cores (... MiB free)
      CPU-NUMA1: ...
      CPU-NUMA2: ...
      CPU-NUMA3: ...

If it prints `(none)`, the backend is absent and every tuning knob you try is
operating on one socket's worth of bandwidth.

**The environment variable is required for the devices to appear at all.**
Without `GGML_CPU_NUMA_DEVICES=1`, a build that fully supports the backend still
reports `(none)`. Do not conclude the backend is missing from an unset-env run —
that mistake costs a rebuild.

## Confirming the source really lacks it

Cheap discriminator between "not compiled in" and "not in the tree":

    grep -rl 'GGML_CPU_NUMA_DEVICES' ggml/src | wc -l    # 0 means absent
    wc -l ggml/src/ggml-cpu/ggml-cpu.cpp                 # ~700 vs ~1350

The NUMA device implementation adds roughly 650 lines to `ggml-cpu.cpp`. A tree
with the backend is about twice the size in that one file. Note that
`ggml/src/ggml-backend-meta.cpp` may be **present but unpatched** — its presence
is not evidence the backend is available, which is a misleading signal if you
check only for the file.

## Applying

[`llama.cpp-b249-cpu-numa.patch`](llama.cpp-b249-cpu-numa.patch) touches five
files and is mostly additive:

    common/arg.cpp                    |  34 +-
    docs/backend/CPU-NUMA.md          |  95 +++++
    ggml/src/ggml-backend-meta.cpp    |  26 +
    ggml/src/ggml-cpu/ggml-cpu.cpp    | 677 ++++++++++++++++++++++++++++++--
    tools/llama-bench/llama-bench.cpp |   5 +

Because it is additive and concentrated in files that model-support work rarely
touches, it tends to apply across branch boundaries. Verified on the
`glm5next` model-support branch (Unsloth PR 27754 line, base commit `2e0e57f1`),
which is a different lineage from the patch's own base:

    git apply --check ../llama-llama-duck/patches/llama.cpp-b249-cpu-numa.patch
    # applied cleanly, no fuzz

Always `--check` first. If it rejects, the conflict is almost certainly in
`ggml-cpu.cpp` device registration rather than in the collective itself.

## Building

Match the flags of the build you are replacing, or you will A/B a compiler
change while thinking you are A/B-ing the backend. Read them out of the old
build rather than guessing:

    grep -E 'CMAKE_BUILD_TYPE|GGML_NATIVE|GGML_CPU_REPACK|GGML_OPENMP' \
      <old-build>/CMakeCache.txt

Then:

    cmake -B build-numa -S . \
      -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_OPENMP=ON \
      -DGGML_CPU_REPACK=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
    cmake --build build-numa --target llama-server -j

`GGML_NATIVE=ON` is doing real work here: the explicit `GGML_AVX512*` options
read `OFF` in the cache even on an AVX-512 host, because native detection sets
the ISA without flipping them. Do not "fix" them.

Building into a tmpfs (`/dev/shm`) is worth it on a host whose root filesystem
is tight — these trees are several GB per build and you will make several.

## Then re-derive the knobs, do not port them

A ported backend does not come with a ported configuration. Per-node thread
count in particular is model-specific and the falloff above the optimum is
sharp:

| model | best threads/node | cost of using 16 instead |
| --- | ---: | ---: |
| Qwen3.8-Flash-Next (176.9B MoE, Q2_K_XL) | 12 | −14% |
| Qwen3.8-27B (dense, Q8_K_XL) | 16 | — (16 is best) |

Sweep `GGML_CPU_NUMA_THREADS` before reporting any number. Sweep it again after
a quant change.
