#!/usr/bin/env bash
# Qwen3.8-Flash-Next on a 4-socket CPU host, tensor-sharded across NUMA nodes.
#
# Measured 8.98 tok/s decode vs 6.68 for the previous single-node profile
# (+34%) on a 4x Xeon Gold 6242 / 755 GiB host. See
# benchmarks/qwen38-flash-next-numa.md for the sweep behind every default here.
#
# Requires an engine built with the CPU-NUMA device backend. Verify with:
#   GGML_CPU_NUMA_DEVICES=1 llama-server --list-devices
# which must list CPU-NUMA0..N. Without it, fall back to SINGLE_NODE=1 below.
set -euo pipefail

: "${LLAMA_SERVER:?set LLAMA_SERVER to llama-server}"
: "${MODEL:?set MODEL to the first GGUF shard}"

mmproj_args=()
if [[ -n "${MMPROJ:-}" ]]; then
  mmproj_args=(--mmproj "$MMPROJ")
fi

# Speculation is OFF by default and that is a measured choice, not an omission:
# the published MTP head scored 0.000 draft acceptance on this model (-65%),
# and ngram-mod cost -46% on novel prose. Only enable SPEC_TYPE for traffic you
# have confirmed is repetitive, and check acceptance before keeping it.
spec_args=(--spec-type "${SPEC_TYPE:-none}")
if [[ "${SPEC_TYPE:-none}" == *draft-mtp* ]]; then
  : "${MTP_MODEL:?draft-mtp requested but MTP_MODEL is unset}"
  spec_args+=(
    --spec-draft-model "$MTP_MODEL"
    --spec-draft-ngl all
    --spec-draft-n-max "${SPEC_DRAFT_N_MAX:-2}"
    --spec-draft-n-min 0
    --spec-draft-p-min "${SPEC_DRAFT_P_MIN:-0}"
  )
fi
if [[ "${SPEC_TYPE:-none}" == *ngram-mod* ]]; then
  spec_args+=(
    --spec-ngram-mod-n-match "${SPEC_NGRAM_MATCH:-8}"
    --spec-ngram-mod-n-min "${SPEC_NGRAM_MIN:-48}"
    --spec-ngram-mod-n-max "${SPEC_NGRAM_MAX:-64}"
  )
fi

# ---- single-node fallback for engines without the CPU-NUMA backend ----
if [[ "${SINGLE_NODE:-0}" == "1" ]]; then
  command -v numactl >/dev/null || { echo "numactl is required" >&2; exit 1; }
  node="${NUMA_NODE:-0}"
  exec numactl --cpunodebind="$node" --membind="$node" \
    "$LLAMA_SERVER" \
    --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" \
    --model "$MODEL" "${mmproj_args[@]}" \
    --load-mode none --gpu-layers 0 \
    --ctx-size "${CTX_SIZE:-65536}" \
    --cache-type-k f16 --cache-type-v f16 \
    --threads "${THREADS:-32}" --threads-batch "${THREADS_BATCH:-32}" \
    "${spec_args[@]}" "$@"
fi

# ---- default: tensor-sharded across all NUMA nodes ----
# THREADS is per node, not total. 12 measured best; 14 and 16 are both worse
# than 8. The falloff above the optimum is sharp, so undersubscribe if unsure.
THREADS="${THREADS:-12}"

export GGML_CPU_NUMA_DEVICES=1
export GGML_CPU_NUMA_THREADS="$THREADS"
# Polling aggressiveness, a PERCENTAGE clamped to 0-100 in ggml-cpu.cpp:
#   poll = std::min(100, env("GGML_CPU_NUMA_POLL", 50))
# Measured +6.7% going 50 -> 100 on this MoE (8.98 -> ~9.36). Values above 100
# are silently clamped and buy nothing -- do not "tune" them.
export GGML_CPU_NUMA_POLL="${NUMA_POLL:-100}"
export GGML_CPU_NUMA_DIRECT_ALLREDUCE=1
export GGML_CPU_NUMA_REPACK=1

# --mlock with mmap measured 9.645 vs 9.452 (+2%). Note --load-mode none, which
# also puts weights in anonymous memory, was *worse* (9.473) -- so the gain is
# from pinning, not from avoiding mmap. Needs the memlock rlimit raised.

devices="${DEVICES:-CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3}"
splits="${TENSOR_SPLIT:-1,1,1,1}"

exec "$LLAMA_SERVER" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" \
  --model "$MODEL" "${mmproj_args[@]}" \
  --load-mode mmap --mlock --fit off --gpu-layers 999 \
  --ctx-size "${CTX_SIZE:-65536}" \
  --device "$devices" --split-mode tensor --tensor-split "$splits" \
  --cache-type-k f16 --cache-type-v f16 \
  --threads "$THREADS" --threads-batch "$THREADS" \
  "${spec_args[@]}" "$@"
