#!/usr/bin/env bash
set -euo pipefail

: "${LLAMA_SERVER:?set LLAMA_SERVER to the patched llama-server binary}"
: "${MODEL:?set MODEL to the main GGUF}"
: "${MTP_MODEL:?set MTP_MODEL to the detached MTP GGUF}"

export GGML_CPU_NUMA_DEVICES=1
export GGML_CPU_NUMA_THREADS=16
export GGML_CPU_NUMA_POLL=50
export GGML_CPU_NUMA_HUGEPAGES=1
export GGML_CPU_NUMA_DIRECT_ALLREDUCE=1
export GGML_CPU_NUMA_REPACK=1
export GGML_CPU_IQ2_XS_REPACK=1
export GGML_CPU_IQ3_XXS_REPACK=1
export GGML_CPU_Q5_K_REPACK=1
export GGML_CPU_REPACK_LOAD_THREADS="${REPACK_LOAD_THREADS:-16}"
export GGML_CPU_MOE_GATE_UP_FUSION=1
export GGML_CPU_FFN_GATE_UP_FUSION=1
export GGML_CPU_MOE_WEIGHTED_SUM_FUSION=1

mtp_device_args=()
if [[ -n "${MTP_DEVICE:-}" ]]; then
  mtp_device_args=(--spec-draft-device "$MTP_DEVICE" --spec-draft-ngl all)
fi

exec "$LLAMA_SERVER" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8080}" \
  --model "$MODEL" \
  --load-mode mmap \
  --fit off \
  --ctx-size "${CTX_SIZE:-32768}" \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --flash-attn on \
  --batch-size "${BATCH_SIZE:-512}" \
  --ubatch-size "${UBATCH_SIZE:-256}" \
  --threads 16 \
  --threads-batch 16 \
  --gpu-layers 999 \
  --device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 \
  --split-mode tensor \
  --tensor-split 1,1,1,1 \
  --spec-type draft-mtp \
  --spec-draft-model "$MTP_MODEL" \
  --spec-draft-n-max 32 \
  --spec-draft-n-default 2 \
  --spec-draft-n-min 0 \
  --spec-draft-p-min 0 \
  --spec-draft-type-k q8_0 \
  --spec-draft-type-v q8_0 \
  "${mtp_device_args[@]}" \
  "$@"
