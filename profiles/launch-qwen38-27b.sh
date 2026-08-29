#!/usr/bin/env bash
set -euo pipefail

: "${LLAMA_SERVER:?set LLAMA_SERVER to the patched llama-server binary}"
: "${MODEL:?set MODEL to the main GGUF}"
: "${MTP_MODEL:?set MTP_MODEL to the detached MTP GGUF}"

export GGML_CPU_NUMA_DEVICES=1
export GGML_CPU_NUMA_THREADS=12
export GGML_CPU_NUMA_POLL=100
export GGML_CPU_NUMA_HUGEPAGES=0
export GGML_CPU_NUMA_DIRECT_ALLREDUCE=1

spec_types="${SPEC_TYPES:-draft-mtp}"
spec_args=(
  --spec-type "$spec_types"
  --spec-draft-model "$MTP_MODEL"
  --spec-draft-n-max "${SPEC_DRAFT_N_MAX:-3}"
  --spec-draft-n-min 0
  --spec-draft-p-min "${SPEC_DRAFT_P_MIN:-0.2}"
)
if [[ "$spec_types" == *ngram-mod* ]]; then
  spec_args+=(
    --spec-ngram-mod-n-match "${SPEC_NGRAM_MATCH:-8}"
    --spec-ngram-mod-n-min "${SPEC_NGRAM_MIN:-48}"
    --spec-ngram-mod-n-max "${SPEC_NGRAM_MAX:-64}"
  )
fi

exec "$LLAMA_SERVER" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8080}" \
  --model "$MODEL" \
  --fit off \
  --ctx-size "${CTX_SIZE:-262144}" \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --flash-attn on \
  --threads 12 \
  --threads-batch 12 \
  --gpu-layers 999 \
  --device CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3 \
  --split-mode tensor \
  --tensor-split 1,1,1,1 \
  "${spec_args[@]}" \
  "$@"
