#!/usr/bin/env bash
set -euo pipefail

: "${LLAMA_SERVER:?set LLAMA_SERVER to the compatible llama-server binary}"
: "${MODEL:?set MODEL to the main GGUF}"
command -v numactl >/dev/null || {
  echo "numactl is required" >&2
  exit 1
}

mmproj_args=()
if [[ -n "${MMPROJ:-}" ]]; then
  mmproj_args=(--mmproj "$MMPROJ")
fi

spec_args=()
if [[ -n "${MTP_MODEL:-}" ]]; then
  spec_args=(
    --spec-type draft-mtp
    --spec-draft-model "$MTP_MODEL"
    --spec-draft-n-max "${SPEC_DRAFT_N_MAX:-3}"
    --spec-draft-n-min 0
    --spec-draft-p-min "${SPEC_DRAFT_P_MIN:-0.6}"
  )
fi

node="${NUMA_NODE:-0}"
exec numactl --cpunodebind="$node" --membind="$node" \
  "$LLAMA_SERVER" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8080}" \
  --model "$MODEL" \
  "${mmproj_args[@]}" \
  --load-mode none \
  --gpu-layers 0 \
  --ctx-size "${CTX_SIZE:-65536}" \
  --threads "${THREADS:-32}" \
  --threads-batch "${THREADS_BATCH:-32}" \
  "${spec_args[@]}" \
  "$@"
