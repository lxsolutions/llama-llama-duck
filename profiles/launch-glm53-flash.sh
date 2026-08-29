#!/usr/bin/env bash
set -euo pipefail

: "${LLAMA_SERVER:?set LLAMA_SERVER to llama-server}"
: "${MODEL:?set MODEL to the main GGUF}"
command -v numactl >/dev/null || {
  echo "numactl is required" >&2
  exit 1
}

mmproj_args=()
if [[ -n "${MMPROJ:-}" ]]; then
  mmproj_args=(--mmproj "$MMPROJ")
fi

node="${NUMA_NODE:-0}"
exec numactl --cpunodebind="$node" --preferred="$node" \
  "$LLAMA_SERVER" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8080}" \
  --model "$MODEL" \
  "${mmproj_args[@]}" \
  --load-mode mmap \
  --gpu-layers 0 \
  --ctx-size "${CTX_SIZE:-65536}" \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --flash-attn off \
  --threads "${THREADS:-32}" \
  --threads-batch "${THREADS_BATCH:-32}" \
  "$@"
