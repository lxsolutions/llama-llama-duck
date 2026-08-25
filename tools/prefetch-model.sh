#!/usr/bin/env bash
# Warm a GGUF shard set into page cache before llama-server mmaps it.
#
# WHY: the backing device has read_ahead_kb=128, the kernel default.
# llama.cpp's mmap load pattern does not trigger useful readahead at that size,
# so a 147 GB model streams at ~46 MB/s. Sequential parallel reads pull the same
# bytes at full device speed and the subsequent mmap faults hit page cache.
#
# Measured on a 147 GB MoE model over the CPU_REPACK path:
#   without prefetch : 2.7 GB/min of repack progress  (~55 min projected)
#   with prefetch    : 7.3 GB/min                     (~20 min)  = 2.7x
#
# Only useful when free RAM exceeds the model size. Checks that first.
set -euo pipefail
first="${1:?usage: prefetch-model.sh <path-to-shard-00001-of-000NN.gguf> [--wait]}"
wait_flag="${2:-}"
dir="$(dirname "$first")"
base="$(basename "$first" | sed 's/-[0-9]\{5\}-of-[0-9]\{5\}\.gguf$//')"
shopt -s nullglob
shards=("$dir/$base"-*.gguf)
[[ ${#shards[@]} -eq 0 ]] && shards=("$first")

total=0
for f in "${shards[@]}"; do total=$(( total + $(stat -Lc %s "$f") )); done
avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
printf 'prefetch: %d shards, %.1f GB; MemAvailable %.1f GB\n' \
  "${#shards[@]}" "$(echo "$total/1000000000"|bc -l)" "$(echo "$avail_kb/1048576"|bc -l)"
if (( total/1024 > avail_kb )); then
  echo "prefetch: model larger than available RAM - skipping (would thrash)" >&2
  exit 0
fi

pids=()
for f in "${shards[@]}"; do
  dd if="$f" of=/dev/null bs=16M status=none & pids+=($!)
done
if [[ "$wait_flag" == "--wait" ]]; then
  for p in "${pids[@]}"; do wait "$p"; done
  echo "prefetch: complete"
else
  echo "prefetch: ${#pids[@]} streams running in background"
fi
