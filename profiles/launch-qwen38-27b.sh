#!/usr/bin/env bash
# Qwen3.8-27B on a 4-socket CPU host, tensor-sharded across NUMA nodes.
#
# Measured on 4x Xeon Gold 6242 / 755 GiB (see benchmarks/ten-tokens-per-second.md):
#   novel prose   8.33 tok/s   (Q4_0 + MTP draft, depth 2)
#   replay/agentic 10.90 mean, 17.13 warm   (Q4_0 + ngram-mod)
#
# QUANT MATTERS AND THE CURVE IS NOT MONOTONIC. Measured, same host, same day:
#   UD-Q3_K_XL 13.1 GB -> 5.25 raw     (Q3_K extraction collapses to 69 GB/s)
#   Q4_0       16.1 GB -> 7.42 raw     <- optimum
#   UD-Q4_K_XL 17.6 GB -> 7.86 w/ MTP  (vs Q4_0's 8.33; worse repack coverage)
#   UD-Q8_K_XL 31.4 GB -> 6.12 raw
# Plain Q4_0 wins because ~86% of its bytes actually repack. Do not "upgrade" to
# a UD Q4 variant or "speed it up" with Q3 without re-measuring.
set -euo pipefail

: "${LLAMA_SERVER:?set LLAMA_SERVER to the patched llama-server binary}"
: "${MODEL:?set MODEL to the main GGUF (plain Q4_0 recommended)}"

# THREADS is per NUMA node, not total. 16 measured best for this model; 8/12/20/24
# all land 6.8-7.0 against 7.42. Note Qwen3.8-Flash-Next prefers 12 on the same
# host -- the optimum does not transfer between models. Re-sweep after any change.
THREADS="${THREADS:-16}"

export GGML_CPU_NUMA_DEVICES=1
export GGML_CPU_NUMA_THREADS="$THREADS"
export GGML_CPU_NUMA_POLL="${NUMA_POLL:-100}"
export GGML_CPU_NUMA_DIRECT_ALLREDUCE=1
export GGML_CPU_NUMA_REPACK=1
# Hugepages measured 7.75 vs 8.33 -- off is deliberate.
export GGML_CPU_NUMA_HUGEPAGES="${NUMA_HUGEPAGES:-0}"

# Speculation. Pick by workload; there is no setting that wins both:
#   SPEC_TYPE=draft-mtp   novel prose      8.33 (needs MTP_MODEL)
#   SPEC_TYPE=ngram-mod   replay/agentic  10.90 mean, 17.13 warm, no draft model
# ngram-mod on novel prose is a large regression on some models -- do not default
# to it blindly, and check the acceptance line in the server log after switching.
spec_types="${SPEC_TYPE:-draft-mtp}"
spec_args=(--spec-type "$spec_types")

if [[ "$spec_types" == *draft-mtp* ]]; then
  : "${MTP_MODEL:?draft-mtp requested but MTP_MODEL is unset}"
  # Depth 2 measured best: 7.46 / 8.33 / 7.04 for n_max 1 / 2 / 3.
  # Draft on all four devices beat one node (8.33 vs 7.05); cutting draft threads
  # to 4 or 8 made it worse. Defaults below are the measured optimum.
  spec_args+=(
    --spec-draft-model "$MTP_MODEL"
    --spec-draft-device "${DEVICES:-CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3}"
    --spec-draft-ngl all
    --spec-draft-n-max "${SPEC_DRAFT_N_MAX:-2}"
    --spec-draft-n-min 0
    --spec-draft-p-min "${SPEC_DRAFT_P_MIN:-0}"
  )
fi
if [[ "$spec_types" == *ngram-mod* ]]; then
  # n_match=8 beats the default 24. The engine warns 8 is "too small"; that
  # warning is about draft hit-rate, not correctness -- accepted tokens are
  # verified against the target and are exactly what it would have emitted.
  spec_args+=(
    --spec-ngram-mod-n-match "${SPEC_NGRAM_MATCH:-8}"
    --spec-ngram-mod-n-min "${SPEC_NGRAM_MIN:-48}"
    --spec-ngram-mod-n-max "${SPEC_NGRAM_MAX:-64}"
  )
fi

exec "$LLAMA_SERVER" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" \
  --model "$MODEL" \
  --load-mode mmap --fit off --gpu-layers 999 \
  --ctx-size "${CTX_SIZE:-262144}" \
  --device "${DEVICES:-CPU-NUMA0,CPU-NUMA1,CPU-NUMA2,CPU-NUMA3}" \
  --split-mode tensor --tensor-split "${TENSOR_SPLIT:-1,1,1,1}" \
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
  --threads "$THREADS" --threads-batch "$THREADS" \
  "${spec_args[@]}" \
  "$@"
