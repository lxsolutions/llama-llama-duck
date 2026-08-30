#!/usr/bin/env bash
# Launch llama-server under one env/arg combination, benchmark it, tear it down.
#
# Every arm gets a fresh process, so an arm cannot inherit warmed state from the
# previous one -- which matters, because a server that has already generated
# tokens will beat a cold one and silently flatter whichever arm you ran second.
#
# Also prints the resulting NUMA page placement and RssAnon/RssFile while the
# process is still alive. Those two lines catch the two most common ways a
# sharded run is not doing what you think: skewed placement, and weights left on
# the shared mmap instead of copied into node-bound memory.
#
# Usage:
#   sweep_config.sh <label> <port> <bin_dir> <model> "<env>" "<server_args>" [bench args...]
#
# Example:
#   sweep_config.sh t12 8181 /opt/llama-numa /models/m.gguf \
#     "GGML_CPU_NUMA_DEVICES=1 GGML_CPU_NUMA_THREADS=12" \
#     "--gpu-layers 999 --device CPU-NUMA0,CPU-NUMA1 --split-mode tensor" \
#     --workloads prose --reps 2
set -uo pipefail

if [ $# -lt 6 ]; then
  sed -n '2,20p' "$0" >&2
  exit 2
fi

LABEL="$1"; PORT="$2"; BIN="$3"; MODEL="$4"; EXTRA_ENV="$5"; EXTRA_ARGS="$6"; shift 6

OUTDIR="${OUTDIR:-.}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG="$OUTDIR/srv_${LABEL}.log"
mkdir -p "$OUTDIR"
rm -f "$LOG"

# shellcheck disable=SC2086
nohup env LD_LIBRARY_PATH="$BIN/bin" $EXTRA_ENV \
  "$BIN/bin/llama-server" --host 127.0.0.1 --port "$PORT" --model "$MODEL" \
  --alias bench --no-webui --metrics $EXTRA_ARGS \
  > "$LOG" 2>&1 &
PID=$!

ready=0
for i in $(seq 1 "${READY_TRIES:-180}"); do
  if curl -s --max-time 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok; then
    ready=1; echo "[$LABEL] ready after $((i*5))s"; break
  fi
  if ! ps -p $PID >/dev/null 2>&1; then
    echo "[$LABEL] SERVER DIED"; tail -15 "$LOG"; exit 1
  fi
  sleep 5
done
if [ $ready -eq 0 ]; then
  echo "[$LABEL] TIMEOUT waiting for health"; kill -9 $PID 2>/dev/null; exit 1
fi

python3 "$HERE/decode_bench.py" --port "$PORT" --model bench --label "$LABEL" \
  --out "$OUTDIR/bench_${LABEL}.json" "$@"
rc=$?

# Placement and residency, read while the process still exists.
awk '{for(i=1;i<=NF;i++) if($i ~ /^N[0-9]+=/){split($i,a,"=");n[a[1]]+=a[2]}}
     END{printf "[placement] "; for(k in n) printf "%s=%.1fGB ", k, n[k]*4096/1e9; print ""}' \
     /proc/$PID/numa_maps 2>/dev/null
grep -E 'RssAnon|RssFile' /proc/$PID/status 2>/dev/null | tr '\n' ' '; echo

kill $PID 2>/dev/null
for _ in $(seq 1 60); do ps -p $PID >/dev/null 2>&1 || break; sleep 1; done
ps -p $PID >/dev/null 2>&1 && kill -9 $PID 2>/dev/null
sleep 3
exit $rc
