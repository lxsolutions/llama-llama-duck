# SR950 launch profiles

These launchers capture measured starting points for a four-socket Xeon Gold
6242 host. They use environment variables for every path and contain no
machine-specific directories, credentials, or network addresses.

| Profile | Placement | Starting point |
| --- | --- | --- |
| [`launch-glm53-full.sh`](launch-glm53-full.sh) | four `CPU-NUMA` devices | 16 cores/socket, poll 50, direct collective, 32K, Q8_0 KV, MTP default 2 |
| [`launch-glm53-flash.sh`](launch-glm53-flash.sh) | one NUMA node, preferred memory | 32 cores, mmap, Flash Attention off, F16 KV |
| [`launch-qwen38-27b.sh`](launch-qwen38-27b.sh) | four `CPU-NUMA` devices | plain Q4_0, 16 cores/node, poll 100, hugepages off, MTP depth 2 (prose) or ngram-mod (replay) |
| [`launch-qwen38-flash-next.sh`](launch-qwen38-flash-next.sh) | four `CPU-NUMA` devices | 12 cores/node, poll 100 (max), repack, F16 KV, speculation off (`SINGLE_NODE=1` falls back to the old one-node profile) |

Set `LLAMA_SERVER` and `MODEL` before running a script. Full GLM and Qwen-27B
also require `MTP_MODEL`. `PORT` defaults to 8080, `HOST` to 127.0.0.1, and
additional llama-server arguments can be appended on the command line.

These are benchmark profiles, not universal defaults. Re-sweep thread count,
context, batch geometry, speculative depth, and confidence threshold after any
model, quant, compiler, firmware, or llama.cpp change.
