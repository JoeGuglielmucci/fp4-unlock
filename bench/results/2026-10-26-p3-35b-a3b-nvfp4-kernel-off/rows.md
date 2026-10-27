# 2026-10-26-p3-35b-a3b-nvfp4-kernel-off

| key | value |
|---|---|
| date | 2026-10-26T22:20:12-04:00 |
| tree | p3 |
| llama_cpp_commit | e15eef193 |
| fork_describe | v0.7.0-242-ge15eef193-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2857 clocks_mem_mhz=14001 max_sm_mhz=3090 power_limit_w=575.00 temp_c=62 |
| clocks_locked | no   # owner runs: sudo nvidia-smi -lgc 3090,3090 -lmc 14001; set CLOCKS_LOCKED=yes |
| model | 35b-a3b-nvfp4 path=/home/crockettgugg/models/michaelw9999--Qwen3.6-35B-A3B-NVFP4-MTP-GGUF/Qwen3.6-35B-A3B-NVFP4-MTP-HQ.gguf |
| model_sha256 | 777564174a7ccf01a2e9d171ac73206ec3da6b6f6b0124e71a9628ac19f61aa9 |
| kv | f16 |
| ctx | 8192 |
| n_predict | 512 |
| runs | 3 |

## Rows 4-7, 10: llama-server (greedy, cache_prompt off, median of 3 timed runs after 1 warm-up)

| config | slots | n_predict | tok/s | acceptance | mean accepted len | avg power W |
|---|---|---|---|---|---|---|
| none | 0 | 512 | 279.22 | 0.0000 | nan | 351.8 |
| mtp-n3 | 0 | 512 | 427.46 | 0.6738 | 2.663 | 371.6 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| none | 279.5 | 279.2 | 279.2 | 279.3 | 278.6 | 279.1 |
| mtp-n3 | 440.4 | 462.6 | 451.9 | 385.0 | 414.5 | 370.6 |
