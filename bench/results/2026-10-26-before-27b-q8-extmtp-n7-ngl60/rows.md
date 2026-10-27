# 2026-10-26-before-27b-q8-extmtp-n7-ngl60

| key | value |
|---|---|
| date | 2026-10-26T21:03:27-04:00 |
| tree | before |
| llama_cpp_commit | 63b4b303b |
| fork_describe | v0.7.0-242-ge15eef193-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2407 clocks_mem_mhz=14001 max_sm_mhz=3090 power_limit_w=575.00 temp_c=52 |
| clocks_locked | no   # owner runs: sudo nvidia-smi -lgc 3090,3090 -lmc 14001; set CLOCKS_LOCKED=yes |
| model | 27b-q8 path=/home/crockettgugg/models/unsloth--Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf |
| model_sha256 | a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348 |
| kv | f16 |
| ctx | 4096 |
| n_predict | 512 |
| runs | 3 |

## Rows 4-7, 10: llama-server (greedy, cache_prompt off, median of 3 timed runs after 1 warm-up)

| config | slots | n_predict | tok/s | acceptance | mean accepted len | avg power W |
|---|---|---|---|---|---|---|
| extmtp-n7 | 0 | 512 | 37.73 | 0.3283 | 3.093 | 227.7 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| extmtp-n7 | 36.8 | 45.0 | 38.6 | 32.8 | 43.0 | 35.6 |
