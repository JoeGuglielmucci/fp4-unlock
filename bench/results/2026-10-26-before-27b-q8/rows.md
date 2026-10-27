# 2026-10-26-before-27b-q8

| key | value |
|---|---|
| date | 2026-10-26T19:33:08-04:00 |
| tree | before |
| llama_cpp_commit | 63b4b303b |
| fork_describe | v0.7.0-242-ge15eef193-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 180 clocks_mem_mhz=405 max_sm_mhz=3090 power_limit_w=575.00 temp_c=54 |
| clocks_locked | no   # owner runs: sudo nvidia-smi -lgc 3090,3090 -lmc 14001; set CLOCKS_LOCKED=yes |
| model | 27b-q8 path=/home/crockettgugg/models/unsloth--Qwen3.8-27B-GGUF/Qwen3.8-27B-Q8_0.gguf |
| model_sha256 | a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348 |
| kv | f16 |
| ctx | 4096 |
| n_predict | 512 |
| runs | 3 |

## Rows 1-2: llama-bench (batch 1)

| test | t/s | stddev | samples |
|---|---|---|---|
| pp512 | 4687.51 | 10.03 | 4687.0 4697.8 4677.7 |
| pp8192 | 4453.99 | 8.83 | 4463.2 4453.2 4445.6 |
| tg128 | 53.46 | 0.02 | 53.5 53.5 53.5 |

## Rows 4-7, 10: llama-server (greedy, cache_prompt off, median of 3 timed runs after 1 warm-up)

| config | slots | n_predict | tok/s | acceptance | mean accepted len | avg power W |
|---|---|---|---|---|---|---|
| none | 0 | 512 | 53.05 | 0.0000 | nan | 499.1 |
| extmtp-n3 | 0 | 512 | 115.82 | 0.5729 | 2.630 | 544.4 |
| np8-none | 8 | 300 | 280.01 | 0.0000 | nan | 537.5 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| none | 53.0 | 53.1 | 53.0 | 53.0 | 53.0 | 53.0 |
| extmtp-n3 | 117.4 | 131.6 | 108.8 | 114.2 | 126.3 | 114.3 |
