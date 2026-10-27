# 2026-10-26-after-27b-nvfp4-smallmall-q8-bench

| key | value |
|---|---|
| date | 2026-10-26T20:21:31-04:00 |
| tree | after |
| llama_cpp_commit | e15eef193 |
| fork_describe | v0.7.0-242-ge15eef193-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2865 clocks_mem_mhz=14001 max_sm_mhz=3090 power_limit_w=575.00 temp_c=67 |
| clocks_locked | no   # owner runs: sudo nvidia-smi -lgc 3090,3090 -lmc 14001; set CLOCKS_LOCKED=yes |
| model | 27b-nvfp4 path=/home/crockettgugg/models/cdiamond--Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF/Qwen3.8-27B-iMatrix-NVFP4-MTP.gguf |
| model_sha256 | 15b6bb8d0eb15cb6b081c562f8daea86330ec64174cb0f9e1655f3d467484007 |
| kv | f16 |
| ctx | 8192 |
| n_predict | 512 |
| runs | 3 |

## Rows 1-2: llama-bench (batch 1)

| test | t/s | stddev | samples |
|---|---|---|---|
| pp512 | 3811.19 | 4.90 | 3810.1 3816.6 3807.0 |
| pp8192 | 3690.63 | 5.57 | 3696.3 3690.4 3685.2 |
| tg128 | 76.31 | 0.03 | 76.3 76.3 76.3 |

## Rows 4-7, 10: llama-server (greedy, cache_prompt off, median of 3 timed runs after 1 warm-up)

| config | slots | n_predict | tok/s | acceptance | mean accepted len | avg power W |
|---|---|---|---|---|---|---|
| none | 0 | 512 | 75.42 | 0.0000 | nan | 501.6 |
| mtp-n3 | 0 | 512 | 136.32 | 0.5774 | 2.640 | 505.5 |
| mtp-n7 | 0 | 512 | 131.24 | 0.3467 | 3.013 | 478.9 |
| mtp-n8 | 0 | 512 | 125.66 | 0.3142 | 3.047 | 493.7 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| none | 75.5 | 75.5 | 75.4 | 75.4 | 75.4 | 75.4 |
| mtp-n3 | 138.3 | 140.2 | 152.8 | 132.4 | 134.3 | 133.8 |
| mtp-n7 | 161.4 | 141.8 | 155.9 | 115.4 | 120.7 | 117.7 |
| mtp-n8 | 153.2 | 139.0 | 148.8 | 109.3 | 112.3 | 110.3 |
