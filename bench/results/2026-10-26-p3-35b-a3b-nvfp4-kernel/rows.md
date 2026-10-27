# 2026-10-26-p3-35b-a3b-nvfp4-kernel

| key | value |
|---|---|
| date | 2026-10-26T22:18:51-04:00 |
| tree | p3 |
| llama_cpp_commit | e15eef193 |
| fork_describe | v0.7.0-242-ge15eef193-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2865 clocks_mem_mhz=13801 max_sm_mhz=3090 power_limit_w=575.00 temp_c=58 |
| clocks_locked | no   # owner runs: sudo nvidia-smi -lgc 3090,3090 -lmc 14001; set CLOCKS_LOCKED=yes |
| model | 35b-a3b-nvfp4 path=/home/crockettgugg/models/michaelw9999--Qwen3.6-35B-A3B-NVFP4-MTP-GGUF/Qwen3.6-35B-A3B-NVFP4-MTP-HQ.gguf |
| model_sha256 | 777564174a7ccf01a2e9d171ac73206ec3da6b6f6b0124e71a9628ac19f61aa9 |
| kv | f16 |
| ctx | 8192 |
| n_predict | 512 |
| runs | 3 |

## Rows 1-2: llama-bench (batch 1)

| test | t/s | stddev | samples |
|---|---|---|---|
| pp512 | 14042.02 | 272.30 | 13731.8 14152.5 14241.7 |
| pp8192 | 13552.10 | 18.99 | 13538.1 13544.5 13573.7 |
| tg128 | 284.84 | 0.22 | 285.0 284.6 284.9 |

## Rows 4-7, 10: llama-server (greedy, cache_prompt off, median of 3 timed runs after 1 warm-up)

| config | slots | n_predict | tok/s | acceptance | mean accepted len | avg power W |
|---|---|---|---|---|---|---|
| none | 0 | 512 | 274.90 | 0.0000 | nan | 350.6 |
| mtp-n3 | 0 | 512 | 425.74 | 0.6844 | 2.760 | 370.2 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| none | 275.0 | 274.9 | 274.7 | 274.9 | 274.8 | 274.9 |
| mtp-n3 | 433.0 | 450.2 | 418.5 | 386.4 | 452.2 | 381.5 |
