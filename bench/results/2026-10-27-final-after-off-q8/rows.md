# 2026-10-27-final-after-off-q8

| key | value |
|---|---|
| date | 2026-10-27T12:34:24-04:00 |
| tree | after |
| llama_cpp_commit | 63b4b303b |
| fork_describe | v0.7.0-241-g63b4b303b-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2842 clocks_mem_mhz=13801 max_sm_mhz=3090 power_limit_w=575.00 temp_c=67 |
| clocks_locked | no   # to lock: sudo nvidia-smi -lgc 3090,3090 -lmc 14001, then set CLOCKS_LOCKED=yes |
| model | 27b-nvfp4 path=~/models/cdiamond--Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF/Qwen3.8-27B-iMatrix-NVFP4-MTP.gguf |
| model_sha256 | 15b6bb8d0eb15cb6b081c562f8daea86330ec64174cb0f9e1655f3d467484007 |
| kv | q8_0 |
| ctx | 8192 |
| n_predict | 512 |
| runs | 3 |

## Rows 4-7, 10: llama-server (greedy, cache_prompt off, median of 3 timed runs after 1 warm-up)

| config | slots | n_predict | tok/s | acceptance | mean accepted len | avg power W |
|---|---|---|---|---|---|---|
| mtp-n7 | 0 | 512 | 131.82 | 0.3537 | 3.237 | 522.3 |
| dflash-n7 | 0 | 512 | 184.01 | 0.4008 | 3.520 | 504.4 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| mtp-n7 | 178.2 | 139.1 | 143.6 | 110.1 | 120.9 | 124.5 |
| dflash-n7 | 255.3 | 199.2 | 203.3 | 145.6 | 151.6 | 168.9 |
