# 2026-10-27-ab-before2-f16

| key | value |
|---|---|
| date | 2026-10-27T11:02:50-04:00 |
| tree | before |
| llama_cpp_commit | 63b4b303b |
| fork_describe | v0.7.0-241-g63b4b303b-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2812 clocks_mem_mhz=13801 max_sm_mhz=3090 power_limit_w=575.00 temp_c=69 |
| clocks_locked | no   # to lock: sudo nvidia-smi -lgc 3090,3090 -lmc 14001, then set CLOCKS_LOCKED=yes |
| model | 27b-nvfp4 path=~/models/cdiamond--Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF/Qwen3.8-27B-iMatrix-NVFP4-MTP.gguf |
| model_sha256 | 15b6bb8d0eb15cb6b081c562f8daea86330ec64174cb0f9e1655f3d467484007 |
| kv | f16 |
| ctx | 8192 |
| n_predict | 512 |
| runs | 3 |

## Rows 4-7, 10: llama-server (greedy, cache_prompt off, median of 3 timed runs after 1 warm-up)

| config | slots | n_predict | tok/s | acceptance | mean accepted len | avg power W |
|---|---|---|---|---|---|---|
| mtp-n7 | 0 | 512 | 121.67 | 0.3311 | 3.073 | 529.1 |
| dflash-n7 | 0 | 512 | 164.34 | 0.3718 | 3.300 | 513.1 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| mtp-n7 | 117.4 | 143.8 | 170.0 | 115.2 | 123.4 | 120.0 |
| dflash-n7 | 151.1 | 202.2 | 229.1 | 159.9 | 167.6 | 161.1 |
