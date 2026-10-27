# 2026-10-27-ab-p3-2-f16

| key | value |
|---|---|
| date | 2026-10-27T11:05:40-04:00 |
| tree | p3 |
| llama_cpp_commit | 63b4b303b |
| fork_describe | v0.7.0-241-g63b4b303b-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2767 clocks_mem_mhz=13801 max_sm_mhz=3090 power_limit_w=575.00 temp_c=69 |
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
| mtp-n7 | 0 | 512 | 126.65 | 0.3151 | 3.293 | 464.6 |
| dflash-n7 | 0 | 512 | 168.94 | 0.3674 | 3.500 | 442.9 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| mtp-n7 | 109.7 | 137.0 | 160.8 | 121.2 | 121.5 | 131.8 |
| dflash-n7 | 152.2 | 194.7 | 221.3 | 162.1 | 165.1 | 172.8 |
