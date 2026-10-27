# 2026-10-27-ab-before-f16

| key | value |
|---|---|
| date | 2026-10-27T10:43:56-04:00 |
| tree | before |
| llama_cpp_commit | 63b4b303b |
| fork_describe | v0.7.0-241-g63b4b303b-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 180 clocks_mem_mhz=405 max_sm_mhz=3090 power_limit_w=575.00 temp_c=37 |
| clocks_locked | no   # owner runs: sudo nvidia-smi -lgc 3090,3090 -lmc 14001; set CLOCKS_LOCKED=yes |
| model | 27b-nvfp4 path=~/models/cdiamond--Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF/Qwen3.8-27B-iMatrix-NVFP4-MTP.gguf |
| model_sha256 | 15b6bb8d0eb15cb6b081c562f8daea86330ec64174cb0f9e1655f3d467484007 |
| kv | f16 |
| ctx | 8192 |
| n_predict | 512 |
| runs | 3 |

## Rows 4-7, 10: llama-server (greedy, cache_prompt off, median of 3 timed runs after 1 warm-up)

| config | slots | n_predict | tok/s | acceptance | mean accepted len | avg power W |
|---|---|---|---|---|---|---|
| none | 0 | 512 | 82.09 | 0.0000 | nan | 520.7 |
| mtp-n7 | 0 | 512 | 121.90 | 0.3311 | 3.073 | 527.3 |
| dflash-n7 | 0 | 512 | 164.28 | 0.3718 | 3.300 | 513.1 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| none | 82.2 | 82.2 | 82.1 | 82.0 | 82.0 | 82.1 |
| mtp-n7 | 117.3 | 143.9 | 170.2 | 115.4 | 123.3 | 120.5 |
| dflash-n7 | 150.5 | 202.6 | 229.2 | 160.0 | 167.6 | 161.0 |
