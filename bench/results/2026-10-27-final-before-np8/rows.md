# 2026-10-27-final-before-np8

| key | value |
|---|---|
| date | 2026-10-27T12:37:05-04:00 |
| tree | before |
| llama_cpp_commit | 63b4b303b |
| fork_describe | v0.7.0-241-g63b4b303b-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2790 clocks_mem_mhz=14001 max_sm_mhz=3090 power_limit_w=575.00 temp_c=70 |
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
| np8-none | 8 | 300 | 350.99 | 0.0000 | nan | 501.6 |
| np8-mtp-n3 | 8 | 300 | 396.54 | 0.6550 | 3.010 | 402.1 |
| np8-mtp-n7 | 8 | 300 | 327.14 | 0.4050 | 3.990 | 414.4 |
