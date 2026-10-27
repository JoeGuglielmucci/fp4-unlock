# 2026-10-27-final-after-f16

| key | value |
|---|---|
| date | 2026-10-27T11:57:09-04:00 |
| tree | after |
| llama_cpp_commit | 63b4b303b |
| fork_describe | v0.7.0-241-g63b4b303b-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2407 clocks_mem_mhz=14001 max_sm_mhz=3090 power_limit_w=575.00 temp_c=41 |
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
| none | 0 | 512 | 82.16 | 0.0000 | nan | 519.0 |
| mtp-n3 | 0 | 512 | 149.61 | 0.5737 | 2.590 | 544.7 |
| mtp-n5 | 0 | 512 | 136.31 | 0.4116 | 2.963 | 486.5 |
| mtp-n7 | 0 | 512 | 130.15 | 0.3172 | 3.327 | 461.8 |
| dflash-n3 | 0 | 512 | 166.39 | 0.6152 | 2.650 | 545.0 |
| dflash-n5 | 0 | 512 | 171.23 | 0.4502 | 3.040 | 468.3 |
| dflash-n7 | 0 | 512 | 176.92 | 0.3674 | 3.500 | 430.4 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| none | 82.3 | 82.3 | 82.2 | 82.2 | 82.2 | 82.1 |
| mtp-n3 | 138.8 | 169.8 | 183.2 | 144.9 | 152.3 | 146.9 |
| mtp-n5 | 135.6 | 157.1 | 166.0 | 127.5 | 130.4 | 137.1 |
| mtp-n7 | 112.5 | 139.8 | 166.4 | 123.6 | 124.1 | 136.2 |
| dflash-n3 | 161.5 | 188.1 | 193.5 | 160.9 | 171.3 | 160.1 |
| dflash-n5 | 179.0 | 190.1 | 203.3 | 157.3 | 159.7 | 163.4 |
| dflash-n7 | 160.2 | 202.6 | 231.7 | 167.9 | 172.8 | 181.1 |
