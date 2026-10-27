# 2026-10-27-final-before-f16

| key | value |
|---|---|
| date | 2026-10-27T11:21:09-04:00 |
| tree | before |
| llama_cpp_commit | 63b4b303b |
| fork_describe | v0.7.0-241-g63b4b303b-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2482 clocks_mem_mhz=13801 max_sm_mhz=3090 power_limit_w=575.00 temp_c=39 |
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
| none | 0 | 512 | 82.10 | 0.0000 | nan | 518.8 |
| mtp-n3 | 0 | 512 | 148.79 | 0.5737 | 2.590 | 543.6 |
| mtp-n5 | 0 | 512 | 137.96 | 0.4244 | 3.123 | 548.5 |
| mtp-n7 | 0 | 512 | 121.82 | 0.3311 | 3.073 | 529.0 |
| dflash-n3 | 0 | 512 | 166.02 | 0.6152 | 2.650 | 541.9 |
| dflash-n5 | 0 | 512 | 174.52 | 0.4804 | 3.280 | 538.2 |
| dflash-n7 | 0 | 512 | 163.48 | 0.3718 | 3.300 | 505.5 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| none | 82.2 | 82.1 | 82.1 | 82.1 | 82.1 | 82.1 |
| mtp-n3 | 138.5 | 169.9 | 182.6 | 144.9 | 152.3 | 145.3 |
| mtp-n5 | 132.0 | 158.6 | 181.9 | 128.5 | 131.7 | 143.9 |
| mtp-n7 | 116.4 | 143.2 | 169.7 | 114.2 | 123.3 | 120.4 |
| dflash-n3 | 161.7 | 188.5 | 193.6 | 160.5 | 170.3 | 160.1 |
| dflash-n5 | 174.6 | 203.5 | 225.1 | 160.1 | 164.3 | 174.4 |
| dflash-n7 | 149.0 | 199.4 | 226.0 | 157.3 | 166.7 | 160.3 |
