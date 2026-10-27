# 2026-10-27-final-after-off-f16

| key | value |
|---|---|
| date | 2026-10-27T12:07:08-04:00 |
| tree | after |
| llama_cpp_commit | 63b4b303b |
| fork_describe | v0.7.0-241-g63b4b303b-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2880 clocks_mem_mhz=14001 max_sm_mhz=3090 power_limit_w=575.00 temp_c=66 |
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
| none | 0 | 512 | 82.04 | 0.0000 | nan | 534.6 |
| mtp-n3 | 0 | 512 | 149.62 | 0.5737 | 2.590 | 548.8 |
| mtp-n5 | 0 | 512 | 138.66 | 0.4244 | 3.123 | 550.5 |
| mtp-n7 | 0 | 512 | 120.83 | 0.3311 | 3.073 | 526.6 |
| dflash-n3 | 0 | 512 | 164.56 | 0.6152 | 2.650 | 537.6 |
| dflash-n5 | 0 | 512 | 173.14 | 0.4804 | 3.280 | 534.0 |
| dflash-n7 | 0 | 512 | 162.20 | 0.3718 | 3.300 | 505.1 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| none | 82.1 | 82.0 | 82.0 | 82.1 | 82.0 | 82.0 |
| mtp-n3 | 139.0 | 169.7 | 182.8 | 144.7 | 152.2 | 147.0 |
| mtp-n5 | 133.3 | 158.7 | 181.9 | 128.5 | 132.0 | 144.0 |
| mtp-n7 | 116.5 | 142.8 | 168.1 | 114.3 | 122.3 | 119.3 |
| dflash-n3 | 159.4 | 186.1 | 192.2 | 159.2 | 169.8 | 158.4 |
| dflash-n5 | 172.4 | 200.6 | 222.7 | 159.2 | 163.8 | 173.8 |
| dflash-n7 | 149.2 | 199.8 | 226.0 | 157.9 | 165.5 | 158.9 |
