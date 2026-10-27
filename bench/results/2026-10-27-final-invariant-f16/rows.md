# 2026-10-27-final-invariant-f16

| key | value |
|---|---|
| date | 2026-10-27T12:17:59-04:00 |
| tree | invariant |
| llama_cpp_commit | 63b4b303b |
| fork_describe | v0.7.0-241-g63b4b303b-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2880 clocks_mem_mhz=14001 max_sm_mhz=3090 power_limit_w=575.00 temp_c=69 |
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
| none | 0 | 512 | 74.16 | 0.0000 | nan | 466.0 |
| mtp-n3 | 0 | 512 | 141.45 | 0.5733 | 2.627 | 477.6 |
| mtp-n5 | 0 | 512 | 134.22 | 0.4216 | 2.890 | 487.7 |
| mtp-n7 | 0 | 512 | 119.21 | 0.3246 | 3.080 | 494.5 |
| dflash-n3 | 0 | 512 | 157.07 | 0.6055 | 2.630 | 474.9 |
| dflash-n5 | 0 | 512 | 166.99 | 0.4646 | 3.100 | 487.8 |
| dflash-n7 | 0 | 512 | 159.20 | 0.3727 | 3.280 | 497.4 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| none | 74.2 | 74.2 | 74.1 | 74.2 | 74.2 | 74.2 |
| mtp-n3 | 134.3 | 154.2 | 170.8 | 141.3 | 141.1 | 141.6 |
| mtp-n5 | 125.3 | 158.9 | 180.2 | 136.2 | 132.2 | 132.2 |
| mtp-n7 | 108.5 | 144.8 | 168.8 | 118.6 | 117.7 | 119.8 |
| dflash-n3 | 151.1 | 170.0 | 188.9 | 161.8 | 152.3 | 151.6 |
| dflash-n5 | 151.6 | 197.3 | 215.5 | 171.2 | 161.1 | 162.8 |
| dflash-n7 | 142.6 | 199.8 | 223.8 | 163.8 | 154.5 | 154.6 |
