# 2026-10-27-p3-27b-nvfp4-final-default

| key | value |
|---|---|
| date | 2026-10-27T02:05:30-04:00 |
| tree | p3 |
| llama_cpp_commit | e15eef193 |
| fork_describe | v0.7.0-242-ge15eef193-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2880 clocks_mem_mhz=14001 max_sm_mhz=3090 power_limit_w=575.00 temp_c=65 |
| clocks_locked | no   # owner runs: sudo nvidia-smi -lgc 3090,3090 -lmc 14001; set CLOCKS_LOCKED=yes |
| model | 27b-nvfp4 path=/home/crockettgugg/models/cdiamond--Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF/Qwen3.8-27B-iMatrix-NVFP4-MTP.gguf |
| model_sha256 | 15b6bb8d0eb15cb6b081c562f8daea86330ec64174cb0f9e1655f3d467484007 |
| kv | f16 |
| ctx | 8192 |
| n_predict | 512 |
| runs | 3 |

## Rows 4-7, 10: llama-server (greedy, cache_prompt off, median of 3 timed runs after 1 warm-up)

| config | slots | n_predict | tok/s | acceptance | mean accepted len | avg power W |
|---|---|---|---|---|---|---|
| none | 0 | 512 | 82.08 | 0.0000 | nan | 534.2 |
| mtp-n3 | 0 | 512 | 149.82 | 0.5737 | 2.590 | 550.2 |
| mtp-n5 | 0 | 512 | 138.81 | 0.4299 | 3.073 | 492.4 |
| mtp-n7 | 0 | 512 | 127.07 | 0.3151 | 3.293 | 468.2 |
| mtp-n8 | 0 | 512 | 113.62 | 0.2805 | 3.020 | 443.9 |
| dflash-n3 | 0 | 512 | 166.51 | 0.6152 | 2.650 | 545.6 |
| dflash-n5 | 0 | 512 | 173.31 | 0.4800 | 3.170 | 479.2 |
| dflash-n7 | 0 | 512 | 169.05 | 0.3674 | 3.500 | 444.7 |
| synth4 | 0 | 512 | 145.04 | 0.3757 | 3.950 | 440.3 |
| synth6 | 0 | 512 | 211.35 | 0.6095 | 5.810 | 430.0 |
| synth8 | 0 | 512 | 282.67 | 0.8593 | 7.860 | 418.2 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| none | 82.1 | 82.1 | 82.0 | 82.0 | 82.1 | 82.1 |
| mtp-n3 | 139.1 | 169.9 | 183.2 | 145.0 | 152.4 | 147.3 |
| mtp-n5 | 137.1 | 165.0 | 173.4 | 126.5 | 132.1 | 140.6 |
| mtp-n7 | 109.8 | 137.1 | 160.8 | 121.1 | 121.6 | 132.5 |
| mtp-n8 | 104.3 | 139.1 | 142.0 | 113.0 | 114.3 | 111.6 |
| dflash-n3 | 161.5 | 188.1 | 193.5 | 161.2 | 171.6 | 160.2 |
| dflash-n5 | 179.4 | 197.4 | 223.0 | 157.7 | 161.2 | 167.2 |
| dflash-n7 | 152.9 | 195.4 | 221.5 | 162.5 | 165.2 | 172.9 |
| synth4 | 144.9 | 145.0 | 144.7 | 145.1 | 145.1 | 145.1 |
| synth6 | 211.3 | 211.6 | 211.2 | 211.6 | 211.4 | 211.3 |
| synth8 | 282.4 | 283.0 | 282.0 | 282.8 | 282.7 | 282.7 |
