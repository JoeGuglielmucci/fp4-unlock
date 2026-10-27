# 2026-10-27-p3-27b-nvfp4-final-off

| key | value |
|---|---|
| date | 2026-10-27T02:20:45-04:00 |
| tree | p3 |
| llama_cpp_commit | e15eef193 |
| fork_describe | v0.7.0-242-ge15eef193-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2872 clocks_mem_mhz=13801 max_sm_mhz=3090 power_limit_w=575.00 temp_c=65 |
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
| none | 0 | 512 | 82.15 | 0.0000 | nan | 534.8 |
| mtp-n3 | 0 | 512 | 149.70 | 0.5737 | 2.590 | 550.5 |
| mtp-n5 | 0 | 512 | 132.38 | 0.4215 | 2.923 | 562.3 |
| mtp-n7 | 0 | 512 | 113.62 | 0.3219 | 3.283 | 542.3 |
| mtp-n8 | 0 | 512 | 113.44 | 0.2795 | 3.010 | 443.7 |
| dflash-n3 | 0 | 512 | 166.62 | 0.6152 | 2.650 | 545.2 |
| dflash-n5 | 0 | 512 | 159.06 | 0.4651 | 3.110 | 556.6 |
| dflash-n7 | 0 | 512 | 146.86 | 0.3747 | 3.570 | 530.8 |
| synth4 | 0 | 512 | 144.85 | 0.3757 | 3.950 | 438.7 |
| synth6 | 0 | 512 | 211.40 | 0.6095 | 5.810 | 430.5 |
| synth8 | 0 | 512 | 282.21 | 0.8593 | 7.860 | 421.4 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| none | 82.2 | 82.2 | 82.0 | 82.2 | 82.2 | 82.2 |
| mtp-n3 | 139.0 | 169.9 | 183.2 | 144.9 | 152.2 | 147.2 |
| mtp-n5 | 137.0 | 146.2 | 172.5 | 121.3 | 123.3 | 127.7 |
| mtp-n7 | 101.8 | 135.5 | 141.3 | 104.3 | 109.5 | 117.7 |
| mtp-n8 | 103.6 | 139.0 | 140.7 | 112.9 | 114.0 | 110.8 |
| dflash-n3 | 161.5 | 188.3 | 193.7 | 161.4 | 171.7 | 160.5 |
| dflash-n5 | 163.1 | 175.6 | 210.8 | 149.7 | 154.2 | 155.0 |
| dflash-n7 | 141.2 | 180.3 | 183.9 | 139.7 | 141.0 | 152.6 |
| synth4 | 144.9 | 144.9 | 144.6 | 144.9 | 144.8 | 144.4 |
| synth6 | 211.4 | 211.6 | 210.9 | 211.3 | 211.4 | 211.4 |
| synth8 | 282.2 | 282.6 | 281.7 | 282.2 | 282.2 | 282.3 |
