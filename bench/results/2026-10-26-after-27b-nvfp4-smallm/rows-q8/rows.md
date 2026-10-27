# rows-q8

| key | value |
|---|---|
| date | 2026-10-26T19:51:40-04:00 |
| tree | after |
| llama_cpp_commit | e15eef193 |
| fork_describe | v0.7.0-242-ge15eef193-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2647 clocks_mem_mhz=14001 max_sm_mhz=3090 power_limit_w=575.00 temp_c=69 |
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
| none | 0 | 512 | 82.00 | 0.0000 | nan | 532.6 |
| mtp-n3 | 0 | 512 | 136.32 | 0.5761 | 2.640 | 505.9 |
| mtp-n7 | 0 | 512 | 133.44 | 0.3480 | 3.010 | 482.7 |
| mtp-n8 | 0 | 512 | 128.95 | 0.3164 | 3.040 | 495.1 |
| dflash-n3 | 0 | 512 | 148.87 | 0.6055 | 2.740 | 503.6 |
| dflash-n7 | 0 | 512 | 183.21 | 0.4020 | 3.490 | 458.6 |
| synth4 | 0 | 512 | 143.71 | 0.3757 | 3.950 | 492.4 |
| synth8 | 0 | 512 | 280.65 | 0.8590 | 7.860 | 472.3 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| none | 82.1 | 82.1 | 82.0 | 82.0 | 82.0 | 82.0 |
| mtp-n3 | 138.2 | 140.8 | 153.5 | 132.8 | 134.2 | 134.4 |
| mtp-n7 | 162.6 | 143.8 | 158.3 | 115.7 | 123.1 | 118.6 |
| mtp-n8 | 157.1 | 143.3 | 150.0 | 108.9 | 114.6 | 111.2 |
| dflash-n3 | 150.5 | 153.0 | 166.4 | 146.4 | 145.4 | 147.2 |
| dflash-n7 | 227.5 | 190.6 | 210.1 | 158.3 | 175.9 | 171.8 |
| synth4 | 143.8 | 143.9 | 143.3 | 143.8 | 143.7 | 143.7 |
| synth8 | 280.8 | 281.2 | 280.2 | 280.7 | 280.6 | 280.5 |
