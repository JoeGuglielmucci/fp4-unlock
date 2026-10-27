# rows-q4

| key | value |
|---|---|
| date | 2026-10-26T20:03:37-04:00 |
| tree | after |
| llama_cpp_commit | e15eef193 |
| fork_describe | v0.7.0-242-ge15eef193-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2865 clocks_mem_mhz=13801 max_sm_mhz=3090 power_limit_w=575.00 temp_c=68 |
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
| none | 0 | 512 | 82.00 | 0.0000 | nan | 535.2 |
| mtp-n3 | 0 | 512 | 129.32 | 0.5617 | 2.437 | 438.8 |
| mtp-n7 | 0 | 512 | 141.72 | 0.3799 | 3.713 | 428.9 |
| mtp-n8 | 0 | 512 | 135.38 | 0.3470 | 3.750 | 439.8 |
| dflash-n3 | 0 | 512 | 143.18 | 0.6032 | 2.480 | 429.2 |
| dflash-n7 | 0 | 512 | 197.53 | 0.4350 | 4.220 | 393.5 |
| synth4 | 0 | 512 | 144.53 | 0.3757 | 3.950 | 434.7 |
| synth8 | 0 | 512 | 282.38 | 0.8593 | 7.860 | 419.5 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| none | 82.1 | 82.0 | 81.9 | 82.0 | 82.0 | 82.0 |
| mtp-n3 | 128.7 | 142.1 | 164.2 | 130.0 | 126.0 | 121.6 |
| mtp-n7 | 176.5 | 138.8 | 162.9 | 117.9 | 123.6 | 144.6 |
| mtp-n8 | 184.6 | 133.1 | 161.4 | 114.0 | 118.9 | 137.7 |
| dflash-n3 | 142.2 | 157.7 | 180.1 | 144.1 | 142.2 | 131.5 |
| dflash-n7 | 254.8 | 191.4 | 234.0 | 160.3 | 158.0 | 203.6 |
| synth4 | 144.6 | 144.7 | 144.4 | 144.6 | 144.5 | 144.5 |
| synth8 | 282.3 | 282.9 | 281.8 | 282.3 | 282.6 | 282.5 |
