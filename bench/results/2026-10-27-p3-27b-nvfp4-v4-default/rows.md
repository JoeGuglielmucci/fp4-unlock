# 2026-10-27-p3-27b-nvfp4-v4-default

| key | value |
|---|---|
| date | 2026-10-27T00:44:05-04:00 |
| tree | p3 |
| llama_cpp_commit | e15eef193 |
| fork_describe | v0.7.0-242-ge15eef193-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2857 clocks_mem_mhz=13801 max_sm_mhz=3090 power_limit_w=575.00 temp_c=65 |
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
| mtp-n3 | 0 | 512 | 142.50 | 0.5696 | 2.627 | 491.7 |
| mtp-n7 | 0 | 512 | 129.95 | 0.3322 | 3.323 | 474.0 |
| mtp-n8 | 0 | 512 | 113.14 | 0.2864 | 3.047 | 444.0 |
| dflash-n3 | 0 | 512 | 158.89 | 0.6020 | 2.590 | 487.6 |
| dflash-n7 | 0 | 512 | 174.37 | 0.3744 | 3.470 | 453.2 |
| synth4 | 0 | 512 | 144.94 | 0.3757 | 3.950 | 439.0 |
| synth8 | 0 | 512 | 282.76 | 0.8593 | 7.860 | 419.2 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| none | 82.1 | 82.0 | 81.9 | 82.0 | 82.0 | 82.0 |
| mtp-n3 | 135.8 | 156.3 | 170.1 | 141.8 | 138.9 | 143.2 |
| mtp-n7 | 125.8 | 136.8 | 163.6 | 122.8 | 124.2 | 134.1 |
| mtp-n8 | 110.3 | 135.7 | 159.2 | 114.2 | 105.7 | 112.1 |
| dflash-n3 | 150.4 | 169.1 | 185.6 | 160.3 | 157.5 | 149.1 |
| dflash-n7 | 177.6 | 191.2 | 215.9 | 165.4 | 157.2 | 171.1 |
| synth4 | 144.9 | 144.9 | 144.7 | 145.0 | 144.9 | 144.9 |
| synth8 | 282.6 | 283.2 | 282.2 | 282.8 | 282.8 | 282.9 |
