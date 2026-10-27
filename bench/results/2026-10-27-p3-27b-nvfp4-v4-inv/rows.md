# 2026-10-27-p3-27b-nvfp4-v4-inv

| key | value |
|---|---|
| date | 2026-10-27T00:55:37-04:00 |
| tree | p3 |
| llama_cpp_commit | e15eef193 |
| fork_describe | v0.7.0-242-ge15eef193-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2880 clocks_mem_mhz=14001 max_sm_mhz=3090 power_limit_w=575.00 temp_c=64 |
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
| none | 0 | 512 | 73.13 | 0.0000 | nan | 471.7 |
| mtp-n3 | 0 | 512 | 145.12 | 0.5840 | 2.727 | 489.2 |
| mtp-n7 | 0 | 512 | 128.49 | 0.3377 | 3.353 | 504.7 |
| mtp-n8 | 0 | 512 | 124.95 | 0.3127 | 2.947 | 440.4 |
| dflash-n3 | 0 | 512 | 162.00 | 0.6217 | 2.850 | 485.1 |
| dflash-n7 | 0 | 512 | 166.26 | 0.3899 | 3.570 | 509.2 |
| synth4 | 0 | 512 | 143.38 | 0.3757 | 3.950 | 434.7 |
| synth8 | 0 | 512 | 279.68 | 0.8593 | 7.860 | 417.1 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| none | 73.2 | 73.1 | 73.1 | 73.1 | 73.1 | 73.1 |
| mtp-n3 | 134.2 | 158.4 | 168.8 | 142.3 | 142.9 | 147.4 |
| mtp-n7 | 111.6 | 143.7 | 163.7 | 116.5 | 126.5 | 130.5 |
| mtp-n8 | 184.6 | 151.0 | 139.4 | 110.5 | 101.8 | 107.8 |
| dflash-n3 | 152.0 | 173.8 | 186.7 | 154.1 | 159.7 | 164.3 |
| dflash-n7 | 150.8 | 202.4 | 230.3 | 156.1 | 164.0 | 168.5 |
| synth4 | 143.3 | 143.4 | 143.2 | 143.4 | 143.5 | 143.6 |
| synth8 | 279.6 | 280.2 | 279.3 | 279.6 | 279.8 | 279.8 |
