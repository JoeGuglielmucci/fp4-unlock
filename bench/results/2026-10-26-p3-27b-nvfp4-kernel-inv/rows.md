# 2026-10-26-p3-27b-nvfp4-kernel-inv

| key | value |
|---|---|
| date | 2026-10-26T22:01:06-04:00 |
| tree | p3 |
| llama_cpp_commit | e15eef193 |
| fork_describe | v0.7.0-242-ge15eef193-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2647 clocks_mem_mhz=14001 max_sm_mhz=3090 power_limit_w=575.00 temp_c=73 |
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
| none | 0 | 512 | 66.26 | 0.0000 | nan | 539.0 |
| mtp-n3 | 0 | 512 | 129.58 | 0.5586 | 2.487 | 537.0 |
| mtp-n7 | 0 | 512 | 108.56 | 0.3143 | 2.827 | 546.1 |
| mtp-n8 | 0 | 512 | 130.68 | 0.3528 | 3.350 | 443.5 |
| dflash-n3 | 0 | 512 | 147.45 | 0.6076 | 2.610 | 531.3 |
| dflash-n7 | 0 | 512 | 144.04 | 0.3668 | 3.150 | 551.4 |
| synth4 | 0 | 512 | 141.99 | 0.3757 | 3.950 | 442.0 |
| synth8 | 0 | 512 | 277.46 | 0.8593 | 7.860 | 423.3 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| none | 66.3 | 66.3 | 66.2 | 66.3 | 66.2 | 66.3 |
| mtp-n3 | 133.0 | 140.6 | 158.5 | 126.2 | 125.7 | 124.3 |
| mtp-n7 | 110.7 | 127.7 | 158.4 | 102.2 | 106.4 | 102.2 |
| mtp-n8 | 183.2 | 134.8 | 154.6 | 120.0 | 126.5 | 120.8 |
| dflash-n3 | 148.8 | 156.2 | 175.5 | 141.7 | 146.1 | 139.4 |
| dflash-n7 | 146.7 | 172.0 | 208.7 | 141.4 | 141.1 | 137.7 |
| synth4 | 142.0 | 142.2 | 141.9 | 142.0 | 142.0 | 141.9 |
| synth8 | 277.5 | 278.0 | 277.0 | 277.4 | 277.5 | 277.4 |
