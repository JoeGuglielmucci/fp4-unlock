# 2026-10-26-dev-27b-nvfp4-invariant-off

| key | value |
|---|---|
| date | 2026-10-26T21:32:52-04:00 |
| tree | dev |
| llama_cpp_commit | e15eef193 |
| fork_describe | v0.7.0-242-ge15eef193-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2872 clocks_mem_mhz=14001 max_sm_mhz=3090 power_limit_w=575.00 temp_c=71 |
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
| none | 0 | 512 | 82.03 | 0.0000 | nan | 535.7 |
| mtp-n7 | 0 | 512 | 113.60 | 0.3219 | 3.283 | 544.8 |
| dflash-n7 | 0 | 512 | 146.82 | 0.3747 | 3.570 | 533.0 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| none | 82.1 | 82.0 | 82.0 | 82.0 | 82.0 | 82.0 |
| mtp-n7 | 101.5 | 134.8 | 141.2 | 104.4 | 109.5 | 117.7 |
| dflash-n7 | 141.2 | 180.1 | 183.9 | 139.6 | 141.0 | 152.5 |
