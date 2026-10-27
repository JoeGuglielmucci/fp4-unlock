# 2026-10-26-dev-27b-nvfp4-invariant-on

| key | value |
|---|---|
| date | 2026-10-26T21:24:26-04:00 |
| tree | dev |
| llama_cpp_commit | e15eef193 |
| fork_describe | v0.7.0-242-ge15eef193-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2820 clocks_mem_mhz=13801 max_sm_mhz=3090 power_limit_w=575.00 temp_c=75 |
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
| none | 0 | 512 | 81.60 | 0.0000 | nan | 533.0 |
| mtp-n3 | 0 | 512 | 153.42 | 0.5860 | 2.730 | 549.9 |
| mtp-n7 | 0 | 512 | 106.80 | 0.3104 | 2.860 | 542.5 |
| dflash-n3 | 0 | 512 | 170.08 | 0.6343 | 2.840 | 543.0 |
| dflash-n7 | 0 | 512 | 142.97 | 0.3619 | 3.210 | 533.2 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| none | 81.7 | 81.6 | 81.5 | 81.6 | 81.6 | 81.6 |
| mtp-n3 | 143.5 | 165.2 | 179.7 | 149.0 | 151.7 | 155.2 |
| mtp-n7 | 104.8 | 134.6 | 147.5 | 108.8 | 98.5 | 103.0 |
| dflash-n3 | 165.8 | 180.7 | 198.4 | 168.8 | 168.7 | 171.4 |
| dflash-n7 | 139.8 | 193.6 | 191.8 | 146.2 | 124.7 | 137.4 |
