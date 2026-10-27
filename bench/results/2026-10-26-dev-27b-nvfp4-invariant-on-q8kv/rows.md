# 2026-10-26-dev-27b-nvfp4-invariant-on-q8kv

| key | value |
|---|---|
| date | 2026-10-26T21:38:28-04:00 |
| tree | dev |
| llama_cpp_commit | e15eef193 |
| fork_describe | v0.7.0-242-ge15eef193-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2865 clocks_mem_mhz=14001 max_sm_mhz=3090 power_limit_w=575.00 temp_c=72 |
| clocks_locked | no   # owner runs: sudo nvidia-smi -lgc 3090,3090 -lmc 14001; set CLOCKS_LOCKED=yes |
| model | 27b-nvfp4 path=/home/crockettgugg/models/cdiamond--Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF/Qwen3.8-27B-iMatrix-NVFP4-MTP.gguf |
| model_sha256 | 15b6bb8d0eb15cb6b081c562f8daea86330ec64174cb0f9e1655f3d467484007 |
| kv | q8_0 |
| ctx | 8192 |
| n_predict | 512 |
| runs | 3 |

## Rows 4-7, 10: llama-server (greedy, cache_prompt off, median of 3 timed runs after 1 warm-up)

| config | slots | n_predict | tok/s | acceptance | mean accepted len | avg power W |
|---|---|---|---|---|---|---|
| none | 0 | 512 | 80.24 | 0.0000 | nan | 525.3 |
| mtp-n3 | 0 | 512 | 160.11 | 0.6226 | 2.760 | 538.7 |
| dflash-n3 | 0 | 512 | 174.65 | 0.6546 | 2.850 | 537.2 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| none | 80.3 | 80.2 | 79.9 | 80.2 | 80.2 | 80.3 |
| mtp-n3 | 189.9 | 166.0 | 170.7 | 148.8 | 140.1 | 154.2 |
| dflash-n3 | 208.9 | 179.9 | 187.9 | 161.8 | 155.0 | 169.4 |
