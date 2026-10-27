# 2026-10-27-p3-27b-nvfp4-v4-default-T2

| key | value |
|---|---|
| date | 2026-10-27T01:29:38-04:00 |
| tree | p3 |
| llama_cpp_commit | e15eef193 |
| fork_describe | v0.7.0-242-ge15eef193-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2835 clocks_mem_mhz=13801 max_sm_mhz=3090 power_limit_w=575.00 temp_c=70 |
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
| none | 0 | 512 | 81.99 | 0.0000 | nan | 534.1 |
| mtp-n3 | 0 | 512 | 144.22 | 0.5732 | 2.680 | 483.3 |
| mtp-n7 | 0 | 512 | 131.04 | 0.3248 | 3.380 | 467.1 |
| dflash-n7 | 0 | 512 | 177.33 | 0.3815 | 3.700 | 442.3 |
| synth8 | 0 | 512 | 282.64 | 0.8593 | 7.860 | 420.1 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| none | 82.1 | 82.0 | 81.9 | 82.0 | 82.0 | 82.0 |
| mtp-n3 | 134.3 | 156.3 | 170.2 | 142.7 | 142.7 | 145.7 |
| mtp-n7 | 120.4 | 140.1 | 154.3 | 117.0 | 126.1 | 135.9 |
| dflash-n7 | 170.6 | 195.3 | 212.9 | 160.3 | 171.8 | 182.8 |
| synth8 | 282.6 | 283.1 | 282.1 | 282.6 | 282.6 | 282.7 |
