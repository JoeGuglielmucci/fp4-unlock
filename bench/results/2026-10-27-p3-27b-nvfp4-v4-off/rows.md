# 2026-10-27-p3-27b-nvfp4-v4-off

| key | value |
|---|---|
| date | 2026-10-27T01:07:16-04:00 |
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
| none | 0 | 512 | 82.08 | 0.0000 | nan | 534.6 |
| mtp-n3 | 0 | 512 | 149.73 | 0.5737 | 2.590 | 549.8 |
| mtp-n7 | 0 | 512 | 113.68 | 0.3219 | 3.283 | 541.7 |
| dflash-n7 | 0 | 512 | 146.98 | 0.3747 | 3.570 | 531.3 |
| synth8 | 0 | 512 | 282.20 | 0.8593 | 7.860 | 421.5 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| none | 82.1 | 82.1 | 82.0 | 82.1 | 82.1 | 82.1 |
| mtp-n3 | 139.0 | 169.9 | 183.2 | 144.9 | 152.3 | 147.2 |
| mtp-n7 | 101.7 | 135.5 | 141.4 | 104.5 | 109.5 | 117.8 |
| dflash-n7 | 141.2 | 180.2 | 183.9 | 139.8 | 141.2 | 152.7 |
| synth8 | 282.1 | 282.6 | 281.6 | 282.2 | 282.2 | 282.2 |
