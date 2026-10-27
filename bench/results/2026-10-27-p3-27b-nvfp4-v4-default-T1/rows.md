# 2026-10-27-p3-27b-nvfp4-v4-default-T1

| key | value |
|---|---|
| date | 2026-10-27T01:14:52-04:00 |
| tree | p3 |
| llama_cpp_commit | e15eef193 |
| fork_describe | v0.7.0-242-ge15eef193-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2880 clocks_mem_mhz=14001 max_sm_mhz=3090 power_limit_w=575.00 temp_c=65 |
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
| none | 0 | 512 | 81.97 | 0.0000 | nan | 532.5 |
| mtp-n3 | 0 | 512 | 152.05 | 0.6001 | 2.810 | 476.5 |
| dflash-n7 | 0 | 512 | 186.93 | 0.4042 | 3.340 | 437.1 |
| synth8 | 0 | 512 | 282.77 | 0.8593 | 7.860 | 417.0 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| none | 82.1 | 82.0 | 81.9 | 82.0 | 82.0 | 82.0 |
| mtp-n3 | 147.0 | 159.3 | 171.2 | 146.3 | 149.1 | 155.0 |
| dflash-n7 | 231.0 | 201.1 | 236.1 | 172.7 | 161.1 | 167.2 |
| synth8 | 282.7 | 283.1 | 282.2 | 282.9 | 282.7 | 282.9 |
