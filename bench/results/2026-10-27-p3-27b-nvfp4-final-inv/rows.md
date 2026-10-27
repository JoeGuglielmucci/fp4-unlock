# 2026-10-27-p3-27b-nvfp4-final-inv

| key | value |
|---|---|
| date | 2026-10-27T02:36:44-04:00 |
| tree | p3 |
| llama_cpp_commit | e15eef193 |
| fork_describe | v0.7.0-242-ge15eef193-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2880 clocks_mem_mhz=14001 max_sm_mhz=3090 power_limit_w=575.00 temp_c=64 |
| clocks_locked | no   # owner runs: sudo nvidia-smi -lgc 3090,3090 -lmc 14001; set CLOCKS_LOCKED=yes |
| model | 27b-nvfp4 path=~/models/cdiamond--Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF/Qwen3.8-27B-iMatrix-NVFP4-MTP.gguf |
| model_sha256 | 15b6bb8d0eb15cb6b081c562f8daea86330ec64174cb0f9e1655f3d467484007 |
| kv | f16 |
| ctx | 8192 |
| n_predict | 512 |
| runs | 3 |

## Rows 4-7, 10: llama-server (greedy, cache_prompt off, median of 3 timed runs after 1 warm-up)

| config | slots | n_predict | tok/s | acceptance | mean accepted len | avg power W |
|---|---|---|---|---|---|---|
| none | 0 | 512 | 74.32 | 0.0000 | nan | 467.4 |
| mtp-n3 | 0 | 512 | 142.27 | 0.5733 | 2.627 | 479.2 |
| mtp-n7 | 0 | 512 | 119.24 | 0.3246 | 3.080 | 497.5 |
| dflash-n3 | 0 | 512 | 157.34 | 0.6055 | 2.630 | 475.3 |
| dflash-n7 | 0 | 512 | 159.21 | 0.3727 | 3.280 | 497.1 |
| synth8 | 0 | 512 | 280.41 | 0.8593 | 7.860 | 417.9 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| none | 74.3 | 74.3 | 74.3 | 74.3 | 74.3 | 74.3 |
| mtp-n3 | 135.2 | 155.5 | 171.2 | 141.9 | 141.2 | 142.6 |
| mtp-n7 | 108.5 | 144.9 | 169.0 | 118.6 | 117.7 | 119.8 |
| dflash-n3 | 151.4 | 170.7 | 189.5 | 162.0 | 152.7 | 152.1 |
| dflash-n7 | 142.7 | 200.1 | 223.7 | 163.8 | 154.6 | 154.6 |
| synth8 | 280.3 | 280.7 | 279.9 | 280.4 | 280.4 | 280.4 |
