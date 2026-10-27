# 2026-10-27-p3-27b-nvfp4-final-T3

| key | value |
|---|---|
| date | 2026-10-27T03:01:33-04:00 |
| tree | p3 |
| llama_cpp_commit | e15eef193 |
| fork_describe | v0.7.0-242-ge15eef193-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2730 clocks_mem_mhz=13801 max_sm_mhz=3090 power_limit_w=575.00 temp_c=70 |
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
| mtp-n7 | 0 | 512 | 123.09 | 0.3134 | 2.940 | 476.8 |
| dflash-n7 | 0 | 512 | 168.25 | 0.3693 | 3.100 | 454.7 |

### Per-prompt median tok/s

| config | code-1 | code-2 | code-3 | prose-1 | prose-2 | prose-3 |
|---|---|---|---|---|---|---|
| mtp-n7 | 109.9 | 148.9 | 159.8 | 120.5 | 125.6 | 118.5 |
| dflash-n7 | 154.7 | 222.4 | 218.0 | 164.5 | 172.0 | 153.6 |
