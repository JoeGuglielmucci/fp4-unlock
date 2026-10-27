# 2026-10-27-p3-27b-nvfp4-final-np

| key | value |
|---|---|
| date | 2026-10-27T02:45:34-04:00 |
| tree | p3 |
| llama_cpp_commit | e15eef193 |
| fork_describe | v0.7.0-242-ge15eef193-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2865 clocks_mem_mhz=13801 max_sm_mhz=3090 power_limit_w=575.00 temp_c=64 |
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
| np8-none | 8 | 300 | 360.01 | 0.0000 | nan | 426.7 |
| np8-mtp-n3 | 8 | 300 | 428.94 | 0.6501 | 2.975 | 381.4 |
| np8-mtp-n7 | 8 | 300 | 352.33 | 0.4516 | 4.352 | 403.3 |
| np16-none | 16 | 300 | 495.86 | 0.0000 | nan | 373.6 |
| np16-mtp-n3 | 16 | 300 | 466.71 | 0.6452 | 2.975 | 382.4 |
