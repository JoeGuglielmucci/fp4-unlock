# 2026-10-26-after-35b-a3b-nvfp4-np-v2

| key | value |
|---|---|
| date | 2026-10-26T22:17:28-04:00 |
| tree | after |
| llama_cpp_commit | e15eef193 |
| fork_describe | v0.7.0-242-ge15eef193-dirty |
| driver | 595.99.02 |
| cuda | release 13.2 |
| gpu | NVIDIA GeForce RTX 5090 |
| clocks_sm_mhz | 2865 clocks_mem_mhz=14001 max_sm_mhz=3090 power_limit_w=575.00 temp_c=62 |
| clocks_locked | no   # owner runs: sudo nvidia-smi -lgc 3090,3090 -lmc 14001; set CLOCKS_LOCKED=yes |
| model | 35b-a3b-nvfp4 path=/home/crockettgugg/models/michaelw9999--Qwen3.6-35B-A3B-NVFP4-MTP-GGUF/Qwen3.6-35B-A3B-NVFP4-MTP-HQ.gguf |
| model_sha256 | 777564174a7ccf01a2e9d171ac73206ec3da6b6f6b0124e71a9628ac19f61aa9 |
| kv | f16 |
| ctx | 8192 |
| n_predict | 512 |
| runs | 3 |

## Rows 4-7, 10: llama-server (greedy, cache_prompt off, median of 3 timed runs after 1 warm-up)

| config | slots | n_predict | tok/s | acceptance | mean accepted len | avg power W |
|---|---|---|---|---|---|---|
| np8-none | 8 | 300 | 785.08 | 0.0000 | nan | 358.9 |
| np8-mtp-n3 | 8 | 300 | 829.98 | 0.7657 | 3.312 | 303.0 |
| np16-none | 16 | 300 | 841.63 | 0.0000 | nan | 300.1 |
| np16-mtp-n3 | 16 | 300 | 901.97 | 0.7160 | 3.159 | 302.4 |
