== default (MMQ_PREC auto)
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | CUDA       |  99 |           pp512 |   15898.49 ± 2120.60 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | CUDA       |  99 |            tg32 |       293.82 ± 57.46 |
== GGML_CUDA_MMQ_PREC=q8
| model                          |       size |     params | backend    | ngl |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | CUDA       |  99 |           pp512 |    15371.16 ± 758.33 |
| gpt-oss 20B MXFP4 MoE          |  11.27 GiB |    20.91 B | CUDA       |  99 |            tg32 |        384.64 ± 5.23 |
exit 0
