# Findings

Measurements and decisions that did not become code. Newest at the bottom.

## 2026-10-26 Phase 0.2 toolchain

| Tool | Version |
|---|---|
| GPU / driver | RTX 5090 32607 MiB, driver 595.99.02 |
| CUDA | 13.2 (nvcc V13.2.86), /usr/local/cuda-13.2 |
| cmake / ninja / gcc | 3.28.3 / 1.11.1 / 13.3.0 |
| python / hf CLI | 3.12.3 / huggingface_hub 1.11.0 |
| docker | 29.9.2 |
| nsys | absent (owner: apt install cuda-nsight-systems-13-2) |
| ncu | 2022.4.1 (too old for sm_120; owner: apt install cuda-nsight-compute-13-2) |
| clang-format | absent (owner: apt install clang-format) |
| vLLM local image | voipmonitor/vllm id 820181fbbc97, 0.11.2.dev280 custom b12x cu132 build |
| vLLM official | vllm/vllm-openai:v0.31.0 pull attempted 2026-10-26, result below |

Already in the HF cache (not GGUF unless noted): unsloth/Qwen3.8-27B-NVFP4 (safetensors 22.6 GB + model_mtp.safetensors 849 MB, vLLM target),
QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 (safetensors), gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 (README only cached).

## 2026-10-26 Phase 0.2 vLLM 0.31.0 on sm_120

`vllm/vllm-openai:v0.31.0` (vllm 0.31.0, torch 2.13.0+cu130, arch list includes sm_120) runs
unsloth/Qwen3.8-27B-NVFP4 on the 5090. Kernel selection from the engine log:
FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM, FLASHINFER attention, FlashInfer GDN prefill,
CutlassFP8ScaledMM for the W8A8 parts. Weights 21.15 GiB; KV cache 5.48 GiB at max_model_len 4096.

- First attempt (gpu_memory_utilization 0.85, CUDA graphs on) OOMed during profiling (12.25 GiB alloc
  on top of 23.3 GB). Second attempt in eager mode, util 0.92, max_num_seqs 8, max_num_batched_tokens
  2048, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: ran, greedy output coherent.
- The 4.8 tok/s printed by the smoke script is NOT a throughput number: the first generate call
  included a FlashInfer JIT build of the xqa attention kernel. A real vLLM number needs CUDA graphs
  back on (lower max_model_len or util), a warm-up generate, and the JIT cache mounted
  (`/root/.cache/vllm`, `/root/.cache/flashinfer`). That is Phase 1.3, not Phase 0.
- Decision: 0.31.0 is the vLLM comparator. The local 0.11.2-dev image is kept only as a fallback.
- Note the GPU drives the desktop; vLLM saw 30.86 of 31.36 GiB free at startup.

## 2026-10-26 Phase 0.2 cuBLAS for CUDA 13

No cuBLAS 13 dev package is installed (only the CUDA 12 runtime from Ubuntu and ollama's bundled
libcublas.so.13). CMake fails with "Target ggml-cuda links to CUDA::cublas but the target was not
found". Workaround used for Phase 0: `.cuda-overlay/{lib,include}` symlinks to the pip
`nvidia-cublas` cu13 wheel already present in ~/hyperqwen-fidelity/quantlab/.venv-quant, passed via
`-DCUDA_cublas_LIBRARY`, `-DCUDA_cublasLt_LIBRARY`, `-I` flags and `CMAKE_BUILD_RPATH` (see
scripts/build.sh). Clean fix, owner: `sudo apt install libcublas-dev-13-2` (13.4.1.3-1, 433 MB); the
overlay block in build.sh becomes a no-op once /usr/local/cuda-13.2/lib64/libcublas.so exists.

## 2026-10-26 Phase 0.3 builds and baseline tests (upstream 63b4b303, unpatched)

All three builds pass: build-before, build (fork branch fp4-smallm, identical to upstream so far),
build-cpu. Logs: *.cmake.log, *.build.log in the repo root (gitignored).

`test-backend-ops test -b CUDA0 -o MUL_MAT,MUL_MAT_ID -p "nvfp4|mxfp4"`: 280/280 passed
(bench/results/phase0/tbo-test-before.log). Note: type names are lower-case in the test names.

`test-backend-ops perf -b CUDA0 -o MUL_MAT -p "type_a=nvfp4"` (m=4096, k=14336, f32 activations;
bench/results/phase0/tbo-perf-before.log):

| n (batch) | us/run | TFLOPS | path on this commit |
|---|---|---|---|
| 1 | 9.37 | 12.5 | MMVQ (dp4a) |
| 2 | 10.63 | 22.1 | MMVQ |
| 3 | 16.22 | 21.7 | MMVQ |
| 4 | 15.79 | 29.7 | MMVQ |
| 5 | 21.13 | 27.8 | MMVQ |
| 8 | 30.22 | 31.1 | MMVQ |
| 512 | 99.73 | 602.9 | MMQ native FP4 MMA (GGML_PREC_Q4) |

Reading:
- The weight tensor is 4096 x 14336 x 4.5 bits = 33.0 MB. At 9.37 us that is 3.5 TB/s, above the
  card's 1.79 TB/s, so repeated runs are L2-resident. The DRAM floor for one read is ~18.4 us.
- n=8 costs 3.2x n=1 on the MMVQ path. A bandwidth-bound kernel would cost ~1x (same weight read).
  Even against the DRAM floor (18.4 us), MMVQ at n=8 (30.2 us) is 1.6x over the floor: the verify
  step is compute-bound on dp4a exactly where the PRD predicted (Section 4).
- The native FP4 MMA path is real and fast at n=512 (603 TFLOPS). It is simply never used for
  2 <= n <= 8 because of MMVQ_MAX_BATCH_SIZE 8 (mmvq.cuh:3, ggml-cuda.cu:1888).
- Upper bound for a tensor-core small-M kernel on this shape: ~1.6x per matmul vs MMVQ at n=8 in the
  DRAM-resident case, ~3x in the L2-resident case. Real models are DRAM-resident, so plan on the
  1.3-1.6x per-matmul number for the verify step, not 3x. Attention and non-matmul time dilute it
  further; Phase 1.5 profiling decides.
- The one-line routing experiment (route NVFP4 2<=ne11<=8 to MMQ J=8) is the first thing to run on
  the fork branch: it gives the n=2..8 MMQ number on the same table without writing a kernel.

## 2026-10-26 Phase 0.4 gpt-oss-20b MXFP4 smoke (build-before)

`llama-bench -ngl 99 -p 512 -n 32 -r 3`, native FP4 MMQ (default) vs `GGML_CUDA_MMQ_PREC=q8`:
pp512 15.9k vs 15.4k t/s, tg32 294 +/- 57 vs 385 +/- 5 t/s. Smoke only: a CUDA build was running
on all CPU cores during the first run, hence the variance. Both paths load and run; MXFP4 is a
secondary model (PRD 7.4), not re-measured until Phase 1 hygiene applies.

## 2026-10-26 Phase 0.3 batch-invariance diagnostic (PR #28488 tests on CUDA0, upstream 63b4b303)

Ankk98's test cases (PR #28488, head cc21fe83b) were 3-way-applied onto a detached scratch worktree
(`third_party/llama.cpp-pr28488`, gitignored, not carried). Two conflict regions were purely
additive and resolved by keeping both sides. Two edits of our own for the diagnostic:
`GGML_TYPE_NVFP4` and `GGML_TYPE_MXFP4` added to the MMVQ loop; the three `test_fa_naive`
registrations skipped because they build an FA shape CUDA does not support and the test does not
gate on `supports_op`, so the binary aborts at fattn.cu:761 (`BEST_FATTN_KERNEL_NONE`). That abort
is a test-harness gap in the PR, not a CUDA bug; worth a comment on the PR from the owner.

Result (bench/results/phase0/tbo-pr28488-batch-all.log, 22/24):

| Test | N=1 vs N=2 on CUDA | Note |
|---|---|---|
| MMVQ_BATCH q4_0 / q4_K / q6_K / q8_0 | bit-exact | 2 shapes each |
| MMVQ_BATCH nvfp4 / mxfp4 | bit-exact | our addition; MMVQ for FP4 is invariant |
| RMS_BATCH m=512..5120 | bit-exact | fused partials vs regular |
| FA_BATCH (hs 64, 128; GQA 2) | bit-exact | small shapes, f16 K/V |
| DMMV_VIEW_BATCH, DMMV_BATCH | bit-exact | |
| P021_BATCH nh=4, nh=8 | FAIL, max 2.7e-3 | FA-off K x Q path (p021 kernel vs GEMM) |

Reading for this project:
- At this commit the CUDA MMVQ path is batch-invariant between N=1 and N=2, including NVFP4 and
  MXFP4. So the dp4a kernel itself is not the #25618 culprit; the known divergence sources are the
  kernel switches (MMVQ -> MMQ at N>8, FA vec -> tile/mma by Q->ne[1], f16 vs quantized K/V).
  These tests do not cross those thresholds (N=2 only, small FA shapes).
- A small-M tensor-core kernel for 2<=N<=8 adds a new kernel switch at N=2 unless it also serves
  N=1, or unless N=1 keeps MMVQ and the new kernel is proven bit-compatible, which it cannot be
  (different accumulation order). Design rule stands: serve N=1..8 with one kernel, or accept and
  measure a bounded divergence (Phase 3.0 decision, test 1.8 measures the status quo first).
- The P021 failure only matters with flash attention off; the benchmark runs with `-fa on`.
- Phase 1.8 should extend this harness to N in {1,2,4,8,9,16} for nvfp4 so the MMVQ->MMQ switch at
  N=9 and any new kernel are covered by the same test.

## 2026-10-26 Phase 0.4 cdiamond Qwen3.8-27B NVFP4 + MTP loads (build-before)

- `llama-completion -ngl 99 --fit off -c 8192 -n 20 --temp 0`: loads; 20 greedy tokens coherent.
  Decode 81.9 tok/s (19 runs), prompt 469 tok/s on 13 tokens, load 1.2 s from page cache.
  Note: `--fit off` without `-c` tries the model's 262144 training context and OOMs on the rs cache
  (149 MiB alloc failed with ~23 GB already allocated). Always pass `-c` with `--fit off`.
- The spec-decode flags (`--spec-type` etc.) are llama-server / llama-speculative-simple flags, not
  llama-completion flags. Load checks for drafters go through scripts/run-27b.sh.
- llama-server, `--spec-type draft-mtp --spec-draft-n-max 8 --spec-draft-backend-sampling`,
  greedy 64 tokens on a code prompt: 157.0 tok/s, draft acceptance 0.4375 (49 of 112), mean
  accepted length 4.50 per verify. Output coherent (Qwen3.8 thinking preamble). GPU idle otherwise,
  no hang on this short run.
- So the "before" picture on this card: plain 82 tok/s, MTP n=8 already 157 tok/s (1.9x) with the
  verify step on dp4a MMVQ. The project's lever is the cost of that verify step at N=9 (8 drafts +
  1), which today runs through MMVQ at 3.2x the N=1 cost (perf table above).

## 2026-10-26 Phase 0.4 DFlash2 against the NVFP4 target, and a live #25618 observation

- llama-server, `-md z-lab Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 8`,
  same greedy code prompt, 64 tokens: 172.3 tok/s, acceptance 49 of 96 (0.51). Loads and runs.
- Same prompt, no drafter: 82.0 tok/s.

Same weights, same prompt, temp 0, top-k 1, seed 7, three different continuations after
"1. Recursive (naive) - ":

| Drafter | Continuation |
|---|---|
| none | "simple but exponential time O(2^n)" |
| MTP n=8 | "O(2^n) time, simple but slow" |
| DFlash2 n=8 | "simple" (captured text truncated there; may match none) |

That is issue #25618 reproduced on this card in the first hour: greedy output depends on the
verify batch size because N=1 and N>1 take different kernels (MMVQ vs MMQ at N>8, FA vec vs
tile/mma by Q->ne[1], different reduction orders). It is the exact thing Phase 1.8 measures and
Phase 3.0 must not make worse. Logs: bench/results/phase0/server-cdiamond-{none,mtp,dflash}.log.

Phase 0 "before" summary on the RTX 5090, upstream 63b4b303, cdiamond NVFP4 27B, 8K ctx, f16 KV:

| Mode | tok/s | acceptance |
|---|---|---|
| plain | 82.0 | - |
| MTP n_max=8 | 157.0 | 0.44 (mean len 4.5) |
| DFlash2 n_max=8 | 172.3 | 0.51 |

Single prompt, 64 tokens, no hygiene (PRD 7.1), not benchmark numbers. They set expectations:
the baseline already has 1.9-2.1x from speculation on dp4a; the project claims the next step on
top of that, and "lossless 2-3x" must be measured against plain with 7.1 hygiene in Phase 1.

## 2026-10-26 Phase 0.4/1.1 cdiamond GGUF tensor types, Q8_0 memory, harness dry runs

**What is actually NVFP4 in the cdiamond 27B GGUF** (gguf dump, arch qwen35, 65 blocks incl. the
nextn MTP block 64, n_embd 5120, n_ff 17408, 24 q heads / 4 kv heads, ctx 262144):

| Tensor group | Type | Shape |
|---|---|---|
| ffn_gate, ffn_up | NVFP4 | [5120, 17408] |
| ffn_down | NVFP4 | [17408, 5120] |
| ssm_alpha, ssm_beta, nextn.eh_proj | NVFP4 | small |
| attn_q/k/v/output, ssm_out (and the rest of the GDN projections) | Q5_K | - |
| output.weight | Q8_0 | [5120, vocab] |

About 65% of the per-token weight bytes are NVFP4; the rest is Q5_K/Q8_0 and goes through the
k-quant MMVQ/MMQ paths whatever we do. Consequence for the kernel plan: the attainable step-time
gain from a small-M FP4 kernel is bounded by that 65% share (Amdahl), and the 27B perf shapes for
test-backend-ops are (m=17408, k=5120) and (m=5120, k=17408). `bench/perf-shapes.patch` adds those
shapes for NVFP4/MXFP4/Q4_0/Q8_0 at n in {1,2,3,4,6,8,12,16,32,64,512} behind
`FP4_UNLOCK_PERF_SHAPES=1`; scripts/build.sh applies it to both trees (benchmark-only change).

**Q8_0 reference target memory** (unsloth Qwen3.8-27B-Q8_0, build-before, -ngl 99, f16 KV):

| Config | Peak VRAM (telemetry memory.used) | Result |
|---|---|---|
| Q8_0 alone, -c 4096 | 27,560 MiB | ok, 52.8 tok/s plain (16-token dry run) |
| Q8_0 + unsloth MTP head (-md mtp-Qwen3.8-27B-Q4_0), -c 4096, n_max 3 | ~27.7 GiB | 129 tok/s, acc 10/11, mean len 3.50 (dry) |
| Q8_0 + DFlash2 Q4_K_M drafter, -c 8192 or 4096, -ngl 99 | OOM (cudaMalloc 1079.61 MiB fails) | - |
| Q8_0 + DFlash2, -ngl 60 (5 target layers on CPU), -c 4096 | 30,814 MiB | 46 tok/s, acc 11/17, mean len 4.67 (dry) |

So the Q8_0 column is attention-only for 1.6 attribution: acceptance on the Q8_0 target is valid
at -ngl 60 (the math is the same), its tok/s is not comparable and is never reported as speed.
The harness takes `NGL=60` for exactly that pair. The dry acceptance numbers above are 1-prompt,
16-token runs; the 1.2/1.6 table replaces them.

**Harness validation (bench/ scripts, DRY=1 on the NVFP4 target, build-before).** Bugs found and
fixed before the real baseline:

- llama-bench jsonl has no `test` field; the label is built from n_prompt / n_gen / n_depth.
- llama-perplexity needs `--kl-divergence` in addition to `--kl-divergence-base FNAME`; without it
  the flag is an alias of `--save-all-logits` and the run silently overwrote the reference file
  with the quantized model's logits. The reference was deleted and is regenerated by kld.py.
- llama-batched-bench with `-pps` (shared prompt) fails on qwen35 (M-RoPE position check:
  "last position stored ... 512, batch starting position 512, required X < Y"). The scaling row
  uses per-sequence prompts; -c is sized to max(M) * (pp + tg).
- The server holds port 8080 for a few seconds after SIGTERM; server_start waits for the port.
- The KL summary line is `Mean ln(PPL(Q)/PPL(base))     :` (one space, padded before the colon).

Dry KL (2 chunks of 512, kl-set-5k, Q8_0 reference; NOT a result, chunk count too small):

| NVFP4 path | Mean KLD | Median | 99.9% | Max | Same top-1 |
|---|---|---|---|---|---|
| mmvq (-b 8 -ub 8) | 0.0257 | 0.0047 | 0.681 | 0.832 | 96.1% |
| mmq-auto (-b 512 -ub 512) | 0.0343 | 0.0066 | 0.856 | 0.869 | 94.1% |

The two paths already differ on the same weights, which is the batch-size dependence of #25618
seen through KL rather than through greedy divergence. Phase 1.2 runs the full set (all chunks)
and 1.8 records whether the difference is inside the chunk-to-chunk noise.

Dry invariance (code-1, 32 tokens, f16 KV): spec-off run A vs run B identical; MTP n_max 3 vs
spec-off identical (acc 22/25). The 64-token Phase 0 divergence appeared with n_max 8; the full
1.8 run covers n_max 3/7, both drafters, both KV types, six prompts, 512 tokens.

Dry llama-bench, build-before, cdiamond NVFP4, f16 KV: pp512 6663 t/s, tg128 83.2 t/s (3 reps,
no clock lock, telemetry recorded).

## 2026-10-26 Phase 1.2 note: DFlash2 draft length is capped at 7 on this drafter

`z-lab/Qwen3.8-27B-DFlash2-GGUF` carries `dflash.block_size=8`; llama.cpp clamps any `--spec-draft-n-max` above 7 to 7
with a warning (`common_speculative_impl_draft_dflash: requested draft size (n_max=15, n_min=0) exceeds the trained block
size 8 -- clamping to 7`). The planned `dflash-n15` row therefore duplicates `dflash-n7` exactly (same acceptance
6627/17826 in the before baseline). PRD 7.2 row 5 is amended: DFlash rows are `dflash-n3` and `dflash-n7`; the verify
batch for DFlash on the 27B is at most M = 8, which is inside the kernel's M in [2,8] scope. The 35B-A3B DFlash (if any)
must be checked for its own block size before assuming M > 8.

## 2026-10-26 Phase 1.2 frozen "before" baseline: 27B NVFP4 (cdiamond), build-before 63b4b303, driver 595.99.02

Hygiene: clocks NOT locked (needs sudo; owner), GPU cooled below 60 C before each row, nothing else on the GPU,
telemetry sampled every 1 s, greedy (temp 0, top-k 1, seed 7), cache_prompt off, f16 KV, -c 8192, 512 predicted tokens
per prompt, 3 timed runs after warm-up, median. Full tables: bench/results/2026-10-26-before-27b-nvfp4/rows.md.

### Rows 1-3: llama-bench and batched-bench

| row | number |
|---|---|
| pp512 | 6653 t/s |
| pp8192 | 6409 t/s |
| tg128 (M=1) | 83.2 t/s |
| batched-bench step(1) | 12.07 ms |
| step(2)/step(1) | 1.049 |
| step(4)/step(1) | 1.183 |
| step(8)/step(1) | 1.472 |
| step(16)/step(1) | 1.852 |
| step(64)/step(1) | 4.370 |

### Rows 4-7: llama-server end to end

| config | slots | tok/s | acceptance | mean accepted len | power W |
|---|---|---|---|---|---|
| none | 1 | 82.01 | - | - | 535 |
| mtp-n3 | 1 | 149.79 | 0.574 | 2.59 | 548 |
| mtp-n7 | 1 | 122.03 | 0.331 | 3.07 | 530 |
| mtp-n8 | 1 | 113.50 | 0.281 | 3.02 | 443 |
| dflash-n7 | 1 | 159.15 | 0.372 | 3.30 | 499 |
| dflash-n15 (= n7, clamped) | 1 | 157.69 | 0.372 | 3.30 | 495 |
| synth4 (random accept, mean 3.95) | 1 | 141.94 | - | 3.95 | 429 |
| synth8 (random accept, mean 7.86) | 1 | 279.58 | - | 7.86 | 412 |
| np8-none | 8 | 352.06 | - | - | 499 |
| np8-mtp-n3 | 8 | 372.28 | 0.648 | 2.98 | 402 |
| np16-none | 16 | 496.50 | - | - | 369 |
| np16-mtp-n3 | 16 | 443.41 | 0.645 | 2.99 | 381 |

Per prompt, mtp-n3 ranges 139 (code-1) to 183 (code-3) tok/s; dflash-n7 146 to 222. Code prompts accept more.
Speculation stops paying at 16 slots (443 vs 497 aggregate), as expected: the verify batch competes with real batching.

### Phase 1.3 KL vs the Q8_0 target (bench/kld.py, all wikitext chunks at ctx 512, reference = unsloth Q8_0 on mmvq)

| path | batch | env | Mean KLD | Median | 99% | Max | ln(PPL ratio) | Same top |
|---|---|---|---|---|---|---|---|---|
| mmvq (M<=8 path, Q8_1 activations) | -b 8 | | 0.0431 | 0.0022 | 0.727 | 8.70 | 0.0342 | 95.32% |
| mmq-q8 (tensor cores, Q8_1 activations) | -b 512 | GGML_CUDA_MMQ_PREC=q8 | 0.0429 | 0.0021 | 0.742 | 8.99 | 0.0348 | 95.35% |
| mmq-auto (= q4 on Blackwell for NVFP4) | -b 512 | | 0.0618 | 0.0030 | 0.972 | 9.08 | 0.0500 | 94.33% |
| mmq-q4 (native W4A4 FP4 MMA) | -b 512 | GGML_CUDA_MMQ_PREC=q4 | 0.0618 | 0.0030 | 0.972 | 9.08 | 0.0500 | 94.33% |

- The NVFP4 weights themselves cost Mean KLD 0.043 against Q8_0 on the Q8_1-activation paths, and mmvq and mmq-q8 agree
  to within 0.0002, so the tensor-core path with 8-bit activations is numerically equivalent to the dp4a path. This is
  the "lossless" bar: the small-M path must keep Q8_1 activations (W4A8) or re-use exactly this q8 MMQ kernel.
- Native W4A4 (auto = q4 on this card) adds 43% to the Mean KLD (0.043 -> 0.062) and drops same-top-1 by a point. It is
  faster, but it does not meet the headline. Phase 1.5 must be run under GGML_CUDA_MMQ_PREC=q8 for the lossless column and
  q4 only as a labelled "fast" column. PRD 7.3 K-column rule amended accordingly.
- Phase 1.3 result vs PRD K2 (KL budget): the before-tree batch-size dependence is 0.043 (M=8 path) vs 0.062 (M=512 auto
  path); the after tree must show no KL increase on the q8 column at any M.

### Phase 1.4 roofline, first cut (f16 KV, 8K ctx)

- Weight bytes: 17,125,207,136 (15.95 GiB) over 1.79 TB/s = 9.57 ms memory floor per step; measured step(1) 12.07 ms,
  so M=1 decode runs at about 79% of the bandwidth roofline (the rest is KV reads, activations, GDN state and launch
  gaps). Batch-1 is near the floor: no batch-1 speedup is claimed.
- Speculative verify step cost from the synthetic rows (drafting = 8 sequential MTP-head passes, accept random):
  synth8 = 279.6 tok/s at 8.86 tokens per step = 31.7 ms per step; synth4 = 141.9 tok/s at 4.95 tokens per step = 34.9 ms
  per step. batched-bench says an M=9 batch of separate sequences costs 1.5x step(1) = 18 ms, so the verify step in the
  speculative loop costs about 14 to 17 ms more than the batched forward alone. The split between the MTP-head drafting
  passes, the M=9 target forward on the MMVQ path, and host overhead is the nsys job (1.4/1.6) below.
- Kernel-level cliff (test-backend-ops perf, NVFP4, build-before, 27B FFN shapes), the number the project is about:

| shape (m x k) | n=1 | n=2 | n=4 | n=8 (MMVQ, dp4a) | n=12 (MMQ, FP4 MMA) | n=16 |
|---|---|---|---|---|---|---|
| 17408 x 5120 | 15.43 us | 16.24 | 23.21 | 39.76 | 19.55 | 19.78 |
| 5120 x 17408 | 13.74 us | 15.21 | 23.88 | 44.45 | 25.12 | 25.45 |

  n=8 on the dp4a path costs 2.6x to 3.2x n=1, while n=12 on the tensor-core path costs 1.27x to 1.83x n=1. The routing
  cut at MMVQ_MAX_BATCH_SIZE=8 is a cliff: every draft length a drafter actually uses (verify M = 4 to 9) lands on the slow
  side of it. Phase 1.5 tests moving the cut (FP4_UNLOCK_SMALLM) before any kernel is written. Caveat: these MMQ numbers
  are the auto (q4) precision; the q8 column is measured in Phase 1.5.

### Phase 1.8 divergence rates (bench/invariance.sh, 6 prompts x 512 greedy tokens, build-before)

| KV | none vs none (A/B) | mtp-n3 | mtp-n7 | dflash-n3 | dflash-n7 |
|---|---|---|---|---|---|
| f16 | 0/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| q8_0 | 0/6 | 6/6 | 5/6 | 6/6 | 5/6 |

- Plain decode is bit-stable run to run. Every drafter changes the greedy output, with f16 KV too, so #25618 is not only
  a quantized-KV flash-attention problem on this card: the M>1 target forward itself is not batch-invariant.
- Every first divergence is a near-tie flip: reference top-2 logprob gaps 0.005 to 0.15.
- The first-divergence positions are identical for mtp-n3 and dflash-n3 (f16: 164, 475, 377, 283, 95, 136; q8_0: 396, 231,
  351, 114, 294, 135) and identical for mtp-n7 and dflash-n7 (f16: 134, 213, 25, 218, 95, 103). The verify batch size is
  M = n_max + 1 regardless of what was drafted, so the divergence is a function of M and the KV type, not of the drafter.
  This is the cleanest evidence yet that a batch-invariant small-M path (same numerics for M in [1,8]) removes it, and it
  gives Phase 2 a deterministic reproducer: `bench/invariance.sh before 27b-nvfp4` (results in
  bench/results/2026-10-26-before-27b-nvfp4-invariance/invariance.csv).
- Apoze's 6263445 (quantized-FA stabilization) can only touch the q8_0 column; it is built as tree `apoze` and will be
  measured against this table.
- Q8_0 target: batched-bench row 3 OOMs at the harness minimum ctx 16384 x 64 sequences (28 GB weights); row 3 is skipped
  for the Q8_0 target, which is the acceptance reference only.


### Incident 2026-10-26 19:26 (recorded for hygiene): a duplicate invariance run contaminated the first Q8_0 stage

A notes-writing shell command with an unquoted heredoc executed a backtick-quoted command inside the text and started a
second `bench/invariance.sh before 27b-nvfp4` while the baseline chain was on the Q8_0 target. Consequences and fixes:
the duplicate held port 8080 (the Q8_0 `none` row failed to start) and shared the GPU with the Q8_0 llama-bench row,
so the whole Q8_0 stage was discarded and re-run; it appended to the NVFP4 invariance report.csv and overwrote
summary.txt, both regenerated from the untouched per-run json files (identical rates, which is one more determinism
check). New guard: `gpu_exclusive` in bench/lib.sh refuses to start a measurement when the GPU already has more than
2000 MiB used or more than 5% utilisation (override FP4_UNLOCK_SHARED_GPU=1).

## 2026-10-26 Phase 2.1 pre-analysis: where batch size enters the numerics (27B, build-before 63b4b303)

Read from the CUDA backend source, before the Phase 1.5 and Apoze runs finished. Model geometry from the GGUF: 65 blocks, full attention every 4th block (16 attention + 49 GDN), 24 heads, 4 KV heads (gqa_ratio 6), head dim 256, FFN 17408, embd 5120.

Decode with M = n_max+1 tokens in one batch (M=1 plain, M=4 for n3, M=8 for n7) passes M through four kernel families. Which of them changes its floating-point reduction order with M:

| Op | Kernel on this GPU | Depends on M? | Why |
|---|---|---|---|
| MUL_MAT NVFP4 (all weight matmuls) | MMVQ dp4a for M<=8 (`ggml_cuda_should_use_mmvq`, Blackwell default) | **yes, at M 4->5** | `calc_nwarps` (GENERIC table on sm_120): nwarps=4 for ncols 1..4, 2 for 5..8. `blocks_per_iter = vdr*nwarps*warp_size/qi` sets each thread's k-stride, so the per-thread partial-sum sequence differs between M<=4 and M>=5. `rows_per_cuda_block` 1 vs 2 (ncols 1 vs 2..8) does not change the k-order. So MMVQ alone is invariant within {1..4} and within {5..8}, not across. |
| MUL_MAT NVFP4 with `FP4_UNLOCK_SMALLM=all` | MMQ for every M | expected **no** for M<=8 | mmq_x tile is chosen from ne11 (`ntiles_x`), but every M<=8 fits one J=8 tile, so the tile count and the stream-k split (which depends on tile count vs SM count) are identical for all M<=8. Chain3 step 8 tests this directly with f16 KV. |
| FLASH_ATTN_EXT, f16 KV | MMA-f16 for every M (vec is skipped: gqa_ratio 6 > 4 and D=256) with ncols2=8 (gqa>4) and ncols1 from M: M=1 -> ncols 8, M=2 -> 16, M=3,4 -> 32, M=5..8 -> 64 | **yes** | D=256 configs: nbatch_fa 64 for ncols 8 and 16, 32 for ncols 32 and 64. nbatch_fa is the number of KV rows per online-softmax rescale, so M=1 and M=4 rescale at different points. On top of that `launch_fattn` picks parallel_blocks / stream-k blocks from ntiles_dst (depends on ncols1), changing the split-KV combine order. Matches the 6/6 f16 divergence at both n3 and n7. |
| FLASH_ATTN_EXT, q8_0 KV | vec for M<=2 (Ada+ rule), MMA-f16 above | **yes** | kernel family switches at M=3, and parallel_blocks is clamped/tuned by ntiles_KV. Apoze's commit: vec with cols_per_block=1 for M<=4 and a fixed partition (no clamp to ntiles_KV, no efficiency search), quantized KV on Ada+ only. It cannot cover M=8 (n7) or f16 KV by construction. |
| GDN recurrent / conv / norm / rope / sampler | per-token or per-row kernels | to check | Not analysed yet; the SMALLM=all + f16 result decides whether they need to be. If step 8 still diverges with MMVQ removed, FA is the remaining known source; if FA is then pinned and it still diverges, GDN is next. |

Consequences for the Phase 2.1 design:

1. The Phase 3 small-M kernel removes the MMVQ source by construction (one kernel, one k-order for M in [1,8]). Until it exists, `FP4_UNLOCK_SMALLM=all` with `GGML_CUDA_MMQ_PREC=q8` is the stand-in (cost at M=1 measured in chain3 step 9).
2. FA needs a mode that picks ncols1 and the KV partition from a fixed cap, not from M. Candidate: for M <= 8 always take the ncols1 = 8/ncols2 ... 64/ncols2 template that M=8 would take (ncols 64 at D=256, nbatch_fa 32) with the existing OOB masking of unused columns, and freeze parallel_blocks. Cost at M=1: the MMA tile is 1/8 used; FA decode is KV-bandwidth bound so this may be cheap, but it must be measured at 8K and 64K (makoni's regression). Alternative: extend Apoze's vec path to M<=8 and f16 KV; the vec kernel was not chosen at M=1 for this geometry for speed reasons, so it needs the same measurement.
3. Apoze's approach as-is covers quantized KV at M<=4 only. The chain3 Apoze run (q8_0 KV, n3 and n7) should show n3 fixed, n7 still diverging; that is the expected outcome, not a failure of the commit.
4. Regression test for 2.1: test-backend-ops bitwise-equality case over column 0 for N=1 vs N=2..8, for MUL_MAT NVFP4 (fails today at N=5..8 only, if the analysis above is right; the test will confirm) and FLASH_ATTN_EXT D=256 gqa 6 at f16 and q8_0 KV (fails today at N=2..8).

## 2026-10-26 Phase 1.2 addenda: DFlash-n3 on NVFP4, the Q8_0 target, 35B-A3B, and kill criterion K3

Build-before 63b4b303, driver 595.99.02, same harness and prompts as the frozen baseline (greedy, 512 tokens, 3 runs, median).

**NVFP4 27B, new row.** `dflash-n3` (DFlash2 drafter, n_max 3) = **166.48 tok/s**, acceptance 5958/9684 = 61.5 %, mean draft length 2.65, 537 W.
It is now the best single-stream NVFP4 row (dflash-n7 159.15, mtp-n3 149.79, none 82.01). Shorter drafts win because the
verify step at M=4 costs 1.18x a single token while M=8 costs 1.47x (batched-bench ratios), and acceptance at n3 is higher.

**Q8_0 27B target (cdiamond Q8_0 GGUF, same prompts).** Full offload rows:

| row | tok/s | acceptance | mean len | power |
|---|---|---|---|---|
| none | 53.05 | | | 499 W |
| extmtp-n3 (unsloth external MTP head) | 115.82 | 5806/10134 = 57.3 % | 2.63 | 544 W |
| extmtp-n7 | FAILED: out of memory allocating the 212 MiB compute buffer at full offload | | | |
| np8-none | 280.01 | | | 537 W |
| np8-mtp-n3 | FAILED: out of memory allocating 4788 MiB | | | |

Q8_0 with `-ngl 60` (partial offload, so the absolute tok/s are not comparable to the NVFP4 rows, only the acceptance is):
dflash-n3 37.66 tok/s, acceptance 5934/9741 = 60.9 %, len 2.81; dflash-n7 42.36 tok/s, acceptance 6663/17568 = 37.9 %, len 3.32.
The OOM rows are re-queued at `-ngl 60` (chain4) so the acceptance numbers exist; their tok/s will be a partial-offload number and is
labelled as such.

**K3 decision (NVFP4 acceptance vs Q8_0): no NVFP4-specific acceptance problem.**
MTP acceptance 57.4 % (NVFP4) vs 57.3 % (Q8_0); DFlash-n3 61.5 % vs 60.9 %; DFlash-n7 37.2 % vs 37.9 %. All within one point.
The acceptance goal in the PRD is closed as "no bug": the speculative speedup on NVFP4 is bounded by the verify-batch cost, not by
the drafter being fed worse logits. The Phase 3 small-M kernel is the lever; acceptance is not.

**35B-A3B NVFP4 (michaelw9999 Qwen3.6-35B-A3B-NVFP4-MTP-HQ, qwen35moe, 41 blocks incl. 1 NextN, 256 experts / 8 used, 19.07 GiB).**
`mtp-n3` = **426.65 tok/s**, acceptance 6147/9123 = 67.4 %, mean len 2.66, 368 W (per prompt 369-461 tok/s).
Every row without `--spec-type draft-mtp` (llama-bench, batched-bench, server `none`) failed to load:
`done_getting_tensors: wrong number of tensors; expected 1101, got 1079`.

Root cause (a loader bug on the pinned commit, not a model-file problem): with `load_mtp == false`, `nextn_flags()` adds
`TENSOR_SKIP` to the NextN layer's main tensors, and the loader counts skipped tensors as created. But the generic pass that
creates the NVFP4 `.scale` / `.input_scale` tensors (`src/llama-model.cpp`, "generic pass: load optional per-tensor/per-expert
.scale tensors") only creates a scale tensor `if (layer.X)`, i.e. when the base tensor exists. For the skipped layer the base
tensors are null, so its 22 scale tensors (attn q/k/v/out x2, gate/down/up_exps x2, gate/down/up_shexp x2, nextn.eh_proj x2) are
never created or counted, and the file holds 22 more tensors than the loader created. 1101 - 1079 = 22. Any NVFP4 GGUF with a
NextN layer and per-tensor scales hits this whenever MTP is off, which is the default for llama-bench, batched-bench, and plain
`llama-server`.

Fix in the fork (`patches/nextn-scale-skip.patch`, also uncommitted in `third_party/llama.cpp`): in the generic scale pass,
detect a skipped NextN layer (`!ml.load_mtp`, `layer.nextn.eh_proj == nullptr`, and the file has `blk.N.nextn.eh_proj.weight`)
and create its scale tensors with `TENSOR_NOT_REQUIRED | TENSOR_SKIP` so they are counted the same way as the main tensors.
Verified on the CPU build: `llama-bench -p 0 -n 1` on the 35B file now loads and runs (tg1 18.88 tok/s on 8 CPU threads) where
it failed before. GPU rows for 35B on the patched tree are queued (chain6: llama-bench, batched-bench scaling, server none and
mtp-n3). For the unpatched `build-before` tree, server rows can be run with `--mtp` (keeps the NextN tensors loaded) via the new
`SERVER_EXTRA` hook in `bench/lib.sh`; llama-bench and batched-bench have no such flag, so the before-column plain-decode numbers
for 35B are the `--mtp` server rows only.

Upstream candidate: this is a self-contained fix with a reproducible trigger (any NextN NVFP4 GGUF, MTP off). The owner writes
the PR; the diff is mechanical (42 conditions, 42 flag arguments) and could be reduced to a table-driven loop if upstream prefers.

## 2026-10-26 Phase 1.5 small-M experiment (fork `FP4_UNLOCK_SMALLM`, build 63b4b303 + mmvq.cu gate), 27B NVFP4

Kernel timings, test-backend-ops perf, FFN shapes (m,k) = weight rows, cols; n = batch. `stock` = upstream dispatch (MMVQ for
n <= 8, MMQ above). `smallm` = route n >= 2 to MMQ with default precision (auto = W4A4 FP4 MMA), `smallm_q8` = same with
`GGML_CUDA_MMQ_PREC=q8` (W4A8), `smallm_q4` = `GGML_CUDA_MMQ_PREC=q4`.

| m x k | n | stock us | smallm us | smallm_q8 us | smallm_q4 us |
|---|---|---|---|---|---|
| 5120x17408 | 1 | 13.62 | 13.75 | 13.78 | 13.84 |
| 5120x17408 | 2 | 15.07 | 23.06 | 26.63 | 23.13 |
| 5120x17408 | 3 | 28.37 | 23.18 | 26.70 | 23.27 |
| 5120x17408 | 4 | 23.71 | 23.17 | 26.75 | 23.24 |
| 5120x17408 | 6 | 39.21 | 23.47 | 27.00 | 23.53 |
| 5120x17408 | 8 | 44.34 | 23.41 | 27.02 | 23.47 |
| 5120x17408 | 12 | 25.09 | 25.09 | 32.10 | 25.14 |
| 5120x17408 | 32 | 28.84 | 28.97 | 43.62 | 28.97 |
| 5120x17408 | 512 | 140.24 | 140.29 | 422.02 | 140.45 |
| 17408x5120 | 1 | 15.23 | 15.39 | 15.42 | 15.48 |
| 17408x5120 | 2 | 16.09 | 17.49 | 26.17 | 17.54 |
| 17408x5120 | 3 | 19.59 | 17.49 | 26.21 | 17.53 |
| 17408x5120 | 4 | 23.12 | 17.46 | 26.24 | 17.51 |
| 17408x5120 | 6 | 30.55 | 17.61 | 26.45 | 17.67 |
| 17408x5120 | 8 | 39.61 | 17.64 | 26.49 | 17.68 |
| 17408x5120 | 12 | 19.45 | 19.43 | 31.54 | 19.49 |
| 17408x5120 | 32 | 22.74 | 22.80 | 42.87 | 22.77 |
| 17408x5120 | 512 | 137.06 | 136.00 | 399.33 | 136.39 |

Readings:
1. **Caveat first: these are L2-resident timings.** test-backend-ops repeats one op; a 17408x5120 NVFP4 weight is 50 MB and the
   5090 L2 holds it, so n=1 "runs" at 3.3 TB/s, above HBM bandwidth. The table measures kernel compute/latency structure, not the
   DRAM-bound decode step. The end-to-end truth is batched-bench (step(1) 12.07 ms, 79 % of the 9.57 ms roofline) and the server rows.
2. MMQ W4A4 (`smallm`, `smallm_q4`) is flat from n=2 to n=8 (23 / 17.5 us) and beats MMVQ from n=3 up (1.2x at n=3 to 1.9-2.3x at
   n=8). At n=2 it loses to MMVQ (23.1 vs 15.1, 17.5 vs 16.1), so the right gate is n >= 3, not n >= 2.
3. **MMQ W4A8 (`smallm_q8`) is not a free lossless option.** It is slower than stock MMVQ at n=2..4 for the 17408x5120 shape
   (26.2 vs 16.1-23.1 us) and only wins from n=6. Worse, `GGML_CUDA_MMQ_PREC=q8` is global: it makes every MMQ call W4A8, which costs
   **3x at prefill** (n=512: 399-422 us vs 136-140 us). The "lossless column" in the PRD therefore cannot be "set the env var"; the
   lossless small-M path must be gated to M <= 8 inside the dispatcher, and the existing MMQ q8 path at M <= 8 is too slow to be it.
4. End-to-end confirms it: with `FP4_UNLOCK_SMALLM=all` + `GGML_CUDA_MMQ_PREC=q8`, server `none` = 82.00 tok/s (unchanged, M=1 is
   MMVQ either way) but `mtp-n3` = **136.32 tok/s vs 149.79 stock** (acceptance 57.6 %, identical). Routing the M=4 verify batch
   through MMQ-q8 loses 9 %. The `mtp-n7` and q4 rows are still running and are appended below when done.
5. Consequence for Phase 3: the deliverable is a dedicated small-M (M=2..8) W4A8 kernel that keeps the MMVQ streaming structure
   (one weight read, N accumulators) rather than a tile GEMM; "flip the existing dispatch" is ruled out as the lossless product.
   The W4A4 flip at n >= 3 remains a labelled "fast" column (KL 0.0618 vs 0.0431 lossless).

### Phase 1.5 end-to-end columns, complete (server rows 4-6, f16 KV, 8K ctx, 6 prompts x 512 greedy tokens x 3 runs)

`stock` = build-before dispatch. `q8` = `FP4_UNLOCK_SMALLM=all GGML_CUDA_MMQ_PREC=q8` (every M >= 2 on MMQ W4A8).
`q4` = `FP4_UNLOCK_SMALLM=all GGML_CUDA_MMQ_PREC=q4` (every M >= 2 on MMQ W4A4). Acceptance = accepted / drafted.

| row | stock tok/s (acc, len) | q8 tok/s (acc, len) | q4 tok/s (acc, len) |
|---|---|---|---|
| none | 82.01 | 82.00 | 82.00 |
| mtp-n3 | 149.79 (57.4 %, 2.59) | 136.32 (57.6 %) | 129.32 (56.2 %, 2.44) |
| mtp-n7 | 122.03 | 133.44 | 141.72 (38.0 %, 3.71) |
| mtp-n8 | 113.50 | 128.95 | 135.38 (34.7 %, 3.75) |
| dflash-n3 | 166.48 (61.5 %, 2.65) | 148.87 | 143.18 (60.3 %, 2.48) |
| dflash-n7 | 159.15 (37.2 %, 3.30) | 183.21 (40.2 %, 3.49) | 197.53 (43.5 %, 4.22) |
| synth4 | 141.94 | 143.71 | 144.53 |
| synth8 | 279.58 | 280.65 | 282.38 |

Readings:
1. The MMQ columns lose on the short-draft rows (M = 4: mtp-n3 -9 % / -14 %, dflash-n3 -11 % / -14 %) and win on the long-draft
   rows (M = 8: mtp-n7 +9 % / +16 %, dflash-n7 +15 % / +24 %). This is exactly the kernel table: MMQ is flat in n, MMVQ grows with n,
   and the crossover sits at n = 3-6 for L2-resident weights but higher end to end because the verify batch also pays the
   drafter and the sampling.
2. The synth rows (a drafter that always proposes the right token, verify batch M = 5 / 9) move < 2 % in either column: at the
   same acceptance the FFN GEMM precision is not what the step time is made of. The spec rows move because acceptance
   moves and because the verify batch of the real drafter is longer.
3. Acceptance is not invariant under target precision: dflash-n7 goes 37.2 -> 40.2 -> 43.5 % stock -> q8 -> q4, and mean accepted
   length 3.30 -> 3.49 -> 4.22. Not explained yet. It is in the direction "a less precise verify pass agrees with the drafter more
   often", which is the same symptom as Phase 1.8 (the verify batch at M > 1 does not reproduce the M = 1 argmax). It is one more
   reason the lossless column has to be bitwise batch-invariant before any kernel speed-up is claimed: an acceptance gain that
   comes from numerics drift is not a speed-up, it is a different model.
4. Product reading unchanged: the fast column (W4A4 at M >= 3) exists today in upstream code behind a gate and is worth up to
   +24 % on the best DFlash row; the lossless column needs the Phase 3 kernel, and the Phase 2 invariance mode is a precondition
   for calling either "lossless".

## 2026-10-26 Phase 2.1 design: batch-invariant mode for M <= 8 (fork, `FP4_UNLOCK_INVARIANT`), from source at 63b4b303

Goal: the verify step at batch M = n_max + 1 (2..8) must produce bitwise the same logits for a token as the M = 1 decode step
would, for f16 and q8_0 KV, so that greedy speculative output equals greedy spec-off output (Phase 1.8 found 5-6/6 prompts
diverging for every drafter, with the first divergence position a function of M and the KV type only).

**Switch inventory (every place M enters the numerics of the 27B decode graph on CUDA):**

| op | where M enters | batch-invariant already? |
|---|---|---|
| MUL_MAT NVFP4/Q8_0 at M <= 8 (MMVQ) | `calc_nwarps` GENERIC: 4 warps for ncols 1..4, 2 warps for 5..8; `blocks_per_iter = vdr*nwarps*warp_size/qi` sets the k partition per thread | no for M in 5..8 (the k reduction order changes at the 4 -> 2 warp switch); `rows_per_block`, `small_k`, `ncols_dst` do not change the k order |
| MUL_MAT at M > 8 (MMQ) | tile GEMM, different reduction | out of scope (M <= 8 only), also why n_max = 7 is the cap |
| FLASH_ATTN_EXT | (a) kernel family: VEC at ne01 == 1 for f16 KV (unless gqa > 4 and (D >= 256 or nkv >= 8192)), VEC at ne01 <= 2 for quantized KV on Ada+, else MMA; (b) MMA ncols1 tile: 8/16/32/64 / ncols2 by ne01; (c) stream-k split: `nblocks_stream_k = min(max_blocks, ntiles_KV * ntiles_dst)` with `kbc = blockIdx.x * W / gridDim.x`, so the KV range each block reduces, and the fixup combine order, depend on ntiles_dst, i.e. on M; (d) sparse decode variant (ncols1 = 1) checked first | no: (a) flips at M = 1 vs 2 (q8_0 KV) or M = 1 vs 2 with gqa 6 and D = 256 (f16 KV at 27B: MMA already at M = 1 since gqa 6 > 4 and D = 256), (b) and (c) change with M in all cases |
| GDN (gated delta net) recurrence | chunked kernel, chunk size 64 | invariant for M < 64 by construction (one chunk, same order) |
| RMS norm, rope, softmax, adds | per-row ops | invariant (row-local reductions, fixed block size) |

**Mechanism chosen (option "pin the kernel choices for M <= 8", a runtime mode, FA + MMVQ):**
1. FA kernel family for M <= 8 is chosen as if M = 8: MMA for f16 and q8_0 KV. Consequence: q8_0 KV at M = 1 no longer takes the
   VEC path and pays the MMA path's full-cache f16 conversion; this is the cost to measure (estimate ~4 % of the attention time
   at 8K, a fraction of a percent of the step). Alternative not taken: VEC for all M <= 8 (VEC's ncols is also M-dependent and
   its split differs, so it would need its own pinning and is slower at M = 8).
2. FA MMA ncols1 tile pinned to the M = 1 choice (8 / ncols2) for all M <= 8. Knob `FP4_UNLOCK_FA_TIER=16` pins to the M = 8
   choice (16 / ncols2) instead; both are measured.
3. FA stream-k split replaced by a fixed split for M <= 8: `blocks_num.x = ntiles_dst * (ntiles_KV / (256 / nbatch_fa))`, i.e. every
   block reduces exactly one 256-position KV chunk (KV length is padded to a multiple of 256 = FATTN_KQ_STRIDE, so the chunks tile
   exactly), aligned the same way for every M. The existing uniform fixup kernel then combines the same partials in the same
   order for every M. Masked-only chunks contribute exact zeros (KQ max initialised to -HALF_MAX_HALF, FTZ threshold), so extra
   masked tiles do not change bits. No kernel code changes, only the launch geometry. Predicted cost: more partial buffers
   (one per 256 KV positions instead of one per stream-k block): at 64K ~+12 % attention traffic from the fixup (~1.5 % of the
   step), negligible at 8K.
4. Sparse decode variant disabled for M <= 8 (it is M = 1 only, so it would break invariance for SWA models at >= 4K ctx).
5. MMVQ `calc_nwarps` GENERIC pinned to 4 warps for ncols 1..8 (compile-time `-DFP4_UNLOCK_INVARIANT_MMVQ`, only in build-dev
   until measured: the 5..8 cases change from 2 to 4 warps, which costs or gains at M = 5..8 and nothing at M <= 4).
6. Everything else (GDN, norms, rope) needs no change.

Runtime: `FP4_UNLOCK_INVARIANT=1` enables 1-4 (default off in the fork until measured, so the queued chain runs on `build/` are
not contaminated; the PRD says to flip the default on once the cost is within 2 %). Touched: `fattn-common.cuh` (helpers +
fixed split), `fattn.cu` (family and tile pinning, sparse off), `fattn-mma-f16.cuh` (sparse off), `mmvq.cu` (nwarps pin).
Not touched: pre-Turing branches, Vulkan/Metal/CPU.

**Validation plan (chain7, build-dev, results in bench/results/2026-10-26-{opinv-dev,perf-dev,dev-27b-nvfp4-*}):**
opinv bitwise column-0 equality n = 1..16 at kv 1000 / 4096 / 8192 / 65536, env off vs on vs tier 16 (expect: MUL_MAT NVFP4/Q8_0
same for n <= 8 in build-dev; FA f16 and q8_0 KV same for n <= 8 with the mode on, DIFF above 8 as before; the kv = 1000 case checks
the padded-tail handling); test-backend-ops perf FA D = 256 gqa 6 at kv 8192 / 65536, M 1 / 4 / 8, f16 / q8_0 KV, env off / on /
tier 16, and MUL_MAT n = 1..8 build-dev vs build (nwarps pin cost); end-to-end invariance.sh on dev with the mode on, f16 and q8_0 KV,
mtp-n3/n7 and dflash-n3/n7 (expect 0/6 everywhere); llama-bench tg128 at depth 0 / 8192 / 65536, f16 and q8_0 KV, env off vs on
(budget: <= 2 % at M = 1); server rows 4-5 on dev with the mode on, plus env-off control rows.

Deferred, to decide from the numbers: whether the q8_0-KV M = 1 conversion cost forces a VEC-pinned variant; whether sparse
decode needs an invariant form for SWA models; whether the mode should be default-on in the fork (PRD 2.1 says yes if <= 2 %).
Open hole noted for the regression test (PRD 2.1): the test must run with the env var set, so it is a bench/opinv run for now;
a test-backend-ops case needs the mode to be a backend flag rather than an env var (Phase 2.5 cleanup).

## 2026-10-26 chain3 outputs: Phase 1.5 small-M KL, 1.4 profile split, 1.8 Apoze patch with q8_0 KV, 1.7 stall repro, vLLM dry run

**1.5 KL of the SMALLM routes (bench/kld.py, same protocol as Phase 1.3, results in bench/results/2026-10-26-after-27b-nvfp4-kld/kld.csv):**

| route (M = 2..8 on MMQ) | mean KL | top-1 same |
|---|---|---|
| mmvq (stock, Phase 1.3) | 0.0431 | 95.32 % |
| SMALLM=1 W4A8 (`GGML_CUDA_MMQ_PREC=q8`) | 0.043883 | 95.173 % |
| SMALLM=1 W4A4 (default q4) | 0.061691 | 94.389 % |

Reading: W4A8 through MMQ is at the lossless bar (within the Phase 1.3 noise of the mmvq / mmq-q8 pair), W4A4 is a
43 % KL increase, the same gap as the Phase 1.3 mmq-q4 row. Confirms the precision decision for Phase 3: W4A8 default,
W4A4 only as a labelled opt-in.

**1.4 nsys profile split (bench/profile_split.py; single-stream synth8 verify workload, 6 prompts x 512 tokens; reports in
bench/results/2026-10-26-{before,after}-profile-synth8*.nsys-rep):**

| kernel class | before (stock, M = 8 on MMVQ) | after SMALLM=1 W4A8 (M = 8 on MMQ) |
|---|---|---|
| GPU kernel time | 791.7 ms | 821.8 ms |
| mmq | 47.7 % | 52.2 % (429.0 ms, 29510 launches) |
| mmvq | 29.8 % | 25.7 % (211.2 ms; `mul_mat_vec_q<Q8_0,1>` lm_head 796 us x 211 = 20.4 % alone) |
| quantize | 0.7 % (`quantize_mmq_nvfp4` 10098 x 4.3 us = 5.5 % of total in the q4 run) | 0.4 % |
| flash-attn | 1.1 % | 1.1 % |
| gdn | 3.1 % | 3.0 % |
| norm / rope / other | 12.8 % | 11.9 % |
| unknown (mmq stream-k fixups) | - | 5.7 % |

Readings: (a) GEMM (mmq + mmvq + quantize) is ~78 % of the verify-step GPU time, so a better M <= 8 GEMM is worth
building (Expected-gain input for DESIGN-draft). (b) The top kernel in both runs is `mul_mat_q<NVFP4, J=16>` at 25-28 %:
that is the drafter / MTP-head prefill and the verify-batch MMQ, flat in M. (c) The Q8_0 lm_head at M = 1 is the single
largest MMVQ cost (20 % of the step); it is not NVFP4 and not touched by Phase 3. (d) The W4A4 activation quantizer is
5x the cost of the q8_1 quantizer per launch (4.3 vs 0.85 us) because it searches 5 UE4M3 candidates per sub-block, one
more reason W4A4 is not the default at small M.

**1.8 Apoze's patch (build-apoze = 63b4b303 + cherry-pick 6263445 "cuda: stabilize quantized FA for speculative batches"),
q8_0 KV, bench/invariance.sh, 6 prompts x 512 greedy tokens:**

| drafter | divergence (apoze, q8_0 KV) | note |
|---|---|---|
| none vs none | 0/6 | deterministic reference |
| mtp-n3 (M = 4) | 0/6 | |
| mtp-n7 (M = 8) | 6/6 | diverges at tokens 396/361/309/153/177/3 |
| dflash-n3 (M = 4) | 0/6 | |
| dflash-n7 (M = 8) | 6/6 | diverges at exactly the same tokens as mtp-n7 |

Reading: with the patch and q8_0 KV, every M = 4 run is bitwise the M = 1 run and every M = 8 run diverges, and the two
different drafters diverge at identical token positions. The divergence is therefore a property of the verify batch size,
not of the drafter: at M = 5..8 stock MMVQ switches from 4 warps to 2 (a different reduction order), which Apoze's FA-only
patch does not touch. That is exactly the switch Phase 2 pins (`FP4_UNLOCK_INVARIANT_MMVQ`); chain7 tests it.

Cost of Apoze's patch (llama-bench tg128, q8_0 KV, -r 3, bench/results/2026-10-26-{before,apoze}-tg128-q8kv-depth.jsonl):

| depth | before | apoze |
|---|---|---|
| 0 | 81.71 | 82.11 |
| 65536 | 64.22 | 32.32 (-50 %) |

Reading: the patch forces the non-MMA quantized-KV FA path at every batch size and halves long-context decode. Not
acceptable as shipped; Phase 2 keeps the MMA family and pins the tile instead (cost to be measured by chain7).

**1.7 stall repro (#27329 config: ctx 262144, iq4_nl K/V, b/ub 512, 12 threads, fa auto, no spec; bench/stall_repro.sh;
bench/results/2026-10-26-before-27b-nvfp4-stall-iq4_nl):** 3 reps x 2000 tokens on build-before, 47.6 / 47.5 / 47.7 tok/s,
stalled = 0 with a 180 s watchdog. The issue does not reproduce on 63b4b303 with the cdiamond GGUF and driver 595.99.02;
nothing to fix in the fork for it, note it in the shipping doc as "not reproduced here".

**vLLM dry run (bench/vllm_compare.sh DRY=1):** failed, all three configs: the container exited during startup and, because
the script started it with `--rm`, the log was gone before `docker logs` ran ("No such container"). The image itself runs
(`docker run --gpus all vllm/vllm-openai:v0.31.0` reaches the vllm CLI). Fixed the script (no `--rm`, crash log saved to
`vllm-<cfg>-crash.log`, explicit `docker rm`, version probe via python) and queued the dry run for the next GPU-free window.

## 2026-10-26 chain4 outputs: Q8_0 target at NGL=60 (PRD 1.5/1.8 invariance on the lossless target)

The Q8_0 27B target does not fit at full offload with a 32768 context plus a drafter, so the invariance rows re-ran at
`-ngl 60` (build-before, f16 and q8_0 KV, 6 prompts, run B vs run A of `none` as the control). Results
(`bench/results/2026-10-26-before-27b-q8-*`, `$S/phase1/chain4.log`; each block appears twice in the log because the
runner printed the summary once per KV type while iterating both):

| KV | none-B | extmtp-n3 (M=4) | extmtp-n7 (M=8) | dflash-n3 | dflash-n7 |
|---|---|---|---|---|---|
| f16 | 0/6 | 6/6 | 5/6 | 6/6 | 5/6 |
| q8_0 | 0/6 | 6/6 | 6/6 | 6/6 | 6/6 |

Reading: on the Q8_0 target even the M = 4 verify batch diverges from M = 1 decode, unlike NVFP4 where M = 4 was stable
and only M = 8 moved (Phase 1.8). Q8_0 weights go through MMVQ at every M <= 8 with the same 4-to-2 warp switch at
ncols 5, so the extra M = 4 divergence has to come from somewhere else in the graph (FA's tile choice at M = 4, or the
f16/f32 tensors on the `mul_mat_vec_f` path). The op-level probe below narrows it. Either way the "same tokens with or
without the drafter" property does not hold on stock llama.cpp for any quantization tested, which is the Phase 2 target.

Timing at NGL=60 (CPU offload of 4 blocks, so not comparable with the full-offload NVFP4 column): extmtp-n7 37.73
tok/s (acc 6393/19472, mean_len 3.09, 228 W); dflash-n3 37.66, dflash-n7 42.36 (from the earlier rows). The np8
extmtp-n3 row (row 7) at NGL=60 failed to start: the drafter's MTP context could not allocate its 126 MiB compute buffer
(`cudaMalloc failed: out of memory`, `server-np8-extmtp-n3.log`), i.e. 8 slots x 32768 context on Q8_0 at NGL=60 is at
the VRAM limit. Not re-queued; a Q8_0 multi-slot number needs a smaller context or fewer layers and is not on the
headline path (NVFP4 is).

## 2026-10-26 chain5 outputs: op-level batch invariance (bench/opinv, stock vs the Phase 1.5 routes)

`bench/opinv` runs single ops through the CUDA backend and compares column 0 of the n-column result with the n = 1
result, bitwise, for n = 2..16 (seed 7, KV 8192 for FA; "s" = identical, "D" = differs). Stock (build-before):

| op | 2 | 3 | 4 | 5..8 | 9..16 |
|---|---|---|---|---|---|
| MUL_MAT NVFP4 ffn_up 17408x5120 | s | s | s | D (11148 values, max delta 1.1e-5) | D (17408 values, max delta 8.65: MMQ) |
| MUL_MAT NVFP4 ffn_down 5120x17408 | s | s | s | D | D |
| MUL_MAT MXFP4 | s | s | s | D | D |
| MUL_MAT Q8_0 (lm_head shape) | s | s | s | D | D |
| FLASH_ATTN D=256 24/4 heads f16 KV | s | D | D | D | D |
| FLASH_ATTN q8_0 KV | s | D | D | D | D |

So the MMVQ 4-to-2 warp switch at ncols 5 is real and type-independent (NVFP4, MXFP4, Q8_0 all flip at n = 5), the
M >= 9 MMQ route changes every value (different accumulation precision, delta up to 8.65 on raw logits-scale outputs), and
FlashAttention is M-dependent from n = 3 onward for both KV types, which is the M = 4 divergence seen on the Q8_0 target
above. Same table for `before kv=65536` (identical pattern) and for the Apoze worktree (identical, except the q8_0-KV FA
row becomes "s" through n = 4, which is what that patch pins). The Phase 1.5 routes: `SMALLM=1` (M >= 2 to MMQ, M = 1
stays MMVQ) makes every NVFP4/MXFP4 n >= 2 differ from n = 1, as expected; `SMALLM=all` (everything incl. M = 1 to MMQ)
makes the FP4 MUL_MATs identical for all n = 2..16 (MMQ's k reduction is column-independent at a fixed J tile) but leaves
Q8_0 and FA unchanged. Conclusion for the design: making the FP4 GEMM invariant is necessary but not sufficient; the
Q8_0 lm_head (still MMVQ) and FA need pinning too, which is what the Phase 2 mode does (chain7). Full per-n detail with
value counts and max deltas: `$S/phase1/chain5.log`, `bench/results/*opinv*`.

## 2026-10-26 chain6 outputs: 35B-A3B rows (partial, rebuild failure)

The chain's first step rebuilt the `after` tree so the 35B rows would get the NextN loader fix, and it failed:
`scripts/build.sh after` ran `apply_perf_patch`, whose reverse-apply check no longer matched because the shared source
already carried the perf shapes plus the Phase 3 test-backend-ops edits, so it tried a forward apply, which also failed
("patch failed: tests/test-backend-ops.cpp:11138"), and the script exited 1 before cmake. The `build` tree therefore
stayed at its previous state (no loader fix) and every `none` row died with "wrong number of tensors; expected 1101, got
1079". The "server none via --mtp" step was also wrong (`--mtp` is not a flag on this commit). Fix applied to
`scripts/build.sh`: the FP4_UNLOCK_PERF_SHAPES marker in tests/test-backend-ops.cpp is now the "already applied" test.
Rows that did run (mtp-n3 does not trigger the loader bug because MTP loads the NextN tensors), 35B-A3B NVFP4 on `build`,
ctx 8192, `bench/results/2026-10-26-after-35b-a3b-nvfp4*`:

| row | tok/s | notes |
|---|---|---|
| mtp-n3, 1 slot | 428.22 | acc 6147/9123, 358 W |
| np8 mtp-n3 | 841.81 | |
| np16 mtp-n3 | 880.80 | |

The `none`, llama-bench and batched-bench rows re-run on the rebuilt tree after chain8 (chain9).

## 2026-10-26 Phase 3 kernel: first build, the PDL bug, and op-level verification

**What was built (build-p3, `-DGGML_CUDA_FP4_SMALLM`, runtime gate `FP4_UNLOCK_KERNEL=0/1`):**
`ggml/src/ggml-cuda/fp4-smallm.cu[h]`, `fp4_smallm_w4a8_kernel<nwarps, has_fusion>`, dispatched from
`ggml_cuda_mul_mat_vec_q` for NVFP4 src0, no ids, ncols_dst 1..8 (so it replaces MMVQ for both decode and the verify
batch, as the design requires for invariance). Each warp owns 16 output rows x 8 columns and walks K in NVFP4 blocks of 64
(4 sub-blocks of 16), issuing one int8 `mma.m16n8k16` per sub-block into a fresh int32 tile, then
`acc += (ue4m3_scale * q8_1_scale) * float(D)`; warps split K in a fixed interleaved order (block w, w+nwarps, ...) and
reduce through shared memory in fixed warp order, so the k reduction per output element does not depend on ncols_dst.
Columns >= ncols_dst are zero B fragments and are discarded; rows past nrows are clamped on load and masked on store.
The epilogue replicates MMVQ's fusion (x_scale, x_bias, gate + GLU) so `MUL_MAT_VEC_FUSION` cases also go through it.
nwarps 16 / 8 / 4 for <= 6144 / <= 32768 / more rows, grid = ceil(nrows / 16). Test coverage: `test-backend-ops` cases
added for NVFP4 m in {16, 17408, 5120} x n in 1..9 x k in {64, 256, 1280, 5120, 17408}, PREC_Q8 variants, non-multiple-of-16
row counts, fusion variants (bench/perf-shapes.patch marker FP4_UNLOCK_PERF_SHAPES).

**The bug.** The first build passed most op tests but failed `MUL_MAT_VEC_FUSION` at m = 8 nondeterministically, and
`bench/opinv/mmstress` (new: repeated runs of one shape vs the CPU backend, bitwise and NMSE) showed NMSE up to ~2 on
many seeds, with only column 0 wrong (`--dump`), scattered 32-element activation blocks wrong on `--isolate`, and the
unit-vector map (`--unit`: x = e_k for each k) returning the weight column k - 1 for many k, i.e. the kernel was
multiplying by the *previous* run's quantized activations. compute-sanitizer is not installed (sudo), so mmstress was
the substitute. Root cause: `ggml_cuda_kernel_launch` uses programmatic dependent launch, and `quantize_q8_1` calls
`ggml_cuda_pdl_lc()` at its start, so the next kernel in the stream may begin before the q8_1 buffer is written; MMVQ
protects itself with `ggml_cuda_pdl_sync()` before its first global read (mmvq.cu) and the new kernel did not. The
symptom set (timing dependent, column 0 = the first 32-block the quantizer writes last, changes with allocation when
PREC_Q8 toggles, K = 64 always passing) all follow from reading a pool buffer that is still being written.
Fix: `ggml_cuda_pdl_sync()` before the K loop, `ggml_cuda_pdl_lc()` after the last global read (same placement as MMVQ).
Rule for the design doc: any kernel launched through `ggml_cuda_kernel_launch` that consumes a buffer written by the
previous kernel must call the PDL sync before its first global read.

**Verification after the fix** (mmstress, `FP4_UNLOCK_KERNEL=1`, repeat 200 x 30 seeds per row, kernel vs CPU NVFP4
reference, `$S/phase3/`):

| shape | n | bitwise diffs across repeats | bad seeds | worst NMSE vs CPU |
|---|---|---|---|---|
| K 256, R 32 | 1 / 4 / 8 | 0 | 0 | ~1e-14 |
| K 256, R 32, PREC_Q8 | 8 | 0 | 0 | ~1e-14 |
| K 1280, R 512 | 1 / 8 | 0 | 0 | 2.8e-14 / 1.7e-8 |
| K 5120, R 17408 (ffn_up) | 5 | 0 | 0 | 6.5e-9 |
| K 64, R 48 (partial warp) | 8 | 0 | 0 | 5.4e-15 |

test-backend-ops (`build-p3`, KERNEL=1, run during chain7's non-timed e2e step, `bench/results/20261026-p3-eval-early`):
rep 1 MUL_MAT 2484 OK / 0 FAIL, MUL_MAT_W4A8 31 / 0, MUL_MAT_W4A4 21 / 0, MUL_MAT_VEC_FUSION 1376 / 0, all "2/2 backends
passed" (the chain's live counter printed 0/0 because it grepped for a plain "OK" suffix that the colour escape hides;
recounted from the files). Reps 2-3 and the KERNEL=0 controls: see the chain8 section. Small GPU probes (sub-second
mmstress runs) overlapped chain4/6/7 timing steps while the bug was being chased; their effect on those rows is noise
level but noted here.

## 2026-10-26 chain7 outputs: Phase 2 invariant mode on the dev build

Build: `build-dev` (`-DFP4_UNLOCK_INVARIANT_MMVQ`, FA invariant helpers; `GGML_CUDA_FP4_SMALLM` off). Runtime switch
`FP4_UNLOCK_INVARIANT=1` pins the FA kernel choice / stream-k split and the MMVQ warp count (4 for ncols 1..8).
Model 27B NVFP4, f16 and q8_0 KV. Logs `$S/phase2/chain7.log`; results `bench/results/2026-10-26-opinv-dev`,
`...-dev-27b-nvfp4-invariance-on`, `...-dev-27b-nvfp4-tgcost`, `bench/results/2026-10-26-perf-dev`.

### Op level (opinv, dev, INVARIANT=1)

MUL_MAT NVFP4 / MXFP4 / Q8_0 (17408x5120, 5120x17408) and FA (f16 and q8_0 KV; kv 8192, 65536, 4096, 1000; tier16
variant too) are all bitwise "same" for n = 2..8 against the n = 1 column. Env-off control: FA still DIFF from n = 3,
MUL_MAT DIFF from n = 5 (the 4-to-2 warp switch), exactly the chain5 pattern. So the pins do what they say at op level.

### End to end (6 prompts, greedy, 512 tokens, drafter output vs `none` run A)

| KV | none-B | mtp-n3 (M=4) | mtp-n7 (M=8) | dflash-n3 (M=4) | dflash-n7 (M=8) |
|---|---|---|---|---|---|
| f16  | 0/6 | **0/6** | 6/6 | **0/6** | 6/6 |
| q8_0 | 0/6 | **0/6** | 6/6 | **0/6** | 6/6 |

Stock (Phase 1.8) was 5-6/6 divergent everywhere. M = 4 is now bitwise the M = 1 decode for both KV types. M = 8
still diverges 6/6, and the op-level table says it is not the NVFP4 GEMMs or FA.

Root cause (read from `ggml_cuda_should_use_mmvq`, mmvq.cu ~356-372): on Blackwell the per-type MMVQ limits are
Q2_K/Q3_K/Q4_K ne11 <= 5, Q5_K <= 6, Q6_K <= 7, everything else <= 8. The 27B GGUF's attention and GDN projections
are **Q5_K**, so at ne11 = 8 (n7 drafters: 7 draft + 1) they leave MMVQ for MMQ, whose tile reduction is not the
M = 1 reduction. NVFP4 (65 % of bytes) and the Q8_0 lm_head stay on MMVQ to 8, which is why M = 4 pinned cleanly and
M = 8 did not.

Fix (in the fork, not yet validated end to end at the time of writing): `fp4-unlock.cuh` carries the shared
`fp4_unlock_invariant_enabled()`; `ggml_cuda_should_use_mmvq` returns `ne11 <= MMVQ_MAX_BATCH_SIZE` for every
quantized non-FP4 type when invariant mode is on, so k-quants stay on the (warp-pinned) MMVQ through M = 8. opinv
gained Q5_K / Q4_K / Q6_K rows (attn_q 5120x6144, attn_out 6144x5120) so the op-level table covers them next run.
chain8 step "e2e invariance p3 KERNEL=1 INV=1" is the end-to-end check (expect 0/6 at M = 8).

### What the pins cost

FA perf (test-backend-ops perf, us/run, D=256 nh=4 nr23=[6,1]; columns: env-off | INVARIANT=1 | INVARIANT=1 + FA_TIER=16):

| kv | KV type | nb=1 | nb=4 | nb=8 |
|---|---|---|---|---|
| 8192  | f16  | 12.18 / 11.69 / 18.90 | 17.08 / 34.42 / 29.18 | 23.66 / 58.68 / 30.38 |
| 8192  | q8_0 | 26.74 / 47.30 / 54.74 | 53.88 / 70.32 / 55.65 | 61.78 / 94.51 / 57.11 |
| 65536 | f16  | 166.97 / 196.98 / 230.09 | 178.89 / 331.47 / 232.36 | 183.49 / 539.19 / 249.15 |
| 65536 | q8_0 | 218.10 / 500.48 / 563.46 | 492.01 / 747.70 / 569.81 | 501.43 / 1027.25 / 585.46 |

The default tier pins to the M = 1 kernel, which roughly doubles M = 8 FA time (f16, kv 65536: 2.9x) and doubles
q8_0-KV M = 1 FA at depth. tier16 is cheaper at M >= 4 but +55 % at f16 M = 1 (12.2 -> 18.9 us). FA is ~1 % of
the verify-step GPU time at 8k context (chain3 profile), so at short context neither matters; at 64k the q8_0 M = 1
cost is real (below).

MUL_MAT with the warp pin (dev / stock ratio, test-backend-ops perf): 17408x5120 and 5120x17408:
nvfp4 n1 0.976 / 0.977, n4 0.987 / 0.985, n6 1.001 / 1.397, n8 1.040 / 1.484; q4_0 n6 1.037 / 1.358, n8 1.050 /
1.438; q8_0 n6 1.663 / 1.462, n8 1.739 / 1.613. n <= 4 is free (it already ran 4 warps); n = 6..8 pays 40-74 % on
the ffn_down shape and on Q8_0 (the lm_head). The Phase 3 kernel replaces MMVQ for NVFP4 at n <= 8, so the NVFP4
share of that cost disappears; the Q8_0 lm_head and the k-quant projections keep paying it in invariant mode.

tg128 depth cost (llama-bench, -r 3, tok/s; env-off -> INVARIANT=1):

| KV | d=0 | d=8192 | d=65536 |
|---|---|---|---|
| f16  | 82.95 -> 82.77 (-0.2 %) | 80.88 -> 80.84 (0 %) | 68.25 -> 65.82 (-3.6 %) |
| q8_0 | 81.69 -> 81.65 (0 %) | 79.66 -> 78.42 (-1.6 %) | 64.09 -> 51.68 (-19.4 %) |

Decision on the `FP4_UNLOCK_INVARIANT` default: the M = 1 budget (<= 2 %) holds at d = 0 and 8192 for both KV types,
but q8_0 KV at 64k loses 19 % and f16 loses 3.6 %, from the FA pin alone. It stays **opt-in** (env) for now. The
candidate that would let it default on is an FA invariant tier chosen per KV type (tier16 for q8_0, default for f16),
or an FA fix that keeps the M = 1 kernel's reduction order without forcing its tile; both are post-Phase-3 work.
A user who needs bitwise spec = greedy today sets `FP4_UNLOCK_INVARIANT=1` and pays this.

### Server rows 4+5 under invariant mode (dev build, 27B NVFP4, tok/s; acc = accepted/drafted)

| spec | before (stock) | dev INVARIANT=1 f16 KV | dev env-off control (same session) | dev INVARIANT=1 q8_0 KV |
|---|---|---|---|---|
| none | 82.01 | 81.60 | 82.03 | 80.24 |
| mtp-n3 | 149.79 | 153.42 (5854/9990) | - | 160.11 (5986/9615) |
| mtp-n7 | 122.03 | 106.80 (6285/20247) | 113.60 (6352/19730) | - |
| dflash-n3 | 166.48 | 170.08 (6021/9492) | - | 174.65 (6084/9294) |
| dflash-n7 | 159.15 | 142.97 (6582/18189) | 146.82 (6636/17709) | - |

Reading: at M = 4 (n3 rows) invariant mode costs nothing and lands 2-7 % above the stock column, inside session
noise plus the dev build's MMVQ pin. At M = 8 (n7 rows) it loses 6 % against its own env-off control and 9-12 %
against the stock column; the control itself is 7-8 % under stock on the n7 rows, so part of the n7 gap is the
dev build (Q5_K kept on MMVQ to ne11 = 8 instead of MMQ) rather than the env switch. This is the M = 8 MMVQ cost
Phase 3 replaces. Mean accepted draft length also drops under invariant mode on the n7 rows (3.28 -> 2.86 mtp-n7,
3.57 -> 3.21 dflash-n7) while the accepted/drafted ratio barely moves. Not explained; it is a single session per
cell, so it goes on the chain8 watch list (same rows with the Phase 3 kernel, INVARIANT=1) before being called real.

### Process notes

- The invariance runs at M = 8 were made before the k-quant threshold was found; they are the evidence for it,
  not a failed run.
- A low-priority 4-job rebuild of `build-p3` (mmvq.cu only) overlapped chain7's tg128 and server timing steps; tiny
  GPU probes (opinv builds, nvidia-smi) overlapped too. llama-bench stddev stayed <= 0.17 tok/s, so the timing rows
  are kept; re-run before quoting them in the README table.

## 2026-10-26 Phase 2.1 regression tests: MUL_MAT_BATCH and FA_BATCH in the fork's test-backend-ops

Two batch-invariance cases were added to `third_party/llama.cpp/tests/test-backend-ops.cpp` (pattern from upstream PR
28488's MMVQ_BATCH / FA_BATCH, rewritten against the fork's helpers; `test_case::eval` made virtual so a case can own
its comparison). Both compare the backend against itself, bitwise (ULP 0), and never against a CPU reference:

- `MUL_MAT_BATCH(type_a, m, k, n_max)`: column 0 of a `[k,m] x [k,n]` MUL_MAT for n = 2..n_max against the same
  op at n = 1, same seeds. Registered for NVFP4, MXFP4, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K at (256,1024,8) and
  (512,5120,8), plus NVFP4 (1024,5120,16) which crosses into MMQ at n = 9 and documents the stock boundary.
- `FA_BATCH(hsk, hsv, nh, nr23, kv, type_KV, n_max)`: query batch 0 of FLASH_ATTN_EXT with nb = 2..n_max against
  nb = 1. q seeded per (head, batch), mask per column (about half the positions masked), K/V by name hash, so batch 0
  sees identical inputs at every nb. Registered for F16 and Q8_0 KV at (128,128,4,{4,1},1024) and the 27B head shape
  (256,256,4,{6,1},4096).

Run: `test-backend-ops test -b CUDA0 -o MUL_MAT_BATCH,FA_BATCH`. CUDA matrix queued as chain10 (after chain9):
`build/` env-off is the stock column (expected failures: k-quants / FP4 at the per-type MMVQ->MMQ boundary, FA from
nb = 3), build-p3 KERNEL x INVARIANT are expected all OK with INVARIANT=1.

### CPU backend results (build-cpu, same day)

| case | result |
|---|---|
| MUL_MAT_BATCH nvfp4 / mxfp4 / q8_0 / q4_0 (both shapes), nvfp4 n_max=16 | OK |
| MUL_MAT_BATCH q4_K / q5_K / q6_K (both shapes) | FAIL at exactly n = 8 (e.g. 211/256 values differ, max 7.6e-06) |
| FA_BATCH q8_0 KV (both shapes) | OK |
| FA_BATCH f16 KV (both shapes) | FAIL at nb = 2, every value differs, max 6.5e-04 |

Both CPU failures are properties of the ggml CPU backend, not of the fork and not FP4-related:

- FA f16 KV: `ggml-cpu/ops.cpp` `ggml_compute_forward_flash_attn_ext` takes a split-KV path only when
  `neq1 == 1 && neq3 == 1`, K/V are F32/F16 and `nek1 >= 512` (KV chunks per thread, partials combined at the end);
  nb >= 2 goes through the one-chunk or tiled kernel. The nb = 1 decode row is therefore reduced in a different
  order than the same row inside a batch. q8_0 KV is not eligible for the split path, so it stays invariant.
- k-quant MUL_MAT n = 8: not attributed. Not llamafile tinyBLAS (it has no k-quant kernels), not src1 quantization
  (per row, ne11-independent), and nrc = 1 on x86. Worth a look upstream (the ggml CPU batch-invariance claim does
  not hold for k-quants at n = 8), outside this project's scope.

Consequence for the test design: the cases compare a backend against itself at n = 1, which is the property the
product claims (decode row == verify row on the same backend). A CPU-reference comparison would have reported these
CPU properties as fork failures.

### Process note: chain8 first launch died at the eval summary

`bench/lib.sh` sets `set -euo pipefail`, which the chain scripts inherit when they `source` it. chain8's gate line
`nfail_on=$(... | grep -c FAIL)` exits 1 when there are zero FAIL lines (the good case), so the script stopped right
after a clean eval gate (MUL_MAT 2484, W4A8 31, W4A4 21, VEC_FUSION 1376 OK, 0 FAIL; KERNEL=0 controls identical).
Fixed with `|| true`, relaunched 21:46; the chain9/chain10 waiters were unaffected. Rule for new chain scripts: no
bare `grep -c` or other count-returning command in a top-level command substitution after `source bench/lib.sh`.

## 2026-10-26 chain8 outputs: Phase 3 kernel validation (build-p3, 27B NVFP4), part 1

Kernel under test: `fp4-smallm.cu` first build (PF = 1, one block per warp per loop trip), dispatched for NVFP4 at
ne11 <= 8 ahead of MMVQ; `FP4_UNLOCK_KERNEL=0` restores MMVQ in the same binary. Build-p3 also carries the Phase 2
invariant MMVQ split and the FA pin (`FP4_UNLOCK_INVARIANT=1`).

### Correctness gate (test-backend-ops eval, CUDA0 vs CPU)

MUL_MAT / MUL_MAT_W4A8 / MUL_MAT_W4A4 / MUL_MAT_VEC_FUSION with the kernel on: 2484 / 31 / 21 / 1376 OK, 0 FAIL;
kernel off (W4A8, VEC_FUSION): identical counts, 0 FAIL. Results dir `bench/results/2026-10-26-p3-eval/`.

### Kernel time on the 27B FFN shapes (test-backend-ops perf, us/run, L2-resident weights)

| shape (rows x K)  | n  | MMVQ (KERNEL=0) | kernel v1 | kernel v1 repeat | kernel / MMVQ |
|-------------------|----|-----------------|-----------|------------------|---------------|
| 17408 x 5120 (up) | 1  | 15.55 | 24.50 | 25.36 | +58 % |
|                   | 2  | 16.39 | 25.02 | 25.88 | +53 % |
|                   | 3  | 19.93 | 25.58 | 26.35 | +28 % |
|                   | 4  | 23.33 | 27.83 | 28.60 | +19 % |
|                   | 6  | 31.48 | 29.37 | 30.15 | -7 %  |
|                   | 8  | 42.09 | 30.81 | 31.67 | -27 % |
|                   | 16 (MMQ, untouched) | 19.81 | 19.71 | 19.80 | 0 |
| 5120 x 17408 (down) | 1 | 13.89 | 26.49 | 27.29 | +91 % |
|                   | 2  | 15.35 | 28.81 | 29.61 | +88 % |
|                   | 3  | 28.64 | 30.41 | 31.21 | +6 %  |
|                   | 4  | 24.02 | 33.10 | 33.86 | +38 % |
|                   | 6  | 55.89 | 35.09 | 35.85 | -37 % |
|                   | 8  | 66.93 | 38.75 | 39.55 | -42 % |
|                   | 16 (MMQ, untouched) | 25.48 | 25.28 | 25.46 | 0 |

Full files: `bench/results/2026-10-26-perf-p3/mm-p3-{off,on,on2}.txt` (n = 1..512 for both shapes).

Reading: property 3 (flat in M) holds: n = 1 -> 8 costs +26 % (up) / +46 % (down) against MMVQ's +171 % / +382 %.
Property 2 (M = 1 within 2 % of MMVQ) fails by +58 % / +91 %. The kernel only wins from n = 6 up. As built it
would tax decode (M = 1) to buy the verify step, which the PRD forbids (G1).

### Why M = 1 is slow (diagnosis, no profiler needed)

- Instruction count is not it: per 64-wide block a lane does ~150 instructions for 16 rows x 8 columns, MMVQ does
  ~60 per row for one column; per row the kernel issues fewer instructions than MMVQ at any M.
- Memory level parallelism is. The v1 loop `for (kb = warp; kb < nblocks; kb += nwarps)` loads one block's 10 words
  (scale word + 4 nibble words, two rows) and immediately consumes them in 4 MMAs, so each warp is a serial chain of
  L2 round trips: K = 5120 -> 80 blocks / 8 warps = 10 trips per warp; K = 17408 -> 272 / 16 = 17 trips (the down
  projection, which is the slower one). ptxas: 70 registers (79 fused), 0 spills, 1536-7680 B smem, so one block of
  8-16 warps per SM with ~4.6 KB of weight loads in flight per SM, against MMVQ's tens of KB. Measured: 50 MB of
  weights in 24.5 us = 2.0 TB/s from L2; MMVQ moves the same bytes at 3.2 TB/s.
- Fix (v2, in the tree now): software pipelining. The loop issues the loads of `FP4_SMALLM_PF` (= 4) blocks per
  warp back to back (`fp4_smallm_wstage w[PF]`), then consumes them in the same ascending k order. The per-warp
  float accumulation sequence, the K split across warps and the cross-warp reduction order are untouched, so every
  output is bitwise the v1 output (the MMA integer dot products are exact; only the float order matters, and it is
  unchanged). ptxas for v2: 137 registers (165 fused) at nwarps 4, 121 (128 fused) at nwarps 8 and 16, 0 spills;
  `__launch_bounds__(threads, 2)` for nwarps <= 8 so the 17408-row shape (nwarps 8, 1088 blocks) runs two blocks
  per SM (16 warps, 7 waves -> 3.2 waves of 2) instead of one.
  Expected: ~4x the bytes in flight per warp, M = 1 close to MMVQ. Measured by chain11 after chain10 rebuilds and
  gates build-p3; a cross-version check compares chain11's `none` f16 generations with chain8's v1 ones (must be 0/6).

### Op-level invariance with the kernel (bench/opinv, build-p3)

- KERNEL=1, env-off otherwise, kv 8192: NVFP4 / MXFP4 / Q8_0 same for n = 2..8 (DIFF from n = 9, MMQ, out of scope);
  Q5_K DIFF from n = 7, Q4_K from n = 6, Q6_K from n = 8 (the Blackwell MMQ thresholds, as in chain5); FA DIFF.
- KERNEL=1 INVARIANT=1, kv 8192 and kv 65536: every MUL_MAT row and both FA rows (f16, q8_0 KV) same for n = 2..8;
  DIFF only from n = 9. Files `bench/results/2026-10-26-opinv-p3/`.

### KL vs the Q8_0 reference (bench/kld.py, -b 8 -ub 8)

| path | Mean KLD | Median | 99.9 % | Max | ln(PPL ratio) | same top-1 |
|------|----------|--------|--------|-----|---------------|------------|
| p3 kernel on (KERNEL=1 INV=1) | 0.042772 | 0.001990 | 3.117 | 8.81 | 0.03432 | 95.143 % |
| p3 kernel off (MMVQ)          | 0.043163 | 0.002137 | 2.862 | 9.03 | 0.03494 | 95.294 % |

Both are the W4A8 lossless column (bar 0.043, Phase 1.3 mmvq 0.0431); the kernel changes nothing a user could
measure. (It is not expected to be bitwise equal to MMVQ: different float accumulation order, same exact int dots.)

### End-to-end invariance with the kernel (bench/invariance.sh, KERNEL=1 INVARIANT=1, 6 prompts x 512 greedy)

0/6 divergent for every row: none-B, mtp-n3, mtp-n7, dflash-n3, dflash-n7, for f16 KV and for q8_0 KV
(`bench/results/2026-10-26-p3-27b-nvfp4-invariance-on/`). Property 1 holds end to end: the verify batch at M = 4
and M = 8 produces the decode tokens. (chain7's M = 8 f16 result was already 0/6 after the Q5_K-on-MMVQ fix; this
repeats it with the FP4 GEMMs on the new kernel.)

### Still to land from chain8 (appended below as they arrive)

tg128 depth cost KERNEL=0/1, pp512/pp8192, server rows 4-7 (kernel-inv, kernel-noinv, kernel-off control, np8/16,
q8_0 KV), nsys profile split with the kernel, vLLM dry run.

### tg128 depth cost and pp with kernel v1 (llama-bench, build-p3, f16 KV, INVARIANT=1, r = 5)

| depth | KERNEL=0 (MMVQ) tok/s | KERNEL=1 (v1) tok/s | delta |
|-------|------|------|-------|
| 0     | 82.78 | 67.07 | -19.0 % |
| 8192  | 80.95 | 65.80 | -18.7 % |
| 65536 | 65.84 | 55.47 | -15.7 % |

pp512 6526 tok/s, pp8192 6260 with KERNEL=1 (before: 6653; M > 8 is untouched, the -2 % is run noise / the dev
build's MMVQ pin). Files `bench/results/2026-10-26-p3-27b-nvfp4-tgcost/`.

Reading: the +58 % / +91 % kernel time at M = 1 lands as -19 % decode, i.e. the FP4 FFN GEMMs are ~30 % of the
decode step on this build. This is the number kernel v2 has to erase (G1: M = 1 within 2 %). chain8 was trimmed
after its KERNEL=1 INV=1 server rows (recorded below when they land); the remaining v1 rows were dropped because
chain11 repeats them with v2, and the GPU hours are better spent there.

### Server rows 4-6 with kernel v1 (build-p3, KERNEL=1 INVARIANT=1, f16 KV, 8K ctx, 3 runs x 6 prompts x 512 tokens)

| spec | before (stock) | dev INV=1 (chain7) | p3 kernel v1 INV=1 | v1 vs before | acc / mean len (v1) |
|------|------|------|------|------|------|
| none      | 82.01  | 81.60  | 66.26  | -19 % | - |
| mtp-n3    | 149.79 | 153.42 | 129.58 | -13 % | 0.559 / 2.49 |
| mtp-n7    | 122.03 | 106.80 | 108.56 | -11 % | 0.314 / 2.83 |
| mtp-n8    | 113.50 | -      | 130.68 | +15 % | 0.353 / 3.35 |
| dflash-n3 | 166.48 | 170.08 | 147.45 | -11 % | 0.608 / 2.61 |
| dflash-n7 | 159.15 | 142.97 | 144.04 | -9 %  | 0.367 / 3.15 |
| synth4    | 141.94 | -      | 141.99 | 0 %   | 0.376 / 3.95 |
| synth8    | 279.58 | -      | 277.46 | -1 %  | 0.859 / 7.86 |

Files `bench/results/2026-10-26-p3-27b-nvfp4-kernel-inv/`. Reading: with a -19 % decode step, the M = 8 rows already
break even or win (mtp-n8 +15 %, synth8 -1 %, dflash-n7 level with the invariant dev build) because the verify step
got cheaper by about as much as decode got dearer; the M = 4 rows lose 11-13 % because at M = 4 the v1 kernel is still
+19 % / +38 % over MMVQ. Every row in this table should move up by roughly the decode recovery once v2 lands M = 1
near MMVQ; the composition (decode share vs verify share) is now measured rather than assumed. chain8 was stopped
here (22:14); its remaining steps run in chain11 on v2.

## 2026-10-26 chain9 outputs: 35B-A3B NVFP4 rows (after tree with the loader fix; p3 kernel v1)

The 35B-A3B model (Qwen3.6-35B-A3B-NVFP4-MTP-HQ, MoE, ~3B active) loads and runs on the fork after the
llama-model.cpp loader fix (chain6 had failed to load it). Rows 1-4 and np8/16, f16 KV, 8K ctx, 3 runs.

| row | after (stock routing, build/) | p3 kernel v1 KERNEL=1 | p3 KERNEL=0 control |
|-----|------|------|------|
| pp512 / pp8192 tok/s | 14111 / 13620 | 14042 / 13552 | - |
| tg128 | 292.0 | 284.8 (-2.5 %) | - |
| server none | 279.30 | 274.90 | 279.22 |
| server mtp-n3 (acc, len) | 426.91 (0.674, 2.66) | 425.74 (0.684, 2.76) | 427.46 (0.674, 2.66) |
| np8 none / mtp-n3 | 785.1 / 830.0 | - | - |
| np16 none / mtp-n3 | 841.6 / 902.0 | - | - |

batched-bench step ratios (after): M2 1.21, M4 1.54, M8 2.06, M16 3.46 (dense 27B: 1.05 / 1.18 / 1.47 / 1.85): the
MoE's step cost grows faster with M because every extra sequence pulls in more experts.

Reading: the kernel does nothing for this model, by construction: the expert FFNs are MUL_MAT_ID (routed), and the
dispatch is `!ids && NVFP4 && ncols_dst <= 8`; the dense NVFP4 tensors it does catch are a small share of the step.
The -1.5 % on `none` with the kernel on is the v1 M = 1 tax on those few tensors. This confirms the scope statement in
the PRD (dense NVFP4 first; MoE/ids is a later item) and gives the 35B its own "before" column
(`bench/results/2026-10-26-after-35b-a3b-nvfp4-v2`, `-np-v2`, `-p3-35b-a3b-nvfp4-kernel`, `-kernel-off`).

## 2026-10-26 chain10 outputs: CUDA MUL_MAT_BATCH / FA_BATCH matrix and the kernel v2 gate (build-p3 rebuilt 22:21)

build-p3 was rebuilt from the tree with kernel v2 (PF = 4 prefetch, two blocks per SM at nwarps <= 8) and the Phase
2.1 test cases. `test-backend-ops test -b CUDA0 -o MUL_MAT_BATCH,FA_BATCH`, each case comparing column 0 of the
n-column result against the n = 1 result bitwise (MUL_MAT: NVFP4, MXFP4, Q8_0, Q4_0, Q4_K, Q5_K, Q6_K at 256x1024 and
512x5120; FA: D 128 kv 1024 and D 256 kv 4096, f16 and q8_0 KV, nb = 2..8):

| column | result (first n at which column 0 differs from the n = 1 result) |
|--------|--------|
| build/ env-off (stock routing) | every quantized type diverges at n = 5 (the MMVQ 4-to-2 warp switch, max 1e-5), q4_K/q5_K/q6_K 256-row cases at n = 6/7/8 (MMQ thresholds, max 0.08); all 4 FA cases at nb = 2..3 |
| build-p3 KERNEL=0 INV=0 | fork MMVQ pin removes the n = 5 switch; q4_K at n = 6, q5_K at n = 7, q6_K at n = 8 (Blackwell MMQ thresholds, routing opt-in), all 4 FA cases (tile choice is M-dependent without the pin) |
| build-p3 KERNEL=1 INV=0 | identical to the row above: the kernel covers NVFP4 only and routing stays stock without INV=1 |
| build-p3 KERNEL=0 INV=1 | all 14 MUL_MAT cases with n_max = 8 OK, all 4 FA cases OK |
| build-p3 KERNEL=1 INV=1 | all 14 MUL_MAT cases with n_max = 8 OK (NVFP4 through kernel v2), all 4 FA cases OK |

One registered case, NVFP4 1024x5120 with n_max = 16, failed at n = 9 in every column (max 6.8): that is the MMQ
boundary, which the contract does not cover (M <= 8), so the registration was capped to n_max = 8 after this run (the
gate must be green in invariant mode). chain12 re-runs the two INV=1 columns on the rebuilt binary and expects exit 0.

Kernel v2 full eval (all ops, CUDA0 vs CPU, KERNEL=1, INV unset): 11 "compare failed" cases and nothing else. That
count equals the 7 MUL_MAT_BATCH + 4 FA_BATCH failures of the KERNEL=1 INV=0 column above, and "compare failed" is only
emitted by those two test classes, so the full eval found no CUDA-vs-CPU error; but chain10's log filter dropped the
case names (test-backend-ops prints the batch error to stderr on its own line), so chain12 repeats the all-ops eval with
raw captures (INV=1 kernel on/off, and INV=0) plus the four MUL_MAT op evals that gated kernel v1 in chain8.

## 2026-10-26 chain11 outputs: kernel v2 (prefetch) numbers and the trim decision

build-p3 with kernel v2 (PF = 4 blocks in flight per warp, two blocks per SM at nwarps <= 8), 27B NVFP4, INV=1.

| shape (test-backend-ops, us/run) | MMVQ | v2 KERNEL=1 | v2 repeat |
|---|---|---|---|
| 17408x5120 n=1 | 15.50 | 27.23 | 27.96 |
| 5120x17408 n=1 | 13.85 | 28.24 | 29.00 |
| 17408x5120 n=8 | 41.98 | 35.21 | - |
| 5120x17408 n=8 | 66.81 | 40.05 | - |

v2 is flat in M (27-35 us from n = 1 to 8) but still 1.8-2.0x MMVQ at M = 1: prefetch depth was not the limit. The
cross-version check (v2 `none` f16 generations against chain8's v1 ones) was 0/6 divergent, so v2 is bitwise v1 as
designed. tg128 depth 0 / 8192 / 65536: MMVQ 82.8 / 80.9 / 65.9, v2 68.2 / 67.1 / 56.3 (-18 %, -17 %, -14 %); pp512 /
pp8192 6520 / 6264 (unchanged, M > 8 never enters the kernel). chain11 was trimmed before its server rows (trim11.sh):
property 2 ("M = 1 within 2 %") fails by the same margin as v1, and the server rows would only have repeated chain8's
M = 4 / M = 8 pattern. chain12 (gate re-run and raw all-ops evals) was superseded by chain13 and never ran.

Why v1/v2 are slow (the diagnosis that produced v3): in the MMA fragment layout each 4-byte weight load of a warp touches
8 rows = 8 different 128-byte lines, and one 64-wide NVFP4 block costs 10 such loads per lane (~80 L1 wavefronts per
576 bytes of weights). MMVQ reads 8 consecutive blocks of one row per warp instruction (3 lines). The direct kernels
were bound by L1 transaction throughput at ~2 TB/s, against ~3.2 TB/s from L2 for MMVQ on these shapes, independent of
how many loads were in flight.

## 2026-10-26 Phase 3 kernel v3: int8 MMA with weights staged through shared memory (cp.async)

Each warp copies a chunk of 4 consecutive NVFP4 blocks (144 B, 16-byte aligned) of all its 16 rows into shared memory
with 4.5 cp.async instructions per lane (~5 lines each), `FP4_SMALLM_STAGES` chunks in flight, then reads the A
fragments from shared memory where the 36-word row stride makes the 8-row fragment reads bank-conflict free. The k
order per warp and the float accumulation are v1's, so the batch-invariance argument is unchanged. Correctness: 67/67
MUL_MAT nvfp4 cases OK. Numbers (us/run, `$S/phase3/v3quick.log`, `v3perf/`; "1b" is the same kernel with the
alternative stage count):

| shape | MMVQ | v3 | v3 (1b) |
|---|---|---|---|
| 17408x5120 n=1 | 15.43 | 26.92 | 27.72 |
| 5120x17408 n=1 | 13.78 | 28.00 | 28.78 |
| 17408x5120 n=4 | 23.28 | 27.80 | 28.65 |
| 5120x17408 n=4 | 23.92 | 31.95 | 32.77 |
| 17408x5120 n=8 | 41.96 | 35.65 | 36.58 |
| 5120x17408 n=8 | 66.75 | 43.02 | 43.82 |

No better than v2 at M = 1 although the L1 wavefront problem is gone: the no-compute floor of the same copy pipeline
(weights copied, nothing consumed) is 15.6 us on 17408x5120, so the int8 path is issue bound. Per 64-k block and term
it needs the E2M1-to-int8 table expansion of 10 words, four m16n8k16 MMAs into fresh int32 tiles and the scale FMAs,
all on the SM's integer and FMA pipes; v3 is kept as `FP4_UNLOCK_KERNEL=2` for A/B and for devices without the
block-scaled MMA, and the arithmetic moved to the FP4 tensor core in v4.

## 2026-10-26/27 Phase 3 kernel v4: block-scaled FP4 MMA with residual activation terms

Design (fp4-smallm.cu, "v4" header comment): the Blackwell block-scaled MMA (`mma.m16n8k64 kind::mxf4nvf4`, e2m1 x
e2m1, ue4m3 scales, fp32 accumulate) consumes the raw NVFP4 weight words staged by cp.async, one instruction per 16
rows x 64 k per activation term. Activations are quantized to NVFP4 by a residual expansion y ~= y_0 + y_1 + ...:
y_0 is the NVFP4 quantization of y (per-column fp32 scale, per-16 UE4M3 scale with the quantize_mmq_nvfp4 offset
search, E2M1 nibbles), y_1 the quantization of y - y_0 with the column scale / 4, and so on. `FP4_UNLOCK_T` = 1 is
plain W4A4; T = 3 (default pending KL) puts the activation error below the weight's. A separate quantizer kernel writes
one activation record per 64-k block, laid out for the B fragment loads (columns >= ncols_dst are zero, so the GEMM
never branches on M). The GEMM is 4 warps x 16 rows = 64 rows per block, all four warps walking the same k split in the
same order (a record is fetched from L2 once per block, L1 for the other warps); split-K count is a function of the
weight shape and T only (`fp4_smallm_v4_ksplit`, target 480 CTAs = 170 SMs x 3), the partials are combined by the last
block of a row group in split order 0..ks-1, so the fp32 sequence of an output element is fixed by (shape, T) and never
by ncols_dst. Correctness: 67/67 MUL_MAT nvfp4 OK at T = 1, 2, 3, 4 (`$S/phase3/v4test_T*.log`, `v4dtest_T*.log`).

Versions (us/run, 17408x5120 / 5120x17408; `$S/phase3/v4perf1..5.txt`):

| version | change | T=1 n=1 | T=1 n=8 | T=3 n=1 | T=3 n=8 |
|---|---|---|---|---|---|
| MMVQ | - | 15.0 / 13.3 | 40.9 / 65.3 | - | - |
| v3 int8 | - | 27.1 / 28.1 | 35.8 / 43.1 | - | - |
| v4a | first build: one fp32 scale per column, quantizer block reads the whole column for the amax | 20.0 / 23.1 | 20.2 / 23.1 | 24.9 / 31.6 | 24.9 / 31.8 |
| v4b | column scale per (2048-k segment, column, term): the 128-thread quantizer block reduces the amax of the values it already holds, GEMM applies the scale per 256-k chunk (fresh accumulator per chunk, fixed order) | 18.67 / 18.99 | 18.91 / 19.13 | 20.52 / 23.59 | 20.71 / 23.99 |
| v4c | quantizer with 4 lanes per 16-group (256 threads, 1024-k segments) to cut the per-group chain | 21.06 / 24.89 | 20.65 / 25.12 | 31.59 / 25.87 | 31.45 / 26.95 |
| v4d (live) | v4b quantizer restored; ksplit rounded to a power of two for T >= 3 | 18.71 / 21.57 | 18.91 / 22.28 | 20.54 / 20.41 | 20.78 / 21.31 |

v4c's quantizer alone (EXP=16) took 2.2-3.2 us at T = 1 and 8.3-12.7 us at T = 3 against 1.4-1.6 / 2.4-4.4 for v4b
(`v4ablate5.txt`): spreading a group over four lanes doubles the shuffles per term and the 256-thread blocks halve the
occupancy, so it was reverted. The v4d T = 1 column on 5120x17408 (21.57) is run-to-run noise; the crossover table
below measured the same binary at 19.00.

Ablations (compile-time `FP4_SMALLM_V4_EXP`, results wrong, timing only; bit 1 = skip the B/activation loads, 2 = skip
the MMAs, 4 = skip the weight copies, 16 = quantizer only, 32 = GEMM only, 48 = no launches; `v4ablate*.txt`, v4b,
STAGES = 2 unless stated):

| EXP | meaning | T=1 n=1 | T=1 n=8 | T=3 n=1 | T=3 n=8 |
|---|---|---|---|---|---|
| 0 | full op | 18.7 / 19.0 | 18.9 / 19.1 | 20.5 / 23.6 | 20.7 / 24.0 |
| 1 | no B loads | 19.9 / 25.9 | 19.5 / 26.1 | 21.0 / 25.2 | 20.7 / 27.1 |
| 2 | no MMAs | 20.3 / 26.9 | 20.1 / 27.2 | 22.5 / 28.1 | 22.0 / 30.0 |
| 3 | neither | 18.1 / 18.0 | 18.3 / 18.1 | 19.5 / 19.4 | 19.7 / 19.5 |
| 4 | no weight copies | 8.0 / 10.3 | 8.2 / 11.2 | 11.9 / 14.6 | 12.1 / 15.4 |
| 5 | no copies, no B loads | 6.3 / 8.4 | 6.5 / 9.3 | 8.1 / 10.5 | 8.3 / 11.6 |
| 7 | copies, B loads and MMAs all off | 6.1 / 8.4 | 6.3 / 9.3 | 7.2 / 9.5 | 7.4 / 10.3 |
| 16 | quantizer alone | 1.37 / 1.49 | 1.35 / 1.56 | 2.44 / 4.39 | 2.50 / 2.94 |
| 32 | GEMM alone | 17.1 / 17.2 | 17.1 / 17.3 | 18.8 / 18.7 | 18.7 / 18.9 |
| 39 | GEMM shell (no copies, loads, MMAs) | 3.6 / 3.7 | 3.6 / 3.8 | 4.1 / 4.2 | 4.1 / 4.2 |
| 48 | nothing launched | 0.03 | 0.03 | - | - |

Reading: the B loads and the MMAs are not the cost (EXP 1, 2, 3 are within noise of the full op, and removing them
in the EXP=1/2 rows even loses the overlap that hides the copy latency); the weight copies are (EXP=4 removes 10-11 us
of the 18.7). The GEMM is streaming the weights at MMVQ's rate, and the gap to MMVQ at M = 1 (18.7 vs 15.0 on
17408x5120, 19.0 vs 13.4 on 5120x17408) is fixed cost that MMVQ does not have: the quantizer launch (1.4 us at T = 1,
2.4-4.4 at T = 3, a second launch in the PDL chain), the GEMM shell (3.6 us: launch, prologue, split-K combine, epilogue,
against 15.0 total for MMVQ) and the B record traffic from L2. STAGES sweep at EXP=0 (`v4ablate2.txt`): STAGES 2 is the
best (18.7 / 19.0 at T = 1), 3 gives 22.2 / 18.8, 4 gives 20.1 / 21.3; EXP=3 at STAGES 2 is 18.1 / 18.0, the copy
pipeline's floor for this block shape. KSPLIT sweeps (`v4perf2.txt` v4a T=3; `v4ksweep2.txt` v4b 5120x17408):

| KS (5120x17408) | T=1 n=1 / n=8 | T=2 n=1 / n=8 | T=3 n=1 / n=8 (v4a) |
|---|---|---|---|
| 1 | - | - | 55.8 / 56.2 |
| 2 | - | - | 44.1 / 44.5 |
| 4 | 19.75 / 19.81 | 21.15 / 21.17 | 32.5 / 33.8 |
| 6 | 19.03 / 19.09 | 21.03 / 20.95 | 23.7 / 24.0 (v4b) |
| 8 | 21.37 / 22.20 | 22.96 / 23.58 | 24.3 / 24.9 (v4a), 20.4 (v4d) |
| 12 | 21.00 / 22.48 | 23.16 / 23.60 | - |
| 16 | 21.07 / 21.65 | 22.26 / 22.59 | 25.3 / 25.9 |

so the automatic split (480 / nrg: 2 for 17408 rows, 6 for 5120 rows at T <= 2, rounded to 8 at T >= 3) is at the
measured optimum.

End to end (27B NVFP4, INV=1 so every M goes through v4, `llama-bench -d 0,8192 -r 5`,
`bench/results/2026-10-27-p3-27b-nvfp4-tgcost-v4`):

| build | tg128 d0 | tg128 d8192 | vs MMVQ |
|---|---|---|---|
| KERNEL=0 (MMVQ) | 82.96 (repeat 82.95) | 81.16 (80.98) | - |
| v4 T=1 | 76.44 | 74.88 | -7.9 % |
| v4 T=2 | 75.38 | 73.82 | -9.1 % |
| v4 T=3 | 74.13 | 72.62 | -10.6 % |

Property 2 ("M = 1 within 2 % of MMVQ") fails for v4 as it did for v1-v3, by less. The earlier hypothesis that the
microbench gap would shrink end to end because the real weights stream from HBM (the L2-resident test shapes flatter
MMVQ) is wrong: the step loses ~1 ms, which is the ~2.3 us per-GEMM fixed cost (quantizer launch + shell) times the
~450 dense GEMMs of a 27B step, the same gap as the microbench. nsys per-kernel durations are not usable for this
split: llama.cpp launches with programmatic dependent launch, so a kernel's recorded duration includes the time it
spends in `griddepcontrol.wait` behind its predecessor (`$S/phase3/v4nsys*.nsys-rep` were collected and discarded for
that reason). ncu, which would separate issue from memory stalls, needs `sudo` or the NVreg_RestrictProfilingToAdminUsers
driver setting (ERR_NVGPUCTRPERM, `$S/phase3/ncu1.txt`); left for the owner.

## 2026-10-27 Phase 3 kernel v5: the v4 GEMM with in-block quantization (mode 3, rejected)

Hypothesis: the fixed cost is the second launch plus the activation record round trip, so quantize inside the GEMM
block. v5 has no quantizer kernel and no record in global memory: each block reads the f32 activations of its own k
range (8 columns), quantizes one 256-k chunk per iteration into a shared-memory B buffer (one 16-element group per
thread, T terms, column scale per (chunk, column, term)) while the next weight chunk streams in through cp.async, then
runs the MMAs against the buffer; split-K uses a persistent per-(device, stream) counter array that the last block of
each row group resets, so nothing precedes the GEMM launch. Correctness 67/67 at T = 1..4 (`$S/phase3/v5test_T*.log`).
Numbers (us/run; `v5perf1.txt`, `v5perf2.txt`):

| kernel | 17408x5120 n=1 | 5120x17408 n=1 | 17408x5120 n=8 | 5120x17408 n=8 |
|---|---|---|---|---|
| MMVQ | 15.08 | 13.51 | 41.39 | 65.16 |
| v4 T=1 | 18.69 | 18.97 | 18.91 | 19.12 |
| v4 T=3 | 20.51 | 20.38 | 20.70 | 21.26 |
| v5 T=1 | 19.54 | 19.53 | 25.20 | 26.27 |
| v5 T=2 | 21.20 | 22.16 | 33.38 | 34.02 |
| v5 T=3 | 41.14 | 36.81 | 52.45 | 50.68 |
| v5 T=4 | 51.47 | 45.74 | 66.67 | 64.33 |
| v5 T=1, no scale search (NSEARCH=1) | 18.29 | 18.48 | 22.74 | 24.69 |
| v5 T=3, no scale search | 25.97 | 23.75 | 33.30 | 30.22 |

Diagnosis: every one of the nrg = 80..272 row groups re-quantizes the same activations (a 2.4 us job in v4 is done
80-272 times), and the per-group chain (amax shuffles, the 5-candidate scale-error search, fp4 conversions, the dequant
for the next residual) is latency bound and serial in t, about 10 us per extra term at n = 1 where only the threads of
column 0 have non-zero work but every thread runs the chain. The no-search diagnostic halves the T = 3 cost and is still
behind v4 at n = 8, so the structure is wrong, not the constants: the quantization has to happen once per MUL_MAT, in
its own kernel, and v4's separate quantizer stays. v5 remains in the tree as `FP4_UNLOCK_KERNEL=3` for A/B; the search
switch (`FP4_SMALLM_V5_NSEARCH`) is compile-time and defaults to the full search.

## 2026-10-27 the M = 2/3 crossover and the dispatch rule (`FP4_UNLOCK_MIN_M`)

With v4 flat in M and MMVQ growing in M, the product question is where the two cross (`$S/phase3/v4cross1.txt`, same
binary, us/run, 17408x5120 / 5120x17408):

| n | MMVQ | v4 T=1 | v4 T=2 | v4 T=3 |
|---|---|---|---|---|
| 1 | 15.01 / 13.42 | 18.70 / 19.00 | 19.89 / 20.84 | 20.57 / 20.43 |
| 2 | 15.82 / 14.80 | 18.76 / 18.99 | 19.86 / 20.84 | 20.62 / 20.48 |
| 3 | 19.17 / 27.94 | 18.78 / 19.05 | 19.89 / 20.87 | 20.68 / 20.56 |
| 4 | 22.83 / 23.41 | 18.80 / 19.02 | 19.95 / 20.95 | 20.66 / 20.67 |
| 6 | 30.63 / 54.69 | 18.86 / 19.05 | 19.99 / 20.99 | 20.70 / 20.97 |
| 8 | 41.33 / 65.71 | 18.92 / 19.09 | 20.05 / 21.02 | 20.75 / 21.30 |

MMVQ wins at M = 1 and 2 on both shapes; from M = 3 v4 wins on the FFN-down shape (27.9 vs 20.6) and is level on the
FFN-up shape (19.2 vs 20.7), and at M = 4..8 v4 wins by 1.1x to 3.1x. Decision: in the default mode dense NVFP4 goes to
MMVQ for ncols_dst <= 2 and to v4 for 3 <= ncols_dst <= 8 (`ggml_cuda_fp4_smallm_min_m()`, env `FP4_UNLOCK_MIN_M`,
default 3). Consequences:

- Batch-1 decode in the default mode is upstream's MMVQ, instruction for instruction, so property 2 holds by
  construction and the product makes no batch-1 claim (the headline is the verify step, M = 4..8).
- The default mode is not bitwise invariant across the M = 2 -> 3 boundary (the MUL_MAT_BATCH NVFP4 cases report
  "batch invariance broken at n=3", max abs diff 0.12-0.31 between MMVQ's int8 path and v4's FP4 x T path), and stays
  invariant inside [1, 2] and inside [3, 8].
- `FP4_UNLOCK_INVARIANT=1` sets min_m = 1, so every M in [1, 8] goes through v4 and the verify batch reproduces
  decode bitwise, at the M = 1 cost measured above (-8 % to -11 % tg128 depending on T). That is the same shape of
  trade as the FA tile pin at 64k context: invariance is the opt-in, speed is the default.
- The default T is decided by chain13's KL run (bar 0.043 vs Q8_0): T = 1 is W4A4 (fastest), T = 3 is the candidate
  default, T = 2 if it passes within noise.

## 2026-10-27 chain13 outputs, part 1: gates on the v4 + dispatch build

`test-backend-ops test -b CUDA0 -o MUL_MAT_BATCH,FA_BATCH`: KERNEL=0 INV=1 exit 0, KERNEL=1 INV=1 exit 0, KERNEL=1
INV=0 exit 1 as expected (NVFP4 cases break at n = 3, the dispatch boundary above; q4_K n=6 / q5_K n=7 / q6_K n=8 stock
Blackwell MMQ thresholds; all 4 FA cases without the tile pin). Full eval, all ops, KERNEL=1 INV=1: 19143/19143 passed,
exit 0 (`bench/results/2026-10-27-p3-eval-v4/on-inv-all.txt`); the "STEP FAILED (exit 1)" that follows it in
`chain13.log` is the chain's own `grep -B1 'compare failed'` returning 1 because nothing matched, not a test failure.
The rest of chain13 (per-op evals, KL at T = 3/1/2/4 and the MMVQ control, tg128 depth cost in default and invariant
modes, opinv, invariance 0/6, server rows, q8_0 KV, nsys split) is recorded in part 2.

## 2026-10-27 chain13 outputs, part 2: tg128 depth cost, pp, opinv

All numbers: 27B NVFP4, build-p3 (kernel v4 + the M >= 3 dispatch rule), RTX 5090, `gpu_exclusive`.
Source: `bench/results/2026-10-27-p3-27b-nvfp4-tgcost-v4b/summary.txt` and `pp-kernel1.jsonl`,
`bench/results/2026-10-27-p3-opinv-v4/p3-on-inv-kv8192.txt`.

### tg128 at depth 0 / 8192 / 65536 (tok/s, llama-bench)

| run | env | d=0 | d=8192 | d=65536 | vs kernel0 at d=0/8192/65536 |
|---|---|---|---|---|---|
| kernel0 | `FP4_UNLOCK_KERNEL=0` | 82.99 | 80.88 | 68.30 | control |
| kernel1-default | (defaults) | 83.04 | 80.80 | 68.28 | +0.1 % / −0.1 % / 0.0 % |
| kernel1-inv-T3 | `FP4_UNLOCK_INVARIANT=1` | 74.07 | 72.58 | 60.22 | −10.7 % / −10.3 % / −11.8 % |
| kernel1-inv-T1 | `FP4_UNLOCK_INVARIANT=1 FP4_UNLOCK_T=1` | 76.29 | 74.73 | 61.75 | −8.1 % / −7.6 % / −9.6 % |
| kernel0-inv | `FP4_UNLOCK_KERNEL=0 FP4_UNLOCK_INVARIANT=1` | 82.80 | 80.97 | 65.87 | −0.2 % / +0.1 % / −3.6 % |

Reading:

- **Phase 3.7 bar met.** In default mode M = 1 never enters the FP4 MMA path (dispatch rule: `ncols_dst >= 3`), so
  batch-1 decode is within noise of build-before / MMVQ at every depth. This is by construction, not by the kernel
  being fast at M = 1; see "the M = 2/3 crossover and the dispatch rule".
- **Invariant mode is a documented opt-in with a real cost.** At T = 3 it costs ~10.7 % tg128; ~7 points of that are
  the FP4 MMA kernel at M = 1 (v4 at T = 3 is ~2.3 us slower per GEMM than MMVQ, ~450 GEMMs per token), the rest is
  the FA pin (kernel0-inv: 0 % at short context, −3.6 % at 64k because the pinned FA kernel does not split KV).
- T = 1 in invariant mode is ~2.5 points cheaper than T = 3 but fails the KL bar (0.0570 vs 0.043), so it stays a
  labelled W4A4 fast path, never the default.

### pp512 / pp8192 (KERNEL=1, default)

| pp512 | pp8192 |
|---|---|
| 6533 | 6292 |

Unchanged vs build-before (6653 / ~6300 within run-to-run noise; prompt processing never touches the small-M path,
`ncols_dst <= 8`).

### opinv, KERNEL=1 INV=1, kv 8192, nmax = 16 (column 0 of the n-column result vs the n = 1 result, bitwise)

Every op listed (NVFP4 ffn_up/ffn_down, MXFP4, Q8_0, Q5_K x2, Q4_K, Q6_K, FLASH_ATTN f16 and q8_0 KV) is
`same` for n = 2..8 and `DIFF` from n = 9 on. n = 9 is where llama.cpp's own MMVQ -> MMQ switch and the FA
tile change happen for *every* type, i.e. outside the project's scope (speculative verify never exceeds n = 8 with the
drafters benchmarked here). The NVFP4 max |delta| at n = 9 (8.5 for ffn_up, 15 for ffn_down) is the usual
fp32-accumulation-order difference between MMVQ and MMQ, not an error: the same op passes test-backend-ops
against the CPU reference at every n.

Under INV=1 all of n = 1..8 go through the FP4 MMA kernel with the per-column quantizer, so column 0 is bitwise
identical across n, which is exactly property 1 (DESIGN-draft). Under the default (INV=0) the switch at n = 3 is
visible in opinv and in the f16-KV invariance runs; that is the documented trade.

### invariance (bench/invariance.sh, 6 prompts x 512 greedy tokens, bitwise text compare)

Source: `bench/results/2026-10-27-p3-27b-nvfp4-invariance-v4-inv/summary.txt` (INV=1) and
`...-invariance-v4-def/summary.txt` (INV=0, f16 KV only).

| mode | KV | ref A vs B | mtp-n3 | mtp-n7 | dflash-n3 | dflash-n7 |
|---|---|---|---|---|---|---|
| `FP4_UNLOCK_INVARIANT=1` | f16 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| `FP4_UNLOCK_INVARIANT=1` | q8_0 | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| default (INV=0) | f16 | 0/6 | 5/6 | 6/6 | 5/6 | 6/6 |

Invariant mode delivers property 1 end to end: every drafter, both KV types, speculative output bitwise identical to
non-speculative output over 3072 tokens. Default mode is batch-variant exactly as predicted by the dispatch rule
(M = 1, 2 on MMVQ, M >= 3 on the FP4 MMA kernel, plus the unpinned FA tile choice), so the divergence rates are
the same as build-before's. The one prompt that stays identical with the n3 drafters (code-3) is the usual
"this prompt happens not to hit a tie-break" case, not evidence of anything.

### server rows (bench/bench.sh rows 4-7, 27B NVFP4, ctx 8192, greedy, 3 runs, median tok/s)

Sources: `bench/results/2026-10-27-p3-27b-nvfp4-v4-{default,inv,off,default-T1,default-np,default-q8kv}/summary.csv`
and the 2026-10-26 build-before rows (`...-before-27b-nvfp4`, `...-before-27b-nvfp4-dflash-n3`).

| spec | verify M | before (10-26) | KERNEL=0 control (10-27) | default (v4 T=3, M>=3) | INV=1 T=3 | T=1 (W4A4) |
|---|---|---|---|---|---|---|
| none | 1 | 82.01 | 82.08 | 82.00 | 73.13 | 81.97 |
| mtp-n3 | 4 | 149.79 | 149.73 | **142.50** | 145.12 | 152.05 |
| mtp-n7 | 8 | 122.03 | 113.68 | **129.95** | 128.49 | – |
| mtp-n8 | 9 (MMQ) | 113.50 | – | 113.14 | 124.95 | – |
| dflash-n3 | 4 | 166.48 | (chain15) | **158.89** | 162.00 | – |
| dflash-n7 | 8 | 159.15 | 146.98 | **174.37** | 166.26 | 186.93 |
| synth4 | 5 | 141.94 | – | 144.94 | 143.38 | – |
| synth8 | 9 (MMQ) | 279.58 | 282.20 | 282.76 | 279.68 | 282.77 |

Row 7 (8/16 slots, 300 tokens): np8-none 361.33 (before 352.06, +2.6 %; the only row 7 cell whose batch is <= 8),
np8-mtp-n3 378.71 (372.28; M = 32, MMQ), np16-none 494.90 (496.50; MMQ), np16-mtp-n3 447.75 (443.41; MMQ).
q8_0 KV (rows 4+5, default): none 80.68, mtp-n3 142.60, dflash-n7 176.17 (no build-before q8_0-KV server rows exist;
these agree with the f16-KV rows to within the acceptance-length noise).

Caveats that matter when reading the table:

- The real drafters are *not* a clean per-step measurement: the FP4 MMA path changes the sampled text (default mode is
  batch-variant), so `acc_rate` / `mean_len` move with it (mtp-n7 control 3.283 vs default 3.323, dflash-n7 3.570 vs
  3.470, mtp-n3 2.590 vs 2.627). synth4/synth8 have fixed acceptance (3.95 / 7.86) and are the clean rows; synth8's
  verify batch is 9 tokens and goes through MMQ, so it never touches this project's kernel and is a sanity row only.
- The KERNEL=0 control reproduces the 10-26 build-before numbers to the second decimal for mtp-n3 (149.73 vs 149.79,
  same acceptance), so the two days are comparable; mtp-n7 and dflash-n7 moved down on the control (113.68 vs 122.03,
  146.98 vs 159.15) with different acceptance, which is the same text-path effect in the other direction.

Reading:

- **M = 8 (7-token drafters): +14 % (mtp-n7) and +19 % (dflash-n7) vs the same-day control**, with equal or lower
  acceptance length, i.e. a real per-step gain. At M = 8 MMVQ is 41 / 66 us per GEMM vs v4's 20.8 / 21.3.
- **M = 4 (3-token drafters): −4.8 % (mtp-n3) vs the control** and −4.6 % for dflash-n3 vs build-before, at equal
  acceptance. This contradicts the unfused microbench (v4 20.7 us vs MMVQ 22.8 / 23.4 at M = 4) and is the open
  question chain15 was queued for: the graph runs the FFN as fused [gate|up]+SWIGLU
  (`ggml_cuda_should_fuse_mul_mat_vec_q`), which the env-gated perf list never timed (`test_mul_mat_vec_fusion`
  rows were added to it for chain15). The per-step cost at M = 1..8 (`llama-bench -p M -n 0`) with fusion on/off and
  the n3 rows with `FP4_UNLOCK_MIN_M=5` will settle whether the default `MIN_M` moves to 5.
- synth4 (M = 5, fixed acceptance): +2.1 % vs build-before. Small because the step is not GEMM-bound at M = 5 the way
  it is at M = 8 (MMVQ 22.8 -> 30.6 us from M = 4 to 6).
- T = 1 is the fastest variant everywhere it was run (dflash-n7 186.93, +27 % vs control, at a *shorter* mean
  accepted length) and remains the labelled W4A4 opt-in because of the KL bar (0.0570).
- Invariant mode's real-drafter rows (mtp-n3 145.12, dflash-n7 166.26) are within the acceptance-length noise of the
  default rows; its cost is in the batch-1 `none` row (73.13, −10.9 %) and the FA pin at long context.

### nsys profile, synth8, default mode (`2026-10-27-p3-profile-synth8-v4`, split in `2026-10-27-profile-split.csv`)

Under PDL per-kernel durations are not meaningful (see the v4 section), so only the bucket shares are read:
mmq 47.3 %, mmvq 27.5 %, norm/rope/other 12.8 %, unknown 4.7 %, gdn 3.1 %, **fp4-mma 3.0 %** (1496 launches:
748 quantizer + 748 GEMM), fa 1.1 %, quantize 0.5 % (798 ms of GPU time in the profiled window; `profile_split.py`
got an `fp4-mma` bucket for `fp4_smallm*` so the project's kernels are no longer counted as unknown). Against
build-before's split (mmvq 29.8 % / 3179 launches, quantize 0.7 %) the 748 MMVQ launches that moved to the FP4 MMA
path are exactly the M = 3..8 GEMMs of a synth8 run (drafter-side and tail steps); the verify step itself is M = 9
and stays on MMQ, as in build-before. One number that is not PDL-sensitive because the kernel is
long and memory-bound: the Q8_0 lm_head GEMV (`mul_mat_vec_q<Q8_0,1>`, 151936 x 5120) is 796 us per batch-1 step,
~6.5 % of a 12.2 ms step, identical to build-before (796 us). It is outside this project's scope (the GGUF keeps
`output.weight` in Q8_0) but is the largest single kernel in the decode step and worth a note in the README.

## 2026-10-27 chain14 outputs: the T = 2 pass

Sources: `bench/results/2026-10-27-p3-27b-nvfp4-tgcost-v4b/summary.txt` (rows `kernel1-inv-T2`,
`kernel1-default-T2`), `...-v4-default-T2/summary.csv`, `...-v4-default-np-T2/summary.csv`.

| row | T=2 | T=3 (chain13) | KERNEL=0 control |
|---|---|---|---|
| tg128 default d0/8k/64k | 82.99 / 80.88 / 68.29 | 83.04 / 80.80 / 68.28 | 82.99 / 80.88 / 68.30 |
| tg128 INV=1 d0/8k/64k | 75.31 / 73.82 / 61.08 (−9.2 / −8.7 / −10.6 %) | 74.07 / 72.58 / 60.22 (−10.7 / −10.3 / −11.8 %) | – |
| none | 81.99 | 82.00 | 82.08 |
| mtp-n3 (M=4) | 144.22 (len 2.680) | 142.50 (2.627) | 149.73 (2.590) |
| mtp-n7 (M=8) | 131.04 (3.380) | 129.95 (3.323) | 113.68 (3.283) |
| dflash-n7 (M=8) | 177.33 (3.700) | 174.37 (3.470) | 146.98 (3.570) |
| synth8 (MMQ) | 282.64 | 282.76 | 282.20 |
| np8-none (M=8) | 360.34 | 361.33 | (before 352.06) |
| np8-mtp-n3 / np16-* (MMQ) | 400.54 / 493.62 / 476.74 | 378.71 / 494.90 / 447.75 | (text-path noise; not this kernel) |

T = 2 is never behind T = 3: equal in default mode (M = 1 on MMVQ either way), 1.5 points cheaper in invariant
mode, and ahead by 1-3 tok/s on every real drafter (partly acceptance-length noise; the per-GEMM delta is
~0.7 us). KL is 0.0422 (T=2) vs 0.0426 (T=3) vs 0.0432 (MMVQ control): both at the weight-quantization floor.
**Decision pending chain15's per-step rows (INV=1 T=2 vs T=3 at M = 1..8, clean of acceptance noise): if those
agree, the default becomes T = 2** (`ggml_cuda_fp4_smallm_terms()`), with T = 3 kept as the `FP4_UNLOCK_T=3`
"paranoid" setting and T = 1 the labelled W4A4 fast path.

## 2026-10-27 chain15 outputs: the M = 4 diagnosis, and the final defaults

Sources: `bench/results/2026-10-27-p3-27b-nvfp4-m4-diag/` (`unfused-kernel{0,1-T3}.txt`, `step-*.jsonl`,
`step-summary.txt`), `...-v4-default-minm5/summary.csv`, `...-v4-off-dflash-n3/summary.csv`.

### Fused-GEMM perf cannot be measured with test-backend-ops

The first step added `test_mul_mat_vec_fusion` shapes to the `FP4_UNLOCK_PERF_SHAPES` list; the rows came back at
0.76 us/run and 204 kB/run. `eval_perf` replicates only the graph's final node n_runs times, and for a
`run_whole_graph()` test that node is the trailing `add`, so the fused GEMM itself is never timed. The shapes were
removed again (test-backend-ops.cpp is back to the 299-line perf/batch-test addition). The fused path is measured
end to end instead, below.

### Unfused microbench, same binary (us/run, L2-resident)

| shape | M=1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| 17408x5120 MMVQ | 15.11 | 15.95 | 19.35 | 22.90 | 30.74 | 41.36 |
| 17408x5120 v4 T=3 | 20.56 | 20.59 | 20.65 | 20.63 | 20.67 | 20.71 |
| 5120x17408 MMVQ | 13.51 | 14.94 | 28.20 | 23.55 | 55.08 | 66.05 |
| 5120x17408 v4 T=3 | 20.42 | 20.47 | 20.54 | 20.64 | 20.92 | 21.24 |

Unchanged from the Phase 3 table: here v4 is ahead from M = 3. This is the number that misled the MIN_M = 3 default.

### Per-step cost, end to end (llama-bench `-p M -n 0 -r 20`, ms per forward of M tokens, 27B NVFP4, d = 0)

| config | M=1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| KERNEL=0 (MMVQ, fused) | 12.21 | 12.68 | 12.98 | 13.67 | 15.83 | 18.30 |
| KERNEL=1 default (M>=3 v4 T=3, fused) | 12.21 | 12.67 | 13.90 | 14.19 | 14.65 | 14.88 |
| KERNEL=1 INV=1 T=3 | 13.65 | 13.82 | 13.91 | 14.19 | 14.58 | 15.40 |
| KERNEL=1 INV=1 T=2 | 13.43 | 13.80 | 13.94 | 14.10 | 14.61 | 15.41 |
| KERNEL=0, fusion off | 12.74 | 13.09 | 13.40 | 14.14 | 16.08 | 18.30 |
| KERNEL=1 default, fusion off | 12.74 | 13.09 | 14.30 | 14.67 | 15.03 | 15.31 |

d = 8192 rows (in `step-summary.txt`) show the same pattern shifted by the KV cost (+0.3 to +1.1 ms).

Reading:
- **The M = 4 regression is real and is not the fusion.** v4 is 0.9 ms behind MMVQ at M = 3 and 0.5 ms behind at
  M = 4 with fusion on, and 0.9 / 0.5 ms behind with fusion off (`GGML_CUDA_DISABLE_FUSION=1`). Fusion is worth
  0.4-0.5 ms per step to both kernels equally.
- **The microbench crossover is wrong because it is L2-resident.** Each perf shape is one 50 MB weight that fits the
  5090's 96 MB L2, so MMVQ at M = 3..4 is measured at L2 bandwidth and looks 1.3-2x slower than at M = 1. In the real
  step the weights stream from HBM once per layer and MMVQ's M = 3..4 cost grows far less (12.98 / 13.67 ms vs
  12.21 at M = 1: +6 % / +12 %, not +30-100 %). v4's cost is flat in M in both settings; its fixed overhead
  (quantizer launch + GEMM shell, ~1.2 ms per step at M = 1) is what MMVQ has to outgrow.
- **End-to-end crossover is between M = 4 and M = 6**: v4 is behind by 0.5 ms at M = 4, ahead by 1.2 ms at M = 6
  and 3.4 ms at M = 8. The dispatch threshold moves to **MIN_M = 5** (M <= 4 on MMVQ, M = 5..8 on v4).
- **T = 2 vs T = 3 (INV=1 rows, no acceptance noise)**: T = 2 is 0.22 ms cheaper at M = 1, equal (within 0.1 ms)
  from M = 2. With chain14's server rows (T = 2 never behind) and KL 0.0422 vs 0.0426, **the default becomes T = 2**.
  T = 3 stays as `FP4_UNLOCK_T=3`, T = 1 as the labelled W4A4 fast path.
- Invariant-mode cost at the new default (INV=1 T=2): +1.2 ms per batch-1 step (13.43 vs 12.21, −9 %), which is the
  number the README quotes for the opt-in.

### MIN_M = 5 server rows (n = 3 drafters, M = 4 verify back on MMVQ), vs same-day KERNEL=0 controls

| row | MIN_M=5 (T=3 build) | KERNEL=0 control | chain13 MIN_M=3 |
|---|---|---|---|
| mtp-n3 | 149.67 (len 2.590) | 149.73 (2.590) | 142.50 (2.627) |
| dflash-n3 | 166.82 (2.650) | 166.56 (2.650) | 158.89 |

Both rows recover the control to within 0.3 tok/s with identical acceptance (same mean length, as expected: at
MIN_M = 5 an n = 3 drafter runs upstream's kernels end to end). The −4.8 % at mtp-n3 was entirely the M = 4
dispatch. The 10-26 "before" dflash-n3 row (166.48) also agrees with today's control (166.56).

### Decision applied

`fp4-smallm.cu`: `FP4_SMALLM_T_DEFAULT 2`, `ggml_cuda_fp4_smallm_min_m()` default 5, comment block rewritten with the
per-step table; `fp4-smallm.cuh` comments updated; `docs/DESIGN-draft.md` dispatch rule and precision decision
updated. Consequence for the product table: n = 3 drafters (M = 4) are unchanged by design; the gain is for n >= 4
(M = 5..8), where MMVQ's step cost was growing linearly with M. chain16 re-benches the full table on the shipped
defaults (p3 rebuilt from this source).

## 2026-10-27 chain16 outputs: the product table on the shipped defaults (T = 2, MIN_M = 5)

Everything below is the `build-p3` tree after the chain15 defaults were applied, one GPU, one day (01:49-03:07), spec
decode greedy at 8K context, 512-token answers, median of 3 timed runs after a warm-up. Directories:
`bench/results/2026-10-27-p3-27b-nvfp4-final*` (server rows), `...-final` (llama-bench rows), `...-kld-final`,
`...-invariance-final-{inv,def}`, `2026-10-27-p3-opinv-final`. `bench/final_table.py` renders the server tables.
The same-day `FP4_UNLOCK_KERNEL=0` control is the comparison that matters; the 10-26 "before" column (unpatched
upstream) is kept because it is the frozen baseline, and it differs from the control only by day-to-day drift
(clocks were never locked: `mtp-n7` 122.0 on 10-26 vs 113.6 today with the same code path, `np8-none` 352 vs 324).

### Correctness and invariance

- test-backend-ops batch tests: INV=1 19/19; INV=0 6/19 (the expected stock boundaries: NVFP4 at n = 5 where the
  dispatch changes, k-quants at n = 6..8 where MMVQ changes warp count, FA at n >= 3).
- opinv INV=1 (bitwise, column 0 of the n-column result vs n = 1): every op `same` for n = 2..8 (NVFP4, MXFP4, Q8_0,
  Q5_K, Q4_K, Q6_K, FA f16 and q8_0 KV), DIFF from n = 9 (llama.cpp's own MMVQ -> MMQ switch, out of scope).
- opinv INV=0: NVFP4 DIFF from n = 5 (max |delta| 0.82 on ffn_up, 1.46 on ffn_down: a different, not a worse,
  rounding; KL below), Q4_K from n = 6, Q6_K at n = 8, FA from n = 3.
- Invariance (greedy spec on vs spec off, 6 prompts x 512 tokens, byte diff), INV=1: **0/6 for every drafter
  (mtp-n3/n5/n7, dflash-n3/n5/n7) on both f16 and q8_0 KV**, 14 comparisons, zero diffs.
  Default mode (INV unset), f16: n3 drafters 6/6, n5 5/6, n7 6/6. Default mode is the fast layer, invariant mode is
  the opt-in bitwise layer; both ship.
- KL vs Q8_0 on kl-set-5k, shipped defaults with `-b 8 -ub 8` (every GEMM at M = 8 through v4 T = 2):
  Mean 0.042219, same top-1 95.354 %. Upstream MMQ q8 at `-b 512`: 0.042939 / 95.354 %. Upstream W4A4 (`q4`):
  0.061810 / 94.329 %. The kernel is at the weight-quantization floor.

### Batch-1 and prefill (unchanged by design)

| | tg128 d0 | tg128 d8192 | tg128 d65536 | pp512 | pp8192 |
|---|---|---|---|---|---|
| KERNEL=0 | 83.14 | 81.03 | 68.34 | | |
| default | 83.06 | 80.86 | 68.29 | 6573.8 | 6373.6 |
| INV=1 | 75.27 | 73.76 | 61.08 | | |

M = 1 never takes the kernel (MIN_M = 5), so batch-1 is identical within noise (-0.1 %). Invariant mode costs
-9.5 % at batch 1 (FA pin + 4-warp MMVQ pin), documented and opt-in.

### Per-step cost, llama-bench -p M -n 0 (ms per forward of M tokens, d0 / d8192)

| M | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| KERNEL=0 d0 | 12.22 | 12.68 | 13.01 | 13.69 | 14.84 | 15.99 | 16.92 | 18.35 |
| default d0 | 12.21 | 12.69 | 13.01 | 13.67 | **14.41** | **14.70** | **14.97** | **14.90** |
| INV=1 d0 | 13.43 | 13.80 | 13.94 | 14.11 | 14.33 | 14.62 | 15.00 | 15.42 |
| KERNEL=0 d8192 | 12.54 | 13.07 | 13.48 | 13.85 | 15.36 | 16.29 | 17.32 | 18.50 |
| default d8192 | 12.53 | 13.07 | 13.47 | 13.84 | 15.09 | 15.38 | 15.69 | 15.62 |

The kernel's step is flat from M = 5 to 8 (-2.9 % at M = 5, -8.1 % at 6, -11.5 % at 7, **-18.8 % at M = 8**).
MMVQ grows ~0.9 ms per extra token. This is the mechanism behind every gain below.

### Server rows (decode tok/s, acceptance, mean accepted length, board power W)

| config | before 10-26 | KERNEL=0 control | **default** | vs control | INV=1 | acc (ctl / def) | mean len (ctl / def) | power W (ctl / def) |
|---|---|---|---|---|---|---|---|---|
| none | 82.0 | 82.2 | 82.1 | -0.1 % | 74.3 | | | 535 / 534 |
| mtp-n3 | 149.8 | 149.7 | 149.8 | +0.1 % | 142.3 | 0.574 / 0.574 | 2.59 / 2.59 | 550 / 550 |
| mtp-n5 | | 132.4 | **138.8** | **+4.9 %** | | 0.421 / 0.430 | 2.92 / 3.07 | 562 / 492 |
| mtp-n7 | 122.0 | 113.6 | **127.1** | **+11.8 %** | 119.2 | 0.322 / 0.315 | 3.28 / 3.29 | 542 / 468 |
| mtp-n8 | 113.5 | 113.4 | 113.6 | +0.2 % | | 0.280 / 0.281 | 3.01 / 3.02 | 444 / 444 |
| dflash-n3 | 166.5 | 166.6 | 166.5 | -0.1 % | 157.3 | 0.615 / 0.615 | 2.65 / 2.65 | 545 / 546 |
| dflash-n5 | | 159.1 | **173.3** | **+9.0 %** | | 0.465 / 0.480 | 3.11 / 3.17 | 557 / 479 |
| dflash-n7 | 159.2 | 146.9 | **169.1** | **+15.1 %** | 159.2 | 0.375 / 0.367 | 3.57 / 3.50 | 531 / 445 |
| synth4 | 141.9 | 144.8 | 145.0 | +0.1 % | | 0.376 | 3.95 | 439 / 440 |
| synth6 | | 211.4 | 211.3 | 0.0 % | | 0.610 | 5.81 | 430 / 430 |
| synth8 | 279.6 | 282.2 | 282.7 | +0.2 % | 280.4 | 0.859 | 7.86 | 421 / 418 |
| np8-none | 352.1 | 324.3 | **360.0** | **+11.0 %** | | | | 518 / 427 |
| np8-mtp-n3 | 372.3 | 386.6 | **428.9** | **+10.9 %** | | 0.650 / 0.650 | 2.98 / 2.98 | 406 / 381 |
| np8-mtp-n7 | | 297.6 | **352.3** | **+18.4 %** | | 0.404 / 0.452 | 4.02 / 4.35 | 431 / 403 |
| np16-none | 496.5 | 495.3 | 495.9 | +0.1 % | | | | 374 / 374 |
| np16-mtp-n3 | 443.4 | 457.9 | 466.7 | +1.9 % | | 0.667 / 0.645 | 3.05 / 2.98 | 394 / 382 |
| q8_0 KV none | | 80.6 | 80.7 | +0.1 % | | | | 527 / 526 |
| q8_0 KV mtp-n7 | | 122.8 | **136.0** | **+10.8 %** | | 0.367 / 0.342 | 3.44 / 3.25 | 535 / 462 |
| q8_0 KV dflash-n7 | | 169.6 | **182.8** | **+7.8 %** | | 0.427 / 0.392 | 3.76 / 3.50 | 521 / 438 |

Reading:

- **Where the verify batch is M = 5..8 the kernel wins, 5-18 %, at 60-90 W less board power.** mtp-n7 / dflash-n7
  (M = 8 verify), mtp-n5 / dflash-n5 (M = 6), np8 (8 slots x 1 token = M = 8, with or without a drafter), q8_0 KV n7.
  The power drop is the FP4 tensor pipe doing in ~1 ms what the dp4a path spread over 4-6 ms.
- **M <= 4 and M >= 9 are unchanged by construction.** n3 drafters (M = 4) stay on MMVQ (MIN_M = 5); mtp-n8 (M = 9),
  synth rows (the synthetic drafter always drafts 8 tokens, M = 9, then accepts at random: it is the fixed-acceptance
  control for the MMQ path) and np16 (M = 16) are on MMQ. Nothing moved by more than noise on those rows.
- **np8-mtp-n7 is the biggest row, +18.4 %.** With 8 slots the batch handed to llama_decode is the union of whatever
  each slot needs that step; slots drift apart (prompt, draft, verify, finished), so the M of each call varies and the
  gain says a large share of them land in 5..8. The exact per-call M histogram was not recorded (a follow-up:
  count ne1 in the dispatch under np8). The acceptance shift (0.404 -> 0.452) comes with the kernel's different
  rounding; it is reported, not explained.
- Acceptance and mean length on the n5/n7 rows move by a few hundredths both ways between control and default: the
  kernel rounds differently (not worse, KL above), so greedy drafts diverge at some tokens. Invariant mode removes
  that (0/6) at the cost of -5..-10 %.
- INV=1 rows: none 74.3, mtp-n3 142.3, mtp-n7 119.2, dflash-n3 157.3, dflash-n7 159.2, synth8 280.4. Against
  the kernel-off control, invariant mode is -9.6 % at batch 1, -5 % with n3 drafters, **+4.9 % (mtp-n7) and
  +8.4 % (dflash-n7)** on the n7 rows: at M = 8 the kernel more than pays for the FA pin.
- **Best single-user number: dflash-n7 with q8_0 KV, 182.8 tok/s, 2.27x the 80.7 tok/s batch-1 rate on the same
  settings** (control 169.6). Best f16 KV: dflash-n5 173.3 (2.11x). The 2-3x headline is the spec-decode layer;
  the kernel's own contribution on top of upstream spec decode is the 5-18 % column.
- T sweep on the n7 rows (default T = 2 vs T = 3 extra margin vs T = 1 W4A4): mtp-n7 127.1 / 123.1 / 124.9,
  dflash-n7 169.1 / 168.3 / 171.3. T = 1 is 1 % faster on dflash and fails the KL bar (0.0570); T = 3 buys nothing
  measurable. T = 2 default confirmed.

### Downstream reach check (Phase 4)

- **Ollama 0.20.7** (client + server from /usr/local/bin, isolated model store and port): `ollama create` accepts the
  cdiamond NVFP4 GGUF, the load fails in the runner with `ggml.c:1676: GGML_ASSERT(type >= 0 && type < GGML_TYPE_COUNT)`
  (`file_type=unknown`): Ollama's bundled ggml does not have the NVFP4 tensor type. No NVFP4 GGUF runs in Ollama at
  this version, with or without this project. Log: scratchpad `reach/serve.log` (not kept in the repo).
- **LM Studio**: not installed on this machine; not verified.
- The README therefore claims llama.cpp (llama-server / llama-bench / llama-cli from this fork) only.

### What is still open after chain16

- The before column is from 10-26 and clocks were never locked (sudo); the same-day control makes the table honest
  but a clock-locked rerun of `before` + `default` would tighten the +/-5 % day drift on the n7 rows.
- vLLM column: the dry run on 10-26 failed (container exited); never rerun. README states "not measured".
- MXFP4 and MoE (MUL_MAT_ID) are Phase 3.5, not shipped: the kernel is NVFP4 MUL_MAT only.
