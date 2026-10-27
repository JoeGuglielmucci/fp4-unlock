# fp4-unlock

**Lossless speculative decoding for NVFP4 GGUFs on the RTX 5090: 2.2x over batch 1 with bit-identical output, on
the FP4 tensor cores.**

Four patches on top of [llama.cpp](https://github.com/ggml-org/llama.cpp), pinned at upstream commit `63b4b303`
(2026-10-26). This repository carries only the patches; the build script fetches upstream at the pinned commit and
applies them. Every number below was measured on one RTX 5090 (32 GB, driver 595.99.02, CUDA 13.2) with
Qwen3.8-27B NVFP4 in one session on 2026-10-27, with the commands in `bench/` and the raw output in `bench/results/`.

## Read this first: what it will not do for you

- **Batch 1 without a drafter does not get faster.** Plain single-user decode is memory-bound at about 82 tok/s on
  this model and stays there. Everything here is about speculative decoding and small batches.
- **Ollama and LM Studio do not get it.** This is a llama.cpp build; you run `llama-server`, `llama-cli` or
  `llama-bench` from it. Ollama 0.20.7 cannot load any NVFP4 GGUF at all (its bundled ggml rejects the tensor type).
- **Blackwell only.** The kernel uses the Blackwell block-scaled FP4 MMA and is built for compute 12.0 (`120a`).
  On any other card the patches do nothing useful.
- **MoE models do not take the kernel.** Expert matmuls go through a different op (`MUL_MAT_ID`). On Qwen3.6-35B-A3B
  NVFP4 the kernel on and off measure the same (426 vs 427 tok/s with the MTP drafter).

If you run an NVFP4 GGUF through llama.cpp on an RTX 5090 with an MTP or DFlash drafter, read on.

## What you get

1. **Speculative decoding that does not change the output.** Upstream llama.cpp changes greedy output when a
   drafter is on, because the verify step (a batch of 4 to 8 tokens) runs different kernels and different reduction
   orders than plain decode. With the `invariant` build and `FP4_UNLOCK_INVARIANT=1`, greedy output with a drafter on
   is byte-identical to greedy output with the drafter off: 0 of 6 prompts diverge for every drafter and both KV
   types, where upstream diverges on 5 or 6 of 6. In that mode the DFlash2 drafter decodes at 167 tok/s against
   74 tok/s for the same build at batch 1 (2.25x), or 2.03x upstream's own batch-1 rate of 82 tok/s.
   The price of the pins is 2-5 % against upstream's speculative rows and 9.6 % at batch 1, which is why the mode
   is opt-in.
2. **The verify step on the FP4 tensor cores.** For a decode batch of 5 to 8 tokens, NVFP4 matmuls run through a
   block-scaled FP4 MMA kernel (`mma.sync m16n8k64`, e2m1 x e2m1, ue4m3 scales, fp32 accumulate) instead of the
   int8 dp4a `MMVQ` path. Its step cost is flat from M = 5 to 8 where MMVQ grows about 0.8 ms per token. With a
   7-token drafter that is +6.8 % (MTP) and +8.2 % (DFlash2) end to end, at 60 to 80 W less board power. The
   activations are quantized to NVFP4 with two residual terms, which puts the activation error below the weight's:
   KL against the Q8_0 model is the same as upstream's int8 path.
3. **A loader fix** so an MTP GGUF loads without `--mtp` (the 35B-A3B NVFP4-MTP GGUF fails to load upstream), plus
   batch-invariance cases for `test-backend-ops`.

## Results: Qwen3.8-27B NVFP4, RTX 5090, greedy, 8K context, 512-token answers

Decode tok/s, median of 3 timed runs after a warm-up, 6 prompts (3 code, 3 prose), `llama-server`, 2026-10-27.
Columns: `before` is unpatched upstream at the pinned commit; `default` is this repository's default build;
`kernel off` is the default build with `FP4_UNLOCK_KERNEL=0`; `invariant` is the invariant build with
`FP4_UNLOCK_INVARIANT=1`. Drafters: `mtp-nN` is the MTP head embedded in the GGUF, `dflash-nN` is the DFlash2 Q4_K_M
head, N is the maximum draft length (verify batch M = N + 1). Clocks were not locked. `before` and `kernel off` run
the same code path and agree within 1 % on every single-slot row; that is the noise floor.

| config | before (upstream) | default | kernel off | invariant | default vs before | power W, before / default / invariant |
|---|---|---|---|---|---|---|
| no drafter (batch 1) | 82.1 | 82.2 | 82.0 | 74.2 | +0.1 % | 519 / 519 / 466 |
| mtp-n3 | 148.8 | 149.6 | 149.6 | 141.4 | +0.6 % | 544 / 545 / 478 |
| mtp-n5 | 138.0 | 136.3 | 138.7 | 134.2 | -1.2 % | 548 / 486 / 488 |
| mtp-n7 | 121.8 | **130.2** | 120.8 | 119.2 | **+6.8 %** | 529 / 462 / 494 |
| dflash-n3 | 166.0 | 166.4 | 164.6 | 157.1 | +0.2 % | 542 / 545 / 475 |
| dflash-n5 | 174.5 | 171.2 | 173.1 | 167.0 | -1.9 % | 538 / 468 / 488 |
| dflash-n7 | 163.5 | **176.9** | 162.2 | 159.2 | **+8.2 %** | 506 / 430 / 497 |
| mtp-n7, q8_0 KV | 131.6 | **138.1** | 131.8 | - | **+4.9 %** | 524 / 453 / - |
| dflash-n7, q8_0 KV | 182.8 | **191.2** | 184.0 | - | **+4.6 %** | 502 / 425 / - |
| 8 slots, no drafter (aggregate, 300 tokens) | 351.0 | **365.2** | 350.9 | - | **+4.0 %** | 502 / 414 / - |
| 8 slots, mtp-n3 (aggregate) | 396.5 | 406.2 | 413.9 | - | +2.4 % | 402 / 390 / - |
| 8 slots, mtp-n7 (aggregate) | 327.1 | 369.8 | 379.5 | - | +13.0 % | 414 / 392 / - |

How to read it:

- **Speculative decoding is the 2x.** 82 to 177 tok/s (dflash-n7) is 2.15x batch 1 on the default build; with
  q8_0 KV, 191 tok/s is 2.33x. Upstream already gets most of that. This repository makes it lossless (invariant
  column) and faster at the longer draft lengths (n7 rows).
- **The kernel row is n7.** The n3 rows (M = 4) are routed to upstream's kernel by design and are unchanged. The n5
  rows (M = 6) are the crossover: per-step cost is equal within 0.5 ms either way (table below), and the end-to-end
  difference of -1 to -2 % comes from a slightly lower acceptance rate on different generated text, not from the
  kernel; it draws 60-70 W less there. Set `FP4_UNLOCK_MIN_M=7` if you want M <= 6 on the upstream path.
- **The 8-slot drafter rows are noisy.** Acceptance varies by run with 8 concurrent prompts (0.405 / 0.428 / 0.423
  on mtp-n7), so only the no-drafter 8-slot row, where both controls agree, is a kernel result (+4.0 %).
- **Invariant mode** costs 9.6 % at batch 1 and 2-5 % on the drafter rows against upstream. Its 2.2x is measured
  against its own batch-1 rate (74.2 -> 167.0 tok/s, dflash-n5); against upstream batch 1 it is 2.03x.

### Lossless: greedy output with a drafter vs without, 6 prompts x 512 tokens

Prompts diverging from the spec-off reference (the reference run twice against itself: 0/6 in every case).

| drafter | upstream (2026-10-26) | default mode | invariant build + `FP4_UNLOCK_INVARIANT=1` |
|---|---|---|---|
| mtp-n3, f16 KV | 6/6 | 6/6 | **0/6** |
| mtp-n7, f16 KV | 6/6 | 6/6 | **0/6** |
| dflash-n3, f16 KV | 6/6 | 6/6 | **0/6** |
| dflash-n7, f16 KV | 6/6 | 6/6 | **0/6** |
| mtp-n3, q8_0 KV | 6/6 | - | **0/6** |
| mtp-n7, q8_0 KV | 5/6 | - | **0/6** |
| dflash-n3, q8_0 KV | 6/6 | - | **0/6** |
| dflash-n7, q8_0 KV | 5/6 | - | **0/6** |

At the op level (`bench/opinv`): column 0 of an n-column `MUL_MAT` or `FLASH_ATTN_EXT` compared bitwise with the
n = 1 result. Upstream and the default mode differ from n = 3 (flash attention) and n = 5 (NVFP4 matmul) on; the
invariant mode is identical for every op and every n from 2 to 8 (`bench/results/2026-10-27-p3-opinv-final/`).

### The mechanism: per-step cost, ms per forward of M tokens (`llama-bench -p M -n 0`, 10 repetitions)

| M | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| upstream, 0 context | 12.25 | 12.71 | 13.00 | 13.66 | 14.04 | 14.85 | 15.62 | 16.52 |
| default, 0 context | 12.26 | 12.71 | 13.01 | 13.67 | 14.36 | 14.68 | **14.93** | **14.91** |
| invariant, 0 context | 13.46 | 13.83 | 13.96 | 14.13 | 14.36 | 14.64 | 15.02 | 15.44 |
| upstream, 8192 context | 12.65 | 13.20 | 13.59 | 13.96 | 14.63 | 15.39 | 16.10 | 16.87 |
| default, 8192 context | 12.65 | 13.20 | 13.59 | 13.95 | 15.16 | 15.47 | **15.73** | **15.76** |
| invariant, 8192 context | 14.09 | 14.38 | 14.60 | 14.82 | 15.14 | 15.57 | 16.08 | 16.61 |

M = 1..4 run the same kernel in `upstream` and `default` (equal within 0.01 ms). The kernel is flat from
M = 5 to 8; at M = 8 it is 9.7 % (0 context) and 6.6 % (8K context) cheaper per step, which with a 3.3-3.5 token
mean accepted length becomes the 7-8 % in the server table. In invariant mode every M from 1 to 8 runs the kernel,
so M = 1 pays the kernel's fixed cost (+1.2 ms, the 9.6 % at batch 1).

### Quality: KL divergence vs the Q8_0 model (kl-set-5k, 367 lines, every matmul at M = 8 via `-b 8 -ub 8`)

| path | mean KL | top-1 agreement |
|---|---|---|
| upstream int8 MMVQ path (`before`, 2026-10-26) | 0.0431 | 95.32 % |
| default build, kernel off | 0.0432 | 95.29 % |
| **kernel, `FP4_UNLOCK_T=2` (default)** | **0.0422** | **95.35 %** |
| kernel, `FP4_UNLOCK_T=3` | 0.0426 | 95.17 % |
| kernel, `FP4_UNLOCK_T=1` (plain W4A4, not default) | 0.0570 | 94.30 % |
| upstream MMQ at `-b 512`, int8 activations | 0.0429 | 95.35 % |
| upstream MMQ at `-b 512`, FP4 activations | 0.0618 | 94.33 % |

Two residual terms (T = 2) put the FP4 activation error under the weight's own error; one term is the W4A4 fast path
that upstream also offers through `GGML_CUDA_MMQ_PREC=q4`, and it fails the bar. Prefill never takes the kernel.
Batch-1 decode at context depth 0 / 8192 / 65536: upstream 83.0 / 80.9 / 68.3 tok/s, default 83.0 / 80.8 / 68.3,
invariant 75.3 / 73.8 / 61.1 (`bench/results/2026-10-27-p3-27b-nvfp4-tgcost-v4b/`).

## Quick start

Requirements: an RTX 50-series card, the CUDA 13.x toolkit (`nvcc` under `/usr/local/cuda-13.*` or `CUDA_HOME`)
with the cuBLAS development files, cmake, ninja, a C++17 compiler, git, and the Hugging Face CLI (`pip install
huggingface_hub`) for the models. About 20 GB of VRAM at 8K context with the DFlash2 drafter loaded, 18 GB of disk for the two GGUFs.

```
git clone https://github.com/JoeGuglielmucci/fp4-unlock
cd fp4-unlock
scripts/build.sh                     # fetches upstream llama.cpp at docs/PINNED_COMMIT into third_party/ (gitignored),
                                     # applies patches/, builds the default tree -> build/   (JOBS=N limits parallelism)
scripts/build.sh invariant           # the lossless build -> build-invariant/ (adds the compile-time MMVQ pin)
scripts/fetch-models.sh              # Qwen3.8-27B NVFP4-MTP GGUF (17.1 GB) + DFlash2 Q4_K_M drafter (1.1 GB) -> ~/models/
scripts/verify-lossless.sh           # the headline check: 0.0 divergence for mtp-n7 and dflash-n7, 6-8 minutes
```

Then serve (OpenAI-compatible API on port 8080; `PORT`, `CTX` and `KV` override the defaults):

```
FP4_UNLOCK_INVARIANT=1 scripts/run-27b.sh invariant dflash 7   # lossless, 159 tok/s; dflash 5 for 167 tok/s
scripts/run-27b.sh after dflash 7                              # fastest f16 KV row, 177 tok/s (not lossless)
KV=q8_0 scripts/run-27b.sh after dflash 7                      # 191 tok/s
scripts/run-27b.sh after mtp 7                                 # no second model file, 130 tok/s
```

```
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a bash one-liner that counts lines per file type."}],"temperature":0}'
```

Any other NVFP4 GGUF works the same way with `build/bin/llama-server` directly; the relevant upstream flags are
`-ngl 99 -fa on` plus `--spec-type draft-mtp --spec-draft-n-max 7 --spec-draft-backend-sampling` (a GGUF with an MTP
head) or `-md <drafter.gguf> --spec-type draft-dflash --spec-draft-n-max 7` (a DFlash drafter). The model must fit in
VRAM; the kernel does nothing for a MoE model (see the top of this page).

If `scripts/build.sh` stops at configure with cuBLAS missing: install the cuBLAS 13 development package (for
example `libcublas-dev-13-2`), or put the contents of the pip `nvidia-cublas` cu13 wheel into `.cuda-overlay/
{include,lib}` (gitignored); the build script picks the overlay up when it exists.

### Runtime switches (environment, all optional)

| variable | default | meaning |
|---|---|---|
| `FP4_UNLOCK_KERNEL` | 1 | 0 = upstream MMVQ path (the A/B control), 1 = FP4 tensor-core kernel, 2 = int8 MMA fallback, 3 = in-block quantization variant (slower, A/B only) |
| `FP4_UNLOCK_T` | 2 | residual activation terms 1..4; 1 = plain W4A4 (KL 0.057), 3 = extra margin (no measurable gain) |
| `FP4_UNLOCK_MIN_M` | 5 | smallest batch sent to the kernel; M below it stays on MMVQ |
| `FP4_UNLOCK_INVARIANT` | unset | 1 = lossless mode: MIN_M becomes 1 and the flash-attention and MMVQ kernel choices are pinned; needs the `invariant` build (the default build prints a warning and is only partially invariant) |
| `GGML_CUDA_DISABLE_FUSION` | unset | upstream switch; the kernel supports the fused gate/up + GLU path |

### Tests

```
build/bin/test-backend-ops test -b CUDA0 -o MUL_MAT -p nvfp4                                   # kernel vs CPU reference
FP4_UNLOCK_INVARIANT=1 build-invariant/bin/test-backend-ops test -b CUDA0 -o MUL_MAT_BATCH,FA_BATCH   # 19/19 batch-invariance cases
```

The batch-invariance cases pass only on the invariant build; on the default build they fail by design (no pinned
MMVQ k split), which is the warning `FP4_UNLOCK_INVARIANT=1` prints there.

### Reproducing the tables

Build all three trees (`scripts/build.sh all` also builds `build-before/`, the unpatched control, from a detached
worktree of the pinned commit), then:

```
bench/bench.sh before 27b-nvfp4 4,5                                 # server rows; tree = before|after|invariant
bench/bench.sh after 27b-nvfp4 4,5                                  # SPECS="none mtp-n7 dflash-n7" KV=q8_0 SLOTS=8 ... select rows
FP4_UNLOCK_KERNEL=0 OUT=bench/results/$(date +%F)-after-off bench/bench.sh after 27b-nvfp4 4,5
FP4_UNLOCK_INVARIANT=1 bench/invariance.sh invariant 27b-nvfp4      # the divergence table
bench/kld.py after --paths mmvq      # KL table; "mmvq" = -b 8 -ub 8, every matmul at M = 8 (needs scripts/fetch-models.sh ref)
bench/opinv/build.sh invariant && FP4_UNLOCK_INVARIANT=1 build-invariant/bin/opinv   # op-level invariance
python3 bench/final_table.py                                        # renders the results table from bench/results/2026-10-27-final-*
```

Each run writes `meta.txt` (driver, commit, clocks, model hash), `summary.csv`, per-request rows and the server
logs under `bench/results/<date>-<tag>/`. Directories named `2026-10-27-final-*` are the ones in this README;
`ab-*` are earlier A/B runs, and directories labelled `p3` came from the invariant build target before it was
renamed. The vLLM harness (`bench/vllm_compare.sh`) is context only: the DFlash2 drafter does not fit next to the
27B target in vLLM on 32 GB, so the headline row has no vLLM counterpart and no vLLM column is claimed.

## Layout

- `patches/` : the whole change as four patches against the pinned upstream commit (`patches/README.md`): the
  kernel `ggml/src/ggml-cuda/fp4-smallm.cu` and its dispatch in `mmvq.cu`, the invariant mode in `fp4-unlock.cuh`
  and `fattn*`, the loader fix in `src/llama-model.cpp`, the batch tests in `tests/test-backend-ops.cpp`.
- `scripts/` : `build.sh` (after | invariant | before | cpu | all), `fetch-llama.sh`, `fetch-models.sh`,
  `run-27b.sh`, `run-35b-a3b.sh`, `verify-lossless.sh`, `regen-patches.sh`, `sync-upstream.sh` (moves the pin to
  upstream master and re-applies the patches), `profile.sh` (nsys capture of a verify stream).
- `bench/` : the harness (`bench.sh`, `invariance.sh`, `kld.py`, `scaling.sh`, `opinv/`), prompts, and raw results.
- `docs/DESIGN.md` : why a new kernel rather than MMQ, the dispatch rule, the precision decision, the invariance
  argument. `docs/UPSTREAM.md` : upstream status and related work. `docs/PINNED_COMMIT` : the upstream commit.
- `third_party/` : not in git; created by the build script.

## License

MIT for this repository's own code. llama.cpp (fetched at build time, not redistributed here) is MIT,
Copyright (c) The ggml authors.
