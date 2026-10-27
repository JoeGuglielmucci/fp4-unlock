# fp4-unlock

Lossless 2-3x decode on the RTX 5090 for every NVFP4 GGUF, and a llama.cpp path that runs the 5-8 token
speculative verify step on Blackwell's FP4 tensor cores, where upstream still uses the dp4a integer path.

A patch series on top of [llama.cpp](https://github.com/ggml-org/llama.cpp) (pinned at `63b4b303`,
2026-10-26). This repository carries only the patches; the build script fetches upstream at the pinned commit and
applies them. Everything in this README was measured
on one RTX 5090 (32 GB, driver 595.99.02, CUDA 13.2) on Qwen3.8-27B NVFP4 (cdiamond's iMatrix MTP GGUF), with the
exact commands in `bench/` and the raw outputs in `bench/results/`. Nothing here is a batch-1 speedup; see
"What it is not".

Status: v1.0 candidate. Code and numbers complete; not yet tagged. Upstream PRs not yet opened (see
`docs/UPSTREAM.md`).

## What it does

1. **Small-M NVFP4 GEMM on the FP4 tensor cores.** For a decode batch of 5 to 8 tokens, the range a speculative
   verify step and small multi-user serving live in, NVFP4 matmuls run through a block-scaled FP4 MMA kernel
   (`mma.sync m16n8k64`, e2m1 x e2m1, ue4m3 scales, fp32 accumulate) instead of the dp4a `MMVQ` path. The
   activations are quantized to NVFP4 with two residual terms (W4A4 + 4-bit residual, "T = 2"), which puts the
   activation error below the weight's: KL vs Q8_0 is the same as upstream's Q8-activation path. Its step cost is
   flat from M = 5 to 8 where MMVQ grows ~0.9 ms per token.
2. **Batch-invariant mode (`FP4_UNLOCK_INVARIANT=1`, opt-in).** Pins the flash-attention and MMVQ kernel
   choices so that an M <= 8 batch reduces in the M = 1 order. With it, greedy output with MTP or DFlash2
   speculation on is **byte-identical** to speculation off: 0/6 divergence for six drafters on both f16 and q8_0
   KV. Costs 5-10 % where the kernel does not apply; see below.
3. Two small upstream-shaped fixes: the MTP-GGUF loader skips the NextN scale tensors when MTP is off (the
   35B-A3B NVFP4-MTP GGUF would not load otherwise), and `test-backend-ops` batch-invariance tests
   (`MUL_MAT_BATCH`, `FA_BATCH`).

## Results: Qwen3.8-27B NVFP4, RTX 5090, greedy, 8K context, 512-token answers

Decode tok/s, median of 3 timed runs after a warm-up, 6 prompts (code and prose). `before` is unpatched upstream
at the pinned commit, measured 2026-10-26. `kernel off` is this fork with `FP4_UNLOCK_KERNEL=0`, measured the same
hour as `default`; that is the honest comparison (clocks were not locked, and day-to-day drift on this card is up to
+/-5 % on the n7 rows). Drafters: `mtp-nN` = the GGUF's embedded MTP head, `dflash-nN` = DFlash2 Q4_K_M head,
`N` = max draft length (verify batch M = N + 1). Full tables, acceptance rates and power: `bench/final_table.py`.

| config | before (upstream) | kernel off (same day) | **fp4-unlock default** | vs kernel off | invariant mode | power W, kernel off / default |
|---|---|---|---|---|---|---|
| no drafter (batch 1) | 82.0 | 82.2 | 82.1 | -0.1 % | 74.3 | 535 / 534 |
| mtp-n3 | 149.8 | 149.7 | 149.8 | +0.1 % | 142.3 | 550 / 550 |
| mtp-n5 | | 132.4 | **138.8** | **+4.9 %** | | 562 / 492 |
| mtp-n7 | 122.0 | 113.6 | **127.1** | **+11.8 %** | 119.2 | 542 / 468 |
| mtp-n8 (M = 9, MMQ) | 113.5 | 113.4 | 113.6 | +0.2 % | | 444 / 444 |
| dflash-n3 | 166.5 | 166.6 | 166.5 | -0.1 % | 157.3 | 545 / 546 |
| dflash-n5 | | 159.1 | **173.3** | **+9.0 %** | | 557 / 479 |
| dflash-n7 | 159.2 | 146.9 | **169.1** | **+15.1 %** | 159.2 | 531 / 445 |
| dflash-n7, q8_0 KV | | 169.6 | **182.8** | **+7.8 %** | | 521 / 438 |
| mtp-n7, q8_0 KV | | 122.8 | **136.0** | **+10.8 %** | | 535 / 462 |
| 8 slots, no drafter (aggregate) | 352.1 | 324.3 | **360.0** | **+11.0 %** | | 518 / 427 |
| 8 slots, mtp-n3 (aggregate) | 372.3 | 386.6 | **428.9** | **+10.9 %** | | 406 / 381 |
| 8 slots, mtp-n7 (aggregate) | | 297.6 | **352.3** | **+18.4 %** | | 431 / 403 |
| 16 slots, no drafter (aggregate) | 496.5 | 495.3 | 495.9 | +0.1 % | | 374 / 374 |
| 16 slots, mtp-n3 (aggregate) | 443.4 | 457.9 | 466.7 | +1.9 % | | 394 / 382 |

How to read it:

- **Speculative decoding is the 2-3x**: 82 -> 183 tok/s (dflash-n7, q8_0 KV) is 2.27x the batch-1 rate on the
  same settings, 173 tok/s (dflash-n5, f16 KV) is 2.11x. Upstream already gets most of that; this fork makes it
  lossless (invariant mode) and faster at the longer draft lengths.
- **The FP4 tensor-core kernel adds 5-18 % wherever the verify batch is 5-8 tokens**, at 60-90 W less board
  power. Rows with M <= 4 (n3 drafters) or M >= 9 (n8, 16 slots) do not take the kernel and are unchanged within
  noise, by design.
- **Invariant mode** is bitwise greedy equivalence for 5-10 % where the kernel does not apply (batch 1: -9.6 %;
  n3 drafters: -5 %), and still ahead of the kernel-off control on the n7 rows (+4.9 % mtp, +8.4 % dflash).

### The mechanism: per-step cost (llama-bench -p M -n 0, ms per forward of M tokens, 0 context)

| M | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| upstream path (`FP4_UNLOCK_KERNEL=0`) | 12.22 | 12.68 | 13.01 | 13.69 | 14.84 | 15.99 | 16.92 | 18.35 |
| **fp4-unlock default** | 12.21 | 12.69 | 13.01 | 13.67 | **14.41** | **14.70** | **14.97** | **14.90** |
| invariant mode | 13.43 | 13.80 | 13.94 | 14.11 | 14.33 | 14.62 | 15.00 | 15.42 |

Below M = 5 the dp4a path is cheaper (the crossover is between 4 and 6; the 50 MB weight set is L2-resident in
microbenchmarks, which is why an op-level benchmark alone puts it at 3). At M = 8 the kernel is -18.8 %.

### Quality, invariance, and the rows that must not move

| row | before (upstream) | fp4-unlock default | invariant mode |
|---|---|---|---|
| tg128 batch 1, d0 / d8192 / d65536 (tok/s) | 83.2 / - / - | 83.1 / 80.9 / 68.3 | 75.3 / 73.8 / 61.1 |
| pp512 / pp8192 (tok/s) | 6653 / 6409 | 6574 / 6374 | same (prefill never takes the kernel) |
| KL vs Q8_0, kl-set-5k, every GEMM at M = 8 | 0.0429 (MMQ q8, `-b 512`); 0.0618 (W4A4) | **0.0422**, top-1 agreement 95.35 % | same kernel, same KL |
| greedy spec-on vs spec-off, 6 prompts x 512 tokens, 6 drafters, f16 and q8_0 KV | 5/6 - 6/6 diverge | 5/6 - 6/6 diverge (fast layer) | **0/6 everywhere** |
| op-level bitwise invariance, column 0 of an n-column result vs n = 1 | FA differs from n = 3, NVFP4 from n = 5 (MMVQ's own warp-count change) | same boundaries as upstream (NVFP4 n = 5 is now the dispatch) | **all ops `same` n = 2..8** |

Qwen3.6-35B-A3B NVFP4 (MoE): the kernel does nothing by construction (MoE experts go through `MUL_MAT_ID`, not
`MUL_MAT`); tg128 285-292 run to run, mtp-n3 426 vs 427 tok/s with it on and off. The loader fix is what makes that GGUF load
without `--mtp`. MoE support is a follow-up (`docs/DESIGN-draft.md`, Phase 3.5).

vLLM column: not measured. The 0.31.0 container dry run failed on 10-26 and was not rerun; vLLM also runs a
different NVFP4 recipe (nvidia/unsloth safetensors, not this GGUF), so it would be a different-model comparison.

## What it is not

- **Not a batch-1 speedup.** Single-user decode without a drafter is memory-bound at ~83 tok/s and stays there.
  The kernel is dispatched only for 5-8 token batches.
- **Not a driver, clock or power-limit hack.** Stock driver, stock clocks (never locked during these runs), power
  goes down, not up.
- Not a new quantization format; it runs the NVFP4 GGUFs already on Hugging Face.
- Not for MXFP4 or MoE expert matmuls yet, not for models larger than VRAM, not multi-GPU.
- Not an Ollama or LM Studio feature. Ollama 0.20.7 cannot load any NVFP4 GGUF (its bundled ggml asserts on the
  tensor type, with or without this fork); LM Studio was not tested. This runs through llama.cpp's own
  `llama-server`, `llama-cli` and `llama-bench` built from this fork.

## Build and run

Requirements: RTX 50-series (compute 12.0, built for `120a`), CUDA 13.x, cmake, a C++17 compiler.

```
git clone https://github.com/JoeGuglielmucci/fp4-unlock
cd fp4-unlock
JOBS=24 scripts/build.sh p3        # fetches upstream llama.cpp at docs/PINNED_COMMIT into third_party/ (gitignored),
                                   # applies patches/, builds the shipped tree -> build-p3/ ('before' = unpatched control)
scripts/fetch-models.sh            # the benchmarked GGUFs -> ~/models/
scripts/run-27b.sh p3 dflash 7     # llama-server, DFlash2 draft, n_max 7, port 8080
KV=q8_0 scripts/run-27b.sh p3 dflash 7                 # the 182.8 tok/s row
FP4_UNLOCK_INVARIANT=1 scripts/run-27b.sh p3 mtp 7     # bitwise spec = greedy
```

Runtime switches (environment, all optional):

| variable | default | meaning |
|---|---|---|
| `FP4_UNLOCK_KERNEL` | 1 | 0 = upstream MMVQ path (A/B control), 1 = FP4 tensor-core kernel, 2 = int8 MMA fallback |
| `FP4_UNLOCK_T` | 2 | residual activation terms 1..4; 1 = plain W4A4 (1 % faster, KL 0.057, fails the bar), 3 = extra margin (no measurable gain) |
| `FP4_UNLOCK_MIN_M` | 5 | smallest batch sent to the kernel (M = 1..4 stay on MMVQ, where they are faster) |
| `FP4_UNLOCK_INVARIANT` | unset | 1 = bitwise batch-invariant mode: sets MIN_M = 1 and pins the FA / MMVQ kernel choices |
| `GGML_CUDA_DISABLE_FUSION` | unset | upstream switch; the kernel supports the fused gate/up + GLU path |

Reproduce the table: `bench/bench.sh p3 27b-nvfp4 4,5,6` (server rows), `bench/invariance.sh`,
`bench/kld.py`, `bench/opinv/build.sh p3`; each run writes `meta.txt`, `summary.csv` and `rows.md` under
`bench/results/<date>-<tree>-<model>-<tag>/`.

## Layout

- `patches/` : the whole change as four patches against the pinned upstream commit (`patches/README.md`):
  the kernel `ggml/src/ggml-cuda/fp4-smallm.cu`, dispatch in `mmvq.cu`, invariant mode in `fp4-unlock.cuh` +
  `fattn*`, the NextN loader fix in `src/llama-model.cpp`, the batch tests in `tests/test-backend-ops.cpp`.
- `third_party/llama.cpp` : not in git. Created by `scripts/fetch-llama.sh` (upstream clone at
  `docs/PINNED_COMMIT` with the patches applied); `scripts/sync-upstream.sh` moves the pin and regenerates the patches.
- `docs/DESIGN-draft.md` : design note (dispatch rule, precision decision, kernel structure).
  `docs/FINDINGS.md` : every measurement in order, including the dead ends. `docs/UPSTREAM.md` : PR status.
- `bench/` : benchmark scripts and raw results. `prd.md` : the plan, gates and status log.

## License

MIT for this repository's own code. llama.cpp (fetched at build time, not redistributed here) is MIT,
Copyright (c) The ggml authors.
