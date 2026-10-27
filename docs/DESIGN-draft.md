# DESIGN (draft material, PRD 3.0) — small-M NVFP4/MXFP4 MUL_MAT on Blackwell

Status: material supplied by Claude for the owner to rewrite in their own words (PRD 3.0, K5). Not for posting as is.
Numbers come from docs/FINDINGS.md (chain3 profile split, chain7 Phase 2 validation, chain8/11 kernel v1/v2, the Phase 3
kernel v3/v4/v5 sections, the crossover section, chain13 v4 validation).

## Problem

llama.cpp dispatches a quantized MUL_MAT with ne11 <= 8 columns (decode M = 1, speculative verify M = n_max + 1) to
MMVQ: one int8 dp4a dot product per thread per column, weights read once per block, `ncols_dst` accumulators. The
kernel's cost grows with M (L2-resident FFN shape 17408x5120 NVFP4: 15.2 us at M = 1, 23.1 at M = 4, 39.6 at M = 8).
At M >= 9 the same op goes to MMQ, a tile GEMM that on Blackwell uses the block-scaled FP4 MMA (W4A4) or int8 MMA
(W4A8) and is flat in M (17.5 us from M = 2 to 8 at J = 8), but routing M <= 8 to MMQ loses end to end on the M = 4
verify batch (-9 % mtp-n3 W4A8, -14 % W4A4) and wins on M = 8 (+15 % / +24 % dflash-n7). The nsys split (FINDINGS "chain3 outputs") puts
GEMM at ~78 % of the verify-step GPU time (mmq 48-52 %, mmvq 26-30 %, of which the Q8_0 lm_head alone is 20 %), so the
M = 4 loss is the MMQ tile cost itself (J = 16 tile for 4 columns, plus the 4.3 us W4A4 quantizer) and not a side effect.

Also: the MMVQ kernel's reduction order depends on M (4 warps for ncols 1..4, 2 for 5..8), and FA's kernel choice
and stream-k split depend on M, so the verify batch is not bitwise the M = 1 decode (Phase 1.8: 5-6/6 prompts diverge).
Phase 2 pins those (fork `FP4_UNLOCK_INVARIANT`, opt-in). chain7: op level all bitwise same for n = 2..8; end to end
M = 4 is 0/6 divergent for f16 and q8_0 KV, M = 8 stayed 6/6 because Blackwell moves Q4_K/Q5_K/Q6_K from MMVQ to MMQ at
ne11 = 6/7/8 (the 27B's attention and GDN projections are Q5_K); invariant mode now keeps every quantized non-FP4 type
on MMVQ through ne11 = 8. Cost: 0-2 % tg128 at depth <= 8192, but -3.6 % (f16) / -19 % (q8_0 KV) at depth 65536 from
the FA pin, so the switch stays opt-in (FINDINGS "chain7 outputs").

## Evidence (27B NVFP4, RTX 5090, driver 595.99.02, llama.cpp 63b4b303)

- Roofline: 9.57 ms floor per decode step (weights + KV at 1.79 TB/s); measured step(1) 12.07 ms (79 %).
- batched-bench step ratios: M2 1.049, M4 1.183, M8 1.472, M16 1.852 (K1 not triggered: 1.47 > 1.3).
- Phase 1.5 kernel table and end-to-end columns: docs/FINDINGS.md "Phase 1.5".
- Phase 1.3 KL vs Q8_0: mmvq 0.0431, mmq-q8 0.0429, mmq-q4 0.0618 (W4A8 is the lossless bar; W4A4 is a labelled fast path).
- Profile split (synth8 verify stream, before vs SMALLM=1 W4A8): GPU time 791.7 vs 821.8 ms; mmq 47.7 vs 52.2 %, mmvq 29.8
  vs 25.7 % (Q8_0 lm_head 20.4 %), quantize 0.7 vs 0.4 %, flash-attn 1.1 %, gdn 3 %, norm/rope/other 12-13 %. GEMM ~78 %.
- Apoze's FA-only patch with q8_0 KV: M = 4 runs bitwise identical, M = 8 runs 6/6 divergent at drafter-independent token
  positions (the MMVQ 4-to-2 warp switch), and -50 % tg128 at depth 65536. Pin the tile, do not change the family.

## Proposal

One kernel for M in [1, 8], NVFP4 and MXFP4, Blackwell only (cc 12.x), dispatched in `ggml_cuda_mul_mat` ahead of
MMVQ when `type in {NVFP4, MXFP4} && ne11 <= 8 && blackwell`, behind a compile flag `GGML_CUDA_FP4_SMALLM` with an
env override to force the old path for A/B. Properties it must have, in priority order:

1. Bitwise batch-invariant: the k reduction order per output element does not depend on M. M = 1 goes through the
   same kernel, so this removes the MMVQ 4/2-warp switch rather than adding a new one. Delivered in full only in
   `FP4_UNLOCK_INVARIANT=1` mode (see the dispatch rule below); the default mode keeps upstream's M = 1 kernel.
2. M = 1 within 2 % of MMVQ (decode is the product; the kernel may not tax it). Met by the dispatch rule below
   (M <= 2 stays on MMVQ) rather than by the kernel: v1-v5 all tax M = 1 by 8-19 %.
3. Flat in M up to 8, so the verify step at M = 8 costs close to the M = 1 step (weights read once, tensor-core compute).
4. W4A8 (lossless column, default) and W4A4 (fast column, opt-in via `GGML_CUDA_MMQ_PREC=q4`) from the same source.

### Option (a): MMQ with a J <= 8 tile

Already exists (mmq.cu picks J from 8). Measured in Phase 1.5: flat in M but loses end to end at M = 4. Its weight
tile (I = 64..128 rows x K) goes through shared memory, its stream-k split is M-dependent (not invariant), and at
M = 1 it is 2-4x slower than MMVQ on the L2-resident shape. Rejected as the product unless the profile split says the
GEMMs are not where the M = 4 loss is.

### Option (b): MMVQ-shaped kernel issuing the MMA (preferred)

- Block = 4 warps, each warp owns 16 output rows (block 64 rows); grid = nrows / 64. Same row-per-thread streaming
  structure as MMVQ: each warp walks K in steps of 64 (= 4 NVFP4 blocks of 16, = one m16n8k64 A fragment of 16 rows
  x 64 k, 512 bytes of weights + 16 B of UE4M3 scales per row group).
- Activations quantized once per step to the MMA's B layout: W4A4 uses the existing `quantize_mmq_nvfp4` output
  (block_fp4_mmq, per-16 UE4M3 scales + per-column fp32 scale); W4A8 uses block_q8_1 (per-32 fp16 scale) as MMVQ does.
- Compute per k-step: W4A4 = one `mma_block_scaled_fp4` (m16n8k64, e2m1 x e2m1, ue4m3 scales, fp32 accumulate) with
  M padded to 8 in the instruction's N dimension (columns >= M are zero and discarded). W4A8 = four int8 m16n8k16 MMAs
  (one per 16-wide NVFP4 block, weights expanded from E2M1 to int8 by a 16-entry table, the way vecdotq.cuh does it)
  each into a fresh int32 tile, then fp32 FMA with scale_w[row, block] x scale_a[col, block]. The k reduction order is
  fixed by the kernel, independent of M, satisfying property 1 by construction.
- Memory traffic per 64-row block: identical to MMVQ (each weight byte read once); no shared memory for weights; the
  B fragments (activations, <= 8 cols x K) are read from L2 by every block, as MMVQ reads y.
- Output: fp32 tile 16 x 8 per warp, written directly (no fixup kernel; no split-K, so no M-dependent combine).
- Register budget: A fragment 4 regs, B 2 regs per k-step, D 4 regs, plus scales; double-buffer the A loads.
- Compare with upstream #30788 (lane-packed MMVQ MMA) before writing: if it already covers this, build on it with credit.
- Programmatic Dependent Launch: llama.cpp launches CUDA kernels with the PDL attribute and `quantize_q8_1` signals
  completion early (`ggml_cuda_pdl_lc`), so any consumer of the q8_1 pool buffer must call `ggml_cuda_pdl_sync()`
  before its first global read and `ggml_cuda_pdl_lc()` after its last, as mmvq.cu does. The first build of the
  kernel omitted this and read the activation buffer before it was written (wrong results that varied run to run;
  FINDINGS "Phase 3 kernel: the PDL bug").
- Row tiling as built: 16 rows x 8 columns per warp, K in 64-element blocks interleaved across warps, fresh int32 D
  per k16 MMA, `acc += (w_scale * a_scale) * float(D)`, fixed-order cross-warp reduction through shared memory;
  nwarps 16/8/4 chosen by row count. Env `FP4_UNLOCK_KERNEL=0` forces the MMVQ path for A/B.
- Memory-level parallelism (v2): the first build consumed each 64-wide block right after loading it and ran at
  2.0 TB/s from L2 (M = 1: +58 % / +91 % over MMVQ, -19 % tg128). v2 issues the loads of `FP4_SMALLM_PF` = 4
  blocks per warp before consuming them in the same k order, and allows two blocks per SM at nwarps <= 8; the
  float accumulation sequence is unchanged, so v2 is bitwise v1 (checked by chain11 against chain8's generations).
- v3 (shared-memory staging): v1/v2 were bound by L1 transaction throughput (each 4-byte fragment load of a warp
  touches 8 rows = 8 lines, ~80 wavefronts per 576 B of weights; MMVQ reads 3 lines per instruction). v3 copies a
  chunk of 4 consecutive blocks of all 16 rows into shared memory with 16-byte cp.async (bank-conflict-free fragment
  reads from a 36-word row stride), `FP4_SMALLM_STAGES` chunks in flight. Still 27-28 us at M = 1: the int8 path
  (E2M1-to-int8 expansion, four m16n8k16 MMAs per block and term, scale FMAs) is issue bound; the no-compute copy floor
  is 15.6 us. Kept as `FP4_UNLOCK_KERNEL=2` (and the fallback without the block-scaled MMA).
- v4 (FP4 tensor core, the product): the Blackwell block-scaled MMA (m16n8k64, e2m1 x e2m1, ue4m3 scales, fp32
  accumulate) consumes the raw NVFP4 weight words from the cp.async stage. Activations go through a residual
  expansion y ~= y_0 + y_1 + ... with each y_t an NVFP4 quantization (per-column fp32 scale per 2048-k segment, per-16
  UE4M3 scale with the offset search, E2M1 nibbles; column scale / 4 per further term): `FP4_UNLOCK_T` = 1 is W4A4,
  T = 2 or 3 put the activation error under the weight's (KL vs Q8_0 on the 27B: T=1 0.0570, T=2 0.0422, T=3 0.0426,
  T=4 0.0426, MMVQ 0.0432). A separate 128-thread-per-(segment, column) quantizer kernel writes one record per 64-k
  block in B-fragment order with columns >= M zeroed, so the GEMM never branches on M. GEMM block = 4 warps x 16 rows
  with all warps on the same k split; split-K count from (shape, T) only, partials combined by the last block in split
  order: the fp32 sequence per output is fixed by (shape, T). Flat in M (18.7-21.3 us from M = 1 to 8 on both FFN
  shapes); at M = 1 still 3.7-5.6 us behind MMVQ, which the ablations assign to fixed cost (quantizer launch 1.4-4.4
  us, GEMM shell 3.6 us) and not to streaming (the weight copies are 10-11 of the 18.7 us; B loads and MMAs are free).
  End to end that is -8 % (T=1) to -11 % (T=3) tg128 when every M goes through v4.
- v5 (quantization inside the GEMM block, no second launch, no activation record): correct but slower (19.5 at T=1,
  41 us at T=3 at M = 1): every row group re-quantizes the same activations and the per-group chain is latency bound.
  Rejected; kept as `FP4_UNLOCK_KERNEL=3` for A/B. The quantization must happen once per MUL_MAT.
- Dispatch rule (the resolution of properties 1 and 2): the microbench has MMVQ ahead at M = 1 and 2 (15.0/13.4 and
  15.8/14.8 vs 20.6/20.5 us) and behind from M = 3 (19.2/27.9 vs 20.7/20.6; 2-3x at M = 8), but the microbench is
  L2-resident (a 50 MB weight fits the 5090's L2) and overstates MMVQ's M = 3..4 cost. The rule is therefore set from
  the end-to-end verify-step cost (`llama-bench -p M -n 0`, 27B, ms per forward of M tokens): MMVQ 12.21 / 12.68 /
  12.98 / 13.67 / 15.83 / 18.30 at M = 1 / 2 / 3 / 4 / 6 / 8 against v4 (T = 2) 13.43 / 13.80 / 13.94 / 14.10 /
  14.61 / 15.41. v4 is flat in M, MMVQ grows with M, and they cross between M = 4 and 6; the fusion state does not
  move the crossover (same gap with `GGML_CUDA_DISABLE_FUSION=1`). Default mode sends dense NVFP4 with M <= 4 to MMVQ
  (batch-1 decode is upstream's kernel, unchanged, so property 2 holds by construction and no batch-1 speedup is
  claimed) and M = 5..8 to v4 (`ggml_cuda_fp4_smallm_min_m()`, env `FP4_UNLOCK_MIN_M`, default 5). The default mode
  is therefore bitwise invariant inside [1, 4] and inside [5, 8] but not across the boundary;
  `FP4_UNLOCK_INVARIANT=1` sets min_m = 1 so every M in [1, 8] runs v4 and the verify batch reproduces decode
  bitwise, at the measured M = 1 cost (+1.2 ms per step, −9 % tg128). Same trade shape as the FA tile pin at 64k
  context: invariance is the opt-in. Consequence for drafters: an n = 3 drafter (M = 4 verify) runs upstream's
  kernels and is unchanged; the gain is for n >= 4 (M >= 5), where the verify step was growing linearly with M.

### Precision decision (PRD 3.6)

KL vs Q8_0 on the 5k set, decode through the kernel at M = 8 (`-b 8 -ub 8`, kernel on, INV=1): the default T is the
smallest T within noise of the MMVQ control (0.0432): T = 2 (0.0422) and T = 3 (0.0426) both qualify, T = 1 (0.0570)
does not and is the labelled W4A4 fast path (`FP4_UNLOCK_T=1`). T = 2 vs T = 3 was decided by the per-step rows
(INV=1, M = 1..8, clean of acceptance noise): T = 2 is 0.2 ms cheaper at M = 1 and equal from M = 3, and the server
rows (chain14) have T = 2 never behind T = 3. **Default T = 2**; `FP4_UNLOCK_T=3` is the extra-margin setting.
Reported as "FP4 tensor cores at M = 5..8: N % faster on the verify step, KL unchanged".

## Expected gain

From Phase 1.5: the M = 8 FFN GEMM cost can drop from 2.6x the M = 1 cost (MMVQ) to ~1.15x (MMQ-flat). What that is
worth end to end depends on the GEMM share of the verify step, measured at ~78 % (profile split). Upper bound seen so far:
dflash-n7 159 -> 198 tok/s with the (lossy) W4A4 MMQ route.

## Maintainer

The owner, named in the PR. AI disclosure per the llama.cpp policy; the owner can state this design and its trade-offs
unaided before kernel code is written (K5).
