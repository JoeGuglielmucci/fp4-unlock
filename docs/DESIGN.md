# Design: a batch-invariant small-M NVFP4 path for Blackwell

This note explains the two things the patches change in llama.cpp's CUDA backend and why they are shaped the way
they are. Numbers refer to Qwen3.8-27B NVFP4 on an RTX 5090 at the pinned upstream commit; the measurements
themselves are in the README and under `bench/results/`.

## The problem

llama.cpp sends a quantized `MUL_MAT` with up to 8 columns (plain decode is M = 1; a speculative verify step is
M = n_max + 1) to `MMVQ`: one int8 dp4a dot product per thread per column, weights read once per block. Its cost
grows with M. On the 27B's FFN shapes the per-step forward goes from 12.2 ms at M = 1 to 16.5 ms at M = 8, about
0.8 ms per extra token, although the weights are read exactly once either way. From M = 9 the same op goes to `MMQ`,
a tile GEMM that on Blackwell uses the block-scaled FP4 MMA or the int8 MMA and is flat in M, but routing M <= 8 to
MMQ loses at M = 4 (the tile is sized for 8 to 16 columns and the activation quantizer is an extra launch) and only
wins at M = 8. An nsys split of a verify stream puts GEMM at about 78 % of the step's GPU time (MMQ 48-52 %, MMVQ
26-30 %, of which the Q8_0 lm_head alone is 20 %), so the step cost is the GEMM cost.

Separately, three things in the forward pass depend on M in ways that change floating-point reduction order:
MMVQ switches from 4 warps to 2 warps per block between M = 4 and 5; the flash-attention kernel choice and its
stream-k / parallel-blocks split depend on the number of query columns; and on Blackwell the k-quant types move from
MMVQ to MMQ at M = 6, 7 or 8 depending on the type. The verify batch therefore does not reproduce M = 1 decode
bitwise, and greedy output changes when a drafter is on (5 or 6 of 6 prompts diverge, both KV types).

## What was built

### Option considered first: MMQ with a small tile

MMQ already picks a J = 8 tile. Routed at M = 2..8 it is flat in M but loses end to end at M = 4 (-9 % with int8
activations, -14 % with FP4 activations) and wins only at M = 8; its weight tile goes through shared memory, its
stream-k split is M-dependent (not invariant), and at M = 1 it is 2-4x slower than MMVQ. Rejected as the product.
It survives as the one-line experiment behind `FP4_UNLOCK_SMALLM` (`bench/smallm_experiment.sh`).

### The kernel (`fp4-smallm.cu`)

An MMVQ-shaped kernel that issues the Blackwell block-scaled MMA (`mma.sync m16n8k64`, e2m1 x e2m1, ue4m3
scales, fp32 accumulate):

- A block is 4 warps; each warp owns 16 output rows and walks K in 64-element steps (one m16n8k64 A fragment of
  16 rows x 64 k, which is four NVFP4 blocks of 16). A chunk of consecutive blocks for all 16 rows is staged
  through shared memory with 16-byte `cp.async` (bank-conflict-free fragment reads from a 36-word row stride) with
  several chunks in flight, because the direct-from-global version was bound by L1 transaction throughput: each
  4-byte fragment load of a warp touched 8 cache lines.
- Activations are quantized once per `MUL_MAT` by a separate kernel into the MMA's B layout, one record per 64-k
  block in fragment order with columns >= M zeroed, so the GEMM never branches on M. The quantization is a residual
  expansion y ~= y_0 + y_1 + ... with each term an NVFP4 quantization (per-column fp32 scale per 2048-k segment,
  per-16 ue4m3 scale with an offset search, e2m1 nibbles; the column scale is divided by 4 for each further term).
  `FP4_UNLOCK_T` sets the number of terms; T = 1 is plain W4A4.
- Split-K count comes from (shape, T) only; partial sums are combined by the last block in split order. The fp32
  sequence per output element is fixed by (shape, T) and does not depend on M.
- Programmatic Dependent Launch: llama.cpp launches with the PDL attribute and the activation quantizer signals
  completion early, so the GEMM calls `ggml_cuda_pdl_sync()` before its first global read and `ggml_cuda_pdl_lc()`
  after its last, as `mmvq.cu` does. The first version of the kernel omitted this and read the activation buffer
  before it was written, which showed up as results that varied run to run; `bench/opinv/mmstress.cpp` is the
  stress probe written for that bug.

Two earlier variants are kept behind `FP4_UNLOCK_KERNEL`: 2 is an int8 MMA version (e2m1 expanded to int8 through
a 16-entry table, four m16n8k16 MMAs per block) that is issue-bound at 27-28 us per FFN matmul against the FP4
MMA's 18.7-21.3 us; 3 quantizes the activations inside the GEMM block (no second launch) and is correct but slower,
because every row group re-quantizes the same activations. The quantization has to happen once per `MUL_MAT`.

### Precision

KL divergence against the Q8_0 model, every matmul at M = 8 (`-b 8 -ub 8`), kernel on: T = 1 0.0570, T = 2 0.0422,
T = 3 0.0426, T = 4 0.0426; upstream's int8 MMVQ path 0.0431, upstream MMQ with int8 activations 0.0429, with FP4
activations 0.0618. The default is the smallest T within noise of the int8 control: T = 2. T = 3 costs 0.2 ms per
step at M = 1 and buys nothing measurable. T = 1 is the labelled fast path and fails the bar, the same way
upstream's `GGML_CUDA_MMQ_PREC=q4` does.

### Dispatch rule (`mmvq.cu`)

The kernel is flat in M (18.7-21.3 us from M = 1 to 8 on both FFN shapes in isolation) but has a fixed cost that
MMVQ does not: the quantizer launch and the GEMM shell, 3.7-5.6 us at M = 1 on an L2-resident shape. The
microbenchmark overstates MMVQ's M = 3..4 cost (a 50 MB weight set fits the 5090's L2), so the rule is set from the
end-to-end per-step cost (`llama-bench -p M -n 0`): MMVQ and the kernel cross between M = 5 and 7, and the
crossover does not move with the gate/up fusion state. Default mode therefore sends dense NVFP4 `MUL_MAT` with
M <= 4 to MMVQ (so batch-1 decode is upstream's kernel, unchanged, and no batch-1 claim is made) and M = 5..8 to
the kernel; `FP4_UNLOCK_MIN_M` moves the threshold. The condition in `ggml_cuda_mul_mat_vec_q` is: type NVFP4,
`ne1` in [MIN_M, 8], a single 2D matmul (no broadcast over batch dims), not the shared-up fusion, compute capability
12.x. `MUL_MAT_ID` (MoE experts) never takes it.

Consequence for drafters: a draft length of 3 (M = 4) runs upstream's kernels and is unchanged; the gain is at
draft length 4 and above, growing with M. The per-step table in the README is the direct measurement.

## Invariance

In default mode the forward is bitwise invariant inside M = 1..4 and inside M = 5..8 but not across the boundary,
and flash attention and the k-quant MMVQ/MMQ switch are still M-dependent, so default mode is not lossless.
`FP4_UNLOCK_INVARIANT=1` makes every op that a token's logits depend on take the same kernel and the same reduction
order for M = 1..8:

- `MIN_M` becomes 1: every dense NVFP4 matmul in [1, 8] runs the kernel, whose reduction order is fixed by
  construction. M = 1 pays the kernel's fixed cost (+1.2 ms per step, -9.6 % at batch 1).
- Flash attention (`fattn.cu`, `fattn-common.cuh`, `fattn-mma-f16.cuh`): the kernel family and the stream-k /
  parallel-blocks split are pinned to the M = 1 choice for up to 8 query columns. This is a tile pin, not a change
  of kernel family: a stabilization that moves the family instead costs 50 % of decode throughput at 64K context.
  The pin itself costs 0-2 % at depth 8192 and 3.6 % (f16 KV) at depth 65536.
- MMVQ (`mmvq.cu`): under the compile flag `-DFP4_UNLOCK_INVARIANT_MMVQ` the warp count no longer changes with M,
  and every quantized non-FP4 type (the 27B's attention and GDN projections are Q5_K) stays on MMVQ through M = 8
  instead of moving to MMQ at 6, 7 or 8. The flag is compile-time because the pinned warp count costs 7-8 % on
  the verify rows for everyone when compiled in; `scripts/build.sh invariant` produces that tree as a separate
  build directory so the default build keeps upstream's MMVQ.

With all three, the op-level probe (`bench/opinv`) reports every `MUL_MAT` type and both flash-attention KV types
bitwise identical to the n = 1 result for n = 2..8, and the end-to-end check reports 0 of 6 prompts diverging for
each of four drafters on f16 and q8_0 KV. The total cost against upstream is 9.6 % at batch 1 and 2-5 % on the
drafter rows. Invariance is the opt-in, the same shape of trade upstream makes elsewhere.

## Relation to upstream work

Upstream has a lane-packed small-K NVFP4 MMVQ (#30788), NVFP4 MMQ epilogue scale fusion (#26048), batch-invariance
tests and Vulkan MMVQ selection (#28488, #28489), and the user-facing invariance issue (#25618). The answer to
"why a new path instead of fusing into MMVQ" is the FP4 MMA plus a reduction order that does not depend on M; the
int8 dp4a path cannot give either. See `docs/UPSTREAM.md`.
