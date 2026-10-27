# Upstream status

Pinned upstream commit: `docs/PINNED_COMMIT` (ggml-org/llama.cpp master `63b4b303`, 2026-10-26). The patch series
in `patches/` applies to that commit (see `patches/README.md`). No fork or submodule is needed:
`scripts/fetch-llama.sh` clones upstream at the pin and applies the patches into a gitignored `third_party/`, and
`scripts/sync-upstream.sh` moves the pin to current upstream master, re-applies the patches three-way, regenerates
them, rebuilds the default tree and runs the NVFP4 tests (it does not commit).

Nothing has been submitted upstream yet. llama.cpp's contribution policy applies to anything that is: an issue
before a feature PR to gauge interest, a reproducible issue and a regression test with a bug fix, AI-assisted code
disclosed in the PR template, and PR text written by the submitter.

| patch | topic | status |
|---|---|---|
| 0001 | `llama-model`: skip the NextN (MTP head) scale tensors when MTP is off, so an MTP GGUF loads without `--mtp` | ready; needs an issue with the repro (`llama-bench` on the Qwen3.6-35B-A3B NVFP4-MTP GGUF at the pinned commit: "expected 1101, got 1079") |
| 0002 | `tests`: `MUL_MAT_BATCH` / `FA_BATCH` batch-invariance cases, NVFP4 perf shapes | overlaps #28488 (open); coordinate there rather than duplicate |
| 0003 | CUDA: invariant mode, flash-attention pins | not submitted; an alternative quantized-FA stabilization patch that changes the kernel family instead of pinning the tile was measured here and halves batch-1 decode at 64K context with q8_0 KV, so the tile pin was kept |
| 0004 | CUDA: small-M NVFP4 GEMM on the Blackwell FP4 tensor cores, dispatch from MMVQ | open an issue first; a maintainer asked on #30788 why a new path instead of MMVQ fusion, and the answer is the FP4 MMA plus a batch-invariant reduction order (`docs/DESIGN.md`) |

Related upstream work: #25618 (batch invariance, the user-facing symptom), #27329 (Qwen3.8 decode stall; not
reproduced on this card, `bench/stall_repro.sh`), #28488 / #28489 (invariance tests, Vulkan MMVQ selection), #30788
(lane-packed small-K NVFP4 MMVQ), #26048 (NVFP4 MMQ epilogue scale fusion), #27173 (speculative draft performance),
#26704 (SM120 CUTLASS MoE, stalled), #31188 (MoE gate/up + GLU fusion in MMQ).
