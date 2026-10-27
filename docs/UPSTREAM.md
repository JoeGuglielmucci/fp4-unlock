# Upstream status

Pinned upstream commit: `docs/PINNED_COMMIT` (ggml-org/llama.cpp master 63b4b303, 2026-10-26).
Patch series: `patches/` against that commit (see `patches/README.md`). No fork or submodule is needed;
`scripts/fetch-llama.sh` clones upstream at the pin and applies the patches into a gitignored `third_party/`.

Policy (CONTRIBUTING.md / AGENTS.md, binding): every PR description, commit message, issue and review reply is
written by the owner in their own words; AI-assisted code is disclosed in the PR template; the submitter understands
every line; bug fixes carry a reproducible issue and a regression test; features are preceded by an issue to gauge
interest. Nothing in this table is opened until the owner has done that.

| PR | Topic | Files | Status |
|---|---|---|---|
| (not opened) | llama-model: skip NextN scale tensors when MTP is off (35B-A3B NVFP4-MTP fails to load without `--mtp`) | patches/0001 | ready; needs an issue with the repro (`llama-bench` on michaelw9999 HQ GGUF, pinned commit: "expected 1101, got 1079") |
| (not opened) | tests: MUL_MAT_BATCH / FA_BATCH batch-invariance cases | patches/0002 | overlaps #28488 (Ankk98, open); coordinate there rather than duplicate |
| (not opened) | CUDA: batch-invariant small-M NVFP4 GEMM on Blackwell tensor cores | patches/0004 (+0003 for the invariant mode) | design note `docs/DESIGN-draft.md`; numbers in README; open an issue first (maintainer asked #30788 why a new path instead of MMVQ fusion: the answer here is the FP4 MMA + batch invariance, see DESIGN-draft "Dispatch rule") |

Related upstream work tracked: #25618 (batch invariance, the user-facing symptom), #27329 (Qwen3.8 decode stall; not
reproduced here, `bench/stall_repro.sh`), #28488 / #28489 (invariance tests, Vulkan MMVQ selection), #30788
(lane-packed small-K NVFP4 MMVQ), #26048 (NVFP4 MMQ epilogue scale fusion), #27173 (spec draft perf), #26704 (SM120
CUTLASS MoE, stalled), #31188 (MoE gate/up + GLU fusion in MMQ), Apoze 6263445 (quantized-FA stabilization; measured
in `bench/results/2026-10-26-apoze-*`, -50 % on the n7 rows, not adopted).

Sync procedure: `scripts/sync-upstream.sh` (moves the pin to upstream master, re-applies the patches 3-way,
regenerates patches/, rebuilds p3, runs the NVFP4 tests; does not commit).
