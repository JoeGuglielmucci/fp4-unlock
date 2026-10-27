# PRD: Blackwell NVFP4 Speculative-Decode Path for llama.cpp

Status: Draft v3, 2026-10-26. Owner: Joe Guglielmucci (gugg@tutamail.com).
Purpose of this file: single source of truth for the project. Re-read it after any context compaction before doing work. Update the Status Log at the bottom whenever a step lands.

Changes in v3 (all from desk research done 2026-10-26, see Section 3.5):
- The "21.7% MTP acceptance" figure could not be sourced and contradicts every acceptance number in the #25618 thread (0.68-0.94) and cdiamond's own 2.81x MTP speedup. It is removed as a baseline. Acceptance is measured in Phase 1, not assumed.
- #25618 (greedy divergence) is a batch-invariance problem, not a sampling bug: N=1 and N>1 go through different kernels / reduction orders. On CUDA the identified culprit is flash-attention partitioning with quantized KV; a contributor (Apoze) has a 3-file candidate fix; Ankk98 has an open test PR (#28488) and a Vulkan fix (#28489). PR 1 is re-scoped accordingly.
- #27329 (decode hang) reproduces on Ada, Vulkan and SYCL with Q4_K_XL too. It is a Qwen3.8 architecture (Gated DeltaNet) stall, not Blackwell/NVFP4, is labelled stale, and cdiamond reports no hangs on b10454. PR 2 is conditional on Phase 1 reproducing it.
- llama.cpp's CONTRIBUTING.md / AGENTS.md AI policy is now built into the plan: the owner writes every GitHub post, owns the design, and must be able to explain every line; Claude's upstream role is exploration, measurement, suggestions and mechanical work. Private forks are exempt.
- Native Blackwell NVFP4 tensor-core MMQ has been in master since #22196 merged 2026-04-28 and is the default for batch > 8. The unserved range is exactly batch 2-8. A new small-M kernel is itself a batch-dependent numerics switch, so it must be designed batch-invariant (same kernel for M=1..8) or it recreates #25618.
- Toolchain gaps found on this machine: nsys absent, ncu is 2022.4 (too old for sm_120), clang-format absent, local vLLM image is a custom 0.11.2-dev build, not 0.31.
- Benchmark hygiene, prompt set, attribution experiment, gates and kill criteria carried over from v2.

---

## 1. Thesis and decision

**Prediction under test:** "The 5090's next unlock won't come from a driver hack. It'll come from software that finally uses its FP4 tensor cores."

**Research verdict (2026-10-26):** The prediction holds for models that fit in 32 GB of VRAM and fails for the Strata / Project Maya offload class, which is PCIe-bound. On a single RTX 5090, batch-1 decode is memory-bandwidth-bound (~1.79 TB/s) and flat at ~74 tok/s regardless of weight format. FP4 tensor cores only pay when the engine processes a batch: prefill, multi-slot serving, or the verify step of speculative decoding (MTP, DFlash2, DSpark). llama.cpp already uses them for batch > 8 (since April 2026). It does not use them for batch 2-8, which is exactly the MTP verify width (n_max 1-7) and small multi-user serving, and its speculative decoding on quantized targets is not batch-invariant, which makes greedy output change when speculation is on.

**Headline (decided 2026-10-26, two layers):**
- User layer, the hook, true from v0.1: "Lossless 2-3x decode on the RTX 5090 for every NVFP4 GGUF." Greedy output with speculation on is byte-identical to speculation off; the number is measured in Phase 2 and quoted as measured. Survives a kill at G1.
- Thesis layer, the mechanism, only after G3 passes: "The speculative verify step now runs on Blackwell's FP4 tensor cores; single-user decode is unchanged because it was never compute-bound." This is the proof of the prediction and goes in the second paragraph of the write-up and the README design section, never in the title.
- Never claim: "llama.cpp wasn't using FP4 tensor cores" (it has, for batch > 8, since April 2026) or any batch-1 speedup.

**Decision:** Build a Blackwell NVFP4 speculative-decode path for llama.cpp: fix batch invariance for the verify step, then add a batch-invariant tensor-core path for M in [1, 8] on NVFP4/MXFP4. Ship as an independently named repo that vendors llama.cpp, plus upstream PRs where the maintainers want them. This is an engine capability, not a model recipe: it applies to every current and future NVFP4/MXFP4 GGUF on every RTX 50 card.

**Why not the alternatives (decided, do not relitigate):**
- Strata issue #3023 (type-40 EXPDEC coverage, w4a4x2 prefill): 17-21% on NVFP4 Strata, but i-quants still win (IQ3_S 228 tok/s vs FP4-Local 191), does not validate the thesis, and Strata mass-closed ~1,300 issues on 2026-10-19.
- Project Maya NVFP4 experts: PCIe-bound, NVFP4 expert blobs are larger than i-quant blobs, no user demand in issues.
- HyperQwen NVFP4 lane: a two-week vLLM recipe change, not a software upgrade; vLLM b12x on a single 5090 still OOMs / CUDA-graph faults at 256K (#60069).
- From-scratch engine: imp, MegaCapybara, Infernix, NInfer have better 5090 numbers than llama.cpp and sit at 48-150 stars each. No reach.

---

## 2. Goals, non-goals, success metrics

**Goals**
1. Speculative decoding (MTP and DFlash2) on NVFP4 GGUFs is batch-invariant on CUDA: greedy output with speculation on equals greedy output with speculation off, so the speedup is lossless and can be advertised as such.
2. NVFP4/MXFP4 matmuls for batch 1-8 on Blackwell run on a tensor-core MMA through one kernel for the whole range, so the path is batch-invariant by construction. (Batch > 8 already uses native W4A4 FP4 MMA on master; not redone.)
3. A reproducible benchmark table, llama.cpp before/after vs vLLM, on one RTX 5090, with acceptance rates measured, not quoted.
4. Code lands upstream in llama.cpp in small PRs where maintainers accept it; the fork carries everything regardless.

**Non-goals (v1)**
- Windows installer or GUI launcher.
- Expert offload / models larger than VRAM.
- New quantization formats or an NVFP4 ftype in llama-quantize.
- Multi-GPU.
- Any model-specific tuning. Nothing in the headline claim names one model.
- Attention, KV-cache, Gated DeltaNet or sampler performance work. If profiling shows those dominate, that is a finding to publish, not scope to absorb. (The batch-invariance fix touches flash-attention kernel selection, which is correctness, not performance.)

**Success metrics (single RTX 5090, dense 27B unless noted)**

| Metric | Baseline | Target | Measured in |
|---|---|---|---|
| MTP acceptance (n_max 3 and 7), code / prose | measure in Phase 1 (thread reports 0.68-0.94 on IQ quants; NVFP4 unknown) | within 5 points of the Q8_0-target acceptance for the same drafter, i.e. no NVFP4-specific loss | Phase 1 / 2 |
| DFlash2 mean acceptance length (n_max 7) | z-lab reports 5.1-5.4 at temp 1.0 on Q4_K_M target | >= 5.0 on the NVFP4 target, greedy | Phase 1 / 2 |
| Spec-decode tok/s, code, 8K ctx | measure in Phase 1 (cdiamond: 2.81x over target-only on a 24 GB card) | >= 200 (MTP), >= 250 (DFlash2) | Phase 2 / 3 |
| Batch invariance, v0.1 | greedy spec-on != spec-off (reproduced in thread on CUDA sm_120) | spec-on == spec-off byte-for-byte on the prompt set, both drafters, both KV types; regression test in test-backend-ops | Phase 2 |
| Decode hang #27329 | unknown on current master | reproduced and fixed, or shown not to reproduce with the commit range recorded | Phase 1 / 2 |
| Batch 1-8 NVFP4 matmul path | dp4a (MMVQ) | one tensor-core kernel for M in [1,8], passes test-backend-ops incl. a batch-invariance case, faster than dp4a at M=4 and M=8 in test-backend-ops perf | Phase 3 |
| M=1 decode tg128 | measure in Phase 1 (~74 tok/s expected) | not slower than baseline (within 2%) with the new path on | Phase 3 |
| 8-slot aggregate tok/s, NVFP4 27B | measure in Phase 1 | >= 1.5x Phase 1 baseline | Phase 3 |
| Quality, v1.0 default path | n/a | KL vs Q8_0 within noise of dp4a path; greedy spec-on == spec-off still holds | Phase 3 |
| Quality, W4A4 path (if not default) | n/a | KL vs Q8_0 reported; default is the better-KL path | Phase 3 |
| Upstream | 0 PRs | each applicable PR opened by the owner; >= 1 merged | Phase 4 |

---

## 3. Research summary (evidence base)

### 3.1 Demand (Hugging Face, 30-day downloads, 2026-10-26)
- llama.cpp-specific NVFP4 GGUFs: cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF 2.73M; michaelw9999/Qwen3.6-35B-A3B-NVFP4-MTP-GGUF 188k; gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 284k.
- NVFP4 safetensors (vLLM/SGLang users, NOT llama.cpp): nvidia/Qwen3.6-35B-A3B-NVFP4 4.14M (39.9M all-time); unsloth/Qwen3.8-27B-NVFP4 1.51M; RadixArk 983k; nvidia/Qwen3.8-27B-NVFP4 405k. Quote these only as format momentum, never as llama.cpp demand.
- DFlash2 drafters: z-lab/Qwen3.8-27B-DFlash2-GGUF 359k; incoai/Qwen3.8-27B-DFlash2 367k.
- Qwen3.8-Flash-Next family (Strata class): ISTA-DASLab GSQ-RCO GGUF 7.09M; Qwen base 3.16M; Strata packs 260k-395k each; NVFP4 variants 370k-489k. GGUF i-quants dominate.
- Maya: peasantsmith GLM-5.3-Flash-Maya-GGUF 353k; DeepSeek-V4.1-Flash-Maya 98k.

### 3.2 GitHub (stars, 2026-10-26)
ollama 185k; ComfyUI 143k; llama.cpp 132k; vLLM 95.6k; unsloth 78.6k; SGLang 38k; Strata 33,590 (created 2026-09-24); ktransformers 19.6k; TensorRT-LLM 15k; Model-Optimizer 5.4k; nunchaku 3,961; ik_llama.cpp 3.3k; HyperQwen 2,244; project-maya 1,973 (created 2026-10-06); exllamav3 1.75k; vLLM-Moet 550; PolyStrata 537; Xolotl 365; devnen/qwen3.6-windows-server 231; strata-nvfp4 197; imp 150; strata-mlx 104; flash-attention-fp4 69; MegaCapybara 48; lna-lab/blackwell-geforce-nvfp4-gemm 27.

### 3.3 Measured 5090 numbers (Qwen3.8-27B unless noted)
| Engine | 1 user tok/s | Aggregate tok/s | Source |
|---|---|---|---|
| imp (NVFP4, sm_120a only) | 91-94 | 1,964-1,977 @32 | kekzl/imp README |
| vLLM 0.31.0 NVFP4 | 69-70 | 1,355-1,449 @32 | same |
| llama.cpp Q4_K_M | 62-63 | 162-167 @32 | same |
| MegaCapybara MXFP4/MXFP6 + DFlash2, code | 313-540 | 1,254-1,890 @8 | README |
| vLLM NVFP4 + DFlash n=6, code 8K | 304 (221 @60K) | 579 @16 | community |
| vLLM NVFP4 + MTP n=3 | 123 @8K, 90 @60K | | community |
| HyperQwen int4 Marlin + DFlash2 | 286-483; 743 w/ 15 draft tokens | | ~/hyperqwen-fidelity/NOTES.md |
| llama.cpp NVFP4 b8967 no spec | tg128 73.6, pp512 5,547 (+68% from native FP4 MMQ) | | PR benchmarks |
| cdiamond NVFP4-MTP, 24 GB card, 256K ctx | target-only 21.2; MTP n_max 8: 59.5 (2.81x); clean b10454 45.4, with 6 perf PRs 55.4 | | model card |
| michaelw9999 35B-A3B NVFP4-MTP (card, GPU unstated) | tg128 271 (TURBO) / 270 (HQ) / 222 (NVIDIA NVFP4); pp512 11.6k / 10.9k / 10.4k | | model card |

Removed from v2: "llama.cpp NVFP4 + MTP 70.5 tok/s, 21.7% acceptance". Source not recoverable; contradicted by the issue thread. Do not quote it.

Quality: NVFP4 PTQ vs Q4_K_M: q_avg 92.5 vs 93.2, HumanEval 89.6 vs 94.5. FP4-Local QAD GPQA 91.5 vs BF16 92.2. HyperQwen int4 KLD 0.0324 vs EXL3 K4 0.0103 vs FP8 0.0052. DFlash2 acceptance length on Q4_K_M target: BF16 drafter 5.28, Q8_0 5.13, Q4_K_M 5.39 (temp 1.0, GSM8K).

Note on the 12x aggregate gap (llama.cpp 162 vs imp 1,964 @32): at 32 slots llama.cpp is already on MMQ with FP4 tensor cores, so most of that gap is attention, KV layout, scheduling and server overhead, not the matmul. Do not promise to close it. The project's claim is about M in [1,8] and lossless speculation.

### 3.4 Why Strata/Maya are out (from Strata docs/NVFP4.md and strata-nvfp4 README)
- NVFP4 expert blob 2.76 MB vs IQ2_XS 1.51 MB: half the cache hits, double the PCIe bytes per miss.
- Strata w4a4 FP4-MMA prefill: KL 0.0080 vs w4a8 int8 0.0018 (4.5x worse), so int8 stays default.
- strata-nvfp4 on 5090 + 128 GB: IQ2_XS 265, IQ3_S 228, FP4-Local QAD 191, NVFP4 GPTQ 177, NVIDIA NVFP4 167 tok/s.
- Maya v1.1.28: no NVFP4 weight MMA anywhere (only DeepSeek FP4 KV helpers).

### 3.5 Desk research closed on 2026-10-26 (was "check in Phase 0/1" in v2)
- cdiamond GGUF: runs on clean upstream b10454 (commit 4df29be4f) with no patches. The six PRs in recipe/llama.cpp-patches.md (#26001 closed, #26048 open, #26705 merged, #27173 open, #25635 merged, #24891 open) are optional perf patches for the author's benchmark, not loaders. MTP head is NVFP4 (requantizing it to Q5_K cost 3% speed). Card reports no hangs. Card acknowledges #25618 and says n_max 8 is not distribution-preserving; use n_max 1 if equivalence matters. Recommended max-throughput flags: `--spec-type draft-mtp --spec-default --spec-draft-n-max 8 --spec-draft-n-min 0 --spec-draft-p-min 0 --spec-draft-type-k f16 --spec-draft-type-v f16 --spec-draft-threads 8 --spec-draft-threads-batch 8 --spec-draft-backend-sampling`, env `LLAMA_SPEC_CHAIN=1 GGML_CUDA_GRAPH_OPT=1`.
- michaelw9999 35B-A3B: two files, TURBO 18.56 GiB and HQ 18.64 GiB, "experimental", no patches listed, no MTP flags documented, no acceptance numbers. Loads via `-hf michaelw9999/Qwen3.6-35B-A3B-NVFP4-MTP-GGUF:NVFP4`.
- DFlash2 drafter: z-lab/Qwen3.8-27B-DFlash2-GGUF (Q4_K_M 1.14 GB, Q8_0 2.06 GB, BF16 3.86 GB). Support merged as #27342 on 2026-08-27. Flags: `-hfd incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M --spec-type draft-dflash --spec-draft-n-max 7`. Evaluated against ggml-org Q4_K_M target; compatibility with an NVFP4 target is unverified (Phase 1).
- Attribution reference exists: unsloth/Qwen3.8-27B-GGUF has `Qwen3.8-27B-Q8_0.gguf` (29.05 GB) and `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` (1.37 GB). No Q8_0/F16 MTP head is published anywhere found; the Q4_0 head is the common denominator for the Q8_0-target vs NVFP4-target comparison.
- #25618 thread state: 31 comments, reproduced on CUDA (3090, 5080 sm_120), Vulkan (RDNA3, Strix Halo), with MTP, DSpark and DFlash. Divergences are near-tie 1-ULP flips at identifier boundaries; acceptance 0.87 (sm_120, IQ3_S, n_max 3), 0.68-0.94 (Vulkan). Root cause: batch-dependent kernel selection and reduction order. CUDA candidate fix: Apoze commit 6263445 "cuda: stabilize quantized FA for speculative batches" (fattn-common.cuh +10/-3, fattn-vec.cuh +5/-1, fattn.cu +1/-1): keeps the FA vec kernel and a fixed parallel_blocks partition for Q->ne[1] <= 4 with quantized K/V on Ada+, so the reduction order matches N=1. makoni confirms it on RTX 5080; reports it slows long-context decode and that the MoE model needs two more changes (unpublished). Ankk98 PR #28488 (open, 2026-09-06): batch-invariance test cases for test-backend-ops (MMVQ_BATCH, RMS_BATCH, etc.). Ankk98 PR #28489 (open): Vulkan fix making MMVQ selection independent of N. Master has no batch-invariance tests yet.
- #27329 thread state: 6 comments, labels bug-unconfirmed + stale, last comment 2026-09-17. Reports on RTX 4090 (Q4_K_XL, intermittent), 5090D (NVFP4, full hang), dual 5060, Vulkan Strix Halo (crash), SYCL Intel iGPU. Consensus in thread: Qwen3.8 fused Gated DeltaNet path scheduling defect, all backends. Not Blackwell, not NVFP4.
- Upstream policy (CONTRIBUTING.md + AGENTS.md): AI code allowed if the contributor understands every line, can debug it independently and explain it without AI; disclose in the PR template; manual review budget "one hour per 200-400 LOC"; bug-fix PRs must include a reproducible issue and a regression test that fails before / passes after; open an issue to gauge interest before implementing a feature. Prohibited (immediate closure, possible ban): AI-written PR descriptions, commit messages, issue comments, discussion posts, reviewer responses; automated commits or PR submissions. Private forks exempt. Squash-merge, commit title `<module> : <title> (#issue)`. ASCII only in code and comments.
- Native Blackwell NVFP4 (#22196) merged 2026-04-28. MMQ defaults to W4A4 for NVFP4/MXFP4 on Blackwell at batch > 8; `GGML_CUDA_MMQ_PREC=q4|q8|auto` overrides. MMQ token tile J chosen from 8..128 by fewest tiles.
- Ollama: 0.30 (June 2026) bundles llama.cpp for GGUF; no source confirms NVFP4 GGUF loading. LM Studio: no source. Do not claim downstream reach until verified by loading an NVFP4 GGUF in each (Phase 4, 30 min).
- vLLM: local image voipmonitor/vllm (id 820181fbbc97, untagged) is `0.11.2.dev280+gilded.gnosis...b12x...cu132.20260810`, torch 2.12.0+cu132, a custom 5090 build, not 0.31.0. Official vllm-openai images have historically lacked sm_120 kernels; v0.12.0 added NVFP4 CUTLASS MoE for SM120; FlashInfer CUTLASS NVFP4 MoE still reported failing on consumer Blackwell in some configs. Decision: use the local image for the vLLM column, state its version; try `vllm/vllm-openai:v0.31.0` in Phase 0.2 and switch only if it runs NVFP4 on the 5090.

---

## 4. The gap in llama.cpp (verified on master 63b4b303, 2026-10-26)

- `ggml/src/ggml-cuda/mmvq.cuh:3` `#define MMVQ_MAX_BATCH_SIZE 8`. `ggml-cuda.cu:1888` routes any quantized matmul with `src1->ne[1] <= 8` to MMVQ.
- `vecdotq.cuh:334` `vec_dot_nvfp4_q8_1`: int8 dp4a with `get_int_from_table_16` lookups. No tensor cores for batch 1-8.
- `mmq.cu` `ggml_cuda_mmq_get_prec_src1`: NVFP4/MXFP4 on Blackwell default to GGML_PREC_Q4 (native W4A4 FP4 MMA) for batch > 8; env override `GGML_CUDA_MMQ_PREC`. `mmq.cu:228` `use_native_fp4 = prec_src1 == GGML_PREC_Q4`. `mma.cuh` `mma_block_scaled_fp4` (`mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64 ... ue4m3`). `MMQ_ITER_K_FP4 512`. `ggml_cuda_mmq_get_J_best` picks J in 8..128.
- Consequence: the unserved range is M in [2, 8]. A one-line experiment (route NVFP4/MXFP4 with 2 <= ne11 <= 8 to MMQ J=8) bounds the gain before any design work (Phase 1.5).
- Batch invariance: #25618 shows that any kernel switch keyed on N makes greedy speculation lossy. Today's switches on CUDA: FA vec kernel for Q->ne[1] <= 2 vs tile/MMA above (fattn.cu:466), FA parallel_blocks partition depending on ntiles, MMVQ (N <= 8) vs MMQ (N > 8). A new small-M FP4 kernel adds another unless it also serves M=1. Design rule for Phase 3: the new path covers M in [1, 8] with one kernel and one reduction order, so N=1 decode and N<=8 verify produce identical logits. At M=1 this wastes tensor-core lanes but the step is memory-bound, so cost should be nil (verified by the M=1 metric in Section 2). This is the argument that distinguishes the PR from #30788 and from "modest gains not worth maintenance".
- Precision: upstream already accepted W4A4 numerics for prefill. The small-M path compiles both W4A8 (int8 MMA, same activation quantization as dp4a) and W4A4 (FP4 MMA); default decided by KL in Phase 3.6.
- Spec-decode flags on master: `--spec-type`, `--spec-draft-n-max`, `--spec-draft-n-min`, `--spec-draft-n-ubatch`, `--spec-draft-p-min`, `--spec-draft-sampling`, `--spec-draft-backend-sampling`, `--spec-draft-device`, `--spec-draft-type-k/v`, `--spec-synth-len L` / `--spec-synth-rates` (synthetic drafter with fixed mean acceptance length, benchmarking only: isolates verify-step kernel speed from drafter quality).
- Open issues: #25618 (batch invariance, 31 comments, updated 2026-10-18); #27329 (Qwen3.8 decode stall, stale, updated 2026-10-18); #25187 FR-Spec draft-vocab trimming (research).
- Open PRs adjacent: #30788 lane-packed small-K NVFP4 MMVQ (ynankani, review required, active 2026-10-26: maintainer asked why a new path instead of reusing MMVQ fusion); #26048 fuse w_s scale in NVFP4 MMQ epilogue; #26704 SM120 CUTLASS MoE prefill (stalled; maintainer objections: repacking, maintenance burden, MMVQ refactor first, named maintainer); #31188 fuse MoE gate/up + GLU in MMQ; #31031 trimmed draft vocab; #28488 batch-invariance tests; #28489 Vulkan MMVQ selection; #27173 spec draft perf (+10%).
- Converter: #28636 supports compressed-tensor mixed-precision NVFP4 checkpoints in tree. `llama-quantize` has no NVFP4 ftype (only MXFP4_MOE).
- MoE caveat: Qwen3.6-35B-A3B with 8 active experts per token gives each expert 1-2 tokens at M=8, so a small-M dense kernel barely touches MoE decode. The MoE model demonstrates batch invariance, not the kernel. Say so in the README.

---

## 5. Environment (this machine, verified 2026-10-26)

- GPU: RTX 5090 32 GB, driver 595.99.02, max SM clock 3090 MHz, max mem clock 14001 MHz, power limit 575 W. The GPU also drives the desktop (X session): benchmark with the desktop idle and record it.
- CPU 32 threads. RAM 60 GB. Disk 2.8 TB free on /.
- CUDA toolkit 13.2 at /usr/local/cuda-13.2 (nvcc present). cmake 3.28.3, ninja 1.11.1, gcc 13.3.0, python 3.12, pip 24.0, gh authenticated, docker present.
- Missing, install in Phase 0.2 (owner, sudo): `cuda-nsight-systems-13-2` (nsys is absent), `cuda-nsight-compute-13-2` (installed ncu is 2022.4.1, predates sm_120), `clang-format` (apt candidate 18.0; match llama.cpp's .clang-format expectations).
- llama.cpp checkout: /home/crockettgugg/llama.cpp at 4881f3e5 (2026-09-21), 816 commits behind origin/master 63b4b303. Not used by the project; the project uses the submodule.
- vLLM: local image voipmonitor/vllm id 820181fbbc97 (37.8 GB), version in Section 3.5. Run with `docker run --gpus all ... 820181fbbc97`.
- Existing assets: ~/hyperqwen-fidelity (KL harness, NOTES.md with measured 5090 speeds); ~/models/gpt-oss-20b-MXFP4.gguf (12.1 GB, MXFP4 smoke test); ~/project-maya (repo structure reference).
- No torch on host. Python work goes in a venv in the repo.
- Memory budget: Q8_0 27B (29.05 GB) fits in VRAM only with <= 4K context for the KL reference run; use `-c 4096`. Q8_0 of 35B-A3B (~37 GB) does not fit: run the MoE KL reference with `-ncmoe` CPU offload (~3B active params, fine on 32 threads) or use Q6_K and footnote it. Model downloads: 17.1 + 1.1 + 2.1 + 29.1 + 1.4 + 18.6 (+ MoE reference ~30) GB = ~70-100 GB.

---

## 6. Deliverables

### 6.1 The repo (own name, own GitHub)
```
<name>/
  README.md            headline table, one-line build, one-line run, what it is, what it is not
  LICENSE              MIT (own code) + llama.cpp MIT notice retained
  third_party/llama.cpp  NOT in git: scripts/fetch-llama.sh clones upstream at docs/PINNED_COMMIT and applies patches/
                         (changed 2026-10-27 from a submodule on a fork: the owner wants no llama.cpp clone in the repo)
  patches/             patch series on top of upstream, numbered, one topic each (generated, never hand-edited)
  bench/
    bench.sh           produces the single results table (Section 7)
    prompts/           code-{1,2,3}.txt, prose-{1,2,3}.txt, kl-set-5k.txt, system.txt
    kld.py             KL divergence vs Q8_0 reference (ported from hyperqwen-fidelity)
    invariance.sh      spec-on vs spec-off greedy diff across prompts, drafters, KV types; exits non-zero on any diff
    scaling.sh         step-time curve M=1..64 (llama-batched-bench) + test-backend-ops perf for the 27B shapes
    vllm_compare.sh    same prompts on the Docker vLLM image
    results/           dated CSV + markdown per run, never overwritten
  scripts/
    build.sh           builds build-before (pinned, unpatched) and build (patched), sm_120a, tests on
    fetch-models.sh    pulls the GGUFs in 7.4 to ~/models/<repo>/, prints sha256
    run-27b.sh, run-35b-a3b.sh   llama-server with the right spec flags
    sync-upstream.sh   rebase fork branch on upstream master, regenerate patches/, rebuild, test-backend-ops
    profile.sh         nsys + ncu capture for a fixed 200-token spec run
  docs/
    DESIGN.md          small-M batch-invariant FP4 MMA design note (owner-authored; Claude may draft sections the owner then rewrites in their own words before posting)
    UPSTREAM.md        PR links and status
    FINDINGS.md        measurements that did not become code (roofline, attribution, profiling breakdown, gate decisions)
  prd.md               this file (copy of the Desktop version)
```
Rules: llama.cpp license kept intact; no "llama.cpp" in the repo name. **Source of truth for code is the fork branch; `patches/` is regenerated with `git format-patch` at every release.**

**Git policy (owner's instruction, 2026-10-26): the owner makes all commits, pushes and remote/repo creation. Claude scaffolds files and edits the working tree but never runs `git commit`, `git push`, or creates a GitHub repo.** Working name until the owner confirms: `fp4-unlock` (GitHub and PyPI free as of 2026-10-26).

### 6.2 Upstream policy compliance (binding, from CONTRIBUTING.md / AGENTS.md)
- The owner writes every issue comment, PR description, commit message, design-note post and reviewer reply in their own words. Claude supplies measurements, traces, tables and code pointers that the owner uses as material. Claude never produces text intended to be pasted into ggml-org/llama.cpp.
- The owner must understand and be able to explain every line of every upstream diff. Plan for the review budget the maintainers state: one hour per 200-400 LOC of owner review, before opening the PR. For PR 3 (~400-800 LOC) that is 2-4 hours of owner reading plus the time to answer "why this line" questions.
- Claude's upstream-facing role: explain the codebase, propose and compare designs, run experiments, write tests and mechanical code from a design the owner has chosen and can defend, review the owner's code. Before Phase 3 code, Claude checks that the owner can state the design in their own words (AGENTS.md asks agents to verify this).
- Disclose AI assistance in each PR per the PR template.
- Every bug-fix PR carries a regression test that fails before and passes after. Every feature PR is preceded by an issue or discussion gauging interest.
- These rules do not apply inside the project repo (private fork exemption); the project repo can use any workflow the owner wants.

### 6.3 Upstream PRs, in order (each conditional on Phase 1)
1. CUDA: batch-invariant speculative verify for quantized KV (fixes #25618 on CUDA). Builds on or coordinates with Apoze's commit and Ankk98's #28488 tests; owner contacts both in-thread first. Includes a test-backend-ops case asserting N=1 vs N=2..8 logit identity for FA with quantized K/V, and for MMVQ.
2. Only if 1.7 reproduces it on master: fix for the Qwen3.8 decode stall (#27329). Not a Blackwell PR; a Gated DeltaNet scheduling fix in the CUDA (and possibly common) path. Dropped if master no longer reproduces.
3. CUDA: batch-invariant tensor-core path for NVFP4/MXFP4, M in [1,8], on Blackwell, behind a compile flag initially. Preceded by a discussion post and the maintainers' go-ahead (K4).

### 6.4 Public artifacts
- Reproduction comments with numbers on #25618 and #27329 (owner-written, week 1).
- Design note on the kernel posted before writing it (week 3-4), owner-written.
- Release v0.1 (batch invariance, measured acceptance, benchmark table) and v1.0 (kernel).
- One write-up post titled with the user-layer headline and the measured number ("Lossless Nx decode on the RTX 5090 for every NVFP4 GGUF"), before/after/vLLM table first, thesis layer as the second paragraph if G3 passed. If G1 kills the kernel, the write-up is the lossless result plus the roofline finding, and the thesis is reported as not demonstrated at M <= 8.

---

## 7. Benchmark specification (the single table)

### 7.1 Hygiene (every run, before/after/vLLM)
- Lock clocks: `sudo nvidia-smi -lgc 3090,3090 -lmc 14001` if permitted (owner); otherwise record `nvidia-smi -q -d CLOCK` before and after. `sudo nvidia-smi -rgc -rmc` afterwards.
- Desktop idle, no browser video. GPU temperature at start recorded; cool below 60 C before each configuration.
- Warm-up: one untimed run per configuration, then 3 timed runs, median reported, all three kept in the CSV.
- Record: llama.cpp commit, patch tag, driver, CUDA, date, clocks, power (nvidia-smi 1 Hz average), full command line, GGUF sha256, env vars (GGML_CUDA_MMQ_PREC, LLAMA_SPEC_CHAIN, GGML_CUDA_GRAPH_OPT).
- Same GGUF for before/after. "Before" = pinned upstream commit, unpatched, built from the same tree.
- Greedy everywhere (`--temp 0 --top-k 1 --seed 7`) except the sampled KL row. `--fit off` so context is never silently reduced (cdiamond's note).
- Two KV configurations for the invariance rows: `-ctk f16 -ctv f16` and `-ctk q8_0 -ctv q8_0`, because the CUDA culprit is quantized-KV FA; both must pass.

### 7.2 Rows (per model)
1. Batch-1 decode tg128, no spec (llama-bench).
2. Prefill pp512 and pp8192 (llama-bench).
3. Scaling curve: decode step time at M = 1, 2, 3, 4, 6, 8, 12, 16, 32, 64 via `llama-batched-bench -npl ...`; and `test-backend-ops perf -o MUL_MAT -p "NVFP4"` for the 27B weight shapes at the same M. Go/no-go evidence and before/after proof for the kernel.
4. Spec decode MTP: tok/s, acceptance rate and mean accepted length, code and prose sets, 8K context, `--spec-type draft-mtp`, n_max 3 and 7 (cdiamond's n_max 8 flags as a third point).
5. Spec decode DFlash2: same with `--spec-type draft-dflash -hfd incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M`, n_max 3 and 7 (the DFlash2 head is trained with block size 8, so llama.cpp clamps n_max to 7; Phase 1.2 finding).
6. Synthetic drafter: `--spec-synth-len 4` and `8`, so verify-step speed is measured at fixed acceptance. Isolates kernel gains from drafter gains.
7. 8-slot and 16-slot aggregate tok/s via llama-server, 300-token greedy answers, ignore_eos (imp's method).
8. Invariance: invariance.sh, spec-on vs spec-off greedy, byte diff, per drafter, per KV type, per n_max. Must be zero diffs for v0.1 and v1.0.
9. Quality: KL divergence vs Q8_0 reference on kl-set-5k.txt (llama-perplexity `--kl-divergence-base` / `--save-all-logits`), per matmul path (dp4a, W4A8 MMA, W4A4 MMA), spec on.
10. Power during rows 4, 5 and 7, llama.cpp and vLLM alike.

### 7.3 Columns
before (pinned upstream), after v0.1, after v1.0, vLLM (local image version stated; 0.31.0 if it runs NVFP4 on this card).

vLLM parity caveat: vLLM runs nvidia/unsloth NVFP4 safetensors, a different quantization recipe from cdiamond's iMatrix GGUF; footnote it. Single-5090 flags: `--max-model-len 8192`, conservative `--gpu-memory-utilization`; record exact flags.

### 7.4 Models
- Primary (dense, demonstrates everything): cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF (17.1 GB, embedded NVFP4 MTP head) + incoai/z-lab DFlash2 Q4_K_M drafter (1.14 GB).
- Attribution reference: unsloth/Qwen3.8-27B-GGUF Q8_0 (29.05 GB) + unsloth MTP/mtp-Qwen3.8-27B-Q4_0.gguf (1.37 GB) + same DFlash2 drafter.
- MoE (demonstrates invariance and fixes, not the kernel): michaelw9999/Qwen3.6-35B-A3B-NVFP4-MTP-GGUF HQ (18.64 GiB).
- MoE reference: unsloth Qwen3.6-35B-A3B Q6_K or Q8_0 with `-ncmoe`.
- Smoke: ~/models/gpt-oss-20b-MXFP4.gguf (MXFP4 path).
Rule: at ship time, re-run on whatever is the top-downloaded NVFP4 GGUF that week and quote that.

### 7.5 Prompt set (bench/prompts/)
Three code prompts (implement a function, fix a bug in a 150-line file, refactor) and three prose prompts (summary, letter, explanation), one fixed system prompt, n_predict 512. kl-set-5k.txt: 5,000 tokens of mixed code/prose. All committed so anyone can reproduce acceptance and invariance.

---

## 8. Plan: phases, subphases, gates

Conventions: `[O]` owner (git, GitHub, sudo, all upstream text); `[C]` Claude in the working tree; `[O+C]` Claude prepares material, owner acts and writes. Every phase ends with a gate: a written decision in docs/FINDINGS.md and the Status Log with the numbers behind it.

### 8.0 Kill criteria (decided up front)
- K1: If Phase 1 shows the decode step at M=8 takes <= 1.3x the M=1 step time on upstream dp4a, the verify step is bandwidth-bound and a kernel cannot pay. Stop at v0.1. Publish the roofline finding; the headline becomes "lossless speculation on the 5090" and the thesis is reported as false for llama.cpp at M<=8.
- K2: If the one-line MMQ-at-small-M experiment (1.5) is slower than dp4a at M=4 and M=8 AND nsys shows matmul kernels < 40% of verify-step time, the kernel is not the lever. Stop at v0.1, publish the breakdown.
- K3: If 1.6 shows NVFP4-target acceptance equals Q8_0-target acceptance within 5 points, there is no NVFP4-specific acceptance problem; the acceptance goal is closed as "no bug" and the speed goal stands on its own.
- K4: If maintainers decline the design note in writing, PR 3 is not opened; the kernel ships in the fork only.
- K5: If the owner cannot explain a proposed upstream diff in their own words after reviewing it, the diff is not submitted upstream until they can. It still ships in the fork.

### Phase 0: setup (days 1-2)

**0.1 Accounts, repos, tools [O]**
- [ ] Fork ggml-org/llama.cpp on GitHub.
- [ ] Create the project repo on GitHub (name per Section 11), empty, MIT.
- [ ] `sudo apt install cuda-nsight-systems-13-2 cuda-nsight-compute-13-2 clang-format libcublas-dev-13-2` (the last one replaces the `.cuda-overlay` cuBLAS workaround in build.sh, see FINDINGS.md) and confirm `nsys --version`, `ncu --version` (2025.x) and `clang-format --version`.
- [ ] Read CONTRIBUTING.md and AGENTS.md on upstream master in full (required for first-time contributors; 15 minutes).

**0.2 Toolchain check [C]**
- [x] Record versions in FINDINGS.md: nvcc, driver, cmake, ninja, gcc, nsys, ncu, clang-format, docker, vLLM image id and version. (done; see FINDINGS.md "Phase 0.2 toolchain")
- [x] Try `docker pull vllm/vllm-openai:v0.31.0` and a 20-token NVFP4 generation on the 5090. If it fails on sm_120, keep the local image and record why. (done: runs on sm_120 with FlashInfer CUTLASS NVFP4 kernels; it is the comparator. The 4.8 tok/s smoke figure includes a JIT build and is not a number.)
- [x] Python venv in the repo (`.venv/`, gitignored): huggingface_hub, numpy, and the gguf package from the submodule tree. (done)

**0.3 llama.cpp trees [C, owner creates the fork branch]**
- [x] Clone the fork into `third_party/llama.cpp`, add `upstream` remote, fetch, branch `fp4-smallm` at the pinned upstream/master commit. Record the hash in docs/UPSTREAM.md. (clone + upstream remote + branch done at 63b4b303; owner still adds `origin` = fork)
- [x] build-before: `cmake -B build-before -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release` on the unpatched pinned commit. Frozen for the project. (done)
- [x] build: same flags in `build/`, rebuilt as patches land. (done)
- [x] Also a CPU-only build (`-DGGML_CUDA=OFF`) in `build-cpu/` for the test-backend-ops reference backend and for proving upstream diffs build without CUDA. (done)
- [x] `test-backend-ops test -b CUDA0 -o MUL_MAT,MUL_MAT_ID -p "NVFP4|MXFP4"`: record pass counts. `test-backend-ops perf -o MUL_MAT -p "NVFP4"`: keep as the dp4a baseline. (done: 280/280 pass; perf baseline in FINDINGS.md, n=8 costs 3.2x n=1 on MMVQ; note test names use lower-case `nvfp4|mxfp4`)
- [x] Cherry-pick Ankk98's #28488 test cases into a scratch branch and run them on CUDA0: this tells us on day 2 which CUDA ops are already batch-variant (MMVQ, RMS, FA...). Record in FINDINGS.md. Do not carry the branch; it is diagnostic. (done: 22/24 pass on CUDA0 incl. nvfp4/mxfp4 MMVQ; only the FA-off P021 path is variant; the PR's test_fa_naive aborts on CUDA, harness gap. See FINDINGS.md)

**0.4 Models [C]**
- [x] fetch-models.sh downloads Section 7.4 into `~/models/<repo>/`, each in its own directory, prints sha256; treat as untrusted data. (cdiamond NVFP4-MTP and z-lab DFlash2 Q4_K_M done with sha256; unsloth Q8_0 + MTP head and michaelw9999 35B-A3B still downloading)
- [ ] Load each GGUF with `-ngl 99 --fit off -c 8192`, 20 tokens greedy (partly done: cdiamond plain 82 tok/s, MTP 157 tok/s acc 0.44, DFlash2 172 tok/s acc 0.51; greedy outputs already differ by drafter = #25618 live; Q8_0 + external MTP head and 35B-A3B pending download); record tok/s and VRAM. Confirm MTP loads from cdiamond's embedded head (`--spec-type draft-mtp`) and from unsloth's separate head file on the Q8_0 target (`-md`/`-hfd` path), and that DFlash2 loads against the NVFP4 target.
- [x] Smoke: gpt-oss-20b-MXFP4 pp512 with `GGML_CUDA_MMQ_PREC=q8` vs default, to see the FP4-vs-int8 MMQ delta on this machine. (done, smoke only, see FINDINGS.md)

**0.5 Repo skeleton [C], initial commit [O]**
- [x] Section 6.1 layout: README stub, LICENSE, .gitignore (models, build dirs, .venv; results/ is kept), scripts/, bench/prompts/ (six prompts, system prompt, KL set), prd.md copy, docs/ stubs. (done)
- [ ] sync-upstream.sh tested once as a no-op on the pinned commit.
- [ ] Owner commits and pushes (patches-only layout, no submodule; see status 2026-10-27 10:xx).

**Gate G0**: three builds pass; test-backend-ops passes on the unpatched tree for NVFP4/MXFP4; the #28488 diagnostic result is recorded; all models load with their drafters; nsys/ncu/clang-format present; skeleton pushed.

### Phase 1: baseline, reproduce, attribute, decide (days 3-9)

**1.1 Harness [C]**
- [x] bench.sh implements all Section 7.2 rows with 7.1 hygiene; writes `bench/results/<date>-<tree>-<model>/{rows.csv,summary.csv,rows.md,meta.txt,telemetry-*.csv}`. Dry-run at n_predict 16 done on build-before (5 harness bugs found and fixed, see FINDINGS "harness dry runs"). Clock lock still needs the owner (sudo); CLOCKS_LOCKED recorded in meta.
- [x] invariance.sh: for each prompt x drafter x KV type x n_max, run spec-off and spec-on, diff bytes, print first diverging token index and its top-2 logits (server `n_probs`); spec-off run twice as the run-to-run control. Dry run passes.
- [x] scaling.sh (llama-batched-bench per-sequence prompts; `-pps` fails on M-RoPE models) and kld.py (Q8_0 reference logits, `--kl-divergence`) as in Section 6.1. bench/perf-shapes.patch adds the 27B shapes to test-backend-ops perf (applied by scripts/build.sh to both trees).

**1.2 Baseline table [C]**
- [x] 27B NVFP4: frozen (FINDINGS "frozen before baseline" + addenda: dflash-n3 166.48 tok/s is the best single-stream row). 27B Q8_0: none 53.05, extmtp-n3 115.82, np8-none 280.01; extmtp-n7 and np8-mtp-n3 OOM at full offload, re-run at -ngl 60 for acceptance only. 35B-A3B NVFP4: mtp-n3 426.65 tok/s (acc 67.4%); spec-off rows blocked by a loader bug (NextN scale tensors not skipped with MTP off), fixed in the fork (patches/nextn-scale-skip.patch), rows queued (chain6).

**1.3 vLLM column [C]**
- [ ] vllm_compare.sh with the image chosen in 0.2; batch 1 / 8 / 16; DFlash via `--speculative-config '{"method":"dflash",...}'`; MTP where supported; power logged; exact flags and version recorded.

**1.4 Roofline and profile [C]**
- [ ] From row 3: step(M)/step(1) for M = 2, 4, 8. Memory floor = weight bytes / 1.79 TB/s; dp4a compute time for the 27B shapes. Both in FINDINGS.md.
- [ ] nsys on a 200-token synthetic-drafter run (len 8): verify-step time split into MMVQ / MMQ / FA / GDN-recurrent / norm-rope-other / sampler / host gaps, as percentages. Same at 8 slots on llama-server.

**1.5 The one-line experiment [C]**
- [x] Done (`FP4_UNLOCK_SMALLM` in mmvq.cu; FINDINGS "Phase 1.5"). Kernel (L2-resident): MMQ W4A4 flat 17-23 us for n=2..8, beats MMVQ from n=3 (1.2x) to n=8 (1.9-2.3x), loses at n=2. MMQ W4A8 (q8) is slower than MMVQ at n=2..4 and `GGML_CUDA_MMQ_PREC=q8` is global: 3x slower prefill at n=512. End-to-end q8: mtp-n3 136.32 vs 149.79 stock (-9%), mtp-n7 133.44 vs 122.03 (+9%). Conclusion: the existing MMQ path is not the lossless product; Phase 3 needs a dedicated small-M W4A8 kernel gated to M<=8 (option (b)); W4A4 at n>=3 is the labelled fast column.

**1.6 Acceptance attribution [C]**
- [x] (a) 57.4% (b) 57.3%. (c) skipped: not needed for K3.
- [x] n3: 61.5% vs 60.9%; n7: 37.2% vs 37.9% (Q8_0 at -ngl 60). temp 1.0 run not done (greedy is the product setting; z-lab's figure is a different metric, mean accepted length, and ours is 2.65/3.3 at n3/n7).
- [ ] vLLM MTP acceptance on nvidia's NVFP4 safetensors from vLLM's metrics, as the external reference.
- [x] K3 decided: no NVFP4-specific acceptance problem (all within 1 point). Acceptance goal closed as "no bug".

**1.7 Reproduce #27329 (stall) [C]**
- [ ] On build-before with cdiamond's GGUF and with unsloth UD-Q4_K_XL, the issue's command (131K context, FA on, q8_0/q4_0 KV), 2,000 tokens, three times. If it stalls: nsys trace, `gdb -p` backtrace of the spinning thread, GPU utilization log. If it does not: `git bisect` between the issue's reported build (b10488, 9d77fa172) and the pinned commit using a 300-token timed run as the test, record the fixing commit, drop PR 2.

**1.8 Reproduce #25618 (invariance) on this hardware [C]**
- [ ] invariance.sh on build-before: both models, MTP and DFlash2, n_max 3 and 7, KV f16 and q8_0. Record divergence rate, first-divergence indices, logit gaps.
- [ ] Apply Apoze's commit 6263445 on a scratch branch; re-run invariance.sh; measure its cost on tg128 at 8K and 64K context (makoni reports a long-context slowdown). Record whether f16-KV cases still diverge (that would point at MMVQ or RMS, per #28488's list, not FA).
- [ ] Apply #28489's idea on CUDA for comparison: does forcing the same MMVQ ncols template for N=1 and N>1 change anything? (Diagnostic only.)
- [ ] Owner reads the whole #25618 thread and #28488/#28489, decides whether to build on Apoze's commit (with credit, and after contacting them) or write an independent fix.

**1.9 Publish reproductions [O, material from C]**
- [ ] Owner posts, in their own words, on #25618 and #27329: hardware, commit, commands, numbers, trace excerpts. No AI-written text.

**Gate G1** (FINDINGS.md): (a) measured acceptance on NVFP4 vs Q8_0 and the K3 decision; (b) stall reproduces or fixing commit found; (c) invariance divergence rate on this hardware and whether Apoze's approach closes it at what cost; (d) kernel go/no-go against K1/K2 with the 1.4/1.5 numbers; (e) which PRs remain. Status Log records it; later phases are edited to match.

### Phase 2: batch invariance and fixes, release v0.1 (weeks 2-3)

**2.1 Batch-invariance fix on CUDA [O designs, C implements under K5]**
- [ ] Decide the mechanism: fixed FA partition and vec kernel for N <= 8 with quantized KV (Apoze's approach) vs making the FA kernel choice independent of N for all KV types vs a `batch_invariant` flag that pins kernel choices. The owner chooses and writes the rationale in FINDINGS.md.
- [ ] Implement the minimal diff. Keep M=1 and long-context performance within 2% (makoni's regression must not ship).
- [ ] If f16-KV still diverges after the FA fix, extend to the next op #28488 identifies (MMVQ / RMS), one op per commit.
- [ ] Regression test: a test-backend-ops case asserting bitwise equality of column 0 between N=1 and N=2..8 for FLASH_ATTN_EXT with quantized K/V and for MUL_MAT NVFP4 (ULP-0, like #28489's MMVQ_BATCH). Fails on build-before, passes on build.

**2.2 Verify [C]**
- [ ] invariance.sh zero diffs: both models, both drafters, both KV types, n_max 3 and 7, six prompts, plus a 64K-context prompt. Also with `GGML_CUDA_MMQ_PREC=q8`. (Phase 1.3 found q4 = +43% KL, so q8 is the lossless column; q4 is a labelled fast column only.)
- [ ] Full test-backend-ops on CUDA0 and CPU; ctest.

**2.3 Stall fix [O+C], only if 1.7 reproduced it**
- [ ] Root cause from the trace; fix; verify 3 x 2,000 tokens without stall and GPU utilization continuous; regression test if the mechanism allows one.

**2.4 Re-bench [C]**
- [ ] Full table on build. Report acceptance and spec tok/s as measured; no kernel change yet.

**2.5 PR hygiene [C], PRs [O]**
- [ ] clang-format touched files only; build CUDA, CPU-only; full test-backend-ops; ctest. Owner reviews the diff at the stated budget (1 h per 200-400 LOC), then writes the PR description per template with AI disclosure, links the issue, credits Apoze/Ankk98 where their work was used. Owner contacts them in-thread before opening if building on their commits.
- [ ] Owner opens PR 1 (and PR 2 if applicable).

**2.6 Release v0.1 [C prepares, O tags]**
- [ ] patches/ regenerated. README table: before vs v0.1 vs vLLM, with the invariance row. Release title carries the number ("Lossless MTP/DFlash2 on RTX 5090: N tok/s, acceptance A").

**Gate G2**: zero invariance diffs; PRs open (or recorded why fewer); v0.1 tagged.

### Phase 3: the batch-invariant small-M kernel (weeks 4-8), only if G1 said go

**3.0 Design note first [O writes, C supplies], then wait**
- [ ] docs/DESIGN.md content the owner must own: problem (M in [1,8] on dp4a; batch-dependent numerics); evidence (1.4, 1.5, 1.8 numbers); proposal: one kernel for M in [1,8] for NVFP4/MXFP4 on Blackwell, reusing `mma_block_scaled_fp4`, the FP4 activation quantizer and the Q8/Q4 prec switch; dispatched only when type is NVFP4/MXFP4, cc is Blackwell, ne11 <= 8; behind `GGML_CUDA_FP4_SMALLM`. Two implementation options: (a) MMQ with a J <= 8 tile, (b) a dedicated MMVQ-shaped kernel issuing the MMA (compare with #30788's lane-packed approach; do not duplicate it). Precision: W4A8 and W4A4 both compiled, default by KL. Invariance: N=1 goes through the same kernel, so this removes a batch-dependent switch rather than adding one. Expected gain from 1.5. Maintainer: the owner, named.
- [ ] Owner posts it as a discussion or on #30788's thread and asks about MMVQ refactor timing. Wait >= 3 working days; do 3.1 meanwhile. K4 applies.
- [ ] Claude verifies (AGENTS.md) that the owner can state the design and its trade-offs unaided before any kernel code is written (K5).

**3.1 Test harness before kernel [C]**
- [ ] test-backend-ops: MUL_MAT and MUL_MAT_ID cases for NVFP4 and MXFP4 at every M in [1,8] at the 27B / 35B-A3B shapes, vs CPU at existing tolerance; plus the batch-invariance case from 2.1 for the new path. Confirm they pass on dp4a first.
- [ ] perf entries for the same shapes (the microbenchmark maintainers use).

**3.2 Dispatch and flag [C]**
- [ ] ggml-cuda.cu dispatch condition, compile flag, env override to force the old path for A/B. With the flag off, test-backend-ops perf output must match build-before.

**3.3 Option (a): small-tile MMQ, NVFP4, MUL_MAT [C, owner reviews each commit]**
- [ ] J <= 8 configuration or masked loads at ne11 < 8. Perf at M = 1, 2, 4, 8 on both precisions. Pass 3.1. M=1 within 2% of dp4a.
- [ ] If (a) cannot beat dp4a at M=4 or costs > 2% at M=1, switch to (b) and record why.

**3.4 Option (b) fallback [C, owner reviews]**
- [ ] One warp per 16 output rows, m16n8k64 with M padded to 8 in the instruction's N dimension, activations quantized once per step. Same tests.

**3.5 Extend [C]**
- [ ] MXFP4 (gpt-oss-20b smoke), then MUL_MAT_ID (MoE; expect little gain, measure and say so).

**3.6 Quality decision [C measures, O decides]**
- [ ] KL vs Q8_0 on kl-set-5k for dp4a, W4A8 MMA, W4A4 MMA, spec on. Top-1 agreement vs spec-off. Default = the path with KL within noise of dp4a. If that is W4A8, FP4 MMA stays opt-in via `GGML_CUDA_MMQ_PREC=q4` and the README says so. Report the thesis honestly: "FP4 tensor cores at small M: N% faster, KL cost X".

**3.7 Profile and tune [C]**
- [ ] nsys + ncu at M=4 and M=8: tensor pipe utilization, occupancy, smem bank conflicts, DRAM throughput vs 1.79 TB/s. Tune until within 10% of the memory floor or clearly not compute-limited.
- [ ] M=1: perf identical within 2% to build-before.

**3.8 Re-bench [C]**
- [ ] Full table on the v1.0 tree. Targets per Section 2. Invariance row still zero diffs.

**Gate G3**: 3.1 tests pass on both precisions; targets met or shortfall quantified; KL decision recorded; nsys shows tensor-core instructions in the verify step; M=1 unchanged; maintainers' response recorded (K4); owner has reviewed every line (K5).

### Phase 4: ship v1.0 (week 9)
- [ ] [O] PR 3: owner-written description, design note, tests, perf table, flag-gated, under ~800 lines (2-4 h owner review); split if asked.
- [ ] [C] README final: before / v0.1 / v1.0 / vLLM, both models, power column, invariance row, and "what this is not" (batch-1 flat, MoE small gain, offload out of scope).
- [ ] [C] Verify downstream reach: load the cdiamond GGUF in current Ollama and LM Studio; record which work. The write-up claims only what loaded.
- [ ] [O] Write-up post in the owner's words; Hugging Face collection of the benchmarked GGUFs linking the repo.
- [ ] [O] Tag v1.0, release titled by the numbers.
- [ ] [C] docs/UPSTREAM.md current; patches/ regenerated.

**Gate G4**: v1.0 tagged, PR 3 open or K4 recorded, write-up posted, reach claim verified.

### Phase 5: maintenance (weeks 10-12, light)
- [ ] Weekly: sync-upstream.sh, rebuild, test-backend-ops, rows 3, 4 and 8 only; results/ updated; owner commits.
- [ ] Owner answers issues within 48 h and review comments within 24 h, in their own words.
- [ ] When a PR merges: drop its patch, bump the pinned commit, note in UPSTREAM.md.
- [ ] End of Phase 5: re-bench on the week's top NVFP4 GGUF; update the README table.

---

## 9. Upstream coordination rules

- The CUDA maintainer (JohannesGaessler) has stated: no repacking without maintainer consensus, MMVQ refactor comes first, new paths must not interfere with existing code, modest gains are not worth maintenance burden, and (on #30788, 2026-10-26) "why a new code path instead of reusing the existing one". Shape every PR to those points; the batch-invariance argument is the answer to "why".
- am17an and ORippler are the other named CUDA maintainers; am17an is active on SM120 and NVFP4 PRs.
- Active contributors to coordinate with, not around: ynankani (#30788), Ankk98 (#28488, #28489), Apoze (commit 6263445), tczank / makoni / sjvrensburg (reproducers on #25618).
- Ask before building the kernel. Post the design note and wait. Do not open a large PR cold.
- Section 6.2 policy applies to everything posted upstream.
- Every PR: one topic, regression test, test-backend-ops perf numbers, clang-formatted, builds CPU-only and CUDA, ASCII-only comments, commit title `<module> : <title> (#issue)`.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Verify step at M<=8 is still memory-bound; kernel cannot pay | K1/K2 gates on the scaling curve and the one-line MMQ experiment before design; publish either way |
| The new kernel adds a batch-dependent numerics switch and recreates #25618 | Design rule: one kernel for M in [1,8] incl. M=1; invariance test in test-backend-ops; M=1 perf metric guards the cost |
| Acceptance on NVFP4 is a model artifact, not an engine bug | 1.6 attribution vs Q8_0 target and vLLM; K3 closes the goal honestly |
| Upstream closes PRs for AI-policy reasons | Section 6.2: owner writes all text, owns design, reviews at the stated budget; K5 |
| Owner cannot defend a kernel diff line by line | K5; Claude explains until they can; otherwise fork-only |
| Apoze's fix regresses long context (reported) | 2.1 measures tg128 at 64K; the shipped fix must be within 2% |
| f16-KV divergence persists after the FA fix | 2.1 extends op by op using #28488's list |
| W4A4 at small M costs quality on every token | 3.6 KL decides the default; FP4 stays opt-in if worse |
| #27329 is a Gated DeltaNet bug beyond this project's scope | PR 2 conditional on 1.7; if deep, document the reproduction and leave the fix to the architecture owners |
| #27329 already fixed on master | 1.7 bisects the fixing commit; metrics say "fixed in commit X" |
| Kernel PR stalls like #26704 | Fixes first; design note; flag gating; small diffs; named maintainer; fork carries the value (K4) |
| MMVQ refactor or #30788 lands mid-work | Option (a) MMQ-tile shape survives; sync-upstream.sh weekly; coordinate with ynankani |
| MoE shows no kernel gain and the README over-claims | Section 4 caveat in the README; MoE demonstrates invariance only |
| Q8_0 MoE reference does not fit VRAM | `-ncmoe` offload or Q6_K reference, footnoted |
| vLLM column not comparable | Local image version stated; weights differ, footnoted; vLLM is context only |
| Benchmark noise | 7.1 hygiene: locked clocks, warm-up, median of 3, idle desktop, everything recorded |
| Downstream reach over-claimed | Phase 4 loads the GGUF in Ollama and LM Studio before claiming |
| Target models go stale | Engine-level headline; re-bench on the week's top NVFP4 GGUF at ship |
| Chasing vLLM's 32-slot numbers | Non-goal recorded; attention/KV/server overhead out of scope |

---

## 11. Open questions

Resolved 2026-10-26:
- cdiamond GGUF loader: runs on clean upstream; patches optional. Done.
- Q8_0 + MTP head for attribution: unsloth Q8_0 + MTP/mtp-Qwen3.8-27B-Q4_0.gguf. Done.
- DFlash2 minimum build: #27342 merged 2026-08-27. Done.
- vLLM image version: 0.11.2.dev280 custom b12x build; try 0.31.0 in 0.2. Done.
- Nsight: nsys absent, ncu too old; apt packages identified. Done.
- Is #25618 Blackwell-specific: no, all backends; CUDA culprit is quantized-KV FA partitioning. Done.
- Is #27329 Blackwell/NVFP4-specific: no, Qwen3.8 GDN stall on all backends. Done.
- Source of "21.7% acceptance": not found; removed. Done.
- Ollama / LM Studio NVFP4: unverified; verify in Phase 4 before claiming. Decided.

Still open (answered by a phase):
- Repo name: recommended `fp4-unlock`; runners-up fp4boost, fp4lane, tensorverify. Owner to confirm; directory ~/fp4-unlock.
- Product vs credential tone: TBD, affects README only.
- Does master still stall on Qwen3.8 (1.7)? Measured acceptance on NVFP4 vs Q8_0 (1.6)? Does f16 KV diverge too (1.8)? Does vllm-openai:v0.31.0 run NVFP4 on this 5090 (0.2)? Can an external MTP head be loaded against cdiamond's GGUF (1.6c)?

---

## 12. Reference links

- Research PDF: /home/crockettgugg/Desktop/RTX5090-FP4-Market-Research-2026-10-26.pdf
- llama.cpp issues: #25618, #27329, #25187. PRs: #22196 (merged), #27342 (merged), #26704, #26048, #30788, #31188, #31031, #28488, #28489, #27173, #25635 (merged), #26705 (merged).
- Candidate fix: github.com/Apoze/llama.cpp/commit/6263445513416ef93fd5d05391299f33c64702fa
- vLLM: PR #40082 (b12x dense), #47577 (b12x MoE), #60069 (single-5090 OOM draft), #47365, #41834, #50288.
- Strata: docs/NVFP4.md, issues #3023, #3833, #2716. Fork: sergqwer/strata-nvfp4.
- Engines: kekzl/imp, perkel666/MegaCapybara, Wallawalla47/Infernix, Neroued/ninfer, kacper-daftcode/vLLM-Moet, lna-lab/blackwell-geforce-nvfp4-gemm.
- Models: cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF, michaelw9999/Qwen3.6-35B-A3B-NVFP4-MTP-GGUF, z-lab/Qwen3.8-27B-DFlash2-GGUF, incoai/Qwen3.8-27B-DFlash2-GGUF, unsloth/Qwen3.8-27B-GGUF (Q8_0 + MTP/).
- Local: ~/llama.cpp (September tree, not used), ~/hyperqwen-fidelity/NOTES.md, ~/project-maya, ~/models, docker image 820181fbbc97.

---

## 13. Status log (append, newest last)

- 2026-10-26: Research complete, PDF on Desktop, PRD v1 written. Headline agreed: "Your RTX 5090 was never the bottleneck. llama.cpp just wasn't using its FP4 tensor cores." Own git repo; owner commits, Claude does not.
- 2026-10-26: Name recommended: fp4-unlock (owner to confirm). PRD v2: kernel scope narrowed to M in [2,8] after verifying MMQ already uses FP4 tensor cores above 8; subphases, gates, kill criteria added.
- 2026-10-26: Phase 0 [C] items done except the two pending downloads and their load checks: builds pass, 280/280 NVFP4/MXFP4 tests, dp4a perf baseline (n=8 costs 3.2x n=1), vLLM 0.31 confirmed as comparator, #28488 diagnostic 22/24 (MMVQ invariant incl. nvfp4), cdiamond + MTP + DFlash2 load and run, #25618 reproduced. cuBLAS 13 dev package missing, overlay workaround in build.sh. Owner 0.1 list unchanged plus libcublas-dev-13-2. All in docs/FINDINGS.md.
- 2026-10-26: PRD v3 after desk research: 21.7% acceptance figure removed as unsourced; #25618 understood as batch invariance with a CUDA candidate fix and open test PRs; #27329 understood as a Qwen3.8 GDN stall on all backends; upstream AI policy built in (owner writes all posts, owns design); kernel must be batch-invariant across M in [1,8]; toolchain gaps (nsys, ncu, clang-format) and vLLM image version recorded; model-card facts and attribution assets confirmed. Nothing started. Next action: Phase 0.1 (owner: fork, repo, apt installs, read CONTRIBUTING/AGENTS) and 0.2-0.5 (Claude) when the owner says go.
- 2026-10-26: Headline revised after research: old wording ("llama.cpp just wasn't using its FP4 tensor cores") retired as inaccurate. Two-layer headline adopted (Section 1): user layer "Lossless 2-3x decode on the RTX 5090 for every NVFP4 GGUF", thesis layer about the verify step on FP4 tensor cores only after G3. Phase 0 started under working name fp4-unlock at ~/fp4-unlock (owner has not yet confirmed the name).
- 2026-10-26: Phase 1 started autonomously (owner away; owner commits). 1.1 harness complete in bench/ (five bugs fixed in dry runs: llama-bench label, kld.py missing --kl-divergence and key parse, batched-bench -pps on M-RoPE, scaling.csv header, sub-second power). 1.2 "before" baseline chain running on build-before (NVFP4 rows 1-7, invariance, KL, Q8_0 rows). 1.5 one-line experiment implemented in the fork (mmvq.cu, env FP4_UNLOCK_SMALLM, build passes), runner bench/smallm_experiment.sh. 1.4 nsys installed, bench/profile_split.py written. 1.7 bench/stall_repro.sh written (issue #27329 config). 1.8 Apoze 6263445 staged in third_party/llama.cpp-apoze, build target `scripts/build.sh apoze`. vLLM 0.31 confirmed to accept methods "mtp" and "dflash". Hurdle for owner: repo setup ran in the wrong directory (see final report), ~/fp4-unlock still has no .git.
- 2026-10-26: K3 decided (no NVFP4 acceptance bug; MTP 57.4 vs 57.3%, DFlash-n3 61.5 vs 60.9%). dflash-n3 is the best NVFP4 single-stream row at 166.48 tok/s. Q8_0 target rows done at full offload where they fit; two OOM rows re-queued at -ngl 60. 35B-A3B: mtp-n3 426.65 tok/s; spec-off rows exposed a loader bug on the pinned commit (NextN NVFP4 scale tensors not skipped when MTP is off -> "expected 1101, got 1079"); fixed in the fork, verified on the CPU build, patch in patches/nextn-scale-skip.patch, upstream PR candidate for the owner. Phase 1.5 result changes Phase 3: the existing MMQ path at small M is not the lossless product (q8 slower than MMVQ at n<=4, and the q8 env var costs 3x at prefill); the kernel must be a dedicated M<=8 W4A8 path, option (b). bench/lib.sh gained a SERVER_EXTRA hook (used for `--mtp` on the before tree). Still running on the GPU: Phase 1.5 q4 rows, SMALLM invariance + tg128 cost, smallm KL, nsys profiles, Apoze invariance and depth cost, stall repro, vLLM column (chain3); Q8_0 -ngl 60 rows (chain4); op-level invariance probe (chain5); 35B rows after the loader fix (chain6).
- 2026-10-26: Phase 1.5 complete (q8 and q4 end-to-end columns in FINDINGS): MMQ columns lose at M=4 rows, win at M=8 rows (dflash-n7 159 -> 183 q8 / 198 q4); acceptance drifts with target precision (37.2 / 40.2 / 43.5 %), which makes bitwise batch invariance a precondition for any "lossless" claim. Phase 2.1 designed from source and implemented as a draft in the fork (FA: fixed 256-position KV split, pinned tile and kernel family for M<=8, sparse off; MMVQ: nwarps pinned to 4 for ncols 1..8), runtime `FP4_UNLOCK_INVARIANT=1`, compile flag `FP4_UNLOCK_INVARIANT_MMVQ`, built as build-dev (`scripts/build.sh dev`). Design and switch inventory in FINDINGS. Validation chain7 queued behind chain6 (opinv, perf, e2e invariance, tg128 depth cost, server rows). Owner decision pending on the mechanism per 2.1 (Claude's rationale is written; owner may overrule).
- 2026-10-26: Phase 1 closed (1.4-1.8, FINDINGS "chain3/4/5/6 outputs"): nsys split GEMM ~78 % of the verify step (mmq 48-52, mmvq 26-30 of which Q8_0 lm_head 20); K1 not triggered (M8/M1 = 1.47), K2 cannot trigger; stall #27329 not reproduced; Apoze FA patch rejected (-50 % tg128 at depth 65536); Q8_0 target rows at NGL=60 recorded; op-level probe (opinv) shows stock MUL_MAT diverging from n=5 (MMVQ warp switch) and n>=9 (MMQ), FA from n=3; 35B rows partial (chain6 rebuild failure, fixed in scripts/build.sh, re-run queued as chain9). vLLM dry run failed, re-attempt queued in chain8.
- 2026-10-26: Phase 2 validated on build-dev (FINDINGS "chain7 outputs"): op level all bitwise same n=2..8 with `FP4_UNLOCK_INVARIANT=1`; end to end M=4 rows 0/6 divergent (f16 and q8_0 KV), M=8 rows still 6/6 because Blackwell routes Q4_K/Q5_K/Q6_K to MMQ at ne11 6/7/8 (the 27B attention/GDN projections are Q5_K); fix in `ggml_cuda_should_use_mmvq` + shared `fp4-unlock.cuh`, built into build-p3, validated by chain8 step e. Cost: tg128 within 2 % at depth <= 8192, but -3.6 % (f16) / -19 % (q8_0 KV) at depth 65536 from the FA pin, so the switch stays opt-in (2.1 "within 2 %" holds only at short context; makoni-style regression would ship if it defaulted on). 2.1 regression test (test-backend-ops ULP-0 case) still to write; 2.3 closed (no stall).
- 2026-10-26: Phase 3 started ahead of the owner's design sign-off (K5 deviation, recorded): option (b) kernel `fp4_smallm_w4a8_kernel` in the fork (fp4-smallm.cu/.cuh, dispatch in mmvq.cu, compile flag `GGML_CUDA_FP4_SMALLM`, env `FP4_UNLOCK_KERNEL`), build-p3. First build read the q8_1 activation buffer before `quantize_q8_1` finished (Programmatic Dependent Launch); fixed with pdl_sync/pdl_lc, rule added to DESIGN-draft. Correctness: mmstress 0 diffs, test-backend-ops MUL_MAT 2484/0, W4A8 31/0, W4A4 21/0, VEC_FUSION 1376/0 (3 reps + KERNEL=0 controls). Performance, KL, e2e invariance and server rows are chain8 (running); 35B rows chain9 (queued). G1/G2 decided when chain8 lands.
- 2026-10-26: Phase 2.1 regression tests written into the fork's test-backend-ops (`MUL_MAT_BATCH`, `FA_BATCH`, ULP-0, backend compared against its own n=1; FINDINGS "Phase 2.1 regression tests"). CPU run exposed two ggml CPU-backend properties, not fork bugs: f16-KV FA takes a split-KV path only at nb=1 (attributed to ops.cpp), and k-quant MUL_MAT column 0 differs between n=1 and n=8 (unattributed). CUDA matrix queued as chain10 (after chain9): build/ env-off stock column, build-p3 KERNEL x INVARIANT, plus a full eval gate on build-p3. `fp4_unlock_invariant_enabled()` folded into fp4-unlock.cuh (single definition). .gitignore now covers build-dev/ and build-p3/.

- 2026-10-26 22:05 — chain8 part 1 (Phase 3 validation of kernel v1): eval gate clean; op-level and end-to-end
  invariance 0/6 everywhere with KERNEL=1 INV=1 (property 1 holds); KL 0.0428 vs 0.0432 (lossless column holds);
  property 3 holds (n=1->8 +26 %/+46 %); **property 2 fails: M=1 +58 % (up) / +91 % (down) vs MMVQ**, cause = one
  block in flight per warp (serial L2 round trips). Kernel v2 in the tree: FP4_SMALLM_PF=4 prefetch, same float
  order (bitwise same results), 137 regs no spills. chain10 rebuilds/gates build-p3; chain11 (new) re-measures perf,
  tg128, server rows and does a v1-vs-v2 bitwise generation check. G1 stays open until chain11.
- 2026-10-26 22:12 — tg128 with kernel v1: -19 % at depth 0/8192, -16 % at 65536 (FP4 FFN GEMMs ~30 % of the decode
  step). chain8 trimmed after its KERNEL=1 INV=1 server rows (trim8.sh releases chain9); chain11 now also carries the
  nsys profile and vLLM dry run. Sequence: chain8 (trimmed) -> chain9 35B -> chain10 gate -> chain11 v2 numbers.
- 2026-10-26 22:16 — chain8 trimmed after the v1 server rows: none 66.3 (-19 %), M=4 rows -11..-13 %, M=8 rows
  break-even to +15 % (mtp-n8 130.7 vs 113.5). chain9 (35B) running; chain10, chain11 queued.
- 2026-10-26 22:25 — chain9 (35B-A3B) done: model runs after the loader fix; stock none 279 / mtp-n3 427 tok/s,
  np16 842/902; kernel v1 is neutral there (-1.5 % none) because expert FFNs are MUL_MAT_ID, outside the dispatch.
  chain10 (rebuild build-p3 with kernel v2 + gates) running.
- 2026-10-26 22:30 — chain10 done: with FP4_UNLOCK_INVARIANT=1 every MUL_MAT_BATCH (n <= 8) and FA_BATCH case is bitwise
  OK with kernel v2 on and off; the one n_max=16 case failed at n=9 (MMQ boundary, outside the contract) and was capped
  to 8. All-ops eval showed only those batch-test failures under INV=0 (11, names dropped by the log filter); chain12
  (after chain11) rebuilds build-p3 and repeats the gate plus raw all-ops evals. chain11 (kernel v2 numbers) running.
- 2026-10-26 22:35 — chain11 (kernel v2 = v1 + prefetch) trimmed before its server rows: v2 is bitwise v1 (0/6) but
  still 27-28 us at M=1 vs MMVQ 15/13 (tg128 -18 %); diagnosis: the direct MMA fragment loads are L1-wavefront bound
  (8 lines per 4-byte load). chain12 superseded by chain13.
- 2026-10-26 22:47 — kernel v3 (weights staged through shared memory with cp.async): wavefront problem gone, M=1 still
  27-28 us; the int8 path is issue bound (copy floor 15.6 us). Kept as KERNEL=2. Arithmetic moves to the FP4 tensor core.
- 2026-10-27 00:10 — kernel v4 (block-scaled FP4 MMA, activations as T residual NVFP4 terms, separate quantizer,
  shape-only split-K): 67/67 correct at T=1..4; flat in M, 18.7-21.3 us at M=1..8 on both FFN shapes (MMVQ 41/66 at
  M=8). v4a->v4b (per-segment column scales) -2..-8 us; v4c (4-lane quantizer) regressed and was reverted; v4d adds the
  power-of-two ksplit for T>=3. Ablations: B loads and MMAs are free, weight copies are 10-11 of 18.7 us, the rest is
  fixed cost (quantizer launch 1.4-4.4 us, GEMM shell 3.6 us). End to end with every M through v4: tg128 -7.9 % (T=1),
  -9.1 % (T=2), -10.6 % (T=3): property 2 fails for the kernel alone.
- 2026-10-27 00:18 — kernel v5 (quantization inside the GEMM block, no second launch): correct, slower (19.5 us T=1,
  41 us T=3 at M=1; every row group re-quantizes, latency-bound chain). Rejected, kept as KERNEL=3 for A/B.
- 2026-10-27 00:20 — dispatch decision from the crossover table: MMVQ wins at M<=2, v4 from M=3 (FP4_UNLOCK_MIN_M,
  default 3). Default mode = upstream MMVQ for batch-1 decode (property 2 by construction, no batch-1 claim), v4 for
  M=3..8. FP4_UNLOCK_INVARIANT=1 sends all M through v4 (bitwise 1..8, at the measured M=1 cost). chain13 (full v4
  evaluation on this build) started 00:22.
- 2026-10-27 00:30 — chain13 part 1: batch tests K0I1/K1I1 exit 0, K1I0 exit 1 as expected (NVFP4 n=3 boundary +
  stock k-quant/FA); full eval INV=1 19143/19143; INV=0 only the 13 batch cases; MUL_MAT/W4A8/W4A4/VEC_FUSION all
  pass. KL vs Q8_0 (-b 8, all GEMMs through v4): T=1 0.0570 (fails the 0.043 bar: W4A4 is the labelled fast path),
  T=2 0.0422, T=3 0.0426, T=4 0.0426, MMVQ control 0.0432. T=2 and T=3 are at the weight-quantization floor; the
  default T is settled by the end-to-end rows (T=3 in chain13, T=2 pass queued as chain14).
- 2026-10-27 01:20 — chain13 part 2 (so far): tg128 d0/8k/64k kernel0 82.99/80.88/68.30 vs kernel1-default
  83.04/80.80/68.28 (Phase 3.7 bar met, by the dispatch rule); invariant mode T=3 −10.7/−10.3/−11.8 %, T=1
  −8.1/−7.6/−9.6 %, FA pin alone 0/0/−3.6 %. pp512/8192 unchanged (6533/6292). opinv INV=1: every op `same` n=2..8,
  DIFF from n=9 (llama.cpp's own MMVQ→MMQ switch, out of scope). Invariance INV=1 f16+q8_0: 0/6 for all drafters;
  default mode f16 5/6-6/6 with spec on (expected, documented trade). Server rows 4-6 default vs same-day KERNEL=0
  control: mtp-n7 129.95 vs 113.68 (+14 %), dflash-n7 174.37 vs 146.98 (+19 %), synth8 unchanged (M=9 is MMQ),
  **mtp-n3 142.50 vs 149.73 (−4.8 %)** and dflash-n3 158.89 vs the 10-26 before row 166.48. The M=4 verify step is
  slower end to end although the unfused microbench has v4 ahead at M=4. Suspect: the fused [gate|up]+SWIGLU path
  (never timed; perf list only had unfused MUL_MAT). chain15 queued after chain14: fused perf M=1..8, per-step cost
  via llama-bench -p 1..8 (fusion on/off, both kernels), and the n3 rows with FP4_UNLOCK_MIN_M=5. Until that settles,
  the default MIN_M may move to 5 (M=3,4 on MMVQ): kill criteria unaffected, the batch-1 claim was never made.
- 2026-10-27 01:50 — chain15 done (M=4 diagnosis). Fusion is not the cause: per-step cost via llama-bench -p M -n 0
  (ms, M=1/2/3/4/6/8) MMVQ 12.21/12.68/12.98/13.67/15.83/18.30 vs v4 13.90/14.19/14.65/14.88 at M=3/4/6/8, same
  +0.9/+0.5 ms gap at M=3/4 with GGML_CUDA_DISABLE_FUSION=1. The microbench crossover (M=3) was L2-resident (50 MB
  weight fits the 5090's L2); the real crossover is between M=4 and 6. Fused-GEMM perf cannot be timed by
  test-backend-ops perf (it replicates only the final add node); shapes removed again. With FP4_UNLOCK_MIN_M=5 the
  n3 rows recover their controls exactly (mtp-n3 149.67 vs 149.73, dflash-n3 166.82 vs 166.56). INV=1 T=2 vs T=3
  per step: T=2 0.22 ms cheaper at M=1, equal from M=2. **Defaults applied in source: T=2, MIN_M=5** (fp4-smallm.cu/.cuh,
  DESIGN-draft updated, FINDINGS chain15 section written). p3 rebuilding; next: 67 OK + batch + VEC_FUSION re-verify,
  then chain16 = the final product table on the shipped defaults, then README (Phase 4).
- 2026-10-27 03:20 — chain16 done (final product table on T=2 / MIN_M=5; 0 failed steps; FINDINGS section written;
  README rewritten; patches/README.md, docs/UPSTREAM.md, run scripts -> build-p3, bench/final_table.py added).
  Same-day KERNEL=0 control vs default: mtp-n5 +4.9 %, mtp-n7 +11.8 %, dflash-n5 +9.0 %, dflash-n7 +15.1 %,
  q8_0-KV mtp-n7 +10.8 % / dflash-n7 +7.8 % (182.8 tok/s = 2.27x batch 1), np8 none/mtp-n3/mtp-n7 +11.0/+10.9/
  +18.4 %; M<=4 and M>=9 rows (n3, n8, synth = 8-token drafts, np16) unchanged within 0.2 %; board power 60-90 W
  lower on every kernel row. Per step: M=8 18.35 -> 14.90 ms (-18.8 %), flat 14.4-15.0 ms for M=5..8. Batch 1
  83.1 vs 83.1, pp unchanged. KL 0.0422 (bar 0.043). Invariance INV=1: 0/6 for 6 drafters x f16+q8_0 (14 runs);
  INV=1 costs -9.6 % at batch 1, -5 % at n3, but is +4.9/+8.4 % over control at n7. T sweep (n7): T=2 >= T=3,
  T=1 +1 % on dflash but KL 0.057. Reach: Ollama 0.20.7 cannot load any NVFP4 GGUF (GGML_ASSERT on tensor type in
  its bundled ggml); LM Studio not installed. vLLM column: not measured (10-26 dry run failed, not rerun).
  Phase 4 [C] items done: README final, reach check, UPSTREAM.md + patches/. Remaining [O]: review, commit, PR 3
  issue/description, write-up, tag v1.0. Gate G3 note: "maintainers' response recorded (K4)" was never obtained
  (no post was made: owner writes all posts), recorded as the K5/K4 process deviation.
- 2026-10-27 10:20 — repo layout changed to patches-only (owner decision: no llama.cpp clone in the repo). `third_party/` gitignored; `scripts/fetch-llama.sh` clones upstream at docs/PINNED_COMMIT and applies patches/ (build.sh calls it); `scripts/regen-patches.sh` regenerates patches/ (byte-identical to the shipped set); `scripts/sync-upstream.sh` rewritten for a plain upstream clone (3-way re-apply, no fork). Verified from a clean copy of only the tracked files: fetch + apply == development tree byte for byte; build-p3 1m44s; MUL_MAT nvfp4 OK; MUL_MAT_BATCH,FA_BATCH under INVARIANT=1 OK; llama-bench pp8 517.7 tok/s (15.5 ms/step, unlocked clocks). Push simulation in a scratch clone of github.com/JoeGuglielmucci/fp4-unlock: 3695 files, 11.2 MB, nothing under third_party/. The owner's llama.cpp fork is no longer referenced anywhere. Owner commits/pushes.
