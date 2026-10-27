# Design note: batch-invariant small-M tensor-core path for NVFP4/MXFP4 on Blackwell

Owner-authored. Written in Phase 3.0, after Phase 1 numbers exist. Not before.

Sections to fill (from prd.md 3.0): problem, evidence (1.4, 1.5, 1.8), proposal,
options (a) MMQ J<=8 tile / (b) dedicated MMA kernel, precision (W4A8 vs W4A4, KL-decided),
invariance argument (one kernel for M in [1,8]), expected gain, maintainer.
