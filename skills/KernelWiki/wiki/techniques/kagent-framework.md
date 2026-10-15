---
id: technique-kagent-framework
title: "KAF — Kernel Agent Framework (frozen evaluator, formal checks, evidence)"
type: technique
architectures: [gfx950]
tags: [agent, verification, flydsl]
confidence: source-reported
reproducibility: snippet
prerequisites: []
related: [kernel-kda-decode-mi355x, kernel-kda-decode-v8, hw-gfx950-cdna4, hw-gfx950-perf-counters, pattern-pipeline-stalls]
sources: [blog-kernel-lab-mi355x]
aliases: ["KAF", "kagent", "kernel agent framework"]
evidence_basis:
  - source_id: blog-kernel-lab-mi355x
    evidence_type: reproduction
---

## Quick start

```bash
cd /root/amd-workspace/kernel-lab/framework
python3 -m kagent.loop.cli check --pack examples/kda-serving
python3 -m kagent.loop.cli run --pack examples/kda-serving \
    --candidate ../kda-serving/runs/flydsl-decode-v7 --dtype float32
```

## Overview

KAF (`kernel-lab/framework`, package `kagent`) is the driver-agnostic harness the
KDA decode campaign runs on (blog-kernel-lab-mi355x). The agent proposes kernels;
everything that grades or explains them is mechanical.

## Layers

- **spec** — task packs as data: `task.toml` (objective, target, promotion rules),
  `bar.toml` (pre-registered accuracy thresholds, user-approval change policy),
  `cases/*.toml` (every verification case declares class + the bug class it kills
  + the incident that motivated it; `kagent check cases` enforces this).
- **eval** — frozen, hash-locked evaluator crates with hidden holdout seeds;
  reports for holdouts carry counts only. Tamper self-test proves refusal.
- **formal** — FlyDSL mid-level MLIR -> op DAG -> xor-shuffle butterfly coverage
  checks + Z3 exact-real equivalence per queried output element. `exp`/`rsqrt`
  are uninterpreted (shared with the spec), so PROVED means same math modulo the
  abstraction; bf16 rounding points are recorded, not modeled. Verdicts are
  tri-state (PROVED / COUNTEREXAMPLE+witness / UNKNOWN+pattern).
- **evidence** — `isa.json` (math/data + issue/data ratios, occupancy, s_nop),
  `pcs.json` (rocprofv3 stochastic PC sampling, stall-reason aggregation, top PC
  offsets), `roofline.json` (bytes-model floors at spec and baseline-implied BW).
- **loop** — stage machine `verify -> holdout -> formal -> bench -> evidence ->
  report`; bench/evidence refuse to run before the frozen verifier passes.


A verification case is data with a motivation (guidelines 02):

```python
from kagent.spec.loaders import load_pack
pack = load_pack("examples/kda-serving")   # validates task+bar+cases
for c in pack.visible_cases():
    print(c.id, c.klass, c.tolerance_ref, c.kills)
# D1_salts numeric o_rrmse wrong-math-anywhere-samples-reach
# D4_drift drift state_rrmse reduced-precision-state-bias
```

## Measured facts on MI355X (kda-serving / flydsl-decode-v7)

- 19 xor-shuffle reduction chains; all full butterflies; 6 queried output
  elements PROVED in ~1.5 s (z3 4.15.4, no GPU).
- ISA: math/lane 202, data/lane 95 (math/data 2.13, issue/data 4.75),
  64 VGPRs -> ~8 waves/SIMD, 37 s_nop, 49 s_waitcnt -> issue-bound.
- PC sampling: 222/269 samples WAITCNT; hottest unmapped PCs at offsets 380, 260.
- Roofline (serving-pool graphed per-step, fp32 state): 0.25x geomean vs FLA
  serving path (7.5/22.1/66.1/291.6 us) — v7 is slower on the serving path;
  the 41.7%-of-roofline figure was a kernel-only measurement.
- Planted-bug pack (dropped scale*beta, wrong shuffle offset, missing L2N,
  state base off-by-one): 4/4 caught, correct kernel PROVED.

## Traps

- FlyDSL buffer ops in mid-level MLIR index in BYTES (bf16 scalar = 2x element,
  vec4 f32 = 4x element) — misreading this flips index maps.
- FlyDSL compiles lazily on first launch: FLYDSL_DUMP_IR=1 only dumps if the
  kernel actually runs; use the candidate's self-test as the dump workload.
- Loop reorderings of `graphed_us_per_step` (serving-pool) vs
  `kernel_us_median` (public-wrapper) are not comparable; roofline refuses
  mixed pairs.