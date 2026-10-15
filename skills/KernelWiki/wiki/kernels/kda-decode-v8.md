---
id: kernel-kda-decode-v8
title: "Task: KDA decode v8 (native serving pool + issue-bound fixes, fp16 state)"
type: kernel
architectures: [gfx950]
kernel_types: [linear-attention]
languages: [flydsl]
tags: [kda, flydsl, linear-attention, cuda-graphs, agent]
confidence: source-reported
reproducibility: snippet
related: [technique-kagent-framework, kernel-kda-decode-mi355x, hw-gfx950-cdna4, hw-gfx950-perf-counters]
sources: [blog-kernel-lab-mi355x]
aliases: ["v8", "kda-serving v8", "KDA decode v8 task"]
performance_claims:
  - gpu: MI355X
    dtype: "bf16 inputs, fp32 state"
    shape: "B=128, H=96, K=V=128, T=1"
    metric: "graphed_us_per_step (us), FlyDSL v7, serving-pool"
    value: 1361.7
    utilization: "0.25x of FLA serving path; wrapper-dominated (v8 target)"
    source_id: blog-kernel-lab-mi355x
  - gpu: MI355X
    dtype: "bf16 inputs, fp32 state"
    shape: "B=128, H=96, K=V=128, T=1"
    metric: "graphed_us_per_step (us), FLA serving path"
    value: 291.6
    utilization: "~70% of 8 TB/s implied BW"
    source_id: blog-kernel-lab-mi355x
  - gpu: MI355X
    dtype: "fp32"
    shape: "H=96, K=V=128, decode verifier"
    metric: "gates passed (frozen verify / hidden holdout), FlyDSL v7"
    value: 15
    utilization: "15/15 visible + 10/10 holdout; bar selects fp16 over bf16 state"
    source_id: blog-kernel-lab-mi355x
evidence_basis:
  - source_id: blog-kernel-lab-mi355x
    evidence_type: benchmark
---

## Loop commands

```bash
python3 -m kagent.loop.cli run --pack examples/kda-serving \
    --candidate ../kda-serving/runs/flydsl-decode-v8 --dtype float16
```

## Goal (user-decided 2026-10-09)

2x FLA Triton on KDA decode for the Kimi K3 layer (H=96, K=V=128) on MI355X,
geomean over B in {1, 8, 32, 128} on FLA's serving path
(`fused_recurrent_kda_fwd`, `ssm_state_indices`, `inplace_final_state=True`),
no batch size slower than FLA. Language: FlyDSL; v8 may continue from v7 or
restart. Cadence: one candidate, then report, then user input.

## Where v7 stands (measured, framework artifacts)

- frozen verify 15/15 + hidden holdout 10/10 (fp32 state)
- formal: PROVED (structure) — the math is right; the layout is the problem
- serving path (graphed per-step): v7 0.25x of FLA — the gather/scatter wrapper
  around the state pool costs more than the kernel saves
- ISA: issue-bound (math/data 2.13; mov+cndmask+cmp+permlane ≈ shuffles)
- PC sampling: WAITCNT-dominated (state row loads on the critical path)
- fp16-state roofline floors: 3.8/11.2/33.4/148.2 us at FLA's implied BW

## v8 plan (approved)

1. Native in-place indexed state pool (kill the gather/scatter wrapper).
2. Coalesce per-lane work to attack the issue-bound ratio (wider vec4/vec8
   state handling, row-block ownership per lane instead of 2 rows + 8 cols).
3. Then fp16 state (Path B; bar already selects fp16 over bf16).


The done-criteria check the loop engine performs (engine.py):

```python
import json
rep = json.loads((candidate / "verify_decode_report.json").read_text())
assert rep["passed"] and rep["n_gates"] == 15      # frozen verifier
assert formal_report["overall"] != "COUNTEREXAMPLE"
geo = roofline["geomean_speedup"]                  # serving-pool metric
assert geo >= 2.0 and all(p["speedup"] >= 1.0 for p in roofline["per_B"])
```

## Done criteria

geomean >= 2.0 with every B >= 1.0 on the serving-pool metric
(`graphed_us_per_step`), frozen verify 15/15 at the declared dtype, formal
PROVED (or UNKNOWN with the pattern documented), reviewer subagent accepts.