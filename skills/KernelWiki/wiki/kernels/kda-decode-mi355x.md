---
id: kernel-kda-decode-mi355x
title: "Kimi Delta Attention (KDA) Decode on MI355X: FLA Triton vs FlyDSL"
type: kernel
architectures: [gfx950]
tags: [linear-attention, decode, gemv, triton, flydsl, wave64, cuda-graphs]
confidence: experimental
reproducibility: snippet
kernel_types: [linear-attention, decode, gemv]
languages: [triton, flydsl]
related: [kernel-gated-delta-net, hw-gfx950-cdna4, hw-gfx950-perf-counters, migration-sm100-to-gfx950, pattern-pipeline-stalls, pattern-memory-bound]
sources: [blog-kernel-lab-mi355x]
evidence_basis:
  - source_id: blog-kernel-lab-mi355x
    evidence_type: benchmark
  - source_id: blog-kernel-lab-mi355x
    evidence_type: reproduction
blackwell_relevance: "Same operator class as kernel-gated-delta-net; shows which decode bottlenecks carry over to AMD (state bandwidth, dispatch) and which are AMD-specific (wave64 swizzle chains)."
performance_claims:
  - gpu: MI355X
    dtype: "bf16 inputs, fp32 state"
    shape: "B=128, H=96, K=V=128, T=1"
    metric: "kernel time (us), FLA fused_recurrent_kda (Triton)"
    value: 293.8
    source_id: blog-kernel-lab-mi355x
  - gpu: MI355X
    dtype: "bf16 inputs, fp32 state"
    shape: "B=128, H=96, K=V=128, T=1"
    metric: "achieved HBM bandwidth (TB/s), FLA"
    value: 5.6
    utilization: "~70% of 8 TB/s"
    source_id: blog-kernel-lab-mi355x
  - gpu: MI355X
    dtype: "bf16 inputs, fp32 state"
    shape: "B=128, H=96, K=V=128, T=1"
    metric: "kernel time (us), FlyDSL v7"
    value: 486.2
    source_id: blog-kernel-lab-mi355x
  - gpu: MI355X
    dtype: "bf16 inputs, fp32 state"
    shape: "B=1, H=96, K=V=128, T=1"
    metric: "graphed wall time per step (us), FlyDSL v7 (FLA: 8)"
    value: 7
    source_id: blog-kernel-lab-mi355x
  - gpu: MI355X
    dtype: bf16
    shape: "B=1-8, T=1K-64K, H=96, K=V=128"
    metric: "prefill throughput (M tokens/s), FLA chunk_kda"
    value: 3.3
    source_id: blog-kernel-lab-mi355x
---

## Overview

Kimi K3's KDA layers (69 of 93 layers; H=96, K=V=128) keep a fixed-size recurrent state
per head instead of a KV cache. Each decode step reads and writes the whole state:

```
S <- S * exp(g);   u = v - k·S;   S <- S + beta * k ⊗ u;   o = scale * q·S
```

with q and k L2-normalized. At B=128 the fp32 state is 805 MB, read and written every
step, so decode is a bandwidth problem at large batch and a launch-latency problem at
small batch. Per-step latency is flat in context length (measured 1K–32K).

## Baseline: FLA `fused_recurrent_kda` (Triton)

```python
from fla.ops.kda import fused_recurrent_kda

def decode_step(q, k, v, g, beta, state):
    o, new_state = fused_recurrent_kda(q, k, v, g, beta, initial_state=state,
                                       use_qk_l2norm_in_kernel=True,
                                       output_final_state=True)
    return o, new_state
```

FLA launches B·H·(V/32) programs, each holding a 128 × 32 state tile in registers and
reducing locally. On MI355X it reaches 5.6 TB/s at B=128 (~70% of peak HBM bandwidth).

## FlyDSL candidate lineage

| Version | Change | Kernel time vs FLA |
|---|---|---|
| v4 | FLA's tiling ported to FlyDSL, 6 barriers per step | 2.6–3.3× slower |
| v6 | 256 threads; each wave64 owns 8 columns, each lane 2 rows; reductions stay inside the wave, 0 barriers | 1.5–1.7× slower |
| v7 | v6 plus vector `o` stores; captured in graphs | B=1 graphed wall 7 µs vs FLA 8 µs; serving geomean 0.25× (wrapper-dominated) |
| v8 | Native in-place indexed pool (`decode_pool`, zero gather/scatter) | serving geomean 0.74× (B=1 7.28 µs, 1.03× FLA; B=128 468 µs) |
| v9 | Hoisted state loads, interleaved reduction trees, vector FMA state store | serving geomean 0.77× (B=1 7.13 µs, 1.05× FLA; B=128 442 µs) |
| v10 | V-major contiguous state layout (`state_v_first`, 512B burst loads/stores) | serving geomean 1.12× (B=1 5.67 µs 1.32×, B=128 275 µs 1.065×, 74% HBM peak — ALL B beat FLA) |
| v11 | 128-thread blocks, NV=8 (768 blocks at B=1, 3 blocks/CU) | B=1 record latency 5.43 µs (1.38× FLA); B=32 regresses slightly |
| v12 | Adaptive dispatch: NV=8 for B<=8, NV=4 for B>=32 | serving geomean 1.09×, B=1 5.43 µs (1.38× FLA), B=128 277 µs (1.06× FLA) |

The in-wave reduction used by v6 and v7:

```python
_TREE = [(32, 64), (16, 64), (8, 64), (4, 64), (2, 64), (1, 64)]

def _tree(val):
    for i in range_constexpr(6):
        off, width = _TREE[i]
        val = fx.arith.addf(val, val.shuffle_xor(fx.Int32(off), fx.Int32(width)))
    return val
```

## Diagnosis with hardware counters (B=128)

Both kernels move the same ~1.66 GB, launch the same 196,608 waves, and issue nearly
all writes as full 64 B transactions; v7 uses fewer VGPRs (32 vs 44). v7's memory
instructions even finish faster (`VmemLatency` 1,848 vs 2,597 cycles). But v7 waves
spend 74% of their lifetime waiting to issue, against 27% for FLA. The bottleneck is
instruction dependency, not bandwidth or store layout (see hw-gfx950-perf-counters for
the counters). The likely cause is v7's 19 dependent 6-step swizzle trees per step;
stochastic PC sampling has not yet confirmed which instructions stall.

## Lessons

- **Graph-capture decode before comparing.** Without graphs, FlyDSL's ~90 µs and
  Triton's ~38 µs per-launch dispatch hid the kernels entirely at small B.
- **On wave64, reduction structure dominates.** Moving from per-lane redundant trees to
  column-split waves took the gap from 2.6–3.3× to 1.5–1.7×.
- **Store layout was not the bottleneck.** A coalesced-store rewrite regressed, and the
  counters later showed 98.5% full writes already.
- **2× over FLA at B=128 needs a cheaper state format, and it must be fp16.** With fp32
  state that would take ~11 TB/s, beyond HBM peak. Under a pre-registered bar (final
  state within 1% of fp64), bf16 state storage fails long-memory probes (1.6% with
  near-unity gates) while fp16 passes every probe (worst 0.2%), at the same 2 bytes.
- **Benchmark the serving path.** FLA's public wrapper allocates a fresh state per call;
  the in-place indexed pool that serving engines use is 8–19% faster at B ≥ 32, so it is
  the bar to beat.
- **Test with realistic gates.** `-|N(0,1)|` gates decay at most ~90 bits per 64 tokens
  (p99); the real model reaches ~600, where precision shortcuts break.
- **FLA's Triton kernels fault above 2^31 elements per tensor** (e.g. prefill B=8,
  T=32768), consistent with 32-bit index overflow.
