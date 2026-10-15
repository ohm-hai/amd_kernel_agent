---
id: pattern-memory-bound
title: "Memory Bandwidth Bound"
type: pattern
tags: [vectorized-loads, cache-policy, shared-memory-optimization]
symptoms: [memory-bound, low-compute-utilization, high-memory-throughput]
candidate_techniques: [technique-vectorized-loads, technique-swizzling, technique-pipeline-stages]
related: [pattern-compute-bound, kernel-nvfp4-gemv, hw-gfx950-perf-counters]
sources: [blog-yue-nvfp4, blog-amandeep-nvfp4, doc-nvidia-tuning-guide, blog-kernel-lab-mi355x]
---

## Symptom

Nsight Compute shows high DRAM throughput but low tensor core utilization. Arithmetic intensity below the roofline knee point.

## Likely Causes

1. **Low arithmetic intensity**: Operations like GEMV, small batch decode, or reduction kernels
2. **Poor data reuse**: Each data element used only once
3. **Inefficient memory access**: Uncoalesced loads, L1 cache thrashing

## Candidate Techniques

| Technique | Effect |
|---|---|
| [Vectorized loads](../techniques/vectorized-loads.md) | 128/256-bit loads maximize bandwidth utilization |
| [Cache policies](../techniques/vectorized-loads.md) | L1::no_allocate for streaming, L1::evict_last for reuse |
| [Register budgeting](../techniques/vectorized-loads.md) | -maxrregcount increases occupancy |
| [TMA multicast](../hardware/tma.md) | Share loaded data across SMs in cluster |
| [Swizzling](../techniques/swizzling.md) | Eliminate bank conflicts in shared memory |

## Examples

```cuda
// NVFP4 GEMV: memory-bound optimization
// Key insight: profile FIRST to confirm memory-bound behavior
// "The single most important thing could have been running Nsight Compute"
// — Amandeep (12 Attempts at an FP4 Kernel)

// Optimization priorities for memory-bound kernels:
// 1. Maximize memory bandwidth (wide loads, coalescing)
// 2. Reduce register count (higher occupancy)
// 3. Differentiate cache policies per access pattern
// 4. DON'T optimize compute (it's not the bottleneck)
```

## Caveats
- Always profile before optimizing — wrong assumption wastes effort
- B200 has 8 TB/s bandwidth; speed-of-light calculation determines achievable performance
- ILP and compute optimizations have diminishing returns for memory-bound kernels

## On AMD gfx950 (MI355X)

Appended 2026-10-09 from first-party measurements (blog-kernel-lab-mi355x).

- **Measure bytes correctly.** Achieved bandwidth = (`TCC_EA0_RDREQ_sum * 128` +
  `WRITE_SIZE * 1024`) / kernel duration, against ~8 TB/s peak. Do not use `FETCH_SIZE`
  (reports half the bytes read on gfx950) or `BANDWIDTH_EA` (wrong formula); see
  hw-gfx950-perf-counters.
- **HBM back-pressure:** `TCC_EA0_RDREQ_DRAM_CREDIT_STALL_sum` and
  `TCC_EA0_WRREQ_DRAM_CREDIT_STALL_sum`.
- **Write efficiency:** `TCC_EA0_WRREQ_64B_sum / TCC_EA0_WRREQ_sum` (share of full 64 B
  writes).
- **Same bytes is not the same bound.** In KDA decode at B=128, FLA reaches 5.6 TB/s
  (~70% of peak) while a FlyDSL kernel moving the same 1.66 GB reaches 3.4 TB/s; the
  slower one was issue-bound, not memory-bound (pattern-pipeline-stalls,
  kernel-kda-decode-mi355x).
