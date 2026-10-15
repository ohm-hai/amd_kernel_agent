---
id: pattern-low-sm-utilization
title: "Low SM Utilization"
type: pattern
tags: [persistent-kernel, clc, tile-scheduling]
symptoms: [low-sm-utilization, tail-effect, load-imbalance]
candidate_techniques: [technique-persistent-kernels, technique-tile-scheduling, hw-clc]
related: [pattern-tail-effect, pattern-compute-bound, hw-gfx950-perf-counters]
sources: [doc-nvidia-tuning-guide, blog-tcgen05-tutorial, pr-cutlass-2161, blog-kernel-lab-mi355x]
---

## Symptom

SM utilization below 60% despite sufficient occupancy. Nsight Compute shows idle SMs during portions of kernel execution.

## Likely Causes

1. **Tail effect**: Last wave of tiles leaves most SMs idle (see [tail-effect](tail-effect.md))
2. **Load imbalance**: Some tiles take longer than others (variable computation per tile)
3. **Static scheduling**: Fixed tile-to-SM assignment doesn't adapt to runtime conditions
4. **Grid too small**: Fewer threadblocks than SMs

## Candidate Techniques

| Technique | Applicability | Effect |
|---|---|---|
| [CLC](../hardware/clc.md) | SM100 only | Dynamic tile assignment, eliminates load imbalance |
| [Persistent kernels](../techniques/persistent-kernels.md) | SM90+ | Eliminates tail effect, one-time launch overhead |
| [Tile scheduling](../techniques/tile-scheduling.md) | SM90+ | Better L2 locality, reduce load variance |

## Examples

```
// tcgen05 tutorial progression:
// Without persistent/CLC: 86% of cuBLAS (some SMs idle at wave boundaries)
// With persistent + CLC:  98% of cuBLAS (all SMs stay busy)
```

## Caveats
- CLC only available on SM100 datacenter GPUs (not SM120 consumer)
- Persistent kernels complicate debugging and profiling
- For non-persistent kernels, ensure grid size >> SM count

## On AMD gfx950 (MI355X)

Appended 2026-10-09 from first-party measurements (blog-kernel-lab-mi355x).

- **Grid size:** 256 CUs across 8 XCDs. Compare workgroups (`Grid_Size / Workgroup_Size`;
  `Grid_Size` counts work-items) with `CU_NUM`. Idle CUs show as `MeanOccupancyPerCU`
  well under `MeanOccupancyPerActiveCU`.
- **Dispatch-bound short kernels:** on small-batch decode, TraceLens's GPU timeline showed
  74% idle time. Per-launch CPU dispatch (~90 µs FlyDSL, ~38 µs Triton) far exceeded
  kernel time (6–9 µs at B=1); with graph capture both kernels ran at 7–8 µs per step.
- **No CLC equivalent:** persistent kernels on gfx950 schedule work with atomic queues.
