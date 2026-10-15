---
id: hw-gfx950-perf-counters
title: "gfx950 Performance Counters with rocprofv3 (Nsight Compute equivalents and traps)"
type: hardware
architectures: [gfx950]
tags: [wave64, lds, mfma]
confidence: experimental
reproducibility: runnable
related: [hw-gfx950-cdna4, migration-sm100-to-gfx950, kernel-kda-decode-mi355x, pattern-memory-bound, pattern-pipeline-stalls]
sources: [blog-kernel-lab-mi355x, blog-amd-tracelens]
aliases: ["rocprofv3 counters", "AMD performance counters", "rocprof metrics"]
evidence_basis:
  - source_id: blog-kernel-lab-mi355x
    evidence_type: benchmark
  - source_id: blog-kernel-lab-mi355x
    evidence_type: reproduction
blackwell_relevance: "AMD counterpart of the Nsight Compute metric workflow used throughout this wiki; maps Nsight metric names to gfx950 counters."
---

## Overview

On gfx950 with ROCm 7.0, `rocprofv3` exposes 687 counters (487 hardware, 200 derived).
`rocprofv3 -L` lists them, replacing the old `--list-basic` / `--list-derived`. Behavior
below was measured on MI355X (blog-kernel-lab-mi355x) and may change between ROCm
releases, so re-check after upgrades.

## Nsight Compute → rocprofv3

| Nsight Compute | rocprofv3 on gfx950 |
|---|---|
| `gpu__time_duration.sum` | kernel trace `End_Timestamp - Start_Timestamp` |
| `launch__registers_per_thread` | CSV columns `VGPR_Count`, `Accum_VGPR_Count`, `SGPR_Count` |
| `launch__shared_mem_per_block` | column `LDS_Block_Size` |
| local ld/st (spill) | column `Scratch_Size` > 0 |
| `launch__occupancy_limit_*` | `SPI_RA_VGPR_SIMD_FULL_CSN`, `SPI_RA_LDS_CU_FULL_CSN`, `SPI_RA_WAVE_SIMD_FULL_CSN`, `SPI_RA_BAR_CU_FULL_CSN` |
| achieved occupancy | `OccupancyPercent`, `MeanOccupancyPerActiveCU` |
| `dram__bytes_read.sum` | `TCC_EA0_RDREQ_sum * 128` (not `FETCH_SIZE`) |
| `dram__bytes_write.sum` | `WRITE_SIZE * 1024` |
| L2 hit rate | `TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum)` |
| store bytes per sector | `TCC_EA0_WRREQ_64B_sum / TCC_EA0_WRREQ_sum` |
| global ld/st instruction counts | `SQ_INSTS_VMEM_RD`, `SQ_INSTS_VMEM_WR` |
| shared-memory bank conflicts | `SQ_LDS_BANK_CONFLICT`, `LDSBankConflict` |
| tensor-pipe activity | `SQ_VALU_MFMA_BUSY_CYCLES`, `MfmaUtil` |
| stall-reason ratios | no 1:1: `VmemLatency` plus `SQ_WAIT_INST_ANY / SQ_WAVE_CYCLES`, then PC sampling |
| per-PC source counters | `--pc-sampling-beta-enabled --pc-sampling-method stochastic --pc-sampling-unit cycles` |

## Traps (all measured)

1. **`FETCH_SIZE` reports half the bytes read.** Reads are 128 B requests, but the
   formula credits 128 B only to `TCC_BUBBLE` requests, which stay at 0. A 1 GiB copy
   reported `FETCH_SIZE` = 0.500 GiB with `TCC_EA0_RDREQ_sum` = 8,388,792.
2. **`BANDWIDTH_EA` is wrong** for the same reason: its expression subtracts read
   requests when `TCC_BUBBLE_sum` is 0.
3. **Error code 38** ("Request exceeds the capabilities of the hardware to collect") when
   a `--pmc` set doesn't fit one pass; rocprofv3 then **hangs**. Put one derived metric
   per pass and wrap runs in `timeout`.
4. **Quad-cycle units.** `SQ_WAVE_CYCLES`, `SQ_WAIT_INST_ANY`, `SQ_ACTIVE_INST_ANY` are in
   quad-cycles; `SQ_BUSY_CYCLES` and `GRBM_GUI_ACTIVE` in clocks.
5. **`Grid_Size` counts work-items**, not workgroups.
6. Older derived names (`MemUnitBusy`, `L2CacheHit`, `VALUInsts`, `Wavefronts`,
   `KernelBusy`, ...) don't exist; use the raw counters above.

## Collection recipe

```bash
for P in "FETCH_SIZE" "WRITE_SIZE" "TCC_HIT_sum TCC_MISS_sum" "VmemLatency" \
         "SQ_WAVE_CYCLES SQ_WAIT_INST_ANY SQ_ACTIVE_INST_ANY"; do
  tag=$(echo $P | cut -d' ' -f1)
  timeout 150 rocprofv3 --kernel-trace --pmc $P --kernel-include-regex "<kernel>" \
      -d prof/$tag -o run --output-format csv -- python3 bench.py
done
```

`rocprofv3 --att` (thread trace) gives a per-instruction wave timeline on one CU, the
closest tool to Nsight's PM-sampling timelines.
