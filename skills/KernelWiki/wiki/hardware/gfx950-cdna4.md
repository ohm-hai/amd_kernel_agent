---
id: hw-gfx950-cdna4
title: "AMD gfx950 (MI355X / CDNA4) Compute Architecture"
type: hardware
architectures: [gfx950]
tags: [wave64, lds, mfma, dpp]
confidence: source-reported
related: [hw-gfx950-perf-counters, migration-sm100-to-gfx950, kernel-kda-decode-mi355x, pattern-memory-bound]
sources: [blog-kernel-lab-mi355x, blog-amd-primus-moe]
aliases: ["MI355X", "MI350X", "CDNA4", "gfx950"]
evidence_basis:
  - source_id: blog-kernel-lab-mi355x
    evidence_type: benchmark
  - source_id: blog-kernel-lab-mi355x
    evidence_type: reproduction
blackwell_relevance: "Lets Blackwell-tuned kernels and profiling habits be ported to AMD's current Instinct generation; see migration-sm100-to-gfx950 for the concept map."
---

## Overview

gfx950 is the ISA target of AMD Instinct MI355X / MI350X (CDNA4). Facts below come from
rocprofv3 agent info and `hipGetDeviceProperties` on an MI355X (blog-kernel-lab-mi355x),
except memory capacity and bandwidth, which come from AMD (blog-amd-primus-moe).

## Compute hierarchy

| Level | gfx950 | Blackwell (B200) counterpart |
|---|---|---|
| Die | 8 XCDs × 32 CUs = 256 CUs | 148 SMs |
| Compute unit | 4 SIMDs per CU | 4 SM sub-partitions |
| Scheduling unit | wavefront of **64** lanes | warp of 32 |
| Resident waves | up to 8 per SIMD, 32 per CU | up to 64 warps per SM |
| Matrix units | MFMA (BF16, F16, F32, F8, F6/F4, I8, XF32 variants) | tcgen05 tensor cores |
| Clock | up to 2400 MHz | |

## Memory hierarchy

| Level | gfx950 |
|---|---|
| Registers | `regsPerBlock` = 65536; architectural VGPRs plus accumulation VGPRs (AGPRs) for MFMA |
| LDS | **160 KB per CU, all allocatable by one workgroup** (`sharedMemPerBlock` = 163840). MI300X (CDNA3) had 64 KB. |
| vL1D | 32 KB per CU |
| L2 | 4 MB per XCD (32 MB total), not shared across XCDs |
| Infinity Cache | 256 MB, shared by all XCDs |
| HBM3E | 288 GB, up to 8 TB/s (AMD); 8192-bit bus (device query) |

L2-to-fabric transactions measured with rocprofv3: reads are 128-byte requests, writes
are 64-byte or 32-byte transactions.

## What this means for kernels

- **Wave64 changes reduction and divergence math.** An in-wave tree takes 6 steps, and
  a divergent branch idles 64 lanes, not 32. Lane exchange inside a wave uses
  `ds_swizzle` / `ds_bpermute` (LDS pipe) or DPP (`row_shr`, `row_bcast` on
  gfx9-class GPUs; there is no xor-mask DPP mode).
- **Occupancy is rarely register-limited for small kernels.** The cap is 8 waves per
  SIMD; kernels at 32–44 VGPRs measured with zero scratch.
- **No TMA or tensor memory.** Asynchronous movement uses early buffer loads and
  direct-to-LDS buffer loads; the large LDS allows deeper staging than CDNA3.
- **MFMA pads M to 16 or 32**, so GEMV-shaped work (decode with M ≤ 2) is usually faster
  on VALU.
- **Grids under ~256 workgroups leave CUs idle**; grids under ~512 leave little room to
  hide latency.

## Profiling

Use rocprofv3 hardware counters (hw-gfx950-perf-counters) together with framework-level
trace analysis such as TraceLens (blog-amd-tracelens).
