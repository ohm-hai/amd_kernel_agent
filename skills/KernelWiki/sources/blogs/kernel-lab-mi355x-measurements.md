---
id: blog-kernel-lab-mi355x
title: "On-box measurements: MI355X counters and Kimi K3 KDA decode (kernel-lab)"
author: "kernel-lab workspace (first-party measurements on an 8x MI355X node)"
url: file:///root/amd-workspace/kernel-lab/skills/mi355x-kernel-skill/reference/08-mi355x-metric-names.md
source_category: community-note
architectures: [gfx950]
tags: [linear-attention, decode, gemv, triton, flydsl, wave64, lds, mfma]
hardware_features: [wave64, lds, mfma]
kernel_types: [linear-attention, decode, gemv]
languages: [triton, flydsl]
retrieved_at: 2026-10-09
description: "First-party measurements, not a published source: rocprofv3 counter list and calibration on gfx950, HIP device limits, and FLA vs FlyDSL KDA decode counters. Full detail in the local kernel-lab skill (url)."
---

# On-box measurements: MI355X (gfx950)

First-party measurements taken on 2026-10-09 on one GPU of an 8× MI355X node, recorded
in the kernel-lab workspace. Treat as `experimental`: they depend on the ROCm version.

## Environment

ROCm 7.0 (`rocprofv3`, rocprofiler-sdk), PyTorch 2.9.0a0, pytorch-triton-rocm 3.4.0,
flash-linear-attention (FLA) 0.5.2, FlyDSL 0.2.4.

## Hardware limits (rocprofv3 agent info and `hipGetDeviceProperties`)

| Field | Value |
|---|---|
| `Cu_Count` / `Num_Xcc` | 256 / 8 (8 XCDs × 32 CUs) |
| `Simd_Per_Cu` / `Max_Waves_Per_Simd` / `Max_Waves_Per_Cu` | 4 / 8 / 32 |
| `Wave_Front_Size` | 64 |
| `Lds_Size_In_Kb`, `sharedMemPerBlock` | 160 KB, 163840 bytes (all usable by one workgroup) |
| `l2CacheSize` | 4 MB (per XCD) |
| `regsPerBlock` | 65536 |
| `memoryBusWidth` | 8192 bits |
| `Max_Engine_Clk_Fcompute` | 2400 MHz |

## Counter list

`rocprofv3 -L` lists 687 counters on gfx950: 487 hardware counters (blocks SQ, SPI,
TCC, TCP, TA, TD, GRBM, CPC, CPF, TCA) and 200 derived. Several derived names from
older AMD docs are absent: `MemUnitBusy`, `Wavefronts`, `VALUInsts`, `SALUInsts`,
`LDSInsts`, `FlatVMemInsts`, `L2CacheHit`, `WriteUnitStalled`, `ALUStalledByLDS`,
`KernelBusy`, `TD_BUSY`.

## Calibration: FETCH_SIZE reports half the bytes read

A 1 GiB `torch.Tensor.copy_` and a 1 GiB `sum()`, each on random fp32 data:

| Counter | copy | sum |
|---|---|---|
| `FETCH_SIZE` | 0.500 GiB | 0.500 GiB |
| `WRITE_SIZE` | 1.000 GiB | 0.000 GiB |
| `TCC_EA0_RDREQ_sum` | 8,388,792 | 8,389,557 |
| `TCC_BUBBLE_sum` | 0 | 0 |
| `TCC_EA0_RDREQ_32B_sum` | 0 | 0 |

1 GiB / 128 B = 8,388,608 requests, so reads are 128-byte requests. `FETCH_SIZE` credits
128 bytes only to `TCC_BUBBLE` requests and counts the rest as 64 bytes, so it reports
half. Correct read bytes: `TCC_EA0_RDREQ_sum * 128`.

## KDA decode, Kimi K3 layer (B=128, H=96, K=V=128, fp32 state), medians of 5 dispatches

| Metric | FLA `fused_recurrent_kda_fwd_kernel` (Triton) | FlyDSL v7 `kda_decode_step_kern` |
|---|---|---|
| duration (rocprofv3) | 293.8 µs | 486.2 µs |
| bytes read / written | 849.6 MB / 808.5 MB | 850.4 MB / 824.7 MB |
| achieved bandwidth | 5.6 TB/s | 3.4 TB/s |
| full 64 B write transactions | 100.0 % | 98.5 % |
| `VGPR_Count` / `Scratch_Size` | 44 / 0 | 32 / 0 |
| `SQ_WAVES` | 196,608 | 196,608 |
| `VmemLatency` | 2,597 cycles | 1,848 cycles |
| `SQ_WAIT_INST_ANY / SQ_WAVE_CYCLES` | 27 % | 74 % |

## KDA decode at small batch (same layer, fp32 state)

Kernel-only times from torch.profiler, and wall time per step with both sides
captured in CUDA graphs (8 alternating-state steps per graph, replayed):

| B | FLA kernel | v7 kernel | FLA graphed wall | v7 graphed wall |
|---|---|---|---|---|
| 1 | 6.0 µs | 8.6 µs | 8 µs | 7 µs |
| 8 | 18.8 µs | 31.9 µs | 20 µs | 31 µs |
| 128 | 301.7 µs | 483.6 µs | 316 µs | 478 µs |

Without graphs, per-call CPU dispatch was about 90 µs for the FlyDSL launcher and about
38 µs for FLA's Triton launcher, which dominated wall time at small B. A TraceLens
report over 20 back-to-back ungraphed launches at B=8 (10 per kernel) showed the GPU
74% idle.

## Serving path and reduced-precision state (decode)

FLA's serving path (`fused_recurrent_kda_fwd`, in-place state pool indexed by
`ssm_state_indices`), graph-captured: B=1 7.5 µs, B=8 22.1 µs, B=32 66.1 µs, B=128
291.6 µs, 8–19% faster than the public wrapper at B ≥ 32.

A decode verifier with 15 gates (fp64 reference; bar fixed before testing: final state
within 1% relative RMSE, outputs within 0.8%) gave: FLA fp32 15/15; FlyDSL v7 15/15;
bf16 state storage 13/15, failing near-unity gates (1.6%) and zero-input decay from a
warm state (1.4%); fp16 state storage 15/15, worst 0.20%, 0.056% after 8192 steps.

Gate distributions, decay in bits per 64 tokens (p50 / p90 / p99): `-|N(0,1)|` gates
74 / 83 / 91; FLA's KDA parameterization scaled to the real-model p99 of ~600 bits
75 / 313 / 599.

## KDA prefill (FLA `chunk_kda`, same layer)

3.0–3.3 M tokens/s, flat over T = 1K–64K and B = 1–8. Tensors above 2^31 elements
(e.g. B=8, T=32768) triggered a GPU memory-access fault, consistent with 32-bit index
overflow, so cells were capped at 160K tokens.

## Reproduce

```bash
rocprofv3 -L > counters.txt
timeout 150 rocprofv3 --kernel-trace --pmc FETCH_SIZE --kernel-include-regex "kda" \
    -d prof/FETCH_SIZE -o run --output-format csv -- python3 bench.py
```

One derived metric per pass: a set that does not fit one pass fails with error code 38,
after which rocprofv3 hangs until killed.
