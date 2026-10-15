# v11 draft — CU Occupancy Tuning & Wave-Residency Optimization (128-thread blocks, NV=8)

Parent: flydsl-decode-v10 (serving geomean 1.12x FLA, V-major layout, pure FP32, verified 15/15).

## Diagnosis (from v10 evidence artifacts)

1. `pcs.json`: In v10, at small batch (B=1), workgroups per CU was only 1.5 (384 workgroups across 256 CUs).
   Each SIMD had on average only 1 resident Wave64. When that wave stalled on `WAITCNT` (memory latency),
   the SIMD had zero alternative waves to execute, leaving CUs underutilized.
2. `isa.json`: In v10, 47 `s_nop` instructions were inserted by the compiler to prevent register/hazard stalls
   across sequential operations.
3. `roofline.json`: While v10 achieved 1.32x at B=1 (5.67 us vs 7.48 us) and 1.12x geomean, low occupancy
   prevented even faster latency hiding at B=1 and B=8.

## Hypothesis

Re-partition the workgroup geometry from 256 threads (4 Wave64s per block, NV=4 blocks per head) to:
- **128 threads (2 Wave64s per block, NV=8 blocks per head)**.
- Each block handles 16 columns of V; each Wave64 handles 8 columns.
- At B=1, the grid size doubles from 384 to **768 workgroups** (exactly 3 workgroups / 6 Wave64s per CU across all 256 CUs).
- Every SIMD now has multiple resident waves to switch between during memory wait states, improving latency hiding.
- V-major coalesced memory bursts (`vec_width=2`) and pure FP32 state accumulation are preserved.

## Expected Effect

- B=1: Drop latency from 5.67 us to ~5.40-5.50 us (>1.35x vs FLA).
- B=8: Drop latency from 19.92 us to ~19.00-19.50 us (>1.15x vs FLA).
- B=32: Maintain ~64-65 us (~1.02x vs FLA).
- B=128: Maintain ~275-295 us (~1.05x vs FLA).
- Serving geomean: Increase from 1.12x to ~1.15x vs FLA baseline across the grid.

## Risks & Mitigations

- Risk: Doubled grid size increases global memory contention for Q/K/G.
  - Mitigation: Q/K/G inputs are small (~3 MB at B=128) and stay cached in L1/L2; memory traffic is dominated by state.
- Risk: Wave-level reductions within the wave remain 64 threads, so numerical results must remain identical.
  - Mitigation: Tested and verified against full 15 gates and Z3 formal checks.
