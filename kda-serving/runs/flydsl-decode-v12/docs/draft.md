# v12 draft — Adaptive CU Geometry & L1/L2 Spatial Affinity Dispatch (Pure FP32)

Parent: flydsl-decode-v11 (B=1 1.38x FLA, B=8 1.10x FLA) and flydsl-decode-v10 (B=32 1.01x FLA, B=128 1.07x FLA).

## Diagnosis (The Hardware Physics of CUs and Caches on MI355X)

Across our empirical evaluations in `v10` and `v11`, we uncovered a fundamental trade-off between wave residency and L1 cache sharing:

1. **Small Batch Regime (B <= 8) — CU Under-Occupancy:**
   - At B=1, a 256-thread block (NV=4) yields only 384 workgroups across 256 CUs (1.5 workgroups/CU).
   - Half of the CUs get only 1 workgroup (4 Wave64s), leaving each SIMD with only 1 resident wave.
     When that wave stalls on HBM memory latency (`WAITCNT`), the SIMD has zero alternative waves to execute.
   - Using 128-thread blocks (NV=8) doubles the grid to 768 workgroups (exactly 3 workgroups / 6 Wave64s per CU),
     giving every SIMD multiple resident waves to interleave, dropping B=1 latency to **5.43 µs (1.38x vs FLA)**.

2. **Large Batch Regime (B >= 32) — L1 Cache Sharing Across the Wave Group:**
   - At B >= 32, the GPU is already saturated with tens of thousands of blocks.
   - For a given head `nh`, all column tiles share the exact same Q, K, G inputs.
   - In a 256-thread block (NV=4), all 4 Wave64s (cols 0..7, 8..15, 16..23, 24..31) run in the **same workgroup on the same CU**,
     sharing the CU's private 32 KB vL1D vector cache. Q, K, G loaded by Wave 0 are hit in L1 cache by Waves 1..3.
   - In 128-thread blocks (NV=8), tiles are split across different CUs, duplicating L1 cache misses and inflating L2/L3 fabric traffic.
   - Hence, NV=4 achieved **275.2 µs (74% HBM peak bandwidth)** at B=128 vs NV=8's 279.1 µs.

## Design of v12

1. **Adaptive Dual-Geometry Dispatch in `decode_pool`:**
   - **B <= 8:** Dispatch the 128-thread / NV=8 kernel (768 blocks at B=1, 6,144 blocks at B=8),
     saturating all 256 CUs across all 8 XCDs with multiple resident waves per SIMD.
   - **B >= 32:** Dispatch the 256-thread / NV=4 kernel (12,288 blocks at B=32, 49,152 blocks at B=128),
     maximizing CU 32 KB L1 vector cache hits for Q, K, G and minimizing cross-die Infinity Fabric traffic.
2. **Coalesced V-Major Layout (`[P, H, V, K]`):**
   - 100% contiguous 512-byte burst state memory access.
3. **Pure FP32 Numerics:**
   - Exact bit-for-bit FP32 state accumulation and verification.
   - `STATE_DTYPE = torch.float32`, `INPLACE = True`, `STATE_LAYOUT = "v_first"`.

## Expected Effect

- B=1: ~5.40-5.60 µs (>=1.33x vs FLA).
- B=8: ~19.30-19.80 µs (>=1.11x vs FLA).
- B=32: ~62.50-65.00 µs (>=1.05x vs FLA).
- B=128: ~274.00-276.00 µs (>=1.07x vs FLA).
- Serving geomean: **~1.14x-1.16x faster than FLA across all batch sizes, with NO batch size slower than FLA**.
