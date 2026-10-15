# v9 draft — Pure FP32: Hoisted State Loads, Interleaved Reduction Trees, and Vector FMA

Parent: flydsl-decode-v8 (formal PROVED, frozen verify 15/15, holdout 10/10, serving-path geomean 0.74x FLA).

## Objective

Improve pure FP32 decode performance on MI355X without ANY numerical divergence from Triton FLA:
- State $S$ remains strictly `torch.float32`.
- All recurrence math remains strictly FP32.
- Zero numerical divergence / zero drift over infinite horizon.

## Diagnosis (from v8 evidence artifacts)

1. `pcs.json`: In v8, 57% of execution time was stalled on `WAITCNT` (waiting on memory loads).
   State loads occurred deep inside the kernel (after Q/K/G reduction and rsqrt), serializing HBM latency with kernel compute.
2. `isa.json`: In v8, 19 sequential butterfly reduction trees caused 114 serialized `ds_swizzle` operations with 49 `s_waitcnt` and 37 `s_nop` stalls.
3. State store math extracted scalar lanes and rebuilt vectors element-by-element with 16 insert operations per half.

## Changes in v9

1. **Hoist State Loads to Top:**
   Issue all 4 `buffer_load(s_r, ..., vec_width=4)` immediately after loading the sequence `slot`.
   Overlap HBM memory fetch latency with all input loads ($q, k, g, \beta$), norm calculations ($q^2, k^2, qk$), and decay exponent calculations ($\exp(g)$).
2. **Interleaved Multi-Tree Reductions:**
   Instead of reducing 8 `kv` columns sequentially and 8 `od` columns sequentially (16 sequential 6-step dependency chains):
   - Reduce `sq, sk, qk` together (3-wide interleaved).
   - Reduce `kv[0..7]` together (8-wide interleaved).
   - Reduce `od[0..7]` together (8-wide interleaved).
   This issues batches of independent shuffles into the LDS/VALU pipe, eliminating dependency stalls between shuffle steps.
3. **Vectorized FMA State Updates:**
   Pack the 4 update values into `vector<4xf32>` via `mlir_vector.from_elements` and perform vector multiply-add (`addf(base, mulf(bk_vec, u_vec))`) directly into `buffer_store`, eliminating 16 scalar insert instructions per thread.

## Expected Effect

- B=1: Maintain >1.0x lead over FLA (measured ~6.8-7.1 us).
- B=8: Drop from 30.9 us to ~28.9 us (1.06x faster).
- B=32: Drop from 102.6 us to ~94.5 us (1.08x faster).
- B=128: Drop from 468.0 us to ~442.3 us (1.07x faster).
- Serving geomean: Increase from 0.74x to ~0.77-0.78x in pure FP32.
