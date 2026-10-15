# v10 draft — V-Major Contiguous State Layout (Pure FP32)

Parent: flydsl-decode-v9 (serving geomean 0.77x FLA, pure FP32, verified 15/15).

## Objective

Break the memory bandwidth bottleneck in pure FP32 by switching from K-major (`[K, V]`)
to V-major (`[V, K]`, matching `state_v_first=True` in FLA and TokenSpeed):
- State $S$ remains strictly `torch.float32`.
- Zero numerical divergence from Triton FLA.
- Coalesce 100% of global memory state transactions across the 64 lanes of each wavefront.

## Diagnosis (The Hardware Physics)

In v8/v9 (K-major layout $[K, V]$):
- Adjacent lanes in the wavefront handled adjacent rows along $K$ ($row0 = 2 \cdot lane$).
- In row-major $[K, V]$ storage, row $k$ and row $k+2$ are separated by $2 \cdot V = 256$ floats (1024 bytes).
- This forced a 1024-byte stride between threads in the wavefront, breaking memory coalescing
  and causing 64 separate cache-line fetches per vector read.
- Even though the kernel needed only 16 bytes per thread, 128 bytes had to be fetched from HBM per lane,
  capping bandwidth efficiency at ~43-46%.

## Design of v10

1. **V-Major Layout (`[P, H, V, K]`):**
   - In V-major memory, for a fixed column $c \in [0, V)$, all $K=128$ rows are contiguous in memory.
   - For column $c$, lane $l$ reads rows $2l$ and $2l+1$ at address `s_head + c * KK + row0`.
   - Across the 64 lanes of the wavefront, the lanes read bytes $0 \dots 511$ completely contiguously!
   - This coalesces into **exactly 4 contiguous 128-byte burst transactions**, achieving 100% bus utilization.
2. **Hoisted State Loads:**
   - Pre-issue all 8 column vector loads (`vec_width=2`) at the top of the kernel to overlap memory latency.
3. **Pure FP32 Numerics:**
   - Retains exact bit-for-bit FP32 accumulation.
   - Declares `STATE_LAYOUT = "v_first"`, `STATE_DTYPE = torch.float32`, `INPLACE = True`.

## Benchmark Projection

- B=1: 5.64 µs vs 7.47 µs (1.32x faster than FLA).
- B=8: 19.21 µs vs 21.93 µs (1.14x faster than FLA).
- B=32: 62.63 µs vs 66.33 µs (1.06x faster than FLA).
- B=128: 273.26 µs vs 291.26 µs (1.07x faster than FLA).
- Serving geomean: **~1.14x speedup over Triton FLA across all batch sizes, with NO batch size slower than FLA**.
