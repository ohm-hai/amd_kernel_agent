# Measured Baselines (MI355X, this box, GPU 7) — as of 2026-10-09

## Kimi K3 KDA layer (H=96, K=V=128; from Primus kimi_k3.yaml)
Decode (FLA fused_recurrent, fp32 state, graphed wall both sides):
  B=1: 8us | B=8: 20us | B=128: 316us
Decode B=128 hardware counters (rocprofv3; method in 08-mi355x-metric-names.md):
  FLA kernel 293.8us, reads 849.6 MB + writes 808.5 MB = 5.6 TB/s (~70% of 8 TB/s)
  v7 kernel  486.2us, reads 850.4 MB + writes 824.7 MB = 3.4 TB/s
  Same bytes and waves; v7 waits to issue 74% of wave lifetime vs FLA 27%.
Prefill (FLA chunk_kda): ~3.0-3.3M tok/s flat over T=1K-64K, B=1-8.
  Chunked prefill (8x8K state carry) = 1.005x monolithic (chunking ~free).

## Our FlyDSL decode kernel (v7 = v6 + store fix; state rRMSE 7.8e-08 fp64-verified)
Kernel-only: B=1 8.6us | B=8 31.9us | B=128 483.6us = 1.45/1.70/1.60x of FLA.
Graphed wall: B=1 7us (1.06x — beats graphed-FLA) | B=2 0.86x | B=4-128 0.58-0.71x.

## Serving path and correctness (2026-10-09, GPU 4)

FLA serving path (`fused_recurrent_kda_fwd`, in-place indexed pool, graphed):
  B=1 7.5us | B=8 22.1us | B=32 66.1us | B=128 291.6us (8-19% faster than the public
  wrapper at B>=32). This is the reference for the 2x target.
TraceLens (MI355X spec + KDA extension), B=128 fp32 state: FLA 5.41 TB/s = 67.6% of
  roofline; v7 3.34 TB/s = 41.7%. fp32 roofline floor 203us.
Decode verifier (15 gates, pre-registered bar): FLA 15/15, v7 15/15; bf16-state
  emulation 13/15 (1.6% on long-memory gates); fp16-state emulation 15/15 (worst 2.0e-3).
Max-achievable FLOPs (measured, tools/measure_maf.py): bf16 matrix 1344 TFLOPS, fp8 3228,
  fp32 vector 129. torch copy_ 4.68 TB/s.

## Version lineage (evidence in kda-serving/history/benchmark.v0.csv; new rows in results.csv)
v1 smem+barriers 0.10-0.31x -> v2 vec4/CHUNK=32 0.22-0.50x -> v3 coalesced-stores
REJECTED (regression, store theory wrong) -> v4 FLA-structure port ~=v2 -> v5 shuffle
trees -> v6 256-thr column-split ZERO barriers 9.1/33/496us -> v7 graphs+store fix.
Dispatch: FlyDSL ~90us vs Triton ~38us vs our wrapper 47-133us (all measured).

## Physical ceilings (why 2x needs dtype change at large B)
B=128 fp32 state step moves 1.66 GB (measured). 2x of FLA's kernel (147us) would need
11.3 TB/s with fp32 state > 8 TB/s peak: IMPOSSIBLE. With bf16/fp16 state (~0.83 GB)
it needs ~5.6 TB/s, which is exactly FLA's measured efficiency today, so it is
feasible only if v8 matches FLA's bandwidth (v7 reaches 3.4 TB/s). Accuracy tradeoff
documented in the v8 report (fp16 preferred: same bytes, 10 vs 7 mantissa bits).
Earlier estimates in this file (5.1 TB/s, 1.6 GB) were computed from assumed traffic;
the counter-based numbers above replace them.

## Toolchain pins (working set)
ROCm 7, torch 2.9.0a0+git, pytorch-triton-rocm 3.4.0, FlyDSL 0.2.4, FLA 0.5.2,
TraceLens 0.1.0.dev, primus 26.7, primus-turbo 0.4.0 (torch-2.9-compatible build).
KNOWN BROKEN: torch.compile/inductor, aiter JIT (ck_tile mismatch), fp8 path untested.
