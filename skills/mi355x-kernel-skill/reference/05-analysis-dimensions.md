# Six Analysis Dimensions (mapped to AMD/MI355X metrics)

Adapted from kda/skills/ncu-report-skill/reference/05 (B200/sm_100). AMD analogs via
torch-profiler kernel times, TraceLens gpu_timeline, and rocprofv2 counters.

## 1. Occupancy & launch geometry
- Question: grid big enough? regs/LDS limiting waves?
- Read: grid size vs 256 CUs (256 thr/CT -> 1 CT/SIMD when grid < ~1024 CTs);
  kernel registers from rocprofv3 columns `VGPR_Count` / `SGPR_Count` (reference/08);
  LDS is 160 KB per CU and per workgroup (HIP `sharedMemPerBlock` = 163840).
- Our data: B=1 decode = 384 CTs of 256 thr = ~1.5 waves/CU -> latency-exposed.
  This is WHY small-B is a latency problem, not a throughput problem.

## 2. Memory bandwidth & sectors
- Question: achieved TB/s vs 8 TB/s peak; sector efficiency of state r/w.
- Read: (kernel time, state bytes) -> TB/s. graphed-FLA B=128: 0.316ms for
  1.6GB fp32 round-trip = 5.1 TB/s (75% of peak — near-optimal for this pattern).
- Our v6/v7: 3.2-3.3 TB/s -> 1.6x headroom via dtype (v8 bf16/fp16 state) and
  possibly store layout.

## 3. Dispatch/launch overhead (AMD-specific dimension — the one we under-weighted)
- Question: wall vs kernel time gap.
- Read: raw wall loop vs torch.profiler kernel sum. FlyDSL ~90us, Triton ~38us,
  wrapper 47-133us. TraceLens gpu_timeline idle% = the smoking gun (74% at B=8!).
- Fix: CUDA graphs (serving-realistic); both sides graphed for fairness.

## 4. Compute utilization (VALU vs MFMA)
- Question: are we FLOP-bound? (Almost never for attention-state kernels here.)
- Read: useful FLOPs / kernel time vs 157 TFLOP/s fp32 (VALU) or MFMA peak.
- Our decode: ~4 GFLOP/step @B=128 vs 496us -> 0.8% of peak. Never the wall.

## 5. Latency chains & barriers
- Question: dependent-chain depth vs hidden parallelism.
- Read: barriers/token (v2:16 -> v4:6 -> v5/v6:0 via wavefront shuffle trees);
  dependent swizzle steps; load->use distance.
- Trick: issue state loads BEFORE norm-tree math (no dependency) so loads fly
  during compute (v8 candidate micro-opt).

## 6. Numerics / precision
- Question: is speed from precision cheat? (blog hack-A/E)
- Read: fp64-reference rel-RMSE vs context length; state dtype.
- Rule: any speedup >1.5x gets stress-probed (deep decay, near-unity gates,
  boundary lengths) via kda-attn verifier before promotion.
