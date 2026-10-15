# Draft — KDAttn optimization on MI355X (KDA² recreation, AMD edition)

Blog being recreated: https://nvlabs.github.io/kda/blog/2026-09-27-kda-for-kda/
(operator = Kimi Delta Attention; baseline = FLA chunk_kda standing in for FlashKDA;
hardware = 1× AMD MI355X gfx950 instead of NVIDIA B300).

## Status: pipeline live — baseline ACCEPTED, 3 loop iterations recorded

- `runs/baseline-fla/`: verifier **ACCEPTED 23/23 in 3.9s**; bench self-noise band
  **0.996–1.018×** (this calibrates the promote threshold); forward ≈ 2 ms per
  8192-token H64 pass; TraceLens profile: 97.9% compute / 2.1% idle (compute-bound).
- `chunk32` → rejected (0.918×). `fused-beta` → revised (0.992×, noise band).
- See `benchmark.csv` / `candidates.jsonl` for the full evidence trail.

## Operator-spec findings (mirroring the blog's hardening lessons)

1. **L2-normalization is part of the spec, not an option.** Without q,k L2N the
   delta rule is unstable — outputs explode (o_rms → inf by T≈64-128 with weak
   decay). The blog's hack A (hard-coded normalization constant) exists precisely
   because this normalization is load-bearing. Candidates MUST normalize
   (`use_qk_l2norm_in_kernel=True` or equivalent); the fp64 reference normalizes.
2. **Illegal inputs explode, not degrade.** Positive log-decay gates (g>0) make
   outputs grow exponentially — motivated G1/G2 legal-domain enforcement.
3. **Tolerance calibration:** per-element relative error is unattainable on bf16
   outputs near zero; gate = strict per-element abs (5e-2) + scale-aware rRMSE
   (8e-3, calibrated on the baseline's own T=1 boundary precision).

## Current baseline and how it is validated

- `runs/baseline-fla/` wraps `fla.ops.kda.chunk_kda` (Triton, bf16, FLA 0.5.2).
- Triton kernel confirmed working on gfx950 (MI355X): chunk vs fp64-recurrent
  reference diff ~0.004 mean abs at T=128..1024.
- Ground truth: `ref_fp64.py` fp64 recurrence (+ varlen). Verifier: `verifier/verify.py`
  (6-gate hardened acceptance). Bench: `bench/bench.py` (6-workload geomean + `--fast`).

## Key risks and unknowns

1. fp64 reference runtime at T=8192 — loop of 8192 small steps; expect ~1-2 min;
   acceptable (blog test pass was ~0.9 min). Long-state gate currently capped at T=2048.
2. Triton autotune on first call inflates bench warmup — handled (3 warmup iters).
3. Earlier observation: with **positive** gates (illegal input), outputs explode —
   motivated the G1/G2 legal-domain enforcement (blog hack-A analog).

## Candidate directions (ranked, progressive synthesis — blog §05)

1. **chunk_size sweep** (tuning, low risk): fla chunk_kda accepts chunk sizes;
   64 vs 128 changes intra-chunk GEMM shapes. Feedback via `bench --fast`.
2. **safe_gate / lower_bound knobs**: fla's numerical-hardening flags may trade
   accuracy margin for speed; measure both speed and state-rRMSE (G5).
3. **Hand-written Triton chunked KDA** targeting gfx950 (mfma, larger BK tiles,
   fused gate exp): the real "write a kernel" phase, same loop as the blog.
4. **Torch-level pre/post fusion**: fuse L2-normalization + beta sigmoid into the
   kernel epilogue/prologue.

## First concrete steps

1. Run verifier + bench on baseline → record `bench_report.json` as reference row.
2. Candidate 1: chunk-size sweep under `runs/chunk-sweep/`.
3. Record every result in `../benchmark.csv` + `../candidates.jsonl`.

## Evidence required to promote/revise/reject

- verifier_report.json = ACCEPTED (all gates)
- bench_report.json geomean > 1.05× current best (promotion), or noise-band result
  (<±3%) → revise note; any FAIL gate → reject with the failing gate detail.
- "Too good" speedups (>1.5×) must re-run with stress gates explicitly surfaced
  (blog's fake 3.28–3.74× lessons).