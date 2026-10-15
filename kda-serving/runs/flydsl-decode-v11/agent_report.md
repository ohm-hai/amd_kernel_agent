# Agent report — kda-serving / flydsl-decode-v11

Objective: 2x FLA Triton on KDA decode (Kimi K3 layer, H=96, K=V=128) on MI355X, measured on FLA's serving path; geomean over B in {1,8,32,128} >= 2.0 and no batch size slower than FLA

## Frozen verify

- **15/15 gates passed** (PASS)
- thresholds: `{'o_rrmse': 0.008, 'o_max_abs': 0.05, 'state_rrmse': 0.01, 'source': 'accuracy-bar.md (pre-registered 2026-10-09)'}`

## Hidden holdout

- 10/10 (counts only) (counts only)

## Formal verification

- **PROVED** over 19 reduction chains
- rounding points (not modeled; numeric verifier's domain): `['O store truncates to vector<4xbf16> (line 529)', 'O store truncates to vector<4xbf16> (line 532)']`
- scope: writer search bounded by the config in the report; exp/rsqrt are uninterpreted functions shared with the spec — PROVED means same math modulo that abstraction, not bitwise equality

  - [PROVED] equivalence O b=0 h=0 i=0 j=0
  - [PROVED] equivalence O b=0 h=0 i=127 j=127
  - [PROVED] equivalence O b=1 h=3 i=64 j=5
  - [PROVED] equivalence S_OUT b=0 h=0 i=0 j=0
  - [PROVED] equivalence S_OUT b=0 h=0 i=127 j=64
  - [PROVED] equivalence S_OUT b=1 h=3 i=33 j=7

## Candidate metadata (from candidates.jsonl)

- interface: **serving-pool; native**; state dtype: fp32; status: active
- desc: CU wave-residency tuning: 128-thread blocks, NV=8 (768 blocks at B=1, 3 blks/CU), V-Major coalesced bursts, pure FP32

## Roofline (frozen-verified basis; same metric both sides)

- metric `graphed_us_per_step` on `serving-pool`; state dtype `float32`; verdict: **geomean 1.11x (goal 2.0x, every B >= 1.0x)**

| B | baseline µs | candidate µs | speedup | eff % | floor@baseBW µs |
|---|---|---|---|---|---|
| 1 | 7.48 | 5.43 | 1.378 | 29.3 | 7.5 |
| 8 | 22.06 | 20.05 | 1.1 | 63.5 | 22.1 |
| 32 | 65.95 | 69.98 | 0.942 | 72.8 | 66.0 |
| 128 | 293.0 | 279.07 | 1.05 | 73.0 | 293.0 |

## ISA (gfx950)

- math/lane 198 vs data/lane 104 (ratio 1.9); issue/data 4.54; ~7 waves/SIMD; s_nop 47; s_waitcnt 51
- **issue-bound: math/data=1.90 and issue/data=4.54 — shuffles and their adjacent overhead (mov/cndmask/cmp/permlane) dominate issue slots; coalesce work per lane (wider vectors, row-block ownership) before touching memory**

## PC sampling

- 500 samples; stalls: `{'NO_INSTRUCTION_AVAILABLE': 39, 'OTHER_WAIT': 30, 'WAITCNT': 293, 'ALU_DEPENDENCY': 6, 'ARBITER_NOT_WIN': 52, 'ARBITER_WIN_EX_STALL': 25, 'BARRIER_WAIT': 55}`
- top PCs (offset, samples): `[{'offset_from_base': 24108, 'samples': 85}, {'offset_from_base': 23988, 'samples': 80}, {'offset_from_base': 24076, 'samples': 2}, {'offset_from_base': 24068, 'samples': 1}, {'offset_from_base': 24192, 'samples': 1}]`
- **WAITCNT: waiting on memory (LDS/global): overlap more work per wait, wider loads, fewer barriers; BARRIER_WAIT: sync imbalance: rebalance workgroups or remove barriers; ARBITER_NOT_WIN: inspect**

## Artifacts

- `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v11/verify_decode_report.json`
- `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v11/formal_report.json`
- `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v11/isa.json`
- `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v11/pcs.json`
- `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v11/roofline.json`
- `/root/amd-workspace/kernel-lab/kda-serving/results.csv`

## Next hypothesis

TODO(agent): one hypothesis per report, citing the metric it attacks and by how much (guideline 04)
