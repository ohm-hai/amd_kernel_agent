# Agent report — kda-serving / flydsl-decode-v12

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
- desc: Adaptive CU geometry: NV=8 (128-thr, 768 blks) for B<=8; NV=4 (256-thr, L1 vL1D sharing) for B>=32; V-Major bursts, pure FP32

## Roofline (frozen-verified basis; same metric both sides)

- metric `graphed_us_per_step` on `serving-pool`; state dtype `float32`; verdict: **geomean 1.09x (goal 2.0x, every B >= 1.0x)**

| B | baseline µs | candidate µs | speedup | eff % | floor@baseBW µs |
|---|---|---|---|---|---|
| 1 | 7.48 | 5.43 | 1.378 | 29.3 | 7.5 |
| 8 | 22.06 | 20.08 | 1.099 | 63.4 | 22.1 |
| 32 | 65.95 | 73.61 | 0.896 | 69.2 | 66.0 |
| 128 | 293.0 | 277.44 | 1.056 | 73.4 | 293.0 |

## ISA (gfx950)

- math/lane 198 vs data/lane 104 (ratio 1.9); issue/data 4.54; ~7 waves/SIMD; s_nop 47; s_waitcnt 51
- **issue-bound: math/data=1.90 and issue/data=4.54 — shuffles and their adjacent overhead (mov/cndmask/cmp/permlane) dominate issue slots; coalesce work per lane (wider vectors, row-block ownership) before touching memory**

## PC sampling

- 568 samples; stalls: `{'NO_INSTRUCTION_AVAILABLE': 28, 'OTHER_WAIT': 45, 'WAITCNT': 343, 'INTERNAL_INSTRUCTION': 3, 'ALU_DEPENDENCY': 9, 'ARBITER_NOT_WIN': 63, 'ARBITER_WIN_EX_STALL': 30, 'BARRIER_WAIT': 47}`
- top PCs (offset, samples): `[{'offset_from_base': 380, 'samples': 89}, {'offset_from_base': 260, 'samples': 61}, {'offset_from_base': 48, 'samples': 7}, {'offset_from_base': 0, 'samples': 3}, {'offset_from_base': 492, 'samples': 1}]`
- **WAITCNT: waiting on memory (LDS/global): overlap more work per wait, wider loads, fewer barriers; ARBITER_NOT_WIN: inspect; BARRIER_WAIT: sync imbalance: rebalance workgroups or remove barriers**

## Artifacts

- `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v12/verify_decode_report.json`
- `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v12/formal_report.json`
- `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v12/isa.json`
- `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v12/pcs.json`
- `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v12/roofline.json`
- `/root/amd-workspace/kernel-lab/kda-serving/results.csv`

## Next hypothesis

TODO(agent): one hypothesis per report, citing the metric it attacks and by how much (guideline 04)
