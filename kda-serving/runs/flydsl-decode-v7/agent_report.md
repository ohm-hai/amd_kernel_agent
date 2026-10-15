# Agent report — kda-serving / flydsl-decode-v7

Objective: 2x FLA Triton on KDA decode (Kimi K3 layer, H=96, K=V=128) on MI355X, measured on FLA's serving path; geomean over B in {1,8,32,128} >= 2.0 and no batch size slower than FLA

## Frozen verify

- **15/15 gates passed** (PASS)
- thresholds: `{'o_rrmse': 0.008, 'o_max_abs': 0.05, 'state_rrmse': 0.01, 'source': 'accuracy-bar.md (pre-registered 2026-10-09)'}`

## Hidden holdout

- 10/10 (counts only) (counts only)

## Formal verification

- **PROVED** over 19 reduction chains
- rounding points (not modeled; numeric verifier's domain): `['O store truncates to vector<4xbf16> (line 474)', 'O store truncates to vector<4xbf16> (line 477)']`
- scope: writer search bounded by the config in the report; exp/rsqrt are uninterpreted functions shared with the spec — PROVED means same math modulo that abstraction, not bitwise equality

  - [PROVED] butterfly_coverage
  - [PROVED] equivalence O b=0 h=0 i=0 j=0
  - [PROVED] equivalence O b=0 h=0 i=127 j=127
  - [PROVED] equivalence O b=1 h=3 i=64 j=5
  - [PROVED] equivalence S_OUT b=0 h=0 i=0 j=0
  - [PROVED] equivalence S_OUT b=0 h=0 i=127 j=64
  - [PROVED] equivalence S_OUT b=1 h=3 i=33 j=7

## Candidate metadata (from candidates.jsonl)

- interface: **public-wrapper; raw-launch**; state dtype: fp32; status: active
- desc: 256-thread column-split, in-wave shuffle trees, 0 barriers, vector o stores

## Roofline (frozen-verified basis; same metric both sides)

- metric `graphed_us_per_step` on `serving-pool`; state dtype `None`; verdict: **geomean 0.25x (goal 2.0x, every B >= 1.0x)**

| B | baseline µs | candidate µs | speedup | eff % | floor@baseBW µs |
|---|---|---|---|---|---|
| 1 | 7.48 | 21.64 | 0.346 | 7.4 | 7.5 |
| 8 | 22.06 | 76.61 | 0.288 | 16.6 | 22.1 |
| 32 | 65.95 | 348.12 | 0.189 | 14.6 | 66.0 |
| 128 | 293.0 | 1361.72 | 0.215 | 15.0 | 293.0 |

## ISA (gfx950)

- math/lane 202 vs data/lane 95 (ratio 2.13); issue/data 4.75; ~8 waves/SIMD; s_nop 37; s_waitcnt 49
- **issue-bound: math/data=2.13 and issue/data=4.75 — shuffles and their adjacent overhead (mov/cndmask/cmp/permlane) dominate issue slots; coalesce work per lane (wider vectors, row-block ownership) before touching memory**

## PC sampling

- 207 samples; stalls: `{'WAITCNT': 175, 'NO_INSTRUCTION_AVAILABLE': 9, 'ALU_DEPENDENCY': 4, 'OTHER_WAIT': 8, 'ARBITER_WIN_EX_STALL': 5, 'BARRIER_WAIT': 6}`
- top PCs (offset, samples): `[{'offset_from_base': 380, 'samples': 81}, {'offset_from_base': 260, 'samples': 58}, {'offset_from_base': 0, 'samples': 2}, {'offset_from_base': 48, 'samples': 2}, {'offset_from_base': 348, 'samples': 1}]`
- **WAITCNT: waiting on memory (LDS/global): overlap more work per wait, wider loads, fewer barriers; NO_INSTRUCTION_AVAILABLE: tail/occupancy: more parallel work per CU; OTHER_WAIT: misc wait: inspect top PCs**

## Artifacts

- `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v7/verify_decode_report.json`
- `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v7/formal_report.json`
- `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v7/isa.json`
- `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v7/pcs.json`
- `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v7/roofline.json`
- `/root/amd-workspace/kernel-lab/kda-serving/results.csv`

## Next hypothesis

TODO(agent): one hypothesis per report, citing the metric it attacks and by how much (guideline 04)
