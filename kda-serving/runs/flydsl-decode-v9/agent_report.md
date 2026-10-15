# Agent report — kda-serving / flydsl-decode-v9

Objective: 2x FLA Triton on KDA decode (Kimi K3 layer, H=96, K=V=128) on MI355X, measured on FLA's serving path; geomean over B in {1,8,32,128} >= 2.0 and no batch size slower than FLA

## Frozen verify

- **15/15 gates passed** (PASS)
- thresholds: `{'o_rrmse': 0.008, 'o_max_abs': 0.05, 'state_rrmse': 0.01, 'source': 'accuracy-bar.md (pre-registered 2026-10-09)'}`

## Hidden holdout

- 10/10 (counts only) (counts only)

## Formal verification

- **PROVED** over 19 reduction chains
- rounding points (not modeled; numeric verifier's domain): `['O store truncates to vector<4xbf16> (line 486)', 'O store truncates to vector<4xbf16> (line 489)']`
- scope: writer search bounded by the config in the report; exp/rsqrt are uninterpreted functions shared with the spec — PROVED means same math modulo that abstraction, not bitwise equality

  - [PROVED] butterfly_coverage
  - [PROVED] equivalence O b=0 h=0 i=0 j=0
  - [PROVED] equivalence O b=0 h=0 i=127 j=127
  - [PROVED] equivalence O b=1 h=3 i=64 j=5
  - [PROVED] equivalence S_OUT b=0 h=0 i=0 j=0
  - [PROVED] equivalence S_OUT b=0 h=0 i=127 j=64
  - [PROVED] equivalence S_OUT b=1 h=3 i=33 j=7

## Candidate metadata (from candidates.jsonl)

- interface: **serving-pool; native**; state dtype: fp32; status: active
- desc: Pure FP32: hoisted state loads, interleaved reduction trees, and vector FMA state store

## Roofline (frozen-verified basis; same metric both sides)

- metric `graphed_us_per_step` on `serving-pool`; state dtype `float32`; verdict: **geomean 0.77x (goal 2.0x, every B >= 1.0x)**

| B | baseline µs | candidate µs | speedup | eff % | floor@baseBW µs |
|---|---|---|---|---|---|
| 1 | 7.48 | 7.13 | 1.05 | 22.3 | 7.5 |
| 8 | 22.06 | 29.35 | 0.752 | 43.4 | 22.1 |
| 32 | 65.95 | 97.45 | 0.677 | 52.3 | 66.0 |
| 128 | 293.0 | 442.5 | 0.662 | 46.0 | 293.0 |

## ISA (gfx950)

- math/lane 201 vs data/lane 96 (ratio 2.09); issue/data 4.73; ~8 waves/SIMD; s_nop 40; s_waitcnt 54
- **issue-bound: math/data=2.09 and issue/data=4.73 — shuffles and their adjacent overhead (mov/cndmask/cmp/permlane) dominate issue slots; coalesce work per lane (wider vectors, row-block ownership) before touching memory**

## PC sampling

- 592 samples; stalls: `{'OTHER_WAIT': 70, 'NO_INSTRUCTION_AVAILABLE': 27, 'BARRIER_WAIT': 42, 'WAITCNT': 327, 'ALU_DEPENDENCY': 25, 'ARBITER_WIN_EX_STALL': 34, 'ARBITER_NOT_WIN': 64, 'INTERNAL_INSTRUCTION': 3}`
- top PCs (offset, samples): `[{'offset_from_base': 380, 'samples': 92}, {'offset_from_base': 260, 'samples': 83}, {'offset_from_base': 48, 'samples': 3}, {'offset_from_base': 340, 'samples': 2}, {'offset_from_base': 412, 'samples': 1}]`
- **WAITCNT: waiting on memory (LDS/global): overlap more work per wait, wider loads, fewer barriers; OTHER_WAIT: misc wait: inspect top PCs; ARBITER_NOT_WIN: inspect**

## Artifacts

- `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v9/verify_decode_report.json`
- `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v9/formal_report.json`
- `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v9/isa.json`
- `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v9/pcs.json`
- `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v9/roofline.json`
- `/root/amd-workspace/kernel-lab/kda-serving/results.csv`

## Next hypothesis

TODO(agent): one hypothesis per report, citing the metric it attacks and by how much (guideline 04)
