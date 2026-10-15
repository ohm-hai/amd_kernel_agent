# Independent review — kda-serving / flydsl-decode-v9

Reviewer: independent, not a participant in the session that built the framework or the candidate.
Date: 2026-10-15. Method: read-only inspection; the only file written is this `review.md`.
Evaluation re-runs executed on GPU 4 (verified 0% VRAM allocation via `rocm-smi`).

---

## VERDICT

**accept** — as a validated checkpoint, not as a promoted candidate. All 8 checks pass substantively without evidence of tampering, cheating, or reward-hacking. The candidate honestly reports its performance as **0.77x of FLA on the serving-pool metric (a loss vs the baseline, far below the 2.0x target)**. The frozen-evaluation chain is intact, the native serving pool interface is implemented directly in the GPU kernel, all 15 verifier gates and 10 holdout gates pass, and formal equivalence is PROVED across all 19 reduction chains. The candidate does not meet the promotion criteria in `contract.md` (geomean speedup $\ge 2.0\times$ and every batch size $\ge 1.0\times$), and the report does not claim promotion.

---

## 1. Workspace tamper check — PASS

- **Crate edits after build check:**
  Ran `find /root/amd-workspace/kernel-lab/framework/eval-crates -newer /root/amd-workspace/kernel-lab/framework/README.md -type f | head`.
  Files found:
  - `eval.lock.json`
  - `frozen/holdout_runner.py`
  - `frozen/holdouts.json`
  - Bytecode caches (`__pycache__/*.pyc` from python runs).
- **Timestamp analysis:**
  `framework/README.md` has mtime `2026-10-10 08:05:40 UTC`.
  The eval-crate files have mtimes `2026-10-10 08:45:25 UTC`, exactly matching the lock creation timestamp `"created": "2026-10-10T08:45:25"` when the lock was rebuilt to include cryptographic SHA-256 hashes of all `.toml` case files.
- **Lock hash verification:**
  All 14 SHA-256 hashes registered in `/root/amd-workspace/kernel-lab/framework/eval-crates/kda-serving/eval.lock.json` match the on-disk files byte-for-byte:
  - `frozen/holdout_runner.py`: `03e1ee61f61e...` (MATCH)
  - `frozen/holdouts.json`: `bb7cf41cf56f...` (MATCH)
  - `frozen/kda-attn/ref_fp64.py`: `4f9cf62597b9...` (MATCH)
  - `frozen/kda-serving/inputs.py`: `0665afbf899d...` (MATCH)
  - `frozen/kda-serving/verifier/verify_decode.py`: `6a49a61dc785...` (MATCH)
  - `frozen/tools/env_stamp.py`: `a34394df6e20...` (MATCH)
  - `bar:bar.toml`: `60f7e04c6a89...` (MATCH)
  - `cases:cases/D1_salts.toml`: `70c449fab17e...` (MATCH)
  - `cases:cases/D2_warm_state.toml`: `5578190ca80f...` (MATCH)
  - `cases:cases/D3_adversarial_ranges.toml`: `ed3e908ee094...` (MATCH)
  - `cases:cases/D3_beta_and_v_paths.toml`: `edf8f21f4976...` (MATCH)
  - `cases:cases/D4_drift.toml`: `d3ae8de50db4...` (MATCH)
  - `cases:cases/D5_shapes.toml`: `fe08ea6d478a...` (MATCH)
  - `cases:cases/H1_holdout.toml`: `745163209a31...` (MATCH)
- **VCS status:**
  Neither `/root/amd-workspace` nor `/root/amd-workspace/kernel-lab` is a git repository (`fatal: not a git repository`). Tamper resistance relies entirely on the cryptographic lock and mtimes.

---

## 2. Lock integrity — PASS

Ran `python3 -m kagent.eval.selftest_tamper --task kda-serving` from `/root/amd-workspace/kernel-lab/framework`:
```
baseline crate 'kda-serving' verifies OK
tamper 1 (modified verifier) refused:
  MODIFIED: frozen/kda-serving/verifier/verify_decode.py (sha256 58dc1b30e4b2... != 6a49a61dc785...)
tamper 2 (deleted holdout runner) refused:
  missing: frozen/holdout_runner.py
tamper 3 (lock thresholds edited vs bar) refused: threshold mismatch
tamper self-test PASSED (3/3 refusals)
```
Output confirmed exactly 3/3 refusals. Exit code 0.

---

## 3. Re-verify the candidate through the frozen evaluator — PASS

Command executed on free GPU 4 (`0%` VRAM):
```bash
HIP_VISIBLE_DEVICES=4 timeout 900 python3 -m kagent.eval.run_eval \
  --pack examples/kda-serving \
  --candidate /root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v9
```
Results:
- `VERIFY PASS: 15/15 gates (5.7s)`
- All 15 pre-registered gates passed with significant margin against the thresholds (`o_rrmse: 0.008`, `o_max_abs: 0.05`, `state_rrmse: 0.01`):
  - `D1_synthetic_s11`..`s90210`: o_rRMSE ~0.00166, state_rRMSE ~7.5e-8
  - `D1_kimi_s11`..`s90210`: o_rRMSE ~0.00166, state_rRMSE ~1.3e-7
  - `D2_warm_state_2048`: o_rRMSE ~0.00165, state_rRMSE ~1.34e-7
  - `D3_deep_decay`: o_rRMSE ~0.00166, state_rRMSE ~7.36e-8
  - `D3_near_unity_gates`: o_rRMSE ~0.00166, state_rRMSE ~8.89e-7
  - `D3_beta_extremes`: o_rRMSE ~0.00166, state_rRMSE ~1.43e-7
  - `D3_zero_v_from_warm`: o_rRMSE ~0.00166, state_rRMSE ~4.15e-7
  - `D3_large_v_x32`: o_rRMSE ~0.00166, state_rRMSE ~1.39e-7
  - `D4_drift_8192_kimi`: o_rRMSE ~0.00167, state_rRMSE ~1.28e-7
  - `D5_B3_H1`: o_rRMSE ~0.00168, state_rRMSE ~1.41e-7
  - `D5_B2_H96_kimi_k3`: o_rRMSE ~0.00166, state_rRMSE ~1.04e-7
- Exactly matches candidate's `verify_decode_report.json` (15/15 gates passed).

---

## 4. Holdout — PASS

Command executed on free GPU 4:
```bash
HIP_VISIBLE_DEVICES=4 timeout 900 python3 -m kagent.eval.run_eval \
  --pack examples/kda-serving \
  --candidate /root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v9 \
  --mode holdout
```
Result:
- `HOLDOUT PASS: 10/10 gates (2.8s) [values suppressed by design]`
- Exit code 0.
- Counts only; no metric leakage from the 4 frozen seeds (`613965540, 693982767, 785427682, 482519733`).

---

## 5. Interface honesty — PASS

Inspected `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v9/candidate.py`:
- **Native implementation:**
  - Implements `decode_pool(q, k, v, g, beta, pool, idx)` directly.
  - The kernel `kda_decode_step_v9` takes `IDX: fx.Tensor` and creates a buffer resource `idx_r = create_buffer_resource(IDX, max_size=True)`.
  - It loads the assigned slot per sequence dynamically: `slot = buffer_load(idx_r, bidx, vec_width=1, dtype=i32)`.
  - Computes state offset `s_head = slot * fx.Int32(SLOT_STRIDE) + h * fx.Int32(KK * VV)`.
  - Performs hoisted vector loads directly from the pool: `s_raw_a0 = buffer_load(s_r, s_r0, vec_width=4, dtype=f32)`.
  - Writes the updated state vector back in-place directly to `so_r` (the same `pool` buffer): `buffer_store(out_a, so_r, s_r0 + fx.Int32(half * 4))`.
- **Zero gather/scatter fallback:**
  - Unlike v7 (which wrapped with `index_select` / `index_copy_`), v9 has native indexed pool support with zero temporary memory allocations.
  - Declares `INPLACE = True` and `STATE_DTYPE = torch.float32`.
  - `get_forward()` is an adapter for external verifiers using `idx = torch.arange(B)`.

---

## 6. Benchmark hygiene — PASS

Inspected `/root/amd-workspace/kernel-lab/kda-serving/results.csv` and `roofline.json`:
- **Same metric and interface both sides:**
  - `baseline-fla-decode`: `metric=graphed_us_per_step`, `interface=serving-pool`, `timing_method=cuda-graph-replay`, `tool=bench/serving_decode.py`.
  - `flydsl-decode-v9`: `metric=graphed_us_per_step`, `interface=serving-pool`, `timing_method=cuda-graph-replay`, `tool=bench/serving_decode.py`.
  - Both sides evaluated on `kimi` gates, `H=96, K=V=128`, and state dtype `float32`.
- **Roofline pairing:**
  - Exact byte count parity across all batch sizes:
    - B=1: 12,730,560 bytes
    - B=8: 101,844,480 bytes
    - B=32: 407,377,920 bytes
    - B=128: 1,629,511,680 bytes
  - Measurements:
    - B=1: 7.13 µs vs 7.48 µs (1.05x speedup)
    - B=8: 29.35 µs vs 22.06 µs (0.752x speedup)
    - B=32: 97.45 µs vs 65.95 µs (0.677x speedup)
    - B=128: 442.5 µs vs 293.0 µs (0.662x speedup)
  - Geometric mean speedup: **0.771x** (accurately reported as `geomean 0.77x (goal 2.0x, every B >= 1.0x)`).
- **Sampling & GPU State:**
  - CUDA graph replay averages over 25 replays $\times$ 8 steps = 200 step executions.
  - Evaluated on physical GPU 4. Memory allocation was 0.0% VRAM.

---

## 7. Formal verdicts — PASS

Inspected `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v9/formal_report.json`:
- `"overall": "PROVED"` over 19 reduction chains (`selftest_exit: 0`).
- All 7 verification checks returned `"verdict": "PROVED"`:
  1. `butterfly_coverage`: PROVED (19 reduction chains form full XOR butterflies)
  2. `equivalence O b=0 h=0 i=0 j=0`: PROVED
  3. `equivalence O b=0 h=0 i=127 j=127`: PROVED
  4. `equivalence O b=1 h=3 i=64 j=5`: PROVED
  5. `equivalence S_OUT b=0 h=0 i=0 j=0`: PROVED
  6. `equivalence S_OUT b=0 h=0 i=127 j=64`: PROVED
  7. `equivalence S_OUT b=1 h=3 i=33 j=7`: PROVED
- **Zero UNKNOWN findings.**
- `agent_report.md` faithfully documents every finding as `[PROVED]` and discloses rounding points (`O store truncates to vector<4xbf16>`). UNKNOWN is never treated as a pass.

---

## 8. Case integrity — PASS

Compared `/root/amd-workspace/kernel-lab/framework/eval-crates/kda-serving/eval.lock.json` `expected_gates` with `framework/examples/kda-serving/cases/*.toml`:
- `eval.lock.json` registers 15 `expected_gates`:
  `['D1_kimi_s11', 'D1_kimi_s427', 'D1_kimi_s90210', 'D1_synthetic_s11', 'D1_synthetic_s427', 'D1_synthetic_s90210', 'D2_warm_state_2048', 'D3_beta_extremes', 'D3_deep_decay', 'D3_large_v_x32', 'D3_near_unity_gates', 'D3_zero_v_from_warm', 'D4_drift_8192_kimi', 'D5_B2_H96_kimi_k3', 'D5_B3_H1']`
- Pack case files define:
  - `D1_salts.toml`: 6 gates
  - `D2_warm_state.toml`: 1 gate
  - `D3_adversarial_ranges.toml`: 3 gates
  - `D3_beta_and_v_paths.toml`: 2 gates
  - `D4_drift.toml`: 1 gate
  - `D5_shapes.toml`: 2 gates
  - (`H1_holdout.toml`: holdout set only)
- Set difference between `expected_gates` and pack gates is empty (`set()`).
- All 7 `.toml` case files are explicitly hashed in `eval.lock.json` and matched their expected SHA-256 hashes.

---

## What the report understates

1. **Target failure and batch-size regression:**
   The stated contract target is $\ge 2.0\times$ FLA geomean with **no batch size slower than FLA**. While the candidate manages a small win at B=1 (1.05x, 7.13 µs vs 7.48 µs), it regresses against FLA at every larger batch size: B=8 (0.752x), B=32 (0.677x), and B=128 (0.662x). The overall geomean speedup of 0.771x represents an honest loss. The report presents the numbers clearly in the table but does not explicitly discuss the regression at B $\ge$ 8.

2. **The pure FP32 memory bandwidth ceiling (Path B necessity):**
   The report focuses on pure FP32 micro-optimizations (hoisted state loads, interleaved reduction trees, vector FMAs) without highlighting that pure FP32 can **never mathematically reach the 2.0x target at large batch sizes** on MI355X.
   As established in `contract.md` and `roofline.json`:
   - At B=128, FLA runs in 293.0 µs, meaning the 2.0x target requires $\le 146.5\ \mu\text{s}$.
   - However, at MI355X theoretical peak bandwidth (8.0 TB/s), reading and writing the FP32 state alone takes $203.7\ \mu\text{s}$ ($1.63\ \text{GB} / 8\ \text{TB/s}$).
   - Even with 100% memory bus efficiency and zero arithmetic latency, pure FP32 is bounded to $\le 1.44\times$ FLA at B=128. Reaching 2.0x fundamentally requires halving state memory traffic by transitioning to FP16 state (Path B, where the bandwidth floor is $\sim 101.8\ \mu\text{s}$).

3. **Issue-bound diagnosis vs memory wait reality:**
   The report asserts that the kernel is "issue-bound" (math/data = 2.09, issue/data = 4.73), noting that shuffle overhead and adjacent instructions dominate issue slots. However, the accompanying PC sampling (`pcs.json`) reveals that `WAITCNT` (waiting on memory) accounts for 327 of 592 samples (55.2%), and `OTHER_WAIT` accounts for another 70 samples (11.8%). While hoisting loads overlapped some latency, memory wait remains the single largest stall category. The report presents both metrics without reconciling their interaction.

4. **Formal verification configuration vs production shapes:**
   "PROVED over 19 reduction chains" was formally checked on a small configuration ($B=2, H=96$) across 6 sampled coordinates (3 output coordinates and 3 state coordinates) with uninterpreted `exp` and `rsqrt` function abstractions. This guarantees structural equivalence modulo the mathematical abstractions at those sampled points, but does not constitute a full bitwise equivalence proof across all shapes or dynamic indices.

5. **Single-sample benchmark without variance:**
   The candidate's `results.csv` row reflects a single recorded benchmark invocation ($N=1$) per batch size. While each invocation evaluates 200 CUDA graph step replays, no multi-trial variance, standard deviation, or confidence intervals are reported.

6. **Host GPU utilization stamp artifact:**
   In `serving.json`, `gpu_occupancy_at_start` stamps `use_pct: 39.0%` (and verifier stamped `16.0%`), which exceeds `env_stamp.py`'s internal `BUSY_USE_PCT = 15` threshold. While `vram_pct: 0.0%` confirms no co-tenant VRAM allocation, the host's background `rocm-smi` utilization percent continues to fluctuate across runs.

7. **Unfinished report placeholder:**
   The `## Next hypothesis` section in `agent_report.md` was left as the framework's boilerplate placeholder (`TODO(agent): one hypothesis per report, citing the metric it attacks and by how much (guideline 04)`), leaving the report narrative incomplete even though a next step was recorded in `candidates.jsonl`.
