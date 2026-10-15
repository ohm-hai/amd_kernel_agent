# Independent review — kda-serving / flydsl-decode-v10

Reviewer: independent, not a participant in the session that built the framework or the candidate.
Date: 2026-10-15. Method: read-only inspection; the only file written is this `review.md`.
Evaluation re-runs executed on GPU 4 (verified 0% VRAM allocation via `rocm-smi`).

---

## VERDICT

**accept** — as a validated checkpoint / intermediate milestone, not as a promoted candidate meeting the final contract target.

All 8 checklist items pass completely without evidence of tampering, cheating, or reward-hacking. The candidate is an honest, mathematically proven implementation:
- **Accuracy:** Passes all 15 pre-registered frozen verifier gates (15/15) and hidden holdout gates (10/10) with significant error margin against the accuracy bar.
- **Formal Equivalence:** Formally PROVED across 19 reduction chains with zero UNKNOWN findings.
- **Interface:** Native in-place indexed serving pool implementation directly in the FlyDSL GPU kernel (`decode_pool`) with zero gather/scatter copies or host overhead.
- **Performance:** For the first time in the decode campaign, every batch size in the grid beats the FLA baseline (B=1: 1.32x, B=8: 1.107x, B=32: 1.009x, B=128: 1.065x), achieving an overall geometric mean speedup of **1.12x** (1.1194x).

However, the candidate does **not** meet the final contract promotion target in `contract.md` ($\ge 2.0\times$ geomean speedup over $B \in \{1, 8, 32, 128\}$). Neither `agent_report.md` nor `candidates.jsonl` falsely claims promotion; the report clearly and honestly records `verdict: geomean 1.12x (goal 2.0x, every B >= 1.0x)`. Therefore, `flydsl-decode-v10` is accepted as a valid, verified optimization checkpoint.

---

## 1. Workspace tamper check — PASS

- **Crate edits after build check:**
  Command executed:
  ```bash
  find /root/amd-workspace/kernel-lab/framework/eval-crates -newer /root/amd-workspace/kernel-lab/framework/README.md -type f | head
  ```
  Found:
  - `/root/amd-workspace/kernel-lab/framework/eval-crates/kda-serving/eval.lock.json`
  - `/root/amd-workspace/kernel-lab/framework/eval-crates/kda-serving/frozen/holdout_runner.py`
  - `/root/amd-workspace/kernel-lab/framework/eval-crates/kda-serving/frozen/holdouts.json`
  - Python bytecode cache files (`__pycache__/*.pyc` generated during test execution).

- **Timestamp analysis:**
  - `framework/README.md` mtime: `2026-10-10 08:05:40 UTC`.
  - `eval.lock.json`, `frozen/holdout_runner.py`, and `frozen/holdouts.json` mtimes: `2026-10-10 08:45:25 UTC`. This matches the crate creation timestamp recorded inside `eval.lock.json` (`"created": "2026-10-10T08:45:25"`), reflecting the initial build of the eval-crate.
  - Frozen source files (`frozen/kda-attn/ref_fp64.py`, `frozen/kda-serving/inputs.py`, `frozen/kda-serving/verifier/verify_decode.py`, `frozen/tools/env_stamp.py`) have mtimes ranging from `2026-10-09 23:35` to `23:38 UTC`, which predate `README.md`. No frozen source files were edited after crate creation.

- **Lock hash verification:**
  Every SHA-256 hash registered in `/root/amd-workspace/kernel-lab/framework/eval-crates/kda-serving/eval.lock.json` was independently recalculated against the corresponding file on disk. All 14 entries match byte-for-byte:
  - `frozen/holdout_runner.py`: `03e1ee61f61e4536b5ee8d02a9640b4d94bd4cdc6999f14f3b634c1711baeafe` (OK)
  - `frozen/holdouts.json`: `bb7cf41cf56f57b5c0ee8f2b033eb1463060a3b86acb53fe15d8963f8f647c56` (OK)
  - `frozen/kda-attn/ref_fp64.py`: `4f9cf62597b9b466bb32a7ca0e4622fcb033139fe69b993493b7982e6ba903e7` (OK)
  - `frozen/kda-serving/inputs.py`: `0665afbf899dff6697efcb5d746f0d79f3c99b653e31b5be4fd1f200457ff164` (OK)
  - `frozen/kda-serving/verifier/verify_decode.py`: `6a49a61dc7859de127dd038a48fccdcb54f9e8b68969b89284280b7e8bda6372` (OK)
  - `frozen/tools/env_stamp.py`: `a34394df6e20d4e8d967269f1619957bed33244aec8b52d8ec4ec6c70df1ef2c` (OK)
  - `bar:bar.toml`: `60f7e04c6a89b590b0210b420b612854458541f94eb84f60c3598fc18e52fa61` (OK)
  - `cases:cases/D1_salts.toml`: `70c449fab17e6d4786f201f452036068fd3668c813430459118e5a45a54a622a` (OK)
  - `cases:cases/D2_warm_state.toml`: `5578190ca80fe9ef13558b1bcdb64eef753d74d5064aa00335bef561d185d1f7` (OK)
  - `cases:cases/D3_adversarial_ranges.toml`: `ed3e908ee094cb7bee7fa7a373378692a654e9c3c2ef625b5b0c552707b4b027` (OK)
  - `cases:cases/D3_beta_and_v_paths.toml`: `edf8f21f49766fdc608e2a6e0f49efc4305cb587367141feaf7594f816b5d6be` (OK)
  - `cases:cases/D4_drift.toml`: `d3ae8de50db40e0ebf09c8b9790dd08b9c20f14528343e400e339e2b53958f12` (OK)
  - `cases:cases/D5_shapes.toml`: `fe08ea6d478a591f424ee271ad74428958f7a807c60ae93271b98988382963f7` (OK)
  - `cases:cases/H1_holdout.toml`: `745163209a316f58077afce4c1b38e7be3d3cf5196611f47c0edd0ed56324e96` (OK)

- **VCS status:**
  Neither `/root/amd-workspace` nor `/root/amd-workspace/kernel-lab` is under git version control (`fatal: not a git repository`). Tamper resistance is strictly enforced by the cryptographic SHA-256 digest lock.

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
- Output verified: exactly 3/3 refusals. Exit code: 0.

---

## 3. Re-verify the candidate through the frozen evaluator — PASS

Command executed on free GPU 4 (0% VRAM allocated):
```bash
HIP_VISIBLE_DEVICES=4 timeout 900 python3 -m kagent.eval.run_eval \
  --pack examples/kda-serving \
  --candidate /root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v10
```
Results:
- Evaluator reported: `VERIFY PASS: 15/15 gates (5.7s)`
- Bar thresholds (from pre-registered `accuracy-bar.md` / `eval.lock.json`):
  - `o_rrmse`: 0.008
  - `o_max_abs`: 0.05
  - `state_rrmse`: 0.01
- Per-gate measurements:
  - `D1_synthetic_s11`: o_rRMSE 0.00166, o_abs 0.000185, state_rRMSE 8.16e-08 [PASS]
  - `D1_synthetic_s427`: o_rRMSE 0.00166, o_abs 0.000238, state_rRMSE 7.57e-08 [PASS]
  - `D1_synthetic_s90210`: o_rRMSE 0.00166, o_abs 0.000122, state_rRMSE 7.53e-08 [PASS]
  - `D1_kimi_s11`: o_rRMSE 0.00166, o_abs 0.000241, state_rRMSE 1.36e-07 [PASS]
  - `D1_kimi_s427`: o_rRMSE 0.00166, o_abs 0.000221, state_rRMSE 1.30e-07 [PASS]
  - `D1_kimi_s90210`: o_rRMSE 0.00166, o_abs 0.000156, state_rRMSE 1.39e-07 [PASS]
  - `D2_warm_state_2048`: o_rRMSE 0.00165, o_abs 0.000225, state_rRMSE 1.34e-07 [PASS]
  - `D3_deep_decay`: o_rRMSE 0.00166, o_abs 0.000174, state_rRMSE 7.36e-08 [PASS]
  - `D3_near_unity_gates`: o_rRMSE 0.00166, o_abs 0.000488, state_rRMSE 8.89e-07 [PASS]
  - `D3_beta_extremes`: o_rRMSE 0.00166, o_abs 0.000240, state_rRMSE 1.43e-07 [PASS]
  - `D3_zero_v_from_warm`: o_rRMSE 0.00166, o_abs 0.000110, state_rRMSE 4.15e-07 [PASS]
  - `D3_large_v_x32`: o_rRMSE 0.00166, o_abs 0.005850, state_rRMSE 1.39e-07 [PASS]
  - `D4_drift_8192_kimi`: o_rRMSE 0.00167, o_abs 0.000243, state_rRMSE 1.28e-07 [PASS]
  - `D5_B3_H1`: o_rRMSE 0.00168, o_abs 0.000122, state_rRMSE 1.41e-07 [PASS]
  - `D5_B2_H96_kimi_k3`: o_rRMSE 0.00166, o_abs 0.000198, state_rRMSE 1.04e-07 [PASS]
- Exact match: Output values match candidate's `verify_decode_report.json` across all 15 gates (15/15 passed).

---

## 4. Holdout — PASS

Command executed on free GPU 4:
```bash
HIP_VISIBLE_DEVICES=4 timeout 900 python3 -m kagent.eval.run_eval \
  --pack examples/kda-serving \
  --candidate /root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v10 \
  --mode holdout
```
Result:
- `HOLDOUT PASS: 10/10 gates (2.7s) [values suppressed by design]`
- Exit code: 0.
- Counts only; metric values remain confidential per framework invariant 2.

---

## 5. Interface honesty — PASS

Inspected `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v10/candidate.py`:
- **Native in-place serving pool implementation:**
  - Implements `decode_pool(q, k, v, g, beta, pool, idx)` directly.
  - The GPU kernel `kda_decode_step_v10` takes the index tensor `IDX: fx.Tensor` and constructs buffer resource `idx_r = create_buffer_resource(IDX, max_size=True)`.
  - Inside the kernel, each sequence dynamically retrieves its assigned state pool slot:
    ```python
    slot = buffer_load(idx_r, bidx, vec_width=1, dtype=i32)
    s_head = slot * fx.Int32(SLOT_STRIDE) + h * fx.Int32(VV * KK)
    ```
  - State loads and stores are executed directly against `s_r` (`S_IN`) and `so_r` (`S_OUT`), where `pool` is passed for both `S_IN` and `S_OUT`:
    ```python
    s_raw[c] = buffer_load(s_r, s_offset, vec_width=2, dtype=f32)
    # ... computation ...
    buffer_store(new_v, so_r, s_offset)
    ```
  - Zero temporary host or device tensor allocations, zero indexing wrappers (`index_select`/`index_copy_`), and zero gather/scatter operations occur inside `decode_pool`.
  - Candidate metadata sets `INPLACE = True`, `STATE_DTYPE = torch.float32`, and `STATE_LAYOUT = "v_first"`.
  - The function `get_forward()` is strictly a verifier adapter (transposing the verifier's standard K-major `initial_state` to V-major and back). The serving path benchmark invokes `decode_pool` directly with pre-allocated pool slots.

---

## 6. Benchmark hygiene — PASS

Inspected `/root/amd-workspace/kernel-lab/kda-serving/results.csv`, `serving.json`, and `roofline.json`:
- **Strict metric and interface alignment:**
  - `baseline-fla-decode`:
    - `metric`: `graphed_us_per_step`
    - `interface`: `serving-pool`
    - `timing_method`: `cuda-graph-replay`
    - `tool`: `bench/serving_decode.py`
  - `flydsl-decode-v10`:
    - `metric`: `graphed_us_per_step`
    - `interface`: `serving-pool`
    - `timing_method`: `cuda-graph-replay`
    - `tool`: `bench/serving_decode.py`
  - Identical workload conditions: $H=96, K=128, V=128$, `gates="kimi"`, state dtype `float32`, evaluated on physical GPU 4 (0% VRAM allocated).
- **Roofline pairing and byte accounting:**
  Byte accounting is identical on both sides:
  - $B=1$: 12,730,560 bytes
  - $B=8$: 101,844,480 bytes
  - $B=32$: 407,377,920 bytes
  - $B=128$: 1,629,511,680 bytes
- **Latency and speedup breakdown:**
  - $B=1$: candidate 5.67 µs vs baseline 7.48 µs (1.320x speedup; 28.1% of 8 TB/s peak bandwidth)
  - $B=8$: candidate 19.92 µs vs baseline 22.06 µs (1.107x speedup; 63.9% efficiency)
  - $B=32$: candidate 65.38 µs vs baseline 65.95 µs (1.009x speedup; 77.9% efficiency)
  - $B=128$: candidate 275.23 µs vs baseline 293.00 µs (1.065x speedup; 74.0% efficiency)
  - Geometric mean speedup: **1.1194x** (reported as `geomean 1.12x`).
- Neither side conflates `public-wrapper` numbers with `serving-pool` numbers.

---

## 7. Formal verdicts — PASS

Inspected `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v10/formal_report.json`:
- Overall status: `"overall": "PROVED"` over 19 reduction chains (`selftest_exit: 0`).
- Individual verification findings:
  1. `butterfly_coverage`: PROVED (19 reduction chains form full XOR butterflies)
  2. `equivalence O b=0 h=0 i=0 j=0`: PROVED
  3. `equivalence O b=0 h=0 i=127 j=127`: PROVED
  4. `equivalence O b=1 h=3 i=64 j=5`: PROVED
  5. `equivalence S_OUT b=0 h=0 i=0 j=0`: PROVED
  6. `equivalence S_OUT b=0 h=0 i=127 j=64`: PROVED
  7. `equivalence S_OUT b=1 h=3 i=33 j=7`: PROVED
- **Zero UNKNOWN findings.**
- In `agent_report.md`:
  - Every finding is faithfully documented as `[PROVED]`.
  - Explicit disclosure of rounding points: `['O store truncates to vector<4xbf16> (line 529)', 'O store truncates to vector<4xbf16> (line 532)']`.
  - Explicit disclosure of scope and uninterpreted functions (`exp`, `rsqrt`).
  - UNKNOWN is never treated as a pass.

---

## 8. Case integrity — PASS

Compared `expected_gates` in `/root/amd-workspace/kernel-lab/framework/eval-crates/kda-serving/eval.lock.json` with pack case files in `framework/examples/kda-serving/cases/*.toml`:
- `eval.lock.json` expected gates (15 total):
  `['D1_kimi_s11', 'D1_kimi_s427', 'D1_kimi_s90210', 'D1_synthetic_s11', 'D1_synthetic_s427', 'D1_synthetic_s90210', 'D2_warm_state_2048', 'D3_beta_extremes', 'D3_deep_decay', 'D3_large_v_x32', 'D3_near_unity_gates', 'D3_zero_v_from_warm', 'D4_drift_8192_kimi', 'D5_B2_H96_kimi_k3', 'D5_B3_H1']`
- Pack case files define:
  - `D1_salts.toml`: 6 gates
  - `D2_warm_state.toml`: 1 gate
  - `D3_adversarial_ranges.toml`: 3 gates
  - `D3_beta_and_v_paths.toml`: 2 gates
  - `D4_drift.toml`: 1 gate
  - `D5_shapes.toml`: 2 gates
  - (`H1_holdout.toml`: `holdout_quick_set` holdout gate)
- Set differences:
  - `set(expected_gates) - set(pack_gates) == set()`
  - `set(pack_gates) - set(expected_gates) == set()`
- Exact 1:1 match. All 7 `.toml` case files are explicitly hashed in `eval.lock.json` with matching SHA-256 digests.

---

## What the report understates

1. **Contract target shortfall:**
   The objective in `contract.md` specifies a geometric mean speedup of $\ge 2.0\times$ across $B \in \{1, 8, 32, 128\}$ with no batch size slower than FLA. While `flydsl-decode-v10` is a substantial improvement over previous candidates (v9 was 0.77x; v10 achieves $\ge 1.0\times$ across all batch sizes), the resulting **1.12x geomean speedup remains far below the 2.0x target**. At $B=32$, the speedup is effectively at parity ($1.009\times$, 65.38 µs vs 65.95 µs, an improvement of just 0.86%). At $B=128$, the speedup is only $1.065\times$ (275.23 µs vs 293.00 µs). Reaching the 2.0x goal requires an additional ~1.79x speedup.

2. **The mathematical impossibility of 2.0x with pure FP32 state:**
   The report highlights 100% coalesced 512B burst state loads/stores via the V-Major layout, but does not state the fundamental physical ceiling:
   - At $B=128$, the FP32 state alone requires transferring 1.63 GB per step ($2 \times B \times H \times K \times V \times 4\text{ bytes}$).
   - Even at MI355X peak theoretical memory bandwidth ($8000\text{ GB/s} = 8.0\text{ TB/s}$), streaming 1.63 GB takes $203.7\ \mu\text{s}$ (`floor_us_at_spec_bw`).
   - The FLA baseline runs in $293.0\ \mu\text{s}$. A 2.0x speedup requires a latency of $\le 146.5\ \mu\text{s}$.
   - Because $203.7\ \mu\text{s} > 146.5\ \mu\text{s}$, the maximum possible speedup with pure FP32 state at $B=128$ at 100% theoretical bus utilization with zero arithmetic overhead is $293.0 / 203.7 \approx 1.438\times$.
   - Candidate efficiency at $B=32$ (77.9%) and $B=128$ (74.0%) demonstrates that pure FP32 has already entered diminishing returns. Achieving the 2.0x contract target fundamentally necessitates transitioning to FP16 state (Path B), where the state traffic is halved and the theoretical floor drops to $\sim 101.8\ \mu\text{s}$.

3. **Issue-bound diagnosis vs empirical memory stall reality:**
   `agent_report.md` categorizes the kernel as "issue-bound" based on instruction ratios (`math/data = 1.90`, `issue/data = 4.54`), recommending coalescing work per lane to reduce shuffle overheads. However, the empirical PC sampling (`pcs.json`) reveals that runtime execution is heavily dominated by memory stalls:
   - `WAITCNT` accounts for 316 of 649 samples (48.7%).
   - `OTHER_WAIT` accounts for 88 samples (13.6%).
   - Combined wait stalls constitute **62.2%** of all stall samples.
   - The top stalled instructions are `s_waitcnt lgkmcnt(0)` (77 samples), `s_barrier` (58 samples), `s_waitcnt vmcnt(0)` (39 samples), and `global_load_dwordx2` (32 samples). While hoisted loads partially hid latency, memory wait remains the primary execution bottleneck.

4. **Serving framework layout dependency:**
   `v10` changes the state layout from standard K-major `[P, H, K, V]` to V-major `[P, H, V, K]`. While this enables coalesced memory bursts within `decode_pool`, external ML serving frameworks typically allocate KV/state caches in K-major format. To realize the 1.12x speedup in production, the serving framework must natively allocate and manage the state pool in V-major format; otherwise, external transposition overheads (`initial_state.transpose(-1, -2).contiguous()`, as demonstrated in `get_forward()`) would negate the kernel performance gains.

5. **Scope limitations of formal verification:**
   The formal equivalence proof (`PROVED over 19 reduction chains`) verifies equivalence under specific bounded constraints:
   - Evaluated on a single configuration ($B=2, H=96$).
   - Checks 6 sampled coordinate points (3 for $O$, 3 for $S\_OUT$).
   - Relies on uninterpreted function abstractions for `exp` and `rsqrt`.
   - Adopts an identity slot mapping assumption (`slot == b`).
   This is a strong structural proof of mathematical equivalence for those elements, but relies on the numeric verifier (15 gates) to guarantee full precision and general dynamic index validity.

6. **Incomplete report placeholder:**
   In `agent_report.md`, the `## Next hypothesis` section was left as an unedited framework placeholder:
   `TODO(agent): one hypothesis per report, citing the metric it attacks and by how much (guideline 04)`
   While `candidates.jsonl` tracks candidate status, the narrative report was finalized without formulating the next experimental hypothesis.

7. **Single-sample benchmark recording:**
   Each batch size in `results.csv` records a single measurement invocation ($N=1$), capturing 200 CUDA graph step replays (25 replays $\times$ 8 steps). While CUDA graph replay minimizes kernel launch jitter, no multi-trial variance, standard deviation, or confidence intervals across separate GPU initialization runs are reported.
