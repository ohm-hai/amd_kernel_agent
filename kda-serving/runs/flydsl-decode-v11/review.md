# Independent review — kda-serving / flydsl-decode-v11

Reviewer: independent, not a participant in the session that built the framework or the candidate.
Date: 2026-10-15. Method: read-only inspection; the only file written is this `review.md`.
Evaluation re-runs executed on GPU 4 (verified 0% VRAM allocation via `rocm-smi`).

---

## VERDICT

**accept** — as a validated experimental checkpoint (recording a valuable negative result on wave-residency tuning at higher batch sizes), but **strictly rejected for promotion**.

All 8 checklist items pass substantively without evidence of tampering, cheating, or reward-hacking. The candidate is a mathematically sound, reproducible implementation:
- **Accuracy:** Passes all 15 pre-registered frozen verifier gates (15/15) and hidden holdout gates (10/10) within the accuracy bar.
- **Formal Equivalence:** Formally PROVED across 19 reduction chains with zero UNKNOWN findings.
- **Interface:** Native in-place indexed serving pool implementation directly in the FlyDSL GPU kernel (`decode_pool`) with zero gather/scatter copies or host overhead.
- **Integrity:** Lock verification, case matching, and benchmark hygiene are intact.

However, `flydsl-decode-v11` **fails key criteria of `contract.md` and regresses against its parent candidate `flydsl-decode-v10`**:
1. **Regression at B=32 below FLA baseline:** At $B=32$, the candidate runs in 69.98 µs vs FLA baseline 65.95 µs (**0.942x speedup**), violating the contract rule that *"no batch size may be slower than FLA"* and *"every B is >= 1x"*.
2. **Net geomean regression:** Geomean speedup drops from **1.1194x** (v10) to **1.1065x** (v11). Candidate latency regresses at $B=8$ (19.92 µs $\to$ 20.05 µs), $B=32$ (65.38 µs $\to$ 69.98 µs), and $B=128$ (275.23 µs $\to$ 279.07 µs).
3. **Far below contract target:** Geomean speedup of 1.11x remains far below the $2.0\times$ promotion target.

Therefore, `flydsl-decode-v11` is accepted into the experimental record as a valid data point showing that halving block size to increase wave residency hurts performance at $B \ge 8$, but it cannot be promoted. `flydsl-decode-v10` remains the active best checkpoint.

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
  - Python bytecode cache files (`__pycache__/*.pyc` generated during evaluation runs).

- **Timestamp analysis:**
  - `framework/README.md` mtime: `2026-10-10 08:05:40 UTC`.
  - `eval.lock.json`, `frozen/holdout_runner.py`, and `frozen/holdouts.json` mtimes: `2026-10-10 08:45:25 UTC`. This matches the crate creation timestamp recorded inside `eval.lock.json` (`"created": "2026-10-10T08:45:25"`), reflecting the initial build of the eval-crate when case toml hashes were locked.
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
  --candidate /root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v11
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
  --candidate /root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v11 \
  --mode holdout
```
Result:
- `HOLDOUT PASS: 10/10 gates (2.8s) [values suppressed by design]`
- Exit code: 0.
- Counts only; metric values remain confidential per framework invariant 2.

---

## 5. Interface honesty — PASS

Inspected `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v11/candidate.py`:
- **Native in-place serving pool implementation:**
  - Implements `decode_pool(q, k, v, g, beta, pool, idx)` directly.
  - The GPU kernel `kda_decode_step_v11` takes the index tensor `IDX: fx.Tensor` and constructs buffer resource `idx_r = create_buffer_resource(IDX, max_size=True)`.
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
  - `flydsl-decode-v11`:
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
  - $B=1$: candidate 5.43 µs vs baseline 7.48 µs (1.378x speedup; 29.3% efficiency) [paired run: 5.43 µs vs 7.45 µs, 1.372x]
  - $B=8$: candidate 20.05 µs vs baseline 22.06 µs (1.100x speedup; 63.5% efficiency) [paired run: 20.05 µs vs 22.11 µs, 1.103x]
  - $B=32$: candidate 69.98 µs vs baseline 65.95 µs (**0.942x speedup — SLOWER than baseline**; 72.8% efficiency) [paired run: 69.98 µs vs 66.07 µs, 0.944x]
  - $B=128$: candidate 279.07 µs vs baseline 293.00 µs (1.050x speedup; 73.0% efficiency) [paired run: 279.07 µs vs 294.41 µs, 1.055x]
  - Geometric mean speedup: **1.1065x** (reported as `geomean 1.11x`).
- Neither side conflates `public-wrapper` numbers with `serving-pool` numbers.

---

## 7. Formal verdicts — PASS

Inspected `/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v11/formal_report.json`:
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

1. **Performance regression vs parent checkpoint (`flydsl-decode-v10`) and baseline failure:**
   The report understates that `v11` is an overall regression compared to `v10`:
   | Metric | FLA Baseline | v10 Candidate | v10 Speedup | v11 Candidate | v11 Speedup | v11 vs v10 Delta |
   |---|---|---|---|---|---|---|
   | $B=1$ | 7.48 µs | 5.67 µs | 1.320x | 5.43 µs | 1.378x | -0.24 µs (+4.2% faster) |
   | $B=8$ | 22.06 µs | 19.92 µs | 1.107x | 20.05 µs | 1.100x | +0.13 µs (0.7% slower) |
   | $B=32$ | 65.95 µs | 65.38 µs | 1.009x | 69.98 µs | **0.942x** | +4.60 µs (**7.0% slower; drops below FLA**) |
   | $B=128$ | 293.00 µs | 275.23 µs | 1.065x | 279.07 µs | 1.050x | +3.84 µs (1.4% slower) |
   | **Geomean** | — | — | **1.1194x (1.12x)** | — | **1.1065x (1.11x)** | **Regression in geomean** |

   Crucially, at $B=32$, `v11` is slower than FLA (0.942x speedup). In `v10`, every batch size beat FLA ($B \ge 1.0\times$). In `v11`, that property is lost.

2. **Misleading roofline verdict text:**
   In `agent_report.md`, the roofline summary states:
   `verdict: geomean 1.11x (goal 2.0x, every B >= 1.0x)`
   While `(goal 2.0x, every B >= 1.0x)` quotes the contract goal, presenting this immediately after the geomean figure without qualification can easily be misread as claiming that `v11` satisfies the "every B >= 1.0x" requirement, when in reality $B=32$ explicitly violates it (0.942x).

3. **Architectural analysis of the wave-residency trade-off:**
   `v11` halved workgroup size from 256 threads (4 Wave64s) to 128 threads (2 Wave64s), doubling block count ($NV=8$ vs $NV=4$):
   - At $B=1$: Grid size doubled from 384 to 768 blocks across 256 CUs. This eliminated the 1-vs-2 block imbalance (allocating exactly 3 blocks / 6 waves per CU) and improved latency (5.67 µs $\to$ 5.43 µs).
   - At $B \ge 8$: The GPU was already fully saturated with waves (e.g. 12,288 blocks at $B=32$, 49,152 blocks at $B=128$ in v10). Doubling workgroups (to 24,576 and 98,304 blocks) doubled workgroup dispatch overhead and doubled redundant work: every block independently re-reads input tensors ($Q, K, G, \beta$) and re-executes the intra-wavefront $QK$ reduction tree. At $B=32$, this redundant overhead pushed runtime from 65.38 µs to 69.98 µs.
   Occupancy tuning is beneficial when under-saturated ($B=1$), but counterproductive when memory-bound and saturated ($B \ge 8$).

4. **The fundamental physical ceiling of pure FP32 state:**
   At $B=128$, transferring the FP32 state alone ($2 \times B \times H \times K \times V \times 4\text{ bytes}$) requires 1.63 GB per step. At MI355X peak theoretical bandwidth (8.0 TB/s), streaming this data takes a minimum of $203.7\ \mu\text{s}$ (`floor_us_at_spec_bw`). Since the FLA baseline is $293.0\ \mu\text{s}$, a $2.0\times$ speedup requires $\le 146.5\ \mu\text{s}$. Because $203.7\ \mu\text{s} > 146.5\ \mu\text{s}$, a $2.0\times$ speedup at large batch sizes is mathematically impossible with pure FP32 state. Further tuning of FP32 thread geometry will not bridge this gap; transitioning to FP16 state (Path B) is mandatory.

5. **Static instruction ratio vs dynamic PC sampling stalls:**
   `agent_report.md` classifies the kernel as "issue-bound" based on static ISA counts (`math/data = 1.90`, `issue/data = 4.54`). However, empirical PC sampling (`pcs.json`) demonstrates that dynamic execution is overwhelmingly wait-bound:
   - `WAITCNT`: 293 / 500 samples (58.6%)
   - `BARRIER_WAIT`: 55 / 500 samples (11.0%)
   - `OTHER_WAIT`: 30 / 500 samples (6.0%)
   Combined wait stalls represent **75.6%** of all recorded samples. Memory latency and synchronization wait remain the dominant execution bottlenecks.

6. **Incomplete report template and candidate record:**
   - In `agent_report.md`, the `## Next hypothesis` section was left as an unedited framework placeholder: `TODO(agent): one hypothesis per report, citing the metric it attacks and by how much (guideline 04)`.
   - In `candidates.jsonl`, the entry for `flydsl-decode-v11` has `key_results: null`, `reason: null`, and `next: "run full kagent verification and benchmark loop"`.

7. **Single-sample benchmark recording:**
   Each batch size in `results.csv` captures a single measurement invocation ($N=1$) of 200 CUDA graph replays without reporting multi-trial variance or confidence intervals across independent GPU runs.
