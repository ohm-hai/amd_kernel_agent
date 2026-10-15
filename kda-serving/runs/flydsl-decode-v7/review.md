# Independent review — kda-serving / flydsl-decode-v7 (first end-to-end KAF validation)

Reviewer: independent, not a participant in the session that built the framework or the candidate.
Date: 2026-10-10. Method: read-only inspection; the only file written is this `review.md`.
Before re-running the frozen evaluator (which writes reports into the candidate dir), the full
candidate dir was snapshotted to `/tmp/kda-review-snapshot/`; my re-runs were diffed against it.

## VERDICT

**accept** — as a validated checkpoint, not as a promoted candidate. All 8 checks pass
substantively. The candidate is honestly reported as **0.25x of FLA on the serving-pool metric
(a large loss, not a win)**, the frozen-evaluation chain is intact, and my independent re-run
reproduced the session's numbers exactly. The findings below are report-completeness and
tooling issues, none of which change the result or suggest gaming. No promotion rule of
`task.toml` is met (the report does not claim otherwise).

---

## 1. Workspace tamper check — PASS (construction, not tampering; with caveats)

- `git -C /root/amd-workspace status --porcelain` → `fatal: not a git repository`. Same for
  `/root/amd-workspace/kernel-lab`. **The workspace is not under version control**, so there is
  no VCS-level tamper evidence in either direction.
- Git repos exist only for vendored upstreams. Status:
  - `Primus-Turbo`: ` M 3rdparty/composable_kernel` (submodule pointer), `?? 3rdparty/hipkittens/` (untracked dep).
  - `kda`: ` M skills/KernelWiki` — submodule pointer only, `0 insertions, 0 deletions`.
  - `TraceLens`: `README.md | 22 +++++` (additive) + `?? MI355X.json`. The diff is a dated
    note: *"MI355X.json (added 2026-10-09) … max_achievable_tflops entries were measured … with
    `kernel-lab/tools/measure_maf.py`"*. Additive bring-up for a new GPU, method disclosed; not
    in the frozen eval path and not timing-affecting.
- `find framework/eval-crates -newer framework/README.md -type f` →
  `eval.lock.json`, `frozen/holdout_runner.py`, `frozen/holdouts.json`, and three `__pycache__/*.pyc`.
  README.md is 08:05:40, the crate was built 08:10:08 — **these are the framework build itself
  (construction), sequenced after the README**, plus Python bytecode caches.
- Construction vs tampering distinguished by content hashes: frozen copies are byte-identical to
  the task-dir originals and to the lock (check 2). Frozen source mtimes (verifier
  `2026-10-09 23:38`, `inputs.py 23:35`, `env_stamp.py 23:37`) **predate** the measurement rows
  they produced (23:45–23:48) and were preserved into `frozen/` — no post-hoc edit signature.
- Caveat: without git, "no tampering" rests on the hash lock + mtimes, not on VCS history.

## 2. Lock integrity — PASS

```
$ cd framework && python3 -m kagent.eval.selftest_tamper --task kda-serving
baseline crate 'kda-serving' verifies OK
tamper 1 (modified verifier) refused: MODIFIED: frozen/kda-serving/verifier/verify_decode.py
  (sha256 58dc1b30e4b2... != 6a49a61dc785...)
tamper 2 (deleted holdout runner) refused: missing: frozen/holdout_runner.py
tamper 3 (lock thresholds edited vs bar) refused: threshold mismatch
tamper self-test PASSED (3/3 refusals)
```

3/3 refusals, as required. Independently: `sha256sum` of frozen `verify_decode.py` and
`inputs.py` equals both the lock entries and the task-dir originals (`6a49a61d…`, `0665afbf…`);
`tools/env_stamp.py` frozen hash `a34394df…` equals `kernel-lab/tools/env_stamp.py`. Lock
thresholds match the pre-registered `bar.toml` (`o_rrmse 8e-3, o_max_abs 5e-2, state_rrmse 1e-2`).

## 3. Re-verify through the frozen evaluator — PASS (15/15, matches)

```
$ HIP_VISIBLE_DEVICES=4 python3 -m kagent.eval.run_eval --pack examples/kda-serving \
    --candidate /root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v7
VERIFY PASS: 15/15 gates (5.6s) -> .../flydsl-decode-v7/verify_decode_report.json
bar thresholds (from lock): {'o_rrmse': 0.008, 'o_max_abs': 0.05, 'state_rrmse': 0.01}
...
ACCEPTED: 15/15 gates passed in 5.6s
recorded verifier row in /root/amd-workspace/kernel-lab/kda-serving/results.csv
```

Matches the candidate's own `verify_decode_report.json` (15/15). `diff` of my re-run report vs
the pre-run snapshot shows **only** `env.time` (08:39:54 → 08:40:45), `use_pct` (26.0 → 27.0),
and two per-gate wall-clock `seconds` fields; all accuracy values and the gate set are identical.
`run_eval` also verified the lock before and after the run and enforced the lock's
`expected_gates` set (a mismatch would have failed the run).

## 4. Holdout — PASS (counts only)

```
$ HIP_VISIBLE_DEVICES=4 python3 -m kagent.eval.run_eval --pack examples/kda-serving \
    --candidate .../flydsl-decode-v7 --mode holdout
HOLDOUT PASS: 10/10 gates (2.7s) [values suppressed by design]
```

Counts only, per the frozen `holdout_runner.py` (redirects the verifier's stdout, writes
`{passed, n_gates, n_failed, n_holdout_seeds: 4}`; seeds `542411562, 771594925, 188871076,
855232442` live only inside the hash-locked crate). No values leak.

## 5. Interface honesty — PASS with a disclosure gap

- **The candidate wraps; it does not implement the serving interface natively.**
  `candidate.py` defines only `decode_step(q,k,g,v,beta,S)` (allocates `s_out =
  torch.empty_like(S)` per call and returns it) plus a `get_forward()` public wrapper. There is
  no `decode_pool(q,k,v,g,beta,pool,idx)` and no in-place indexed state pool. `bench/
  serving_decode.py` therefore routes it through `candidate_step`'s fallback: `pool.index_select`
  → launch → `pool.index_copy_`, mode `"gather-scatter-wrapper"` (visible in `serving.json`
  rows). The bench docstring is explicit: *"That wrapper is a stand-in, not a kernel property."*
- **No overstatement found.** `agent_report.md` states the verdict verbatim: *"verdict:
  **geomean 0.25x (goal 2.0x, every B >= 1.0x)**"* — i.e. it reports being ~4x *slower* than
  FLA, consistent with the known fact and with `roofline.json` (`geomean_speedup 0.2522`). It
  nowhere claims to be faster than FLA or to implement the native pool; `candidates.jsonl` is
  equally honest (`"interface": "public-wrapper; raw-launch"`, `"kernel 1.45-1.70x slower than
  FLA"`, v8 planned as `"serving-pool (native)"`).
- Gap: the wrapper penalty that inflates the serving-pool number is disclosed in
  `results.csv` notes and the bench docstring, but **not** in the report's roofline section
  (see final section).

## 6. Benchmark hygiene — PASS with one tooling inconsistency

- **Same metric and interface both sides.** `results.csv` rows for `baseline-fla-decode` and
  `flydsl-decode-v7` (kind=measurement, 2026-10-09T23:48:15): both `interface=serving-pool`,
  `metric=graphed_us_per_step`, `timing_method=cuda-graph-replay`,
  `tool=bench/serving_decode.py`, `physical_gpu=4`. `roofline.py` enforces pairing: *"same
  metric for BOTH sides — never mix serving-pool with wrapper numbers"*, and `roofline.json`
  shows identical byte models both sides (e.g. B=1: 12,730,560 bytes for candidate and
  baseline_fp32).
- **Medians: yes at the aggregation layer, with caveats.** `roofline.py::_times` takes the
  **median** of matching `results.csv` rows per B (baseline: median of 2 rows — kimi 7.47 and
  synthetic 7.5 → 7.48 at B=1; candidate: median of 1 row). Within a bench run,
  `graphed_us_per_step` is the **mean over 25 replays × 8 graphed steps = 200 step executions**
  of a CUDA graph — a fine estimator for fixed graph replay, but the report never states the
  statistic or the single-invocation (N=1 per candidate/B) sample size. The FLA baseline was
  measured on two gate distributions (`serving.json` + `serving_fla_synthetic.json`), agreeing
  within ~1.2%; the candidate was measured on kimi only, while the report's baseline column is
  the two-dataset median — a small asymmetry, numerically immaterial here (kimi-only geomean is
  also 0.25x).
- **Free GPU: yes by VRAM, but the stamp's own use% rule is contradicted.**
  `verify_decode_report.json` env: `physical_gpu: 4, gpu_occupancy_at_start: {vram_pct: 0.0,
  use_pct: 26.0}` (session run; my run stamped 27.0). The bench stamps read 44% and 30% use at
  23:45. But the frozen `env_stamp.py` itself defines `BUSY_USE_PCT = 15` and
  `require_free_gpu()` raises above it — so these stamps record `use_pct` values that would
  *fail* that check, while `vram_pct` (0–1%) shows no co-tenant. Either `rocm-smi` "GPU use (%)"
  is unreliable on this host or the stamp samples after the job's own activity. Mitigation: I
  re-ran on a GPU I verified idle (`rocm-smi`: GPU[4] 0% VRAM, 0% use) and reproduced identical
  accuracy values and per-gate timings, so the numbers are not contention artifacts.

## 7. Formal verdicts — PASS

`formal_report.json`: `"overall": "PROVED"`, `"n_chains": 19`, `selftest_exit: 0`; all 7
findings are `"verdict": "PROVED"` (butterfly_coverage + 3 O-equivalence + 3 S_OUT-equivalence).
**No UNKNOWN anywhere.** `agent_report.md` lists every finding as `[PROVED]` and never treats
UNKNOWN (or any other non-PROVED verdict) as a pass. It also carries the rounding-points
disclosure (`O store truncates to vector<4xbf16>`). Caveats the report omits but the JSON
discloses: the verified config is a small spot-check (`B=2, H=4`), and *"exp/rsqrt are
uninterpreted; PROVED means same math modulo the abstraction"*.

## 8. Case integrity — PASS

Lock `expected_gates` (15) vs the pack's case files:

| case file (mtime 08:08) | gates declared |
|---|---|
| D1_salts.toml | D1_synthetic_s11, D1_kimi_s11, D1_synthetic_s427, D1_kimi_s427, D1_synthetic_s90210, D1_kimi_s90210 |
| D2_warm_state.toml | D2_warm_state_2048 |
| D3_adversarial_ranges.toml | D3_deep_decay, D3_near_unity_gates, D3_large_v_x32 |
| D3_beta_and_v_paths.toml | D3_beta_extremes, D3_zero_v_from_warm |
| D4_drift.toml | D4_drift_8192_kimi |
| D5_shapes.toml | D5_B3_H1, D5_B2_H96_kimi_k3 |
| H1_holdout.toml | holdout_quick_set (hidden; seeds only in the lock) |

Visible gates sum to exactly the lock's 15 — the same 15 my re-run executed; no gate added,
removed, or renamed to make things pass. All case files (08:08) predate the lock
(08:10:08), and `run_eval` rejects any gate-set mismatch against the lock at run time.
Residual gap: the lock hashes frozen code and `bar.toml` but **not the case `.toml` contents** —
a post-lock case edit would not be hash-detected (mtime is the only witness today).

---

## What the report understates

1. **The wrapper is inside the number.** The roofline section says "same metric both sides" but
   never says the candidate's serving-pool time includes the gather→launch→scatter wrapper
   (while FLA uses its native in-place pool). That context — which is the difference between
   "kernel 1.5–1.7x slower" and "serving path 4x slower" — lives only in `results.csv` notes
   and the bench docstring. The 0.25x verdict is honest but under-explained in the report body.
2. **Sample size and statistic.** The roofline table gives no reps/statistic: values are medians
   over results.csv rows (N=2 baseline datasets, **N=1 recorded bench invocation** for the
   candidate per B), each being a mean over 200 graphed step executions. One bench pass, no
   variance reported. The two FLA datasets accidentally provide a ~1.2% stability check; the
   candidate has none.
3. **Env-stamp contradiction.** The stamps show `use_pct` 26–44% — above the frozen stamp
   tool's own `BUSY_USE_PCT = 15` — yet the runs are presented as free-GPU. The free-GPU claim
   effectively rests on `vram_pct` alone; the tool's use%-gate is evidently not meaningful on
   this host and should be fixed or its readings explained.
4. **Formal scope.** "PROVED over 19 reduction chains" omits that the formal run used config
   B=2, H=4 (not the Kimi H=96 shape) and that `exp`/`rsqrt` are uninterpreted abstractions.
   Both are disclosed in `formal_report.json` but not in the report.
5. **ISA vs PC-sampling tension.** The report asserts "issue-bound" (math/data 2.13,
   issue/data 4.75) while its own PC sampling shows 260/304 samples in WAITCNT (memory wait).
   Both are relayed verbatim from the artifacts without reconciling which bottleneck dominates
   or how they interact.
6. **Unfinished report.** "Next hypothesis" is still the framework's `TODO(agent)` placeholder —
   the one section that requires the agent's own reasoning was never filled.
7. **No VCS safety net.** The workspace is not a git repo; tamper-evidence for everything
   outside the hash-locked crate (cases, bench, task-dir sources) rests on mtimes alone.
   `eval.lock.json` hashing the case files (not just frozen code + bar) would close this.

None of the above changes the outcome: the candidate fails its own target honestly, and every
claim in `agent_report.md` is supported by the artifacts.
