# Task pack — kda-serving (reference pack)

- Task: 2x FLA Triton on KDA decode (Kimi K3 layer) on MI355X, serving path,
  geomean over B ∈ {1,8,32,128}, no B slower than FLA. Language: FlyDSL.
- Physics: fp32-state B=128 roofline floor sits above 2x of FLA, so the target
  needs reduced-precision state; the bar selects fp16 (passes: worst state
  rRMSE 2.0e-3) over bf16 (fails: 1.6e-2 on drift).
- Full physics + mode table: `../../../kda-serving/contract.md`.

## Files

- `task.toml` — objective, target (metric `graphed_us_per_step`, serving-pool),
  promotion rules, paths into the real task dir.
- `bar.toml` — pre-registered 2026-10-09 thresholds; change policy requires
  user approval + eval-lock rebuild.
- `cases/` — 6 visible cases + 1 hidden holdout; each names the bug class it
  kills and the campaign incident that motivated it.

## Incident register (guideline 05; inherited by every session)

1. **Synthetic-gates lesson** → adversarial deep-decay case (real Kimi gates
   decay ~600 bits / 64 tokens; synthetic -|N(0,1)| reaches p99 91).
2. **bf16-state lesson** → drift case + state-rrmse bar entry.
3. **Tolerance-loosening lesson** → pre-registered bar + frozen eval lock.
4. **Wrapper lesson** → serving-pool metric + interface promotion rule
   (v7 measures 0.25x on the serving path; the wrapper dominates).
5. **Reference-bug lesson** → cases must be able to fail the reference; the
   frozen evaluator runs the baseline through the same gates.
6. **Env-stamp lesson (first review, 2026-10-10)** → rocm-smi "GPU use %" is
   noisy (26–44% on an idle GPU); free-GPU decisions must rest on VRAM%
   (0–1%) plus a repeat check. The reviewer subagent caught this on v7.
7. **Case-tamper lesson (first review, 2026-10-10)** → the eval lock hashes
   the pack's case files, so editing a visible case to pass is detectable.

Rebuilding the eval lock (after an approved bar change):

    python3 -m kagent.eval.build_eval_lock --pack . [--holdout-seeds 4]
    python3 -m kagent.eval.selftest_tamper --task kda-serving