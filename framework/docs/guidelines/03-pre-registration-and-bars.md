# Guideline 03 — Pre-registration and accuracy bars

The bar exists so that "is this accurate enough" is decided before the first
candidate exists, by the user, from physics and reference behavior — not after
seeing a candidate's numbers.

## Rules

1. **The bar is written before any candidate** and lives in `bar.toml` with its
   `pre_registered` date. The frozen evaluator copies thresholds into the eval
   lock; afterwards the bar and the lock must agree.
2. **Changing the bar requires the user's approval** and a lock rebuild, and
   must add a dated note explaining why. `bar.toml.change_policy` states this;
   the loop engine refuses candidates whose report came from a mismatched lock.
3. **Thresholds come from evidence, not comfort.** The kda-serving bar was
   pre-registered from the fp64 reference and FLA behavior: output rRMSE ≤ 8e-3,
   output max-abs ≤ 5e-2, state rRMSE ≤ 1e-2. State which baseline fails which
   threshold and why that is acceptable (or not) in the pack README.
4. **The bar selects the dtype, not vice versa.** When reduced-precision state
   is on the table, run the *emulation* through the bar first (fp16 passed:
   worst state rRMSE 2.0e-3; bf16 failed: 1.6e-2 on drift gates), then build.
   Never loosen a threshold to admit a candidate.
5. **One bar per task, all gates reference it.** A candidate report that does
   not name the bar version is invalid.
6. **Baselines go through the same gates.** If FLA cannot pass the bar, that is
   a finding to record, not a reason to quietly carve out an exemption.

## What pre-registration is not

It is not a substitute for a frozen evaluator. A pre-registered bar that lives
in the same editable workspace as the agent is a social control; the eval lock
makes it a mechanical one (guideline 05).