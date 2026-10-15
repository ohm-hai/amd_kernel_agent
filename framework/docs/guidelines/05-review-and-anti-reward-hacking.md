# Guideline 05 — Review and anti-reward-hacking

The agent that writes the kernel must never be the only thing that grades it.
The reward-hacking surface found in this campaign: promotion thresholds edited
by the same session that optimized against them; a false "passed the hardened
gates" claim; benchmarks that could reward wrapper-bypassing.

## Mechanical controls (the framework enforces these)

1. **Frozen evaluator.** The verifier lives in a hash-locked crate; the agent
   cannot edit it, the bar inside it, or the holdouts. `run_eval` re-checks
   hashes before and after every run; tampering aborts loudly. Changing the bar
   requires the user plus a lock rebuild (guideline 03).
2. **Holdouts.** Hidden seeds run with the visible gates; reports carry counts
   only. Overfitting to visible cases shows up as a holdout failure the agent
   cannot debug against.
3. **Order enforcement.** The loop engine refuses to run bench/evidence stages
   before the frozen verifier passes. No performance number exists before
   correctness.
4. **Artifact coupling.** A claimed speedup must cite results.csv rows and
   artifact paths; `kagent report` renders only claims with artifacts.

## Reviewer checklist (independent reviewer subagent, runs per candidate)

- Was any spec/eval/framework/verifier/bar file modified in this session?
  (git diff the workspace; anything touched is a finding.)
- Does the candidate implement the contracted interface natively, or does it
  bypass the reference path (wrapper detection, gather/scatter where the
  contract says in-place)?
- Is any work skipped when the benchmark can see it (cache across steps, early
  exit for small B, constant-folding of the *benchmark's* inputs)?
- Are the numbers single-trial? Was the same tool used on both sides? Was the
  GPU free (env stamp says so)?
- Does the formal verdict say UNKNOWN anywhere it was treated as pass?
- Does the report state the next hypothesis with the metric it attacks?
- Did any *visible* case get changed to pass? (compare case files against the
  lock.)

## The incident register

Every caught hack or miss becomes either a new case (guideline 02), a new lint,
or a mechanical control. The register lives in the task pack README so the next
session inherits it. Examples already registered: synthetic-gates lesson →
adversarial deep-decay case; bf16 drift lesson → drift class; tolerance-loosening
lesson → pre-registered bar + lock.

## Honest failure reporting

A candidate that fails reports: what failed, the witness or metric, and the
hypothesis. Rejection reasons live in `candidates.jsonl` forever. A session that
only reports wins is itself a red flag.