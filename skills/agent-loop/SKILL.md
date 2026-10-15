---
name: agent-loop
description: >-
  Run the KAF (kagent) per-candidate loop for kernel work in kernel-lab:
  frozen verify, formal verification, bench, ISA/PC-sampling/roofline
  evidence, and the agent report. Use when optimizing KDA decode (or any
  task with a kagent task pack), after writing a candidate, or before
  reporting performance numbers.
---

# Agent loop (KAF binding)

The framework lives in `kernel-lab/framework` (package `kagent`). The Cursor
binding is THIN: everything goes through the `kagent` CLI; you never edit the
pack, the eval crate, the verifier, the bar, or the framework in a session.

## Per-candidate loop (run from `kernel-lab/framework`)

```bash
cd /root/amd-workspace/kernel-lab/framework
export HIP_VISIBLE_DEVICES=4   # or whichever GPU env_stamp.py says is free

python3 -m kagent.loop.cli check --pack examples/kda-serving
python3 -m kagent.loop.cli run --pack examples/kda-serving \
    --candidate ../kda-serving/runs/<name> --dtype float32
```

`run --stages` defaults to `verify,formal,evidence,report` and enforces the
order: no bench/evidence/report before the frozen verifier passes (invariant 3).
Add `bench` explicitly when you want the benchmark to run
(`--stages verify,bench,evidence,report`).

Stage meanings and what to read (guidelines in `framework/docs/guidelines/`):

| stage | command | read |
|---|---|---|
| verify | frozen verifier, 15 gates | `runs/<name>/verify_decode_report.json` |
| holdout | hidden seeds, counts only | `runs/<name>/holdout_report.json` |
| formal | Z3 equivalence + butterfly checks | `runs/<name>/formal_report.json` |
| bench | task bench script via results.csv | `results.csv` rows |
| evidence | isa/pcs/roofline artifacts | `isa.json`, `pcs.json`, `roofline.json` |
| report | renders `agent_report.md` | the report + stage timings |

Prompt packets for each stage live in `framework/kagent/prompts/` — read the
packet for the stage you are in before acting on its output.

## Rules that the CLI enforces (do not fight them)

1. The eval crate is hash-locked; tampering aborts (`kagent.eval.selftest_tamper`).
2. Holdout values never appear in reports (counts only).
3. No performance number before the frozen verify passes.
4. Roofline refuses to mix serving-pool and public-wrapper metrics.
5. Formal verdicts are PROVED / COUNTEREXAMPLE(+witness) / UNKNOWN(+pattern);
   UNKNOWN is never a pass.

## Review (independent)

After the report, spawn a reviewer subagent (generalPurpose) with the prompt in
`framework/bindings/cursor/reviewer-prompt.md`. It writes `review.md` next to
the candidate. Do not implement its findings in the same session without
showing the user first.

## Cadence (user decision 2026-10-09)

One candidate -> report -> wait for the user. The engine prints per-stage
wall-clock times; include them in your message to the user.