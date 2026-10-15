# Task Flow Prompt (KDA basic-flow, filled for the KDA decode task)

Copy this into a new agent session started in `/root/amd-workspace/kernel-lab`. It is
`kda/prompts/basic-flow.md` filled in for `kda-serving`, with TraceLens and rocprofv3 as
the evidence tools. (Until 2026-10-09 this file described a toy GEMM task; that task is
finished and its files remain under `runs/` and `profile/`.)

---

You are working in a task implementation workspace. Your job is to produce the best
correct implementation for the task described below.

## Before anything else

1. Read `AGENTS.md`, especially "Working agreement with the user". Follow it: build one
   version at a time and report to the user after each.
2. Read `kda-serving/contract.md` and `kda-serving/accuracy-bar.md`.
3. Read `skills/mi355x-kernel-skill/SKILL.md` and its `reference/` files 00, 01, 06, 08.
4. Decisions already made (2026-10-09): 2× means geomean over B ∈ {1, 8, 32, 128}
   with no batch size slower than FLA; kernels are written in FlyDSL.
5. Pick a free GPU (`python3 tools/env_stamp.py`, `rocm-smi`) and set `HIP_VISIBLE_DEVICES`.

## Task contract (summary; `kda-serving/contract.md` is authoritative)

- Task name: `kda-serving` decode, Kimi K3 KDA layer (H=96, K=V=128), MI355X.
- Objective: 2× FLA Triton decode, measured on FLA's serving path (in-place, indexed
  state pool), graph-captured: geomean speedup over B ∈ {1, 8, 32, 128} ≥ 2×, every
  B ≥ 1×.
- Correctness: full decode verifier PASS (15 gates) against the pre-registered bar,
  before any performance number is reported. Reduced-precision state must be fp16
  (bf16 fails the bar).
- Allowed approaches: FlyDSL kernels (continue from v7 or start fresh).
  No dependence on unverified input properties: the KDA^2 blog's reward hacks
  (hard-coded normalization constants, hard-coded sequence boundaries, history
  truncation, reduced-precision shortcuts that pass weak tests) are rejected on sight.
- Validation: `HIP_VISIBLE_DEVICES=<gpu> python3 kda-serving/verifier/verify_decode.py --candidate kda-serving/runs/<name>`
- Evaluation: `python3 kda-serving/bench/serving_decode.py --candidate kda-serving/runs/<name> --out kda-serving/profile/<run>/serving.json`
- Efficiency evidence: `kda-serving/tracelens/profile_decode.py` then TraceLens with
  `--extension_file kda-serving/tracelens/kda_extension.py --gpu_arch_platform MI355X`;
  hardware counters per `skills/mi355x-kernel-skill/reference/08-mi355x-metric-names.md`.
- Promotion: contract.md "Promotion criteria" (verifier PASS, native serving interface,
  faster on the serving path with per-B breakdown).

## Current state (2026-10-09)

- Best candidate: `flydsl-decode-v7`, verifier 15/15, kernel 1.45–1.70× slower than FLA.
  rocprofv3 shows it issue-bound (74% of wave lifetime waiting to issue vs FLA 27%), not
  bandwidth-bound. It has no native state-pool support (0.19–0.35× through a
  gather/scatter wrapper).
- Next candidate (v8) plan, in order: native in-place indexed pool; hoist state loads
  above the norm trees; interleave independent reduction trees; DPP sum steps; then
  fp16 state storage.

## Workflow

1. Read the workspace, existing implementations, verifier and benchmarks.
2. Write the plan draft to `kda-serving/docs/draft.md` (baseline, risks, ranked
   candidate directions, exact commands, promotion/rejection evidence).
3. Implement one candidate at a time in `kda-serving/runs/<name>/candidate.py`
   (interface: see the docstring of `verifier/verify_decode.py`).
4. Verify, then measure, then profile. Record every number with `tools/record.py`
   (`results.csv`) and the candidate's status (`candidates.jsonl`).
5. Report to the user and wait for input before starting the next version.

Do not start implementation until the draft exists.
