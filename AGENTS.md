# Kernel Lab — Combined KDA × TraceLens Agent Instructions

This workspace fuses two agent workflows:

1. **KDA (Kernel Design Agents, NVlabs)** — the implementation/decision loop:
   task contract → plan draft → candidates → validate → measure → promote/reject with evidence.
   Reference: `/root/amd-workspace/kda` (`docs/agent-flow.md`, `prompts/basic-flow.md`, `CLAUDE.md`).
2. **TraceLens Agent (AMD)** — the performance-evidence engine:
   PyTorch-profiler trace → perf report → parallel category subagents → prioritized `analysis.md`.
   Reference: `/root/amd-workspace/TraceLens/TraceLens/Agent/` (skills linked in `skills/`).

## Working agreement with the user (standing decisions)

These came from the user in chat. A new session must follow them without re-asking.

- **Goal:** 2× FLA Triton on KDA decode for the Kimi K3 layer on MI355X, measured on
  FLA's serving path: geomean speedup over B ∈ {1, 8, 32, 128} ≥ 2×, with no batch
  size slower than FLA (decided 2026-10-09).
- **Language:** FlyDSL (decided 2026-10-09). v8 may continue from v7 or start fresh.
- **Plan approved:** Path A (dispatch and kernel parity) then Path B (reduced-precision
  state, with an accuracy report). The pre-registered bar selects fp16 over bf16.
- **Cadence:** build one version at a time and report after each one. The user gives
  input between versions; don't run several versions unattended.
- **Constraints:** no Docker, no model downloads (no Hugging Face token).
- **Changing the accuracy bar** (`kda-serving/accuracy-bar.md`) needs the user's approval.

## GPU hygiene and evidence

- This node is shared; other tenants' jobs appear on GPUs without warning (2026-10-09:
  GPUs 0, 1, 3, 5, 6, 7 held 58–90% VRAM). Run `python3 tools/env_stamp.py` or call
  `require_free_gpu()` before measuring, and set `HIP_VISIBLE_DEVICES` to a free GPU.
  The task benchmarks and verifier refuse to run on a busy GPU.
- Record every measurement with `tools/record.py` (`append_result`, `upsert_candidate`).
  It enforces one schema across tasks and stamps GPU and tool versions on each row.

## Active tasks

**`kda-serving/`** — Kimi K3 KDA layer (H=96, K=V=128) on MI355X. Read `contract.md`
(target, official measurements, promotion criteria), `accuracy-bar.md`, then
`docs/draft.md`. Evidence: `results.csv`, `candidates.jsonl`. **Start here.**
Current best: **flydsl-decode-v10** (V-major coalesced state layout, pure fp32 state):
serving geomean 1.12x of FLA (B=1 1.32x, B=8 1.11x, B=32 1.01x, B=128 1.07x — ALL batch sizes beat Triton FLA in pure FP32);
frozen verify 15/15, holdout 10/10, formal PROVED, review ACCEPT. Next: multi-token T=4 verification (pure fp32), or fp16 state (Path B).

**`kda-attn/`** — AMD recreation of the KDA² blog (agentic optimization of Kimi Delta
Attention on MI355X vs the FLA Triton baseline). Contract, hardened fp64 verifier
(6 gates), benchmark harness (`bench --fast` for tight feedback), and evidence files.

## Division of labor

| Phase | Owner | Artifact |
|---|---|---|
| Task contract | KDA (`prompts/task-flow.md`) | filled contract in prompt or `docs/plan.md` |
| Baseline + candidates | KDA loop | `runs/`, `candidates.jsonl` (via `tools/record.py`) |
| Profiling (per candidate) | **TraceLens** (replaces KDA's ncu-report-skill, which is CUDA-only; this box is MI355X/ROCm) | `profile/<candidate>/` (trace.json + perf_report.xlsx + CSVs) |
| Trace analysis | **TraceLens orchestrator skill** (`skills/tracelens-orchestrator/SKILL.md`) | `profile/<candidate>/analysis.md` |
| Promotion decision | KDA promotion rule | `results.csv`, `candidates.jsonl` status field |

## Repository rules (inherited from KDA)

- Keep the reusable workflow here; task-specific code lives under the task's own directory.
- Generated outputs go in `runs/`, `profile/`, `outputs/` — never at workspace root.
- Every measurement is a row in the task's `results.csv` and every candidate a line in `candidates.jsonl` with its parent and status, plus the rejection reason if rejected. Write both through `tools/record.py`, which enforces the schema. (`kda-attn/benchmark.csv` predates the schema.)
- One profiler run = one directory under `profile/<candidate_or_run>/` — never reuse.

## TraceLens usage on this machine (MI355X, ROCm 7, PyTorch 2.9)

- TraceLens is installed in `/opt/venv` (package + repo at `/root/amd-workspace/TraceLens`).
- Generate a trace from any training/eval run with `torch.profiler` (template: `/root/amd-workspace/gen_trace.py`).
- Quick report:
  `TraceLens_generate_perf_report_pytorch --profile_json_path <trace.json> --output_xlsx_path profile/<name>/perf_report.xlsx --output_csvs_dir profile/<name>/csvs`
  (GPU-arch roofline tables: `--gpu_arch_platform MI355X`, added locally 2026-10-09 with measured max-achievable FLOPs.)
- Full agentic analysis: follow `skills/tracelens-orchestrator/SKILL.md` — it spawns category subagents (gemm-analyzer, moe-analyzer, cpu-idle-analyzer, …) and writes `analysis.md`. Subagent specs live inside the skill's `agents/` directory.
- Trace comparison (baseline vs candidate): pass `--comparison_json_path` with the second trace, or use the orchestrator's `comparative` mode — this maps 1:1 onto KDA's "did the candidate improve the metric" question.
- Custom kernels (FLA, FlyDSL, anything without an aten op) need named torch ops and a TraceLens extension, or they land in "other" with no model. For KDA: `kda-serving/tracelens/kda_ops.py` + `kda_extension.py`, profiled with `record_shapes=True`.

## Promotion rule (KDA)

Promote a candidate only when (a) correctness validation passes, and (b) TraceLens evidence shows the target metric improved or held — e.g., lower `idle_time`, higher GEMM TFLOPS/s, smaller exposed-comm share in the GPU-timeline table. Record the exact table rows that justified the decision.
## Domain skill: MI355X

`skills/mi355x-kernel-skill/` is the AMD analog of KDA's B200-focused
`ncu-report-skill`: gfx950 architecture facts, toolchain traps (measured), six
analysis dimensions mapped to rocprof/TraceLens, diagnosis playbook with our
campaign's actual cases, and measured baselines. Read it BEFORE planning any
kernel work on this box. File numbers 05/06/08 match the NVIDIA skill; 08 lists
verified gfx950 counter names (and the `FETCH_SIZE` trap).

KernelWiki (`kda/skills/KernelWiki`) also has gfx950 pages: query it with
`python3 scripts/query.py "MI3555X"` from that directory. (Typo-fixed list:
use `python3 scripts/query.py "MI355X"`.)

## Agent toolchain: KAF (`framework/`, package `kagent`)

Built 2026-10-10. The per-candidate loop is mechanical and binding-agnostic;
Cursor drives it through `skills/agent-loop/SKILL.md`.

- **Evaluator lock rule:** `kda-serving/verifier/verify_decode.py`,
  `accuracy-bar.md`, and the pack files live inside a hash-locked frozen crate
  (`framework/eval-crates/kda-serving/`). An agent must NEVER edit the
  verifier, the bar, the pack, or the framework in a session; bar changes need
  the user's approval plus a lock rebuild (`kagent.eval.build_eval_lock`) and
  the tamper self-test must pass afterwards. This upgrades the pre-registered
  bar from a social control to a mechanical one.
- **Per candidate:** `cd framework && python3 -m kagent.loop.cli run --pack
  examples/kda-serving --candidate ../kda-serving/runs/<name>` — runs frozen
  verify -> formal (Z3 + butterfly checks) -> evidence (isa/pcs/roofline) ->
  report (`agent_report.md`), and refuses bench/evidence/report before the
  frozen verify passes. Stage prompt packets: `framework/kagent/prompts/`.
- **Independent review:** after the report, spawn a reviewer subagent with
  `framework/bindings/cursor/reviewer-prompt.md` (it re-runs the frozen
  evaluator on a hidden holdout and writes `review.md`).
- **Guidelines that steer the loop:** `framework/docs/guidelines/02-defining-
  verification-cases.md` (case classes + the post-mortem rule) and
  `05-review-and-anti-reward-hacking.md`. Read 02 before adding any case.
- Evidence rules: same metric both sides (`graphed_us_per_step` on the
  serving-pool beats `kernel_us_median` on the wrapper for promotion claims);
  roofline refuses mixed pairs; formal verdicts are tri-state and UNKNOWN is
  never a pass.
