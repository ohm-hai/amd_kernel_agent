# KAF — Kernel Agent Framework

Driver-agnostic framework for kernel-optimization AI agents. The competence
lives here, not in any LLM runtime; Cursor is one binding.

```
kagent/
  spec/      task packs: task.toml + bar.toml + cases/*.toml (validated)
  eval/      frozen hash-locked evaluator crates, hidden holdouts
  formal/    FlyDSL MLIR -> DAG -> Z3 equivalence, structural checks, lints
  evidence/  ISA / PC-sampling / roofline analyzers (gfx950 first)
  loop/      stage machine + StageReport + `kagent` CLI
  prompts/   per-stage guidance packets for models
docs/guidelines/   01 specs · 02 verification cases · 03 bars · 04 evidence · 05 review
examples/kda-serving/   reference task pack
```

Read `docs/guidelines/02-defining-verification-cases.md` before defining any
case, and `05-review-and-anti-reward-hacking.md` before running a session.

## Quick start

```bash
cd kernel-lab/framework
python3 -m kagent.loop.cli check --pack examples/kda-serving          # validate pack
python3 -m kagent.loop.cli verify --pack examples/kda-serving --candidate ../kda-serving/runs/flydsl-decode-v7
python3 -m kagent.loop.cli formal --pack examples/kda-serving --candidate ../kda-serving/runs/flydsl-decode-v7
python3 -m kagent.loop.cli evidence --pack examples/kda-serving --candidate ../kda-serving/runs/flydsl-decode-v7
python3 -m kagent.loop.cli run --pack examples/kda-serving --candidate ../kda-serving/runs/flydsl-decode-v7
```

Framework invariants:

1. the agent never edits spec/eval/framework; the eval crate is hash-locked;
2. holdouts are hidden — counts only;
3. no performance number before the frozen verifier passes;
4. every speedup claim carries artifact ids;
5. formal verdicts are PROVED / COUNTEREXAMPLE / UNKNOWN, never silently pass.