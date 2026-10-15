"""KAF — Kernel Agent Framework (package name: kagent).

A driver-agnostic framework for kernel-optimization AI agents.

Layers (import paths):
    kagent.spec      task packs: task.toml + cases/*.toml + bar.toml, validated
    kagent.eval      frozen, hash-locked evaluator crates + hidden holdouts
    kagent.formal    FlyDSL MLIR -> DAG -> Z3 equivalence, structural checks, lints
    kagent.evidence  ISA / PC-sampling / roofline analyzers (gfx950 adapters first)
    kagent.loop      stage machine + StageReport + `kagent` CLI

Framework invariants (see docs/guidelines/):
    1. The agent never edits spec/eval/framework in a session; the eval crate is
       hash-locked and re-checked before every run; tampering is loud.
    2. Holdout cases are hidden; reports show PASS/FAIL counts only.
    3. No performance number is reportable before the frozen verifier passes.
    4. Every claimed speedup carries artifact ids.
    5. Formal verdicts are tri-state: PROVED / COUNTEREXAMPLE / UNKNOWN.
"""

__version__ = "0.1.0"