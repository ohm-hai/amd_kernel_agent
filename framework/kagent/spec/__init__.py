"""Task pack schemas and validators (Phase 1 of the plan).

A task pack is a directory of declarative data:
    task.toml     objective, target, constraints, promotion rules, paths
    bar.toml      pre-registered accuracy thresholds + change policy
    cases/*.toml  verification cases, each declaring the bug class it kills

Nothing here executes kernels; this layer makes the *rules* data so the loop
engine and the frozen evaluator can enforce them mechanically.
"""
from .schema import (  # noqa: F401
    Bar,
    BarThreshold,
    Case,
    CaseClasses,
    TaskSpec,
    check_cases,
    validate_bar,
    validate_task,
)
from .loaders import (  # noqa: F401
    load_bar,
    load_case_dir,
    load_task,
    load_pack,
)