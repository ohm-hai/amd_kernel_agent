"""Schemas for task packs: task.toml, cases/*.toml, bar.toml.

Design rules (docs/guidelines/01, 02, 03):
  - every field that a rule needs is data, so `kagent check` can enforce it;
  - tolerances are always *references* into bar.toml, never inline numbers;
  - every verification case declares the bug class it kills and the incident
    that motivated it (the post-mortem rule).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------- task.toml


@dataclass
class TargetSpec:
    metric: str                  # results.csv metric name, e.g. graphed_wall_us
    aggregation: str             # geomean | mean | max
    baseline: str                # candidate name of the baseline in candidates.jsonl
    grid: list[int]              # batch sizes / context lengths measured
    goal: float                  # geomean (or aggregated) speedup must be >= goal
    per_point_min_ratio: float   # no grid point slower than baseline


@dataclass
class ConstraintSpec:
    gpus: str = "free-only"      # free-only -> require_free_gpu() is mandatory
    no_docker: bool = True
    no_downloads: bool = True
    notes: list[str] = field(default_factory=list)


@dataclass
class PromotionSpec:
    rules: list[str]             # each rule must be checkable by the loop engine


@dataclass
class PathsSpec:
    task_dir: str                # relative to the pack dir
    verifier: str                # relative to task_dir
    bench: str                   # relative to task_dir
    bar: str                     # relative to pack dir
    cases: str                   # dir, relative to pack dir
    results: str                 # inside task_dir
    candidates: str              # inside task_dir
    verifier_deps: list[str] = field(default_factory=list)  # extra files the verifier imports, relative to task_dir


@dataclass
class TaskSpec:
    name: str
    objective: str
    language: str                # implementation language constraint (e.g. flydsl)
    reference: str               # what the baseline is
    target: TargetSpec
    constraints: ConstraintSpec
    promotion: PromotionSpec
    paths: PathsSpec
    pack_dir: str = ""

    def resolve(self, rel: str, *, from_pack: bool = False) -> str:
        """Resolve a path from the pack (from_pack=True) or the task dir."""
        import os
        base = self.pack_dir if from_pack else os.path.join(self.pack_dir, self.paths.task_dir)
        return os.path.normpath(os.path.join(base, rel))


AGGREGATIONS = {"geomean", "mean", "max"}


def validate_task(t: TaskSpec) -> list[str]:
    errs: list[str] = []
    if not t.name or not t.objective:
        errs.append("[task] name and objective are required")
    if len(t.objective) < 20:
        errs.append("[task].objective must state a measurable goal (docs 01)")
    if t.target.aggregation not in AGGREGATIONS:
        errs.append(f"[target].aggregation must be one of {sorted(AGGREGATIONS)}")
    if t.target.goal <= 1.0:
        errs.append("[target].goal must be > 1.0 (a speedup)")
    if not t.target.grid:
        errs.append("[target].grid must list measured points")
    if t.target.per_point_min_ratio < 1.0:
        errs.append("[target].per_point_min_ratio < 1.0 would allow slower-than-baseline wins")
    if not t.promotion.rules:
        errs.append("[promotion].rules must list checkable rules (docs 01)")
    for r in t.promotion.rules:
        if len(r) < 20:
            errs.append(f"[promotion].rule too vague to check: {r!r}")
    return errs


# ---------------------------------------------------------------- bar.toml


@dataclass
class BarThreshold:
    key: str                     # dotted key, e.g. thresholds.o_rrmse
    value: float
    applies_to: str              # output | state | input
    unit: str = "ratio"


@dataclass
class Bar:
    pre_registered: str          # date the bar was frozen
    change_policy: str
    thresholds: dict[str, BarThreshold]  # key -> threshold, e.g. "o_rrmse"
    file: str = ""

    def get(self, key: str) -> BarThreshold:
        if key not in self.thresholds:
            raise KeyError(f"bar has no threshold {key!r} (bar={self.file})")
        return self.thresholds[key]


def validate_bar(b: Bar) -> list[str]:
    errs: list[str] = []
    if not b.pre_registered:
        errs.append("[bar].pre_registered date is required (docs 03)")
    if not b.change_policy or "approval" not in b.change_policy:
        errs.append("[bar].change_policy must require user approval (docs 03)")
    if not b.thresholds:
        errs.append("bar must define at least one threshold")
    for k, th in b.thresholds.items():
        if th.value <= 0:
            errs.append(f"threshold {k} must be > 0")
        if th.applies_to not in {"output", "state", "input"}:
            errs.append(f"threshold {k}.applies_to must be output|state|input")
    return errs


# ---------------------------------------------------------------- cases


class CaseClasses:
    NUMERIC = "numeric"          # sampled correctness vs the fp64 reference
    STRUCTURAL = "structural"    # probes that force a specific computation path
    DRIFT = "drift"              # long-horizon error accumulation
    ADVERSARIAL = "adversarial"  # distribution extremes that kill range hacks
    HOLDOUT = "holdout"          # hidden seeds; counts-only reporting
    ALL = (NUMERIC, STRUCTURAL, DRIFT, ADVERSARIAL, HOLDOUT)


@dataclass
class Case:
    id: str
    klass: str                   # one of CaseClasses.ALL
    generator: str               # dotted path resolvable from the task dir
    params: dict[str, Any]
    tolerance_ref: str           # key into bar.toml, e.g. "o_rrmse"
    kills: str                   # bug class this case kills
    incident: str                # the campaign incident that motivated it
    gates: list[str]             # verifier gate ids covered by this case
    hidden: bool = False         # True -> holdout; counts-only in reports
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_holdout(self) -> bool:
        return self.klass == CaseClasses.HOLDOUT


MIN_INCIDENT_LEN = 40


def validate_case(c: Case, bar: Bar, seen_ids: set[str]) -> list[str]:
    errs: list[str] = []
    if not c.id:
        errs.append("case id required")
    if c.id in seen_ids:
        errs.append(f"duplicate case id {c.id!r}")
    seen_ids.add(c.id)
    if c.klass not in CaseClasses.ALL:
        errs.append(f"{c.id}: class {c.klass!r} not in {CaseClasses.ALL}")
    if c.is_holdout != c.hidden:
        errs.append(f"{c.id}: hidden must be True exactly for class=holdout")
    try:
        bar.get(c.tolerance_ref)
    except KeyError as e:
        errs.append(f"{c.id}: {e} (tolerances are bar references, docs 02)")
    if not c.kills or len(c.kills) < 10:
        errs.append(f"{c.id}: kills must name the bug class (docs 02)")
    if not c.incident or len(c.incident) < MIN_INCIDENT_LEN:
        errs.append(f"{c.id}: incident must cite the motivating incident "
                    f"(>={MIN_INCIDENT_LEN} chars, docs 02)")
    if not c.gates:
        errs.append(f"{c.id}: gates must list the verifier gate ids covered")
    if not c.generator:
        errs.append(f"{c.id}: generator required")
    return errs


def check_cases(cases: list[Case], bar: Bar) -> list[str]:
    """Guideline enforcement (docs 02). Returns a list of violations."""
    errs: list[str] = []
    seen: set[str] = set()
    for c in cases:
        errs.extend(validate_case(c, bar, seen))
    covered = {CaseClasses.NUMERIC, CaseClasses.STRUCTURAL, CaseClasses.DRIFT, CaseClasses.ADVERSARIAL}
    missing = covered - {c.klass for c in cases if not c.hidden}
    if missing:
        errs.append(f"visible case set must cover classes {sorted(covered)}; missing {sorted(missing)}")
    if not any(c.hidden for c in cases):
        errs.append("pack must declare at least one hidden holdout case (docs 02)")
    return errs