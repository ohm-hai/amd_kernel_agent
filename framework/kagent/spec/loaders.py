"""Load and validate task packs (task.toml + bar.toml + cases/*.toml)."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import tomli

from .schema import (
    Bar,
    BarThreshold,
    Case,
    CaseClasses,
    ConstraintSpec,
    PathsSpec,
    PromotionSpec,
    TargetSpec,
    TaskSpec,
    check_cases,
    validate_bar,
    validate_task,
)


@dataclass
class Pack:
    task: TaskSpec
    bar: Bar
    cases: list[Case]
    pack_dir: str

    def visible_cases(self) -> list[Case]:
        return [c for c in self.cases if not c.hidden]

    def holdout_cases(self) -> list[Case]:
        return [c for c in self.cases if c.hidden]


def _req(d: dict, key: str, ctx: str):
    if key not in d:
        raise ValueError(f"{ctx}: missing required key {key!r}")
    return d[key]


def load_task(pack_dir: str) -> TaskSpec:
    path = os.path.join(pack_dir, "task.toml")
    with open(path, "rb") as f:
        d = tomli.load(f)
    tdict = _req(d, "task", path)
    tgt = _req(d, "target", path)
    target = TargetSpec(
        metric=_req(tgt, "metric", "[target]"),
        aggregation=_req(tgt, "aggregation", "[target]"),
        baseline=_req(tgt, "baseline", "[target]"),
        grid=list(_req(tgt, "grid", "[target]")),
        goal=float(_req(tgt, "goal", "[target]")),
        per_point_min_ratio=float(tgt.get("per_point_min_ratio", 1.0)),
    )
    con = d.get("constraints", {})
    constraints = ConstraintSpec(
        gpus=con.get("gpus", "free-only"),
        no_docker=bool(con.get("no_docker", True)),
        no_downloads=bool(con.get("no_downloads", True)),
        notes=list(con.get("notes", [])),
    )
    promo = _req(d, "promotion", path)
    promotion = PromotionSpec(rules=list(_req(promo, "rules", "[promotion]")))
    pth = _req(d, "paths", path)
    paths = PathsSpec(
        task_dir=_req(pth, "task_dir", "[paths]"),
        verifier=_req(pth, "verifier", "[paths]"),
        bench=_req(pth, "bench", "[paths]"),
        bar=_req(pth, "bar", "[paths]"),
        cases=_req(pth, "cases", "[paths]"),
        results=_req(pth, "results", "[paths]"),
        candidates=_req(pth, "candidates", "[paths]"),
        verifier_deps=list(pth.get("verifier_deps", [])),
    )
    task = TaskSpec(
        name=_req(tdict, "name", "[task]"),
        objective=_req(tdict, "objective", "[task]"),
        language=tdict.get("language", ""),
        reference=tdict.get("reference", ""),
        target=target,
        constraints=constraints,
        promotion=promotion,
        paths=paths,
        pack_dir=os.path.abspath(pack_dir),
    )
    errs = validate_task(task)
    if errs:
        raise ValueError("invalid task.toml:\n  " + "\n  ".join(errs))
    return task


def load_bar(pack_dir: str, bar_rel: str) -> Bar:
    path = os.path.join(pack_dir, bar_rel)
    with open(path, "rb") as f:
        d = tomli.load(f)
    raw = _req(d, "thresholds", path)
    thresholds: dict[str, BarThreshold] = {}
    for key, val in raw.items():
        thresholds[key] = BarThreshold(
            key=key,
            value=float(val["value"]),
            applies_to=val.get("applies_to", "output"),
            unit=val.get("unit", "ratio"),
        )
    bar = Bar(
        pre_registered=_req(d, "pre_registered", path),
        change_policy=_req(d, "change_policy", path),
        thresholds=thresholds,
        file=path,
    )
    errs = validate_bar(bar)
    if errs:
        raise ValueError(f"invalid {bar_rel}:\n  " + "\n  ".join(errs))
    return bar


def load_case_dir(pack_dir: str, cases_rel: str) -> list[Case]:
    cases: list[Case] = []
    pattern = os.path.join(pack_dir, cases_rel, "*.toml")
    for path in sorted(glob.glob(pattern)):
        with open(path, "rb") as f:
            d = tomli.load(f)
        ctx = os.path.basename(path)
        cases.append(Case(
            id=_req(d, "id", ctx),
            klass=_req(d, "class", ctx),
            generator=_req(d, "generator", ctx),
            params=dict(d.get("params", {})),
            tolerance_ref=_req(d, "tolerance_ref", ctx),
            kills=_req(d, "kills", ctx),
            incident=_req(d, "incident", ctx),
            gates=list(_req(d, "gates", ctx)),
            hidden=bool(d.get("hidden", False)),
            extra={k: v for k, v in d.items() if k not in {
                "id", "class", "generator", "params", "tolerance_ref",
                "kills", "incident", "gates", "hidden"}},
        ))
    if not cases:
        raise FileNotFoundError(f"no case files under {pattern}")
    return cases


def load_pack(pack_dir: str) -> Pack:
    """Load a pack and run ALL validators; raises ValueError on any violation."""
    pack_dir = os.path.abspath(pack_dir)
    task = load_task(pack_dir)
    bar = load_bar(pack_dir, task.paths.bar)
    cases = load_case_dir(pack_dir, task.paths.cases)
    errs = check_cases(cases, bar)
    if errs:
        raise ValueError("case guideline violations (docs/guidelines/02):\n  " + "\n  ".join(errs))
    return Pack(task=task, bar=bar, cases=cases, pack_dir=pack_dir)


def case_by_id(pack: Pack, case_id: str) -> Case:
    for c in pack.cases:
        if c.id == case_id:
            return c
    raise KeyError(f"case {case_id!r} not in pack {pack.task.name}")