"""Planted-bug regression tests for the formal engine.

    cd kernel-lab/framework && python3 -m pytest kagent/formal/bugs/test_formal.py -x -q

Requires the base v7 candidate (for IR dumps); compiles each variant once and
caches the MLIR dump under bugs/generated/<name>/dump/. GPU: FlyDSL compilation
uses the current HIP device if visible; no kernels are launched.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

FRAMEWORK = Path(__file__).resolve().parents[3]
TASK_RUNS = FRAMEWORK.parent / "kda-serving" / "runs"
BASE = TASK_RUNS / "flydsl-decode-v7"
GEN = Path(__file__).resolve().parent / "generated"

sys.path.insert(0, str(FRAMEWORK))

from kagent.formal.check import run_check  # noqa: E402
from kagent.formal.ir import Unsupported  # noqa: E402


def _dump(variant: Path) -> Path:
    dump = variant / "dump"
    have = dump.exists() and list(dump.rglob("*canonicalize*.mlir"))
    if not have:
        if dump.exists():
            shutil.rmtree(dump)
        os.environ.setdefault("HIP_VISIBLE_DEVICES", "4")
        from kagent.formal.check import dump_candidate_ir
        dump_candidate_ir(variant, dump)
    return dump


def _check(variant: Path) -> dict:
    try:
        return run_check(variant, ir_dir=_dump(variant))
    except Unsupported as e:
        return {"overall": "UNKNOWN", "findings": [], "detail": str(e)}


def test_base_v7_proved():
    report = _check(BASE)
    assert report["overall"] == "PROVED", report["findings"]


@pytest.mark.parametrize("name", [
    "bug_01_no_scale_beta",
    "bug_02_wrong_shuffle",
    "bug_03_no_l2n",
    "bug_04_off_by_one",
])
def test_bug_caught(name: str):
    variant = GEN / name
    expected = (variant / "expected.json")
    assert expected.exists(), f"run gen_bugs first (missing {expected})"
    report = _check(variant)
    assert report["overall"] in ("COUNTEREXAMPLE", "UNKNOWN"), \
        f"{name}: bug NOT caught (overall={report['overall']})"
    if report["overall"] == "COUNTEREXAMPLE":
        # a witness or a named check must exist — no silent failures
        assert report["findings"], name