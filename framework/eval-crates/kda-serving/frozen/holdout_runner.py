#!/usr/bin/env python3
"""FROZEN holdout runner — hash-locked inside the eval crate.

Runs the frozen verifier's gate set with hidden holdout seeds and reports
COUNTS ONLY (framework invariant 2). Values are suppressed on purpose.

Usage: python3 holdout_runner.py --candidate <dir>
"""
import argparse
import contextlib
import importlib.util
import io
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK = json.loads((HERE.parent / "eval.lock.json").read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    args = ap.parse_args()

    seeds = LOCK["holdout"]["seeds"]
    vpath = HERE.parent / LOCK["entrypoints"]["verifier"]
    spec = importlib.util.spec_from_file_location("frozen_verify", vpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.SEEDS = tuple(seeds)  # hidden salts for D1

    cand = Path(args.candidate)
    cspec = importlib.util.spec_from_file_location("candidate", cand / "candidate.py")
    cmod = importlib.util.module_from_spec(cspec)
    cspec.loader.exec_module(cmod)
    fwd = cmod.get_forward()

    t0 = time.perf_counter()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):  # suppress values; counts only
        res = mod.run_gates(fwd, cmod, quick=True)
    n_failed = sum(not r["passed"] for r in res)
    report = {
        "mode": "holdout",
        "candidate": str(cand),
        "passed": n_failed == 0,
        "n_gates": len(res),
        "n_failed": n_failed,
        "n_holdout_seeds": len(seeds),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    out = cand / "holdout_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"HOLDOUT {'PASS' if report['passed'] else 'FAIL'}: "
          f"{report['n_gates'] - n_failed}/{report['n_gates']} gates "
          f"(counts only; values suppressed) -> {out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
