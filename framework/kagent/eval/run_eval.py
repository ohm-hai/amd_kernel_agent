"""Run the frozen evaluator: visible verify (values shown) or holdout (counts only).

    python3 -m kagent.eval.run_eval --pack examples/kda-serving \
        --candidate ../kda-serving/runs/flydsl-decode-v7 [--mode holdout]

Invariant 1: hashes are verified BEFORE and AFTER the run (mid-run tamper is
detected). Invariant 3 is recorded here: a passing verify report is the gate
the loop engine consults before running any performance stage.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from ..spec.loaders import load_pack
from .crate import crate_dir, expected_gates, load_lock, verify_lock

DEFAULT_TIMEOUT_VERIFY = 7200
DEFAULT_TIMEOUT_HOLDOUT = 3600


def _record(pack, candidate_name: str, report: dict, artifact: Path, mode: str) -> None:
    tools = Path(__file__).resolve().parents[3] / "tools"
    sys.path.insert(0, str(tools))
    try:
        import record  # noqa: PLC0415
    except Exception as e:  # pragma: no cover - optional
        print(f"(record.py unavailable: {e}; measurement NOT recorded)")
        return
    results = Path(pack.task.resolve(pack.task.paths.results))
    record.append_result(
        str(results),
        task=pack.task.name,
        candidate=candidate_name,
        kind="verifier",
        mode=mode,
        metric="verify_gates_passed",
        value=report["n_gates"] - report["n_failed"],
        unit="gates",
        tool="kagent.eval.run_eval",
        artifact=str(artifact),
        notes=f"{'PASS' if report['passed'] else 'FAIL'} "
              f"{report['n_gates'] - report['n_failed']}/{report['n_gates']}",
    )
    print(f"recorded verifier row in {results}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--mode", choices=("verify", "holdout"), default="verify")
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--allow-busy-gpu", action="store_true")
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()

    pack = load_pack(args.pack)
    task = pack.task
    name = task.name
    cand = Path(args.candidate).resolve()
    if not (cand / "candidate.py").exists():
        raise SystemExit(f"no candidate.py under {cand}")

    lock = load_lock(name)
    verify_lock(name, lock)  # pre-run check

    if args.mode == "verify":
        entry = crate_dir(name) / lock["entrypoints"]["verifier"]
        timeout = args.timeout or DEFAULT_TIMEOUT_VERIFY
    else:
        entry = crate_dir(name) / lock["entrypoints"]["holdout_runner"]
        timeout = args.timeout or DEFAULT_TIMEOUT_HOLDOUT

    if args.mode == "verify" and not args.allow_busy_gpu:
        tools = Path(__file__).resolve().parents[3] / "tools"
        sys.path.insert(0, str(tools))
        from env_stamp import require_free_gpu  # noqa: PLC0415
        require_free_gpu()

    env = dict(os.environ)
    log = cand / f"eval_{args.mode}.log"
    cmd = [sys.executable, str(entry), "--candidate", str(cand)]
    print(f"[kagent.eval] {args.mode}: {' '.join(cmd)} (timeout {timeout}s)")
    started = Path(report_path_tpl := cand / ("verify_decode_report.json" if args.mode == "verify"
                                              else "holdout_report.json"))
    t_start = started.stat().st_mtime if started.exists() else -1.0
    try:
        proc = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True, env=env)
    except subprocess.TimeoutExpired:
        log.write_text(f"TIMEOUT after {timeout}s\n")
        print(f"TIMEOUT after {timeout}s -> {log}")
        return 2
    log.write_text((proc.stdout or "") + (proc.stderr or ""))
    verify_lock(name, lock)  # post-run check (mid-run tamper detection)

    if proc.returncode != 0:
        print(f"RUN FAILED (exit {proc.returncode}); log -> {log}")
        print((proc.stdout or "")[-2000:])
        print((proc.stderr or "")[-2000:])
        return 1
    report_path = started
    if not report_path.exists() or report_path.stat().st_mtime <= t_start:
        print("REPORT STALE OR MISSING (subprocess exited 0 but wrote nothing new)")
        return 1
    report = json.loads(report_path.read_text())

    if args.mode == "verify":
        want = expected_gates(lock)
        got = sorted(r["gate"] for r in report.get("results", []))
        if want and set(got) != set(want):
            print(f"GATE-SET MISMATCH vs lock: missing={sorted(set(want) - set(got))} "
                  f"extra={sorted(set(got) - set(want))}")
            return 1
        print(f"VERIFY {'PASS' if report['passed'] else 'FAIL'}: "
              f"{report['n_gates'] - report['n_failed']}/{report['n_gates']} gates "
              f"({report.get('seconds', '?')}s) -> {report_path}")
        print(f"bar thresholds (from lock): {lock['thresholds']}")
        tail = (proc.stdout or "").splitlines()[-16:]
        print("\n".join(tail))
        if not args.no_record:
            _record(pack, cand.name, report, report_path, "verify")
        return 0 if report["passed"] else 1

    # holdout: counts only — never print values (invariant 2)
    print(f"HOLDOUT {'PASS' if report['passed'] else 'FAIL'}: "
          f"{report['n_gates'] - report['n_failed']}/{report['n_gates']} gates "
          f"({report.get('seconds', '?')}s) [values suppressed by design]")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())