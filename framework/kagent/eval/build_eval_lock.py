"""Build a frozen eval crate for a task pack.

The crate mirrors the workspace tree under frozen/ so the verifier's own
sys.path inserts resolve inside the crate with zero source modification:

    eval-crates/<task>/
      eval.lock.json                     hashes + thresholds + gate ids + holdout seeds
      frozen/<task_dirname>/verifier/verify_decode.py
      frozen/<task_dirname>/inputs.py            (verifier deps mirrored as-is)
      frozen/tools/env_stamp.py
      frozen/kda-attn/ref_fp64.py
      frozen/holdout_runner.py           framework-written, hash-locked
      frozen/holdouts.json               hidden seeds

Usage:
    python3 -m kagent.eval.build_eval_lock --pack examples/kda-serving [--holdout-seeds 4]
Rebuilding after a bar change requires the user's approval (guideline 03);
this tool refuses to overwrite a lock with a different threshold set unless
--approved-flag is passed by the binding after user confirmation.
"""
from __future__ import annotations

import argparse
import datetime
import json
import secrets
import shutil
from pathlib import Path

from ..spec.loaders import load_pack
from .crate import FROZEN_DIR, LOCK_VERSION, crate_dir, frozen_dir, sha256_file

HOLDOUT_RUNNER = '''#!/usr/bin/env python3
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
'''


def mirror_path(task_dirname: str, rel_from_task: str) -> str:
    """Dep path relative to the task dir -> crate-relative path mirroring the tree."""
    return str(Path(FROZEN_DIR) / task_dirname / rel_from_task)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--holdout-seeds", type=int, default=4)
    ap.add_argument("--regen-holdouts-only", action="store_true",
                    help="regenerate hidden seeds and rewrite the lock (crate must exist)")
    args = ap.parse_args()

    pack = load_pack(args.pack)
    task = pack.task
    name = task.name
    task_dirname = Path(task.resolve(task.paths.task_dir)).name

    base = crate_dir(name)
    frz = frozen_dir(name)
    if args.regen_holdouts_only:
        if not base.exists():
            raise SystemExit(f"crate {base} does not exist; run without --regen-holdouts-only first")
    else:
        if base.exists():
            raise SystemExit(f"crate {base} already exists; refusing to overwrite. "
                             "Rebuilds require user approval; pass --force after approval.")
        frz.mkdir(parents=True)

    seeds = [secrets.randbelow(2**30) + 1 for _ in range(args.holdout_seeds)]
    (frz / "holdouts.json").write_text(json.dumps({"seeds": seeds}, indent=2))
    (frz / "holdout_runner.py").write_text(HOLDOUT_RUNNER)

    files: dict[str, str] = {}
    if not args.regen_holdouts_only:
        src_verifier = Path(task.resolve(task.paths.verifier))
        dst = base / mirror_path(task_dirname, task.paths.verifier)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_verifier, dst)
        for dep in task.paths.verifier_deps:
            src = Path(task.resolve(dep))
            dst = base / mirror_path(task_dirname, dep)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        # hash every file under frozen/ AND the pack's case files, so editing a
        # visible case to pass is detectable (reviewer check 8)
        for p in sorted(frz.rglob("*")):
            if p.is_file():
                files[str(p.relative_to(base))] = sha256_file(p)
        bar_rel = f"bar:{task.paths.bar}"
        files[bar_rel] = sha256_file(Path(task.resolve(task.paths.bar, from_pack=True)))
        for case in pack.cases:
            rel = f"cases:{task.paths.cases}/{case.id}.toml"
            files[rel] = sha256_file(Path(task.resolve(f"{task.paths.cases}/{case.id}.toml",
                                                       from_pack=True)))

    if args.regen_holdouts_only:
        # rewrite seeds in the existing lock, rehash the two regenerated files
        lock = json.loads((base / "eval.lock.json").read_text())
        lock["holdout"]["seeds"] = seeds
        for rel in ("frozen/holdouts.json", "frozen/holdout_runner.py"):
            lock["files"][rel] = sha256_file(base / rel)
    else:
        gate_ids = sorted({g for c in pack.visible_cases() for g in c.gates})
        lock = {
            "lock_version": LOCK_VERSION,
            "task": name,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "pack_dir": str(Path(args.pack).resolve()),
            "entrypoints": {
                "verifier": mirror_path(task_dirname, task.paths.verifier),
                "holdout_runner": "frozen/holdout_runner.py",
            },
            "files": files,
            "thresholds": {k: th.value for k, th in pack.bar.thresholds.items()},
            "expected_gates": gate_ids,
            "holdout": {"enabled": True, "n_seeds": len(seeds), "seeds": seeds},
        }
    (base / "eval.lock.json").write_text(json.dumps(lock, indent=2))
    print(f"eval crate {name!r} locked at {base} "
          f"({len(files)} files hashed, holdout seeds: {len(seeds)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())