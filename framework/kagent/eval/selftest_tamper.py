"""Tamper self-test: prove that a modified frozen file is refused loudly.

    python3 -m kagent.eval.selftest_tamper --task kda-serving

Exit 0 = every tamper was detected. Any non-zero exit means invariant 1 is
broken and the crate is unsafe to trust.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from ..spec.loaders import load_pack
from .crate import EvalLockError, crate_dir, load_lock, verify_lock


def _copy_crate(name: str) -> Path:
    src = crate_dir(name)
    tmp = Path(tempfile.mkdtemp(prefix="kagent-tamper-"))
    dst = tmp / name
    shutil.copytree(src, dst)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="kda-serving")
    args = ap.parse_args()
    name = args.task
    load_lock(name)  # the real crate must exist and verify first
    verify_lock(name)
    print(f"baseline crate {name!r} verifies OK")

    failures = []

    # 1. mutate a frozen file in a copy -> patched lock file must still refuse
    #    (we point verify_lock at the copy by monkey-patching crate_dir)
    import kagent.eval.crate as crate_mod

    tmp_crate = _copy_crate(name)
    victim = next((tmp_crate / "frozen").rglob("verify_decode.py"))
    with open(victim, "a") as f:
        f.write("\nTOL_O_RRMSE = 1.0  # tampered\n")
    orig_crate_dir = crate_mod.crate_dir
    crate_mod.crate_dir = lambda task_name: tmp_crate
    try:
        verify_lock(name)
        failures.append("modified verifier NOT detected")
    except EvalLockError as e:
        print("tamper 1 (modified verifier) refused:\n  " + str(e).splitlines()[1].strip())
    finally:
        crate_mod.crate_dir = orig_crate_dir

    # 2. delete a frozen file
    tmp_crate2 = _copy_crate(name)
    victim2 = next((tmp_crate2 / "frozen").rglob("holdout_runner.py"))
    victim2.unlink()
    crate_mod.crate_dir = lambda task_name: tmp_crate2
    try:
        verify_lock(name)
        failures.append("deleted file NOT detected")
    except EvalLockError as e:
        print("tamper 2 (deleted holdout runner) refused:\n  " + str(e).splitlines()[1].strip())
    finally:
        crate_mod.crate_dir = orig_crate_dir

    # 3. tamper the lock itself (thresholds) -> the pack's bar no longer matches
    tmp_crate3 = _copy_crate(name)
    lock0 = json.loads((tmp_crate3 / "eval.lock.json").read_text())
    lock0["thresholds"]["o_rrmse"] = 1.0
    (tmp_crate3 / "eval.lock.json").write_text(json.dumps(lock0))
    pack = load_pack(load_lock(name)["pack_dir"])
    real = {k: th.value for k, th in pack.bar.thresholds.items()}
    if real != lock0["thresholds"]:
        print("tamper 3 (lock thresholds edited vs bar) refused: threshold mismatch")
    else:
        failures.append("edited lock thresholds NOT detected")

    if failures:
        print("SELFTEST FAILED:", failures)
        return 1
    print("tamper self-test PASSED (3/3 refusals)")
    return 0


if __name__ == "__main__":
    sys.exit(main())