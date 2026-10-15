"""Eval-crate lock: hashing, loading, verification (framework invariant 1)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

LOCK_VERSION = 1
LOCK_NAME = "eval.lock.json"
FROZEN_DIR = "frozen"


class EvalLockError(RuntimeError):
    """Raised when the crate is missing, modified, or inconsistent."""


def framework_root() -> Path:
    return Path(__file__).resolve().parents[2]


def crate_root() -> Path:
    return framework_root() / "eval-crates"


def crate_dir(task_name: str) -> Path:
    return crate_root() / task_name


def frozen_dir(task_name: str) -> Path:
    return crate_dir(task_name) / FROZEN_DIR


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lock_path(task_name: str) -> Path:
    return crate_dir(task_name) / LOCK_NAME


def load_lock(task_name: str) -> dict:
    p = lock_path(task_name)
    if not p.exists():
        raise EvalLockError(
            f"no eval lock for task {task_name!r}: {p} is missing. "
            "Build it with: python3 -m kagent.eval.build_eval_lock --pack <pack-dir>")
    return json.loads(p.read_text())


def verify_lock(task_name: str, lock: dict | None = None) -> dict:
    """Re-verify every hash in the lock. Raises EvalLockError on any mismatch."""
    if lock is None:
        lock = load_lock(task_name)
    if lock.get("lock_version") != LOCK_VERSION:
        raise EvalLockError(f"lock version {lock.get('lock_version')} != {LOCK_VERSION}")
    base = crate_dir(task_name)
    pack_dir = lock.get("pack_dir")
    problems: list[str] = []
    for rel, expected in sorted(lock["files"].items()):
        if rel.startswith("bar:") or rel.startswith("cases:"):
            # synthetic entries: hashed relative to the lock's pack dir
            if not pack_dir:
                problems.append(f"{rel}: lock has no pack_dir to resolve against")
                continue
            p = Path(pack_dir) / rel.split(":", 1)[1]
        else:
            p = base / rel
        if not p.exists():
            problems.append(f"missing: {rel}")
            continue
        got = sha256_file(p)
        if got != expected:
            problems.append(f"MODIFIED: {rel} (sha256 {got[:12]}... != {expected[:12]}...)")
    if lock.get("holdout", {}).get("enabled") and not (base / "frozen" / "holdouts.json").exists():
        problems.append("missing: frozen/holdouts.json (lock declares holdouts enabled)")
    if problems:
        raise EvalLockError(
            f"eval crate {task_name!r} failed verification — refusing to run:\n  "
            + "\n  ".join(problems))
    return lock


def expected_gates(lock: dict) -> list[str]:
    return sorted(lock.get("expected_gates", []))


def thresholds(lock: dict) -> dict:
    return lock.get("thresholds", {})