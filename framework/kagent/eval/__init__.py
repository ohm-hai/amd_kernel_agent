"""Frozen, hash-locked evaluator crates (Phase 2 of the plan).

A crate is built once per task pack (and rebuilt only with the user's approval
when the bar changes). It mirrors the workspace tree under frozen/ so the
verifier's own imports resolve inside the crate with zero source modification,
and records sha256 hashes of every frozen file in eval.lock.json.

Runners:
    run_eval.py --mode verify    full visible verifier (values shown)
    run_eval.py --mode holdout   hidden seeds; counts-only report

The agent is never allowed to write under eval-crates/; `selftest_tamper.py`
proves that tampering aborts loudly.
"""
from .crate import (  # noqa: F401
    EvalLockError,
    crate_root,
    load_lock,
    sha256_file,
    verify_lock,
)