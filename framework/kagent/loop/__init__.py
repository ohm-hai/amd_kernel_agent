"""Loop engine (Phase 5): the stage machine any driver can call.

Stages, in enforced order:
    verify -> holdout -> formal -> bench -> evidence -> report

Framework invariant 3 is enforced here: bench/evidence/report refuse to run
until the frozen verifier has PASSED for the candidate (unless --i-know-verify
is passed explicitly by the user, never by the agent).
"""
from .engine import Loop, StageResult  # noqa: F401
from .report import render_markdown  # noqa: F401