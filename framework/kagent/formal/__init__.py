"""Formal verification engine (Phase 3 of the plan).

Pipeline: FlyDSL mid-level MLIR (from FLYDSL_DUMP_IR=1) -> scalar op DAG ->
(1) structural checks: xor-shuffle butterfly coverage, writer enumeration for a
queried element, guard evaluation; (2) Z3 exact-real equivalence against the
KDA spec with exp/rsqrt abstracted as uninterpreted functions; (3) lints
(constant folding, precision downgrades, magic numbers).

Verdicts are tri-state (framework invariant 5):
    PROVED         equivalence for the queried elements within the modeled scope
    COUNTEREXAMPLE witness found (or a structural check failed)
    UNKNOWN        unsupported pattern; never treated as pass

Scope (v1): straight-line single-wavefront-reduction kernels — element-indexed
buffer ops, xor-shuffle reduction chains, guarded straight-line stores, one
level of scf.if. Anything else raises Unsupported -> UNKNOWN with the pattern.
Precision (bf16/fp16 rounding) is NOT modeled; it stays the numeric verifier's
job (guidelines 02). Approximate ops (rsqrt/exp) are abstracted, so a PROVED
verdict means "same math modulo the abstraction", not "bitwise equal".
"""
from .frontend import FrontEnd, Spec, Verdict  # noqa: F401
from .ir import Unsupported, parse_mlir  # noqa: F401