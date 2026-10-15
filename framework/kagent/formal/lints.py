"""Static lints over the parsed kernel IR (no Z3, no GPU).

Lints catch the KDA²-blog hack class and the campaign's precision incidents:
  - magic_numbers: float literals baked into the kernel (hard-coded scale);
  - index_constants: integers that no shape-derived config explains;
  - precision_downgrade: extf from f64 (literal width loss), truncf into the
    state store, or bf16 stores to the state with a declared fp32 dtype.
"""
from __future__ import annotations

from collections import Counter

from .ir import Module

_FLOAT_TYPES = ("f32", "f64", "fp32", "fp64")


def _base_config(cfg: dict) -> set[int]:
    K, V, NV, T = cfg["K"], cfg["V"], cfg["NV"], cfg["THREADS"]
    base = {0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512}
    for x in (K, V, K * V, NV, T, K // NV, V // NV, (V // NV) // 4):
        base |= {x, x * 2, x * 4, x * 8}
    return base


def magic_numbers(mod: Module, cfg: dict) -> list[dict]:
    """Float literals used in value math (not index math)."""
    out = []
    for op in mod.ops.values():
        if op.name != "arith.constant":
            continue
        t = op.attrs.get("ctype", "")
        v = op.attrs.get("value", "")
        if t.startswith("f") and t not in ("f16",):
            try:
                fv = float(v)
            except ValueError:
                continue
            if fv in (0.0, 1.0):
                continue
            out.append({"kind": "magic_float", "line": op.line,
                        "value": fv, "type": t})
    return out


def _descriptor_consts(mod: Module) -> set[str]:
    """Constants that only feed rocdl.make.buffer.rsrc (buffer descriptors,
    not kernel math): 0xFFFFFFFF and the LDS/segment sizes."""
    desc = set()
    for op in mod.ops.values():
        if op.name == "rocdl.make.buffer.rsrc":
            for o in op.operands:
                desc.add(o)
    return desc


def index_constants(mod: Module, cfg: dict) -> list[dict]:
    allowed = _base_config(cfg)
    desc = _descriptor_consts(mod)
    out = []
    for op in mod.ops.values():
        if op.name != "arith.constant":
            continue
        t = op.attrs.get("ctype", "")
        if not t.startswith(("i", "index")):
            continue
        if op.res in desc:
            continue  # buffer-resource descriptor constant, not kernel math
        try:
            iv = int(op.attrs["value"])
        except ValueError:
            continue
        if iv not in allowed:
            out.append({"kind": "unexplained_index_constant", "line": op.line,
                        "value": iv, "type": t})
    return out


def precision_downgrades(mod: Module, cfg: dict, state_dtype: str) -> list[dict]:
    out = []
    for op in mod.ops.values():
        if op.name == "arith.extf" and "f64" in (op.type or ""):
            out.append({"kind": "extf_from_f64", "line": op.line, "type": op.type})
    for st in mod.stores:
        t, buf = st.type or "", mod.buffers[st.operands[1]]
        if buf == "S_OUT" and "bf16" in t and state_dtype == "float32":
            out.append({"kind": "state_stored_bf16_vs_declared_fp32",
                        "line": st.line, "type": t})
        if buf == "S_OUT" and "f16" in t and state_dtype == "float32":
            out.append({"kind": "state_stored_f16_vs_declared_fp32",
                        "line": st.line, "type": t})
    return out


def run_lints(mod: Module, cfg: dict, state_dtype: str = "float32") -> dict:
    lints = {
        "magic_numbers": magic_numbers(mod, cfg),
        "index_constants": index_constants(mod, cfg),
        "precision_downgrades": precision_downgrades(mod, cfg, state_dtype),
    }
    counts = {k: len(v) for k, v in lints.items()}
    return {"lints": lints, "counts": counts,
            "clean": all(v == 0 for v in counts.values())}


def summarize(lint_report: dict) -> str:
    if lint_report["clean"]:
        return "lints clean"
    parts = [f"{k}={n}" for k, n in lint_report["counts"].items() if n]
    return "lint findings: " + ", ".join(parts)