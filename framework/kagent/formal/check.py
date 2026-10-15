"""Formal checker: FlyDSL MLIR -> structural checks + Z3 equivalence.

    python3 -m kagent.formal.check --candidate ../kda-serving/runs/flydsl-decode-v7 \
        [--ir-dir <dump>] [--state-dtype float32|float16|bfloat16] [--out formal_report.json]

Steps:
  1. obtain stage MLIR (dump via FLYDSL_DUMP_IR=1 by importing candidate.py and
     calling its _compile_decode_kernel(), or reuse --ir-dir);
  2. parse + find reduction chains; butterfly-coverage check on every chain;
  3. for queried elements: enumerate writers (index solve over tid/bid),
     evaluate the stored value symbolically (reductions expanded over the
     wavefront), check the load multiset against the spec, then prove
     cand == spec in Z3 Reals (UNSAT -> PROVED, SAT -> counterexample).

exp/rsqrt are uninterpreted and shared with the spec; bf16 rounding points are
recorded, not modeled (precision is the numeric verifier's domain).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import z3

from .frontend import Verdict
from .ir import Chain, Module, Unsupported, butterfly_ok, find_chains, parse_mlir
from .kda_spec import KdaSpec, build_fns

BUFFER_ORDER = ["Q", "KB", "GB", "Vvin", "BETA", "S_IN", "S_OUT", "O", "SCALE_T"]
BUFFER_ORDER_V8 = BUFFER_ORDER[:8] + ["IDX", "SCALE_T"]   # native indexed pool
BUFFER_ALIAS = {"KB": "K", "GB": "G", "Vvin": "V"}   # spec-facing names


def parse_mlir_checked(text: str) -> tuple[Module, list[str]]:
    """Pick the buffer order by the gpu.func arg count: 9 buffers (18 args) =
    v7 layout, 10 buffers (20 args) = v8 native indexed pool (+IDX)."""
    fm = re.search(r"gpu\.func\s+@[\w.$-]+\(", text)
    if not fm:
        raise Unsupported("no gpu.func found")
    sig_end = text.find("kernel attributes", fm.start())
    if sig_end < 0:
        sig_end = text.find(") {", fm.start())
    sig = text[fm.start():sig_end if sig_end > 0 else fm.start() + 6000]
    n_args = len(re.findall(r"%arg\d+", sig))
    if n_args == 20:
        order = BUFFER_ORDER_V8
    elif n_args == 18:
        order = BUFFER_ORDER
    else:
        raise Unsupported(f"unexpected gpu.func arg count {n_args}")
    return parse_mlir(text, order), order

DEFAULT_CONFIG = {"K": 128, "V": 128, "NV": 4, "THREADS": 256, "WAVEFRONT": 64,
                  "B": 2, "H": 96}   # B/H bound the writer search for queried (b, h)


# ------------------------------------------------------------------ dumping

def dump_candidate_ir(candidate_dir: Path, out_root: Path) -> tuple[Path, int]:
    """Run the candidate's self-test as a subprocess with FLYDSL_DUMP_IR=1.

    FlyDSL compiles lazily on the first launch, so the dump only appears when
    the kernel actually runs; the self-test provides that launch (and a free
    numeric smoke check: exit 0 = its built-in fp64 comparison passed).
    Returns (dump_root, selftest_exit_code).
    """
    env = dict(os.environ)
    env["FLYDSL_DUMP_IR"] = "1"
    env["FLYDSL_DUMP_DIR"] = str(out_root)
    env.setdefault("HIP_VISIBLE_DEVICES", "4")
    # candidates' self-tests import ref_fp64 from <kernel-lab>/kda-attn; make
    # that resolve regardless of where the candidate directory lives
    klab = Path(__file__).resolve().parents[2].parent   # framework/.. = kernel-lab
    ref = klab / "kda-attn"
    env["PYTHONPATH"] = str(ref) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(candidate_dir / "candidate.py")]
    try:
        proc = subprocess.run(cmd, timeout=600, capture_output=True, text=True, env=env)
    except subprocess.TimeoutExpired:
        proc = None
    if not list(out_root.rglob("*canonicalize*.mlir")):
        tail = (proc.stderr if proc else "")[-500:] or (proc.stdout if proc else "")[-500:] \
            if proc else "timeout"
        raise Unsupported(f"no MLIR dump produced by {candidate_dir.name} self-test: {tail}")
    return out_root, (proc.returncode if proc else -1)


def pick_stage(dump_dir: Path) -> Path:
    """Highest-numbered canonicalize stage still holding arith+shuffle+buffer ops."""
    stages = sorted(dump_dir.rglob("*canonicalize*.mlir"))
    if not stages:
        raise Unsupported("no canonicalize stage in dump")
    good = [p for p in stages
            if "arith.addf" in p.read_text() and "gpu.shuffle" in p.read_text()
            and "buffer.load" in p.read_text()]
    return (good or stages)[-1]


# --------------------------------------------------------------- evaluation

class Eval:
    """Symbolic execution with concrete (tid, bid); float values become AST
    node tuples; indices fold to python ints. Reduction chains appear as
    ('red', chain, wf_base) and are expanded over the wavefront later."""

    def __init__(self, mod: Module, chains: list[Chain], config: dict):
        self.m = mod
        self.chains = {c.result: c for c in chains}
        self.member_of: set[str] = {m for c in chains for m in c.members} - set(self.chains)
        self.cfg = config
        self.memo: dict[tuple, Any] = {}

    def value(self, name: str, tid: int, bid: int):
        key = (name, tid, bid)
        if key in self.memo:
            return self.memo[key]
        self.memo[key] = None  # cycle guard
        op = self.m.ops.get(name)
        if op is None:
            raise Unsupported(f"unknown value {name}")
        out = self._value_op(op, tid, bid)
        if out is None:
            raise Unsupported(f"cycle at {name}")
        self.memo[key] = out
        return out

    def _value_op(self, op, tid, bid):
        n, o = op.name, op.operands
        if op.res in self.chains:
            return ("red", self.chains[op.res],
                    (tid // self.cfg["WAVEFRONT"]) * self.cfg["WAVEFRONT"])
        if op.res in self.member_of:
            raise Unsupported(f"intermediate reduction value {op.res} used outside "
                              f"its chain (line {op.line})")
        if n == "gpu.shuffle":
            raise Unsupported(f"shuffle outside a reduction chain (line {op.line})")
        if n == "arith.constant":
            if op.attrs["ctype"].startswith(("i", "index")):
                return int(op.attrs["value"])
            return ("fconst", float(op.attrs["value"]))
        if n == "gpu.thread_id":
            return tid
        if n == "gpu.block_id":
            return bid
        if n == "arith.index_cast":
            return self.value(o[0], tid, bid)
        if n == "rocdl.raw.ptr.buffer.load":
            return self._load(op, tid, bid)
        if n in ("arith.addf", "arith.mulf", "arith.subf", "arith.divf"):
            return self._fbin(n, *(self.value(x, tid, bid) for x in o))
        if n == "math.rsqrt":
            return ("rsqrt", self.value(o[0], tid, bid))
        if n == "math.exp":
            return ("exp", self.value(o[0], tid, bid))
        if n == "arith.extf":
            return self.value(o[0], tid, bid)
        if n == "arith.truncf":
            inner = self.value(o[0], tid, bid)
            if isinstance(inner, tuple) and inner[0] == "vec":
                return ("vec", [("truncf", x, op.type) for x in inner[1]])
            return ("truncf", inner, op.type)
        if n == "vector.broadcast":
            return ("splat", self.value(o[0], tid, bid))
        if n == "vector.extract":
            v = self.value(o[0], tid, bid)
            return v[1][op.attrs["index"]]
        if n == "vector.from_elements":
            return ("vec", [self.value(x, tid, bid) for x in o])
        if n in ("arith.muli", "arith.addi", "arith.remsi", "arith.floordivsi"):
            a, b = (self.value(x, tid, bid) for x in o)
            return {"arith.addi": lambda: a + b, "arith.muli": lambda: a * b,
                    "arith.remsi": lambda: a % b, "arith.floordivsi": lambda: a // b}[n]()
        if n in ("arith.cmpi", "arith.cmpf"):
            pred = o[0]
            a, b = (self.value(x, tid, bid) for x in o[1:])
            if n == "arith.cmpf":
                a, b = _unsplat(a), _unsplat(b)
                if isinstance(a, tuple) and a[0] == "fconst":
                    a = a[1]
                if isinstance(b, tuple) and b[0] == "fconst":
                    b = b[1]
            return ("bool", _cmp(pred, a, b, n))
        if n == "arith.select":
            c, a, b = (self.value(x, tid, bid) for x in o)
            return b if c[1] is False else (a if c[1] is True else ("sel", c, a, b))
        raise Unsupported(f"op {n} (line {op.line})")

    def _load(self, op, tid, bid):
        """buffer.load: the IR index is a BYTE offset (validated against v7's
        numerics); convert to element index."""
        idx = self.value(op.operands[1], tid, bid)
        if not isinstance(idx, int):
            raise Unsupported(f"load index not concrete: {idx!r}")
        buf = self.m.buffers[op.operands[0]]
        es = _elemsize(op.type)
        if idx % es:
            raise Unsupported(f"misaligned load at byte {idx} ({op.type})")
        base = idx // es
        if buf == "IDX":
            # identity-slot assumption (documented in the formal report): the
            # checked mapping is slot == b; arbitrary idx maps to the same
            # per-(slot, head) math and is validated numerically by the
            # candidate's self-test and the verifier.
            return base
        if op.type and op.type.startswith("vector"):
            n = int(re.search(r"vector<(\d+)x", op.type).group(1))
            return ("vec", [("load", buf, base + k) for k in range(n)])
        return ("load", buf, base)

    def _fbin(self, name, a, b):
        a, b = _unsplat(a), _unsplat(b)
        kind = {"arith.addf": "fadd", "arith.mulf": "fmul",
                "arith.subf": "fsub", "arith.divf": "fdiv"}[name]
        va = isinstance(a, tuple) and a[0] == "vec"
        vb = isinstance(b, tuple) and b[0] == "vec"
        if va or vb:
            la = a[1] if va else [a] * len(b[1])
            lb = b[1] if vb else [b] * len(a[1])
            if len(la) != len(lb):
                raise Unsupported(f"vec length mismatch {len(la)} vs {len(lb)}")
            return ("vec", [self._fbin(name, x, y) for x, y in zip(la, lb)])
        return (kind, a, b)


def _elemsize(type_str: str) -> int:
    """Element size in bytes for a (possibly vector) MLIR type string."""
    t = type_str or ""
    if t.startswith("vector"):
        inner = t[len("vector<"):].split(">", 1)[0]   # "4xbf16"
        t2 = inner.split("x", 1)[1]
    else:
        t2 = t
    try:
        return {"bf16": 2, "f16": 2, "f32": 4, "f64": 8, "i32": 4, "i64": 8}[t2]
    except KeyError:
        raise Unsupported(f"unknown element size for {type_str!r}") from None


def _unsplat(x):
    return x[1] if isinstance(x, tuple) and x[0] == "splat" else x


def _cmp(pred, a, b, which) -> bool:
    table = ({"slt": lambda: a < b, "sle": lambda: a <= b, "sgt": lambda: a > b,
              "sge": lambda: a >= b, "eq": lambda: a == b, "ne": lambda: a != b}
             if which == "arith.cmpi" else
             {"olt": lambda: a < b, "ole": lambda: a <= b, "ogt": lambda: a > b,
              "oge": lambda: a >= b, "oeq": lambda: a == b, "one": lambda: a != b})
    if pred not in table:
        raise Unsupported(f"{which} predicate {pred}")
    return table[pred]()


# ----------------------------------------------------------- z3 conversion

class Z3Conv:
    """AST -> z3 Real with a const factory SHARED between candidate and spec so
    both sides name identical variables. `loads` counts only what this conv
    itself materializes."""

    def __init__(self, f_exp, f_rsqrt, consts: dict | None = None):
        self.f_exp, self.f_rsqrt = f_exp, f_rsqrt
        self.consts = consts if consts is not None else {}
        self.loads: Counter = Counter()

    def mk(self, buf: str, idx: int, sink: Counter | None = None):
        key = (buf, idx)
        if key not in self.consts:
            self.consts[key] = z3.Const(f"{buf}[{idx}]", z3.RealSort())
        if sink is not None:
            sink[key] += 1
        return self.consts[key]

    def to_z3(self, node, sink: Counter | None = None):
        if isinstance(node, (z3.ArithRef, z3.BoolRef, int, float)):
            return z3.RealVal(node) if isinstance(node, (int, float)) and not isinstance(node, bool) else node
        kind = node[0]
        if kind == "load":
            buf = BUFFER_ALIAS.get(node[1], node[1])
            return self.mk(buf, node[2], sink if sink is not None else self.loads)
        if kind == "fconst":
            return z3.RealVal(node[1])
        if kind == "fadd":
            return self.to_z3(node[1], sink) + self.to_z3(node[2], sink)
        if kind == "fmul":
            return self.to_z3(node[1], sink) * self.to_z3(node[2], sink)
        if kind == "fsub":
            return self.to_z3(node[1], sink) - self.to_z3(node[2], sink)
        if kind == "fdiv":
            return self.to_z3(node[1], sink) / self.to_z3(node[2], sink)
        if kind == "exp":
            return self.f_exp(self.to_z3(node[1], sink))
        if kind == "rsqrt":
            return self.f_rsqrt(self.to_z3(node[1], sink))
        if kind == "truncf":
            return self.to_z3(node[1], sink)   # rounding recorded separately
        if kind in ("splat", "extf"):
            return self.to_z3(node[1], sink)
        if kind == "vec":
            return [self.to_z3(x, sink) for x in node[1]]
        raise RuntimeError(f"no z3 for node {kind}")


def expand_red(node, ev: Eval, conv: Z3Conv):
    """Replace ('red', chain, base) with the explicit wavefront sum of z3 terms."""
    if not isinstance(node, tuple):
        return node
    kind = node[0]
    if kind == "red":
        chain, base = node[1], node[2]
        if not butterfly_ok(chain.offsets, chain.width):
            raise Unsupported(f"reduction offsets {chain.offsets} not a full "
                              f"butterfly for width {chain.width}")
        total = None
        for l in range(chain.width):
            t = ev.value(chain.seed, base + l, _last_bid[0])
            z = conv.to_z3(expand_red(t, ev, conv))
            total = z if total is None else total + z
        return total
    if kind == "vec":
        return ("vec", [expand_red(x, ev, conv) for x in node[1]])
    if kind in ("fadd", "fmul", "fsub", "fdiv"):
        return (kind, expand_red(node[1], ev, conv), expand_red(node[2], ev, conv))
    if kind in ("exp", "rsqrt", "splat", "extf", "truncf"):
        return (kind, expand_red(node[1], ev, conv)) if kind != "truncf" else \
            (kind, expand_red(node[1], ev, conv), node[2])
    return node


_last_bid = [0]  # seed terms keep the writer's program id; set before expansion


# ------------------------------------------------------------------ checker

@dataclass
class QueryResult:
    buffer: str
    index: int
    verdict: Verdict
    detail: str
    writers: int = 0
    witness: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"verdict": self.verdict.value, "detail": self.detail,
                "writers": self.writers, "witness": self.witness}


def _num(v) -> float | None:
    try:
        if v.is_int_value():
            return float(v.as_long())
        return float(v.numerator_as_long()) / float(v.denominator_as_long())
    except Exception:
        return None


def prove_equal(cand, spec, label: str) -> QueryResult:
    s = z3.Solver()
    s.set("timeout", 60000)
    s.add(z3.Not(cand == spec))
    r = s.check()
    if r == z3.unsat:
        return QueryResult("", 0, Verdict.PROVED, label)
    if r == z3.sat:
        m = s.model()
        wit = {}
        for key in list(m):
            try:
                wit[str(key.name())] = _num(m[key])
            except Exception:
                continue
        cv, sv = _num(m.eval(cand, model_completion=True)), _num(m.eval(spec, model_completion=True))
        ratio = (sv / cv) if (cv not in (None, 0.0) and sv is not None) else None
        return QueryResult("", 0, Verdict.COUNTEREXAMPLE, label,
                           witness={"model": {k: v for k, v in list(wit.items())[:20]},
                                    "spec_over_cand": ratio})
    return QueryResult("", 0, Verdict.UNKNOWN, f"{label}: z3 unknown")


def _ratio_factor(ratio: float) -> str | None:
    """Heuristic missing-factor diagnosis against known KDA factors."""
    import math
    if ratio is None:
        return None
    known = {"2": 2.0, "sqrt(128)": math.sqrt(128.0), "128": 128.0, "1/128": 1 / 128.0}
    for name, val in known.items():
        if abs(ratio - val) <= 1e-6 * max(1.0, abs(val)) or \
           abs(ratio - 1 / val) <= 1e-6 * max(1.0, abs(val)):
            return name
    return None


def check_element(mod: Module, ev: Eval, spec: KdaSpec, buf_out: str, lin: int,
                  f_exp, f_rsqrt, cfg: dict, shared_consts: dict,
                  max_writers: int = 8) -> QueryResult:
    """Enumerate writers of one element; per writer: coverage + Z3 equivalence."""
    tid, bid = z3.Int("tid"), z3.Int("bid")
    store_ops = [st for st in mod.stores if mod.buffers[st.operands[1]] == buf_out]
    if not store_ops:
        raise Unsupported(f"no stores to {buf_out}")

    s = z3.Solver()
    s.set("timeout", 30000)
    s.add(tid >= 0, tid < cfg["THREADS"])
    s.add(bid >= 0, bid < cfg["B"] * cfg["H"] * cfg["NV"])
    per_store = []
    for st in store_ops:
        idx = index_z3(mod, st.operands[2], tid, bid)
        es = _elemsize(st.type)
        n = _store_width(st)
        # byte offset covers elements idx/esz + k; require exact alignment
        match = z3.Or(*[idx == es * (lin - k) for k in range(n)])
        if st.guard is not None:
            g = index_z3(mod, st.guard, tid, bid)
            match = z3.And(g, match)
        per_store.append(match)
    s.add(z3.Or(*per_store))
    writers = []
    while s.check() == z3.sat and len(writers) < max_writers:
        m = s.model()
        w = (m[tid].as_long(), m[bid].as_long())
        writers.append(w)
        s.add(z3.Or(tid != w[0], bid != w[1]))
    if len(writers) >= max_writers:
        return QueryResult(buf_out, lin, Verdict.UNKNOWN,
                           f">={max_writers} writers for element; refusing to model")

    if buf_out == "O":
        nh, rem = lin // cfg["V"], lin % cfg["V"]
        off = {"Q": nh * cfg["K"], "K": nh * cfg["K"], "G": nh * cfg["K"],
               "V": nh * cfg["V"], "S_IN": nh * cfg["K"] * cfg["V"], "BETA": nh}
        i = None
    else:
        nhv = cfg["K"] * cfg["V"]
        nh, rem = lin // nhv, lin % nhv
        if getattr(spec, "layout", "k_first") == "v_first":
            j, i = rem // cfg["K"], rem % cfg["K"]
        else:
            i, j = rem // cfg["V"], rem % cfg["V"]
        off = {"Q": nh * cfg["K"], "K": nh * cfg["K"], "G": nh * cfg["K"],
               "V": nh * cfg["V"], "S_IN": nh * nhv, "BETA": nh}

    spec_conv = Z3Conv(f_exp, f_rsqrt, shared_consts)

    def spec_mk(buf, idx, sink=None):
        return spec_conv.mk(buf, idx)

    if buf_out == "O":
        spec_expr, spec_loads = spec.expected_o(z3, spec_mk, f_exp, f_rsqrt, rem, off)
    else:
        spec_expr, spec_loads = spec.expected_state(z3, spec_mk, f_exp, f_rsqrt, i, j, off)
    spec_ms = Counter(spec_loads)

    results = []
    for (tid_w, bid_w) in writers:
        conv = Z3Conv(f_exp, f_rsqrt, shared_consts)  # per-writer load counting
        _last_bid[0] = bid_w
        for st in store_ops:
            idx = ev.value(st.operands[2], tid_w, bid_w)
            if not isinstance(idx, int):
                raise Unsupported("writer index not concrete")
            es = _elemsize(st.type)
            if idx % es:
                continue  # misaligned store: not this element's writer
            base = idx // es
            n = _store_width(st)
            if st.guard is not None and not ev.value(st.guard, tid_w, bid_w)[1]:
                continue  # this thread does not execute the guarded store
            node = ev.value(st.operands[0], tid_w, bid_w)
            if node[0] == "truncf":
                node = node[1]
            elems = node[1] if node[0] == "vec" else [node]
            for k in range(min(n, len(elems))):
                if base + k != lin:
                    continue
                elem = elems[k]
                if elem[0] == "truncf":
                    elem = elem[1]
                elem_z = conv.to_z3(expand_red(elem, ev, conv))
                cand_keys = set(conv.loads)
                spec_keys = set(spec_ms)
                if cand_keys != spec_keys:
                    missing = sorted(spec_keys - cand_keys)[:4]
                    extra = sorted(cand_keys - spec_keys)[:4]
                    return QueryResult(
                        buf_out, lin, Verdict.COUNTEREXAMPLE,
                        f"load-set mismatch (writer tid={tid_w} bid={bid_w}): "
                        f"missing={missing} extra={extra}")
                res = prove_equal(elem_z, spec_expr, f"{buf_out}[{lin}]")
                res.buffer, res.index, res.writers = buf_out, lin, len(writers)
                if res.verdict == Verdict.COUNTEREXAMPLE and res.witness.get("spec_over_cand"):
                    fac = _ratio_factor(res.witness["spec_over_cand"])
                    if fac:
                        res.detail += f" (missing factor ~= {fac})"
                results.append(res)
                break
    if not results:
        raise Unsupported("no writer actually stored the queried element (guard mismatch)")
    return min(results, key=lambda r: {"PROVED": 0, "UNKNOWN": 1, "COUNTEREXAMPLE": 2}[r.verdict.value])


def _store_width(st) -> int:
    m = re.search(r"vector<(\d+)x", st.type or "")
    return int(m.group(1)) if m else 1


def index_z3(mod: Module, root: str, tid, bid):
    """z3 Int/Bool expression of an integer chain rooted at `root`."""
    memo = {}

    def go(name):
        if name in memo:
            return memo[name]
        memo[name] = None
        op = mod.ops.get(name)
        if op is None:
            raise Unsupported(f"unknown index value {name}")
        n, o = op.name, op.operands
        if n == "arith.constant":
            out = z3.IntVal(int(op.attrs["value"]))
        elif n == "gpu.thread_id":
            out = tid
        elif n == "gpu.block_id":
            out = bid
        elif n == "arith.index_cast":
            out = go(o[0])
        elif n == "arith.addi":
            out = go(o[0]) + go(o[1])
        elif n == "arith.muli":
            out = go(o[0]) * go(o[1])
        elif n == "arith.remsi":
            out = go(o[0]) % go(o[1])
        elif n == "arith.floordivsi":
            out = go(o[0]) / go(o[1])
        elif n == "rocdl.raw.ptr.buffer.load" and mod.buffers.get(o[0]) == "IDX":
            # identity-slot assumption: IDX[b] == b; the IR index is a byte
            # offset, so convert to the element (slot) index
            out = go(o[1]) / _elemsize(op.type)
        elif n == "arith.cmpi":
            a, b = go(o[1]), go(o[2])
            out = {"slt": a < b, "sle": a <= b, "sgt": a > b, "sge": a >= b,
                   "eq": a == b, "ne": a != b}[o[0]]
        else:
            raise Unsupported(f"index chain op {n} (line {op.line})")
        if out is None:
            raise Unsupported(f"cycle at {name}")
        memo[name] = out
        return out

    return go(root)


# --------------------------------------------------------------------- CLI

QUERIES = [("O", 0, 0, 0, 0), ("O", 0, 0, 127, 127), ("O", 1, 3, 64, 5),
           ("S_OUT", 0, 0, 0, 0), ("S_OUT", 0, 0, 127, 64), ("S_OUT", 1, 3, 33, 7)]


def run_check(candidate_dir: Path, out: Path | None = None, ir_dir: Path | None = None,
              state_dtype: str = "float32", cfg: dict | None = None) -> dict:
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    t0 = time.perf_counter()
    dump_root = (ir_dir or Path("/tmp/kagent-formal")) / candidate_dir.name
    selftest_rc = None
    if ir_dir:
        stage = pick_stage(Path(ir_dir))
    else:
        dump_root, selftest_rc = dump_candidate_ir(candidate_dir, dump_root)
        stage = pick_stage(dump_root)
    mod, buf_order = parse_mlir_checked(stage.read_text())
    chains = find_chains(mod)

    cand_src = (candidate_dir / "candidate.py").read_text() if (candidate_dir / "candidate.py").exists() else ""
    state_layout = "v_first" if ('STATE_LAYOUT = "v_first"' in cand_src or "STATE_LAYOUT = 'v_first'" in cand_src) else "k_first"
    m_nv = re.search(r"\bNV\s*=\s*(\d+)", cand_src)
    if m_nv:
        cfg["NV"] = int(m_nv.group(1))
    m_th = re.search(r"\bTHREADS\s*=\s*(\d+)", cand_src)
    if m_th:
        cfg["THREADS"] = int(m_th.group(1))
    if mod.block_size:
        cfg["THREADS"] = mod.block_size[0]

    findings: list[dict] = []
    bad = [c for c in chains if not butterfly_ok(c.offsets, c.width)]

    ev = Eval(mod, chains, cfg)
    spec = KdaSpec(K=cfg["K"], V=cfg["V"], layout=state_layout)
    f_exp, f_rsqrt = build_fns(z3)
    shared: dict = {}
    rounding: list[str] = []

    for (buf, b, h, i, j) in QUERIES:
        try:
            if buf == "O":
                lin = (b * cfg["H"] + h) * cfg["V"] + j
            else:
                if state_layout == "v_first":
                    lin = (b * cfg["H"] + h) * cfg["K"] * cfg["V"] + j * cfg["K"] + i
                else:
                    lin = (b * cfg["H"] + h) * cfg["K"] * cfg["V"] + i * cfg["V"] + j
            r = check_element(mod, ev, spec, buf, lin, f_exp, f_rsqrt, cfg, shared)
            findings.append({"check": f"equivalence {buf} b={b} h={h} i={i} j={j}",
                             **r.as_dict()})
        except Unsupported as e:
            findings.append({"check": f"equivalence {buf} b={b} h={h} i={i} j={j}",
                             "verdict": "UNKNOWN", "detail": f"unsupported: {e}"})

    for st in mod.stores:
        t, buf = st.type or "", mod.buffers[st.operands[1]]
        if t.startswith("vector") and "bf16" in t:
            if buf == "O":
                rounding.append(f"O store truncates to {t} (line {st.line})")
            if buf == "S_OUT" and state_dtype == "float32":
                findings.append({"check": "state_precision_undeclared",
                                 "verdict": "COUNTEREXAMPLE",
                                 "detail": f"S_OUT stored as {t} (line {st.line}) but "
                                           f"declared state dtype is float32"})

    core = [f for f in findings if f["check"].startswith(("equivalence", "butterfly", "state"))]
    vs = [f["verdict"] for f in core]
    overall = ("COUNTEREXAMPLE" if "COUNTEREXAMPLE" in vs
               else ("UNKNOWN" if "UNKNOWN" in vs else "PROVED"))
    report = {
        "candidate": str(candidate_dir), "stage": str(stage), "overall": overall,
        "config": cfg, "state_dtype": state_dtype, "selftest_exit": selftest_rc,
        "n_chains": len(chains), "findings": findings,
        "rounding_points": rounding,
        "seconds": round(time.perf_counter() - t0, 1),
        "note": "exp/rsqrt are uninterpreted; PROVED means same math modulo the "
                "abstraction for the addressed elements; precision is the numeric "
                "verifier's domain (guidelines 02)."
                + (" IDENTITY-SLOT ASSUMPTION: the kernel indexes the pool via "
                   "IDX[b]; the formal check assumes slot == b (the mapping is a "
                   "launch property; arbitrary idx is validated numerically by "
                   "the self-test and the verifier)."
                   if "IDX" in buf_order else ""),
        "buffer_order": buf_order,
    }
    out = out or candidate_dir / "formal_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--ir-dir", default=None, help="existing FLYDSL_DUMP_IR output dir")
    ap.add_argument("--state-dtype", default="float32")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        report = run_check(Path(args.candidate), Path(args.out) if args.out else None,
                           Path(args.ir_dir) if args.ir_dir else None, args.state_dtype)
    except Unsupported as e:
        print(f"FORMAL UNKNOWN: unsupported pattern: {e}")
        return 2
    counts = {v: sum(1 for f in report["findings"] if f["verdict"] == v)
              for v in ("PROVED", "COUNTEREXAMPLE", "UNKNOWN")}
    print(f"FORMAL {report['overall']} "
          f"({counts['PROVED']} proved, {counts['COUNTEREXAMPLE']} failed, "
          f"{counts['UNKNOWN']} unknown) in {report['seconds']}s "
          f"-> {args.out or 'formal_report.json'}")
    for f in report["findings"]:
        line = f"[{f['verdict']}] {f['check']}"
        if f.get("witness", {}).get("spec_over_cand"):
            line += f"  witness spec/cand={f['witness']['spec_over_cand']:.6g}"
        if f.get("detail"):
            line += f"  {f['detail'][:120]}"
        print(line)
    return 0 if report["overall"] == "PROVED" else (1 if report["overall"] == "COUNTEREXAMPLE" else 2)


if __name__ == "__main__":
    sys.exit(main())