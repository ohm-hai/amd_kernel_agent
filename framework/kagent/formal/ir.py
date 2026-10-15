"""FlyDSL mid-level MLIR parser -> op DAG (FrontEnd for the FlyDSL language).

Parses the canonicalize-stage MLIR dumped by FLYDSL_DUMP_IR=1 (e.g. the
09_canonicalize.mlir stage of a decode kernel). Handles the op set our kernels
use: arith (int+float), math.{rsqrt,exp}, gpu.{thread_id,block_id,shuffle},
vector.{extract,broadcast,from_elements}, rocdl.raw.ptr.buffer.{load,store},
one level of scf.if guards. Anything else -> Unsupported.

Addressing convention (validated against the numeric verifier of v7): the
buffer op index counts ELEMENTS of the loaded type; a vector<N x T> access at
element index X covers elements X .. X+N-1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


class Unsupported(Exception):
    """A pattern the formal scope does not model. Never silently skipped."""


# value: python int | float-AST node | ('vec', [nodes]) | ('bool', b)
# AST float nodes: ('load', buf, idx, elem_kind) | ('fadd'|'fmul'|'fsub'|'fdiv', a, b)
#                  ('exp'|'rsqrt', x) | ('extf', x) | ('truncf', x, t) | ('red', chain)
#                  ('sel', cond, a, b) | ('splat', x)

RES_RE = re.compile(r"^\s*(%\S+(?:,\s*%\S+)?)\s*=\s*(.+?)\s+loc\(")
OP_RE = re.compile(r"^\s*(%[\w]+(?:\s*,\s*%[\w]+)*)\s*=\s*([A-Za-z][\w.]+)\s+(.*?)(?:\s+loc\(|\s*$)")
NORES_RE = re.compile(r"^\s*([A-Za-z][\w.]+)\s+(.*?)(?:\s+loc\(|\s*$)")
CONST_RE = re.compile(r"^(-?[\d.]+)\s*:\s*(\S+)$")
LOC_DEF_RE = re.compile(r"^#loc\d+\s*=")


@dataclass
class Op:
    res: str | None          # primary result name (None for stores/returns)
    all_results: list[str]
    name: str
    operands: list[str]
    attrs: dict[str, Any]
    type: str | None
    guard: str | None        # scf.if condition value name, or None
    line: int


@dataclass
class Module:
    func_name: str
    ops: dict[str, Op]                 # by primary result name
    order: list[str]
    stores: list[Op]
    buffers: dict[str, str]            # rsrc ssa name -> buffer name
    arg_buffers: dict[str, str]        # %argN -> buffer name
    block_size: list[int] | None
    guard_ops: dict[str, str]          # ops inside scf.if -> cond value


def _parse_res_names(s: str) -> list[str]:
    return [t.strip() for t in s.split(",")]


def _parse_rhs(name: str, rest: str):
    """Parse (op_name, operand-string after the op name). Returns operands/attrs/type."""
    ty = None
    if " : " in rest:
        rest, ty = rest.rsplit(" : ", 1)
        ty = ty.split(" loc(")[0].strip()
    operands: list[str] = []
    attrs: dict[str, Any] = {}
    if name == "arith.constant":
        val = rest.strip()
        if val.endswith("}"):  # e.g. dense<> constant
            raise Unsupported(f"non-scalar constant: {rest!r}")
        try:
            attrs["value"] = int(val)
            attrs["ctype"] = ty or "i32"
        except ValueError:
            try:
                attrs["value"] = float(val)
                attrs["ctype"] = ty or "f32"
            except ValueError:
                raise Unsupported(f"unparsed constant: {rest!r}") from None
        return operands, attrs, ty
    rest = rest.strip()
    if rest.endswith("}"):
        raise Unsupported(f"attribute dict on op {name}: {rest!r}")
    if name == "gpu.shuffle":
        kind, tail = rest.split(None, 1)
        attrs["kind"] = kind
        operands = [t.strip() for t in tail.split(",")]
        return operands, attrs, ty
    if rest:
        operands = [t.strip() for t in rest.split(",")]
    if name == "vector.extract":
        m2 = re.match(r"^(%\S+)\[(\d+)\]$", operands[0])
        if not m2:
            raise Unsupported(f"vector.extract form: {rest!r}")
        operands = [m2.group(1)]
        attrs["index"] = int(m2.group(2))
    return operands, attrs, ty


def parse_mlir(text: str, buffer_order: list[str]) -> Module:
    """Parse one gpu.func from stage MLIR. buffer_order maps (arg index // 2)
    to buffer names in launch order."""
    lines = text.splitlines()
    ops: dict[str, Op] = {}
    order: list[str] = []
    stores: list[Op] = []
    rsrc: dict[str, str] = {}
    addrcast: dict[str, str] = {}
    arg_buffers: dict[str, str] = {}
    guard_stack: list[str] = []
    guard_ops: dict[str, str] = {}
    func_name = None
    block_size = None
    in_func = False

    for ln, raw in enumerate(lines, 1):
        line = raw.rstrip()
        if not line.strip() or LOC_DEF_RE.match(line.strip()):
            continue
        if func_name is None and line.lstrip().startswith("gpu.func"):
            fm = re.match(r"\s*gpu.func\s+@([\w.$-]+)\(", line)
            if not fm:
                raise Unsupported(f"gpu.func not parsed: {line[:80]}")
            func_name = fm.group(1)
            bs = re.search(r"known_block_size\s*=\s*array<i32:\s*([,\d\s]+)>?", line)
            block_size = [int(x) for x in bs.group(1).split(",")] if bs else None
            in_func = True
            continue
        if not in_func:
            continue
        s = line.strip()
        if s.startswith("gpu.return"):
            in_func = False
            continue
        if s.startswith("}"):
            if guard_stack:
                guard_stack.pop()
            continue
        if s.startswith("scf.if "):
            cond = s[len("scf.if "):].split()[0]
            if guard_stack:
                raise Unsupported("nested scf.if")
            guard_stack.append(cond)
            continue
        if s.startswith(("module ", "gpu.module")):
            continue

        m = OP_RE.match(line)
        if not m:
            nm = NORES_RE.match(line)
            if not nm:
                raise Unsupported(f"unparsed line {ln}: {s[:100]}")
            name = nm.group(1)
            operands, attrs, ty = _parse_rhs(name, nm.group(2))
            if name == "rocdl.raw.ptr.buffer.store":
                stores.append(Op(res=None, all_results=[], name=name,
                                 operands=operands, attrs=attrs, type=ty,
                                 guard=guard_stack[-1] if guard_stack else None, line=ln))
                continue
            raise Unsupported(f"result-less op {name} (line {ln})")
        res_names = _parse_res_names(m.group(1))
        operands, attrs, ty = _parse_rhs(m.group(2), m.group(3))
        name = m.group(2)
        guard = guard_stack[-1] if guard_stack else None
        op = Op(res=res_names[0], all_results=res_names, name=name,
                operands=operands, attrs=attrs, type=ty, guard=guard, line=ln)
        if name == "llvm.addrspacecast":
            addrcast[op.res] = op.operands[0]
            ops[op.res] = op
            order.append(op.res)
            continue
        if name == "rocdl.make.buffer.rsrc":
            base = addrcast.get(op.operands[0])
            if base is None:
                raise Unsupported("make.buffer.rsrc not from addrspacecast")
            ai = int(base[4:].split(":")[0])
            if ai % 2:
                raise Unsupported(f"buffer arg {base} is not a pointer arg")
            buf = buffer_order[ai // 2]
            arg_buffers[base] = buf
            rsrc[op.res] = buf
            ops[op.res] = op          # kept for lints / descriptor analysis
            order.append(op.res)
            continue
        if name == "rocdl.raw.ptr.buffer.store":
            stores.append(op)
            continue
        if guard is not None:
            guard_ops[op.res] = guard
        ops[op.res] = op
        order.append(op.res)

    if func_name is None:
        raise Unsupported("no gpu.func found")
    return Module(func_name=func_name, ops=ops, order=order, stores=stores,
                  buffers=rsrc, arg_buffers=arg_buffers, block_size=block_size,
                  guard_ops=guard_ops)


# ------------------------------------------------------- reduction chains


@dataclass
class Chain:
    seed: str                  # ssa name of the seed value
    offsets: list[int]
    width: int
    result: str                # ssa name of the final addf result
    members: list[str] = field(default_factory=list)   # addf results in order


def find_chains(mod: Module) -> list[Chain]:
    """Detect xor-shuffle reduction chains:
    seed -> shuffle(off0) -> addf -> shuffle(off1) -> addf -> ... -> result."""
    consumers: dict[str, list[Op]] = {}
    for op in mod.ops.values():
        for o in op.operands:
            consumers.setdefault(o, []).append(op)
    shuffles = [op for op in mod.ops.values() if op.name == "gpu.shuffle"]
    for sh in shuffles:
        if sh.attrs.get("kind") != "xor":
            raise Unsupported(f"non-xor shuffle at line {sh.line}")
    discovered: set[str] = set()
    chains = []
    for sh in shuffles:
        src = sh.operands[0]
        src_op = mod.ops.get(src)
        if src_op is None or src_op.name != "arith.addf" or src in discovered:
            continue  # chain-internal shuffle (already walked) or non-chain feed
        seed = src_op.res
        cur, prev_addf = sh, src_op
        offsets = [_const_int(mod, cur.operands[1])]
        width = _const_int(mod, cur.operands[2])
        members: list[str] = []
        result = None
        while True:
            uses = [u for u in consumers.get(cur.all_results[0], []) if u.name == "arith.addf"]
            if len(uses) != 1:
                raise Unsupported(f"shuffle result used {len(uses)} times (line {cur.line})")
            nxt = uses[0]
            if sorted(nxt.operands) != sorted([prev_addf.res, cur.all_results[0]]):
                raise Unsupported(f"chain addf shape unexpected (line {nxt.line})")
            members.append(nxt.res)
            discovered.add(nxt.res)
            nxt_shuffles = [u for u in consumers.get(nxt.res, []) if u.name == "gpu.shuffle"]
            prev_addf = nxt
            if not nxt_shuffles:
                result = nxt.res
                break
            if len(nxt_shuffles) != 1:
                raise Unsupported(f"addf feeds {len(nxt_shuffles)} shuffles")
            cur = nxt_shuffles[0]
            if cur.attrs.get("kind") != "xor":
                raise Unsupported("non-xor shuffle in chain")
            offsets.append(_const_int(mod, cur.operands[1]))
        chains.append(Chain(seed=seed, offsets=offsets, width=width,
                            result=result, members=members))
    return chains


def butterfly_ok(offsets: list[int], width: int) -> bool:
    """A full xor butterfly: xor-combinations of the offsets cover [0, width)
    exactly once. Catches wrong shuffle offsets globally (cheap, no Z3)."""
    if width <= 0 or width & (width - 1):
        return False
    cover = {0}
    for off in offsets:
        cover |= {x ^ off for x in cover}
    return len(cover) == width and len(offsets) == width.bit_length() - 1


def _const_int(mod: Module, name: str) -> int:
    op = mod.ops.get(name)
    if op is None or op.name != "arith.constant":
        raise Unsupported(f"expected constant, got {name}")
    return int(op.attrs["value"])