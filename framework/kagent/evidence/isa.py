"""ISA analyzer for FlyDSL kernels on gfx950 (CDNA4).

Parses the 21_final_isa.s stage of a FLYDSL_DUMP_IR dump:
  - register/LDS metadata -> occupancy estimate;
  - instruction histogram -> ALU-to-DATA-shuffle ratio (the issue-bound
    detector: a wavefront-reduction kernel that moves data with ds_swizzle
    should issue >= 8 ALU ops per data movement; v7 measured ~1.4);
  - s_nop / s_waitcnt counts -> scheduler-stall pressure.

    python3 -m kagent.evidence.isa --candidate <dir> [--ir-dir <dump>]
Writes <candidate>/isa.json.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # framework/

from kagent.formal.check import dump_candidate_ir, pick_stage  # noqa: E402
from kagent.formal.ir import Unsupported  # noqa: E402

META_KEYS = ["vgpr_count", "sgpr_count", "agpr_count", "lds_size",
             "private_segment_fixed_size", "group_segment_fixed_size",
             "wavefront_size", "max_flat_workgroup_size"]

ALU_MATH = ("v_add_", "v_add", "v_mul", "v_fma", "v_fmac", "v_sub", "v_pk_add", "v_pk_mul",
            "v_cvt", "v_div", "v_sqrt", "v_rndne", "v_exp", "v_log", "v_max", "v_min",
            "v_med", "v_trunc", "v_ceil", "v_floor")
ALU_OVERHEAD = ("v_mov", "v_cndmask", "v_cmp", "v_permlane")
ALU_INDEX = ("v_lsh", "v_and", "v_or", "v_xor", "v_not", "v_add_u32", "v_sub_u32",
             "v_lshlrev", "v_lshrev", "v_and_b32", "v_or_b32")
DATA_PREFIXES = ("ds_swizzle", "ds_read", "ds_write", "buffer_load", "buffer_store",
                 "global_load", "global_store", "flat_load", "flat_store")
SCHED_PREFIXES = ("s_waitcnt", "s_nop", "s_barrier", "s_sleep")
CTRL_PREFIXES = ("s_branch", "s_cbranch", "s_endpgm", "s_setpc", "s_getpc")


def _instr_body(text: str) -> list[str]:
    """Instruction lines are indented lines starting with a mnemonic."""
    out = []
    for line in text.splitlines():
        m = re.match(r"^\s+([a-z][\w.]*)", line)
        if not m:
            continue
        t = m.group(1)
        if t.startswith((".", "vgpr", "sgpr", "agpr", "lds", "private", "group",
                         "wavefront", "max_flat", "symbol", "s_code", "kernarg",
                         "amdhsa", "amdgpu", "gf370")):
            continue
        out.append(t)
    return out


def analyze_isa_text(isa_text: str) -> dict:
    meta = {}
    for key in META_KEYS:
        m = re.search(rf"\.{key}:\s*(\d+)", isa_text)
        if m:
            meta[key] = int(m.group(1))
    instrs = _instr_body(isa_text)
    hist = Counter(instrs)
    classes = Counter()
    for ins, n in hist.items():
        if ins.startswith(DATA_PREFIXES):
            classes["data"] += n
        elif ins.startswith(SCHED_PREFIXES):
            classes["sched"] += n
        elif ins.startswith(CTRL_PREFIXES):
            classes["ctrl"] += n
        elif ins.startswith(ALU_MATH):
            classes["alu_math"] += n
        elif ins.startswith(ALU_OVERHEAD):
            classes["alu_shuffle_overhead"] += n
        elif ins.startswith(ALU_INDEX):
            classes["alu_index_int"] += n
        elif ins.startswith("v_"):
            classes["alu_other"] += n
        else:
            classes["scalar_other"] += n
    packed = sum(n for ins, n in hist.items() if ins.startswith("v_pk_"))
    data = classes["data"]
    math_ = classes["alu_math"]
    v_issue = sum(n for k2, n in classes.items() if k2.startswith("alu_"))
    vgpr = meta.get("vgpr_count", 0)
    waves = min(8, 512 // vgpr) if vgpr else None   # CDNA4: 512-entry VGPR file / SIMD
    ratio = round(math_ / data, 2) if data else None
    issue_ratio = round(v_issue / data, 2) if data else None
    if data and ratio is not None:
        limit = ("issue-bound: math/data=%.2f and issue/data=%.2f — shuffles and their "
                 "adjacent overhead (mov/cndmask/cmp/permlane) dominate issue slots; "
                 "coalesce work per lane (wider vectors, row-block ownership) before "
                 "touching memory" % (ratio, issue_ratio)) if ratio < 8 else \
            ("memory-bound direction: math/data=%.2f >= 8 — shuffle issue pressure "
             "is resolved; bandwidth is the next limit" % ratio)
    else:
        limit = "no data-movement instructions found; check the dump"
    if vgpr and vgpr >= 128:
        limit += f"; occupancy limited by VGPRs (est. {waves} waves/SIMD)"
    return {
        "meta": meta, "n_instructions": sum(hist.values()),
        "histogram_top": dict(hist.most_common(24)), "classes": dict(classes),
        "packed_v_pk": packed, "math_per_lane": math_, "data_per_lane": data,
        "math_data_ratio": ratio, "issue_data_ratio": issue_ratio,
        "est_waves_per_simd": waves,
        "s_nop": hist.get("s_nop", 0), "s_waitcnt": hist.get("s_waitcnt", 0),
        "interpretation": limit,
    }


def analyze_isa(candidate_dir: Path, ir_dir: Path | None = None,
                out: Path | None = None) -> dict:
    if ir_dir:
        dump = Path(ir_dir)
    else:
        dump = Path("/tmp/kagent-isa") / candidate_dir.name
        dump_candidate_ir(candidate_dir, dump)
    isa_files = sorted(dump.rglob("*final_isa*.s")) or sorted(dump.rglob("*.s"))
    if not isa_files:
        raise Unsupported(f"no ISA (.s) file under {dump}")
    text = isa_files[-1].read_text()
    report = {"candidate": str(candidate_dir), "isa_file": str(isa_files[-1]),
              **analyze_isa_text(text)}
    out = out or candidate_dir / "isa.json"
    out.write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--ir-dir", default=None)
    args = ap.parse_args()
    try:
        r = analyze_isa(Path(args.candidate), Path(args.ir_dir) if args.ir_dir else None)
    except Unsupported as e:
        print(f"ISA UNKNOWN: {e}")
        return 2
    print(f"ISA: math/lane={r['math_per_lane']} data/lane={r['data_per_lane']} "
          f"ratio={r['math_data_ratio']} issue/data={r['issue_data_ratio']} "
          f"vgpr={r['meta'].get('vgpr_count')} waves={r['est_waves_per_simd']} "
          f"s_nop={r['s_nop']} s_waitcnt={r['s_waitcnt']}")
    print("->", r["interpretation"])
    return 0


if __name__ == "__main__":
    sys.exit(main())