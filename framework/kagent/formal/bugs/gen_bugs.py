"""Generate planted-bug candidate variants from a known-correct kernel.

    python3 -m kagent.formal.bugs.gen_bugs --base ../../../kda-serving/runs/flydsl-decode-v7 --out .

Each variant is a directory with candidate.py (mutated source) + expected.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MUTATIONS = [
    ("bug_01_no_scale_beta",
     "fx.arith.mulf(fx.arith.mulf(scale_f, beta_f), qk_n)",
     "fx.arith.mulf(beta_f, qk_n)",
     {"verdict": "COUNTEREXAMPLE", "on": "O", "hint_factor": "SCALE"},
     "v3 incident: missing scale*beta on the sb term"),
    ("bug_02_wrong_shuffle",
     "_TREE = [(32, 64), (16, 64), (8, 64), (4, 64), (2, 64), (1, 64)]",
     "_TREE = [(16, 64), (16, 64), (8, 64), (4, 64), (2, 64), (1, 64)]",
     {"verdict": "COUNTEREXAMPLE", "on": "O", "check": "butterfly_coverage"},
     "v4 incident: wrong xor offsets break the reduction cover"),
    ("bug_03_no_l2n",
     "q_a = fx.arith.mulf(q_a, inv_q); q_b = fx.arith.mulf(q_b, inv_q)\n"
     "        k_a = fx.arith.mulf(k_a, inv_k); k_b = fx.arith.mulf(k_b, inv_k)",
     "q_a = q_a; q_b = q_b\n"
     "        k_a = k_a; k_b = k_b",
     {"verdict": "COUNTEREXAMPLE", "on": "O"},
     "v1 incident: missing L2 normalization of q/k"),
    ("bug_04_off_by_one",
     "s_r0 = s_head + row0 * fx.Int32(VV) + col0",
     "s_r0 = s_head + (row0 + fx.Int32(1)) * fx.Int32(VV) + col0",
     {"verdict": "COUNTEREXAMPLE", "on": "S_OUT"},
     "v4 incident: state row base off by one"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="path to runs/flydsl-decode-v7")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base_src = (Path(args.base) / "candidate.py").read_text()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for name, old, new, expected, incident in MUTATIONS:
        if old not in base_src:
            raise SystemExit(f"{name}: mutation anchor not found in base source:\n{old[:80]}")
        d = out / name
        d.mkdir(exist_ok=True)
        (d / "candidate.py").write_text(
            base_src.replace(old, new)
            .replace('"""Candidate: flydsl-decode-v7',
                     f'"""Candidate: {name} (planted bug: {incident}; v7-derived'))
        (d / "expected.json").write_text(json.dumps(
            {"name": name, "expected": expected, "incident": incident}, indent=2))
        written += 1
    print(f"generated {written} bug variants under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())