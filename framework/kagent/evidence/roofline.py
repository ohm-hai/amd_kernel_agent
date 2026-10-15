"""Roofline report for decode candidates from results.csv.

bytes_model: state read+write (2*B*H*K*V*esz) + q/k/g/v/o/beta I/O for the
Kimi K3 KDA shape (H=96, K=V=128). floors at spec BW (8000 GB/s) and at the
baseline's implied BW (honest bound). efficiency = implied BW / spec BW.

    python3 -m kagent.evidence.roofline --task-dir ../kda-serving \
        --candidate flydsl-decode-v7 [--baseline baseline-fla-decode] [--dtype float32]
Writes <task-dir>/runs/<candidate>/roofline.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

SPEC_BW_GBPS = 8000.0      # MI355X HBM3E spec
H, K, V = 96, 128, 128
GRID = [1, 8, 32, 128]
STATE_ESZ = {"float32": 4, "float16": 2, "bfloat16": 2}


def kda_decode_bytes(B: int, dtype: str) -> int:
    esz = STATE_ESZ[dtype]
    state = 2 * B * H * K * V * esz                    # read + write
    io = (3 * B * H * K + 2 * B * H * V + B * H * V + B * H) * 2  # q,k,g bf16 + v,o bf16 + beta
    return state + io


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _times(results_csv: Path, candidate: str, metric: str, interface: str | None = None) -> dict[int, float]:
    out: dict[int, list[float]] = {}
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            if row.get("candidate") != candidate or row.get("metric") != metric:
                continue
            if row.get("mode") != "decode":
                continue
            if interface and row.get("interface") != interface:
                continue
            try:
                b = int(row["B"])
                out.setdefault(b, []).append(float(row["value"]))
            except (KeyError, ValueError):
                continue
    return {b: _median(v) for b, v in out.items()}


def roofline_report(task_dir: Path, candidate: str, baseline: str,
                    dtype: str = "float32", out: Path | None = None) -> dict:
    results = task_dir / "results.csv"
    # same metric for BOTH sides — never mix serving-pool with wrapper numbers
    metric, interface = None, None
    t_c = t_b = None
    for metric, interface in (("graphed_us_per_step", "serving-pool"),
                              ("kernel_us_median", "public-wrapper")):
        t_c = _times(results, candidate, metric, interface)
        t_b = _times(results, baseline, metric, interface)
        if t_c and t_b:
            break
    if not (t_c and t_b):
        report = {"task_dir": str(task_dir), "candidate": candidate, "baseline": baseline,
                  "state_dtype": dtype, "per_B": [], "geomean_speedup": None,
                  "verdict": f"no complete {metric}/{interface} pair for both candidates"}
        out = out or task_dir / "runs" / candidate / "roofline.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        return report
    per_b = []
    for B in GRID:
        if B not in t_c or B not in t_b:
            continue
        bytes_c = kda_decode_bytes(B, dtype)
        bytes_b = kda_decode_bytes(B, "float32")
        us_b = t_b[B]
        us_c = t_c[B]
        bw_b = bytes_b / (us_b * 1e-6) / 1e9          # GB/s implied by baseline
        floor_at_b = bytes_c / (bw_b * 1e9) * 1e6     # us at baseline's implied BW
        per_b.append({
            "B": B, "bytes_candidate": bytes_c, "bytes_baseline_fp32": bytes_b,
            "baseline_us": round(us_b, 2), "candidate_us": round(us_c, 2),
            "speedup": round(us_b / us_c, 3),
            "baseline_implied_bw_gbps": round(bw_b, 1),
            "baseline_efficiency_pct": round(100 * bw_b / SPEC_BW_GBPS, 1),
            "candidate_implied_bw_gbps": round(bytes_c / (us_c * 1e-6) / 1e9, 1),
            "candidate_efficiency_pct": round(100 * (bytes_c / (us_c * 1e-6) / 1e9) / SPEC_BW_GBPS, 1),
            "floor_us_at_baseline_bw": round(floor_at_b, 1),
            "floor_us_at_spec_bw": round(bytes_c / (SPEC_BW_GBPS * 1e9) * 1e6, 1),
        })
    if per_b:
        geo = math.exp(sum(math.log(p["speedup"]) for p in per_b) / len(per_b))
        verdict = ("TARGET MET" if geo >= 2.0 and all(p["speedup"] >= 1.0 for p in per_b)
                   else f"geomean {geo:.2f}x (goal 2.0x, every B >= 1.0x)")
    else:
        geo, verdict = None, "no matching results.csv rows"
    report = {"task_dir": str(task_dir), "candidate": candidate, "baseline": baseline,
              "state_dtype": dtype, "metric": metric, "interface": interface,
              "spec_bw_gbps": SPEC_BW_GBPS,
              "per_B": per_b, "geomean_speedup": geo, "verdict": verdict}
    out = out or task_dir / "runs" / candidate / "roofline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-dir", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", default="baseline-fla-decode")
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()
    r = roofline_report(Path(args.task_dir), args.candidate, args.baseline, args.dtype)
    print(f"ROOFLINE {args.candidate} ({args.dtype} state): {r['verdict']}")
    for p in r["per_B"]:
        print(f"  B={p['B']:4d} base={p['baseline_us']:8.1f}us cand={p['candidate_us']:8.1f}us "
              f"speedup={p['speedup']:5.2f}x eff={p['candidate_efficiency_pct']:5.1f}% "
              f"floor@baseBW={p['floor_us_at_baseline_bw']:7.1f}us")
    return 0


if __name__ == "__main__":
    sys.exit(main())