"""PC-sampling analysis via rocprofv3 (stochastic method).

Runs the candidate's self-test under rocprofv3 --pc-sampling (always inside
`timeout`), parses pcs_pc_sampling_stochastic.csv and aggregates stall
reasons (guideline 04 maps reason -> next action) plus the top sampled PCs.
Unrecognized-code-object PCs are kept as offsets relative to the hottest base
and can be matched against the ISA dump's code offsets.

    python3 -m kagent.evidence.pcs --candidate <dir> [--kernel-regex kda] [--gpu 4]
Writes <candidate>/pcs.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # framework/

STALL_PREFIX = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_"
NEXT_ACTION = {
    "WAITCNT": "waiting on memory (LDS/global): overlap more work per wait, wider loads, fewer barriers",
    "ALU_DEPENDENCY": "serial FMA chain: reassociate or split into independent lanes",
    "BARRIER_WAIT": "sync imbalance: rebalance workgroups or remove barriers",
    "NO_INSTRUCTION_AVAILABLE": "tail/occupancy: more parallel work per CU",
    "OTHER_WAIT": "misc wait: inspect top PCs",
    "ARBITER_WIN_EX_STALL": "EX arbiter contention: reduce per-wave issue pressure",
}


def run_pcs(candidate_dir: Path, out_dir: Path | None = None, kernel_regex: str = "kda",
            gpu: str = "4", interval: int = 65536, timeout: int = 900) -> dict:
    out_dir = out_dir or Path("/tmp/kagent-pcs") / candidate_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["HIP_VISIBLE_DEVICES"] = gpu
    env.setdefault("PYTHONPATH", "")
    klab = Path(__file__).resolve().parents[2].parent
    env["PYTHONPATH"] = str(klab / "kda-attn") + os.pathsep + env["PYTHONPATH"]
    cmd = ["timeout", str(timeout), "rocprofv3",
           "--pc-sampling-beta-enabled", "--pc-sampling-method", "stochastic",
           "--pc-sampling-unit", "cycles", "--pc-sampling-interval", str(interval),
           "--kernel-include-regex", kernel_regex,
           "-d", str(out_dir), "-o", "pcs", "--output-format", "csv",
           "--", sys.executable, str(candidate_dir / "candidate.py")]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    (out_dir / "run.log").write_text((proc.stdout or "") + (proc.stderr or ""))
    csvs = sorted(out_dir.rglob("*pc_sampling*.csv"))
    if not csvs:
        return {"candidate": str(candidate_dir), "error": "no pc-sampling csv produced",
                "log": str(out_dir / "run.log"),
                "stderr_tail": (proc.stderr or "")[-800:]}
    rows = list(csv.DictReader(open(csvs[-1])))
    reasons = Counter()
    instrs = Counter()
    unk_pcs = Counter()
    for r in rows:
        reason = r.get("Stall_Reason", "").replace(STALL_PREFIX, "") or "NONE"
        reasons[reason] += 1
        ins = (r.get("Instruction") or "").strip()
        if ins:
            instrs[ins] += 1
        cmt = r.get("Instruction_Comment") or ""
        if cmt.startswith("Unrecognized code object id"):
            try:
                unk_pcs[int(cmt.rsplit(":", 1)[1])] += 1
            except ValueError:
                pass
    top_pcs = []
    if unk_pcs:
        base = min(unk_pcs)
        top_pcs = [{"offset_from_base": pc - base, "samples": n}
                   for pc, n in sorted(unk_pcs.items(), key=lambda kv: -kv[1])[:10]]
    top = reasons.most_common(3)
    action = "; ".join(f"{name}: {NEXT_ACTION.get(name, 'inspect')}" for name, _ in top)
    report = {
        "candidate": str(candidate_dir), "csv": str(csvs[-1]),
        "n_samples": len(rows), "stall_reasons": dict(reasons),
        "top_instructions": dict(instrs.most_common(8)),
        "top_unmapped_pc_offsets": top_pcs,
        "seconds": round(time.perf_counter() - t0, 1),
        "interpretation": action,
    }
    out = candidate_dir / "pcs.json"
    out.write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--kernel-regex", default="kda")
    ap.add_argument("--gpu", default="4")
    args = ap.parse_args()
    r = run_pcs(Path(args.candidate), kernel_regex=args.kernel_regex, gpu=args.gpu)
    if "error" in r:
        print(f"PCS FAILED: {r['error']} (log {r['log']})")
        print(r.get("stderr_tail", ""))
        return 1
    print(f"PCS: {r['n_samples']} samples in {r['seconds']}s; "
          f"top stalls: {dict(list(r['stall_reasons'].items())[:3])}")
    print("top PCs (offset, samples):",
          [(p['offset_from_base'], p['samples']) for p in r['top_unmapped_pc_offsets'][:5]])
    print("->", r["interpretation"])
    return 0


if __name__ == "__main__":
    sys.exit(main())