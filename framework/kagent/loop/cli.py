"""The `kagent` CLI — the surface bindings (Cursor skills, SDK orchestrators) call.

    python3 -m kagent.loop.cli check   --pack examples/kda-serving
    python3 -m kagent.loop.cli verify  --pack ... --candidate ...
    python3 -m kagent.loop.cli holdout --pack ... --candidate ...
    python3 -m kagent.loop.cli formal  --pack ... --candidate ... [--state-dtype fp16]
    python3 -m kagent.loop.cli bench   --pack ... --candidate ...
    python3 -m kagent.loop.cli evidence --pack ... --candidate ... [--dtype float32]
    python3 -m kagent.loop.cli report  --pack ... --candidate ...
    python3 -m kagent.loop.cli run     --pack ... --candidate ... --stages verify,formal,evidence,report
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .engine import Loop


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kagent")
    ap.add_argument("--pack", required=False,
                    help="task-pack dir (accepted before or after the subcommand)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "verify", "holdout", "formal", "bench", "evidence",
                 "report", "run"):
        sp = sub.add_parser(name)
        sp.add_argument("--pack", required=False)
        if name != "check":
            sp.add_argument("--candidate", required=True)
        if name in ("formal", "evidence", "report", "run"):
            sp.add_argument("--dtype", default="float32",
                            help="declared state dtype (float32|float16)")
        if name == "evidence":
            sp.add_argument("--gpu", default="4")
        if name == "run":
            sp.add_argument("--stages", default="verify,formal,evidence,report")
            sp.add_argument("--gpu", default="4")
            sp.add_argument("--force", action="store_true",
                            help="user-only: skip the invariant-3 verify gate")
        if name == "check":
            pass
    args = ap.parse_args(argv)
    pack_dir = getattr(args, "pack", None)
    args.pack = pack_dir
    if not args.pack:
        ap.error("--pack is required (before or after the subcommand)")
    t0 = time.perf_counter()

    if args.cmd == "check":
        from kagent.spec.loaders import load_pack
        pack = load_pack(args.pack)
        n_vis, n_hid = len(pack.visible_cases()), len(pack.holdout_cases())
        print(f"pack OK: {pack.task.name} | {n_vis} visible cases, {n_hid} hidden "
              f"holdouts | thresholds {list(pack.bar.thresholds)}")
        print(f"target: {pack.task.target.aggregation} >= {pack.task.target.goal} "
              f"on {pack.task.target.metric}, grid {pack.task.target.grid}, "
              f"per-point min ratio {pack.task.target.per_point_min_ratio}")
        return 0

    loop = Loop(args.pack)
    candidate = Path(args.candidate).resolve()
    if args.cmd in ("verify", "holdout"):
        r = loop.verify(candidate) if args.cmd == "verify" else loop.holdout(candidate)
        return 0 if r.ok else 1
    if args.cmd == "formal":
        r = loop.formal(candidate, args.dtype)
        return 0 if r.ok else 1
    if args.cmd == "bench":
        r = loop.bench(candidate)
        return 0 if r.ok else 1
    if args.cmd == "evidence":
        r = loop.evidence(candidate, args.dtype, getattr(args, "gpu", "4"))
        return 0 if r.ok else 1
    if args.cmd == "report":
        from .report import build_report, render_markdown
        rep = build_report(loop, candidate, args.dtype)
        out = candidate / "agent_report.md"
        out.write_text(render_markdown(rep))
        print(f"report -> {out}")
        return 0
    if args.cmd == "run":
        stages = [s.strip() for s in args.stages.split(",") if s.strip()]
        results = loop.run(candidate, stages, dtype=args.dtype, gpu=args.gpu,
                           force=args.force)
        print(f"total: {time.perf_counter() - t0:.1f}s; "
              f"{'all OK' if all(r.ok for r in results) else 'SOME FAILED'}")
        return 0 if all(r.ok for r in results) else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())