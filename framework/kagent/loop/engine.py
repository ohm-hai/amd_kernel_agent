"""Stage machine + StageResult plumbing for the kernel agent loop."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..spec.loaders import load_pack

FRAMEWORK = Path(__file__).resolve().parents[2]


@dataclass
class StageResult:
    name: str
    ok: bool
    seconds: float
    summary: dict = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    log: str = ""

    def as_dict(self) -> dict:
        return {"stage": self.name, "ok": self.ok, "seconds": round(self.seconds, 1),
                "summary": self.summary, "artifacts": self.artifacts}


def _run_module(mod: str, args: list[str], timeout: int = 3600) -> tuple[int, str]:
    cmd = [sys.executable, "-W", "ignore", "-m", mod] + args
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=FRAMEWORK,
                           timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 2, f"TIMEOUT after {timeout}s"


def _verify_passed(candidate: Path) -> dict | None:
    p = candidate / "verify_decode_report.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


class Loop:
    def __init__(self, pack_dir: str):
        self.pack = load_pack(pack_dir)
        self.task = self.pack.task
        # the pack's task_dir field points at the task; resolve it ONCE
        # (TaskSpec.resolve joins task_dir itself, so never pass it as rel)
        self.task_dir = Path(os.path.normpath(
            os.path.join(self.task.pack_dir, self.task.paths.task_dir)))

    # ---------------------------------------------------------- stages

    def verify(self, candidate: Path, allow_busy_gpu: bool = False) -> StageResult:
        t0 = time.perf_counter()
        args = ["--pack", str(Path(self.pack.pack_dir)),
                "--candidate", str(candidate)]
        if allow_busy_gpu:
            args.append("--allow-busy-gpu")
        rc, log = _run_module("kagent.eval.run_eval", args, timeout=7300)
        rep = _verify_passed(candidate) or {}
        return StageResult("verify", ok=rc == 0, seconds=time.perf_counter() - t0,
                           summary={"passed": rep.get("passed"),
                                    "n_gates": rep.get("n_gates"),
                                    "n_failed": rep.get("n_failed")},
                           artifacts=[str(candidate / "verify_decode_report.json")]
                           if rep else [], log=log)

    def holdout(self, candidate: Path, allow_busy_gpu: bool = False) -> StageResult:
        t0 = time.perf_counter()
        args = ["--pack", str(Path(self.pack.pack_dir)), "--candidate", str(candidate),
                "--mode", "holdout"]
        if allow_busy_gpu:
            args.append("--allow-busy-gpu")
        rc, log = _run_module("kagent.eval.run_eval", args, timeout=3700)
        p = candidate / "holdout_report.json"
        rep = json.loads(p.read_text()) if p.exists() else {}
        return StageResult("holdout", ok=rc == 0, seconds=time.perf_counter() - t0,
                           summary={"n_gates": rep.get("n_gates"),
                                    "n_failed": rep.get("n_failed")},
                           artifacts=[str(p)] if rep else [], log=log)

    def formal(self, candidate: Path, state_dtype: str = "float32") -> StageResult:
        t0 = time.perf_counter()
        rc, log = _run_module("kagent.formal.check",
                              ["--candidate", str(candidate),
                               "--state-dtype", state_dtype], timeout=1800)
        p = candidate / "formal_report.json"
        rep = json.loads(p.read_text()) if p.exists() else {}
        return StageResult("formal", ok=rc == 0, seconds=time.perf_counter() - t0,
                           summary={"overall": rep.get("overall"),
                                    "n_chains": rep.get("n_chains"),
                                    "selftest_exit": rep.get("selftest_exit")},
                           artifacts=[str(p)] if rep else [], log=log)

    def bench(self, candidate: Path, bench_args: list[str] | None = None) -> StageResult:
        t0 = time.perf_counter()
        bench = self.task_dir / self.task.paths.bench
        out_json = candidate / "serving.json"
        cmd = [sys.executable, str(bench), "--candidate", str(candidate),
               "--out", str(out_json)]
        try:
            p = subprocess.run(cmd + (bench_args or []), capture_output=True, text=True,
                               cwd=str(self.task_dir), timeout=7200)
            rc, log = p.returncode, (p.stdout or "") + (p.stderr or "")
        except subprocess.TimeoutExpired:
            rc, log = 2, "bench TIMEOUT"
        rows = []
        for line in (p.stdout or "").splitlines():
            line = line.strip()
            if line.startswith("{") and '"candidate_us"' in line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        recorded = 0
        if rows and rc == 0:
            tools = FRAMEWORK.parent / "tools"
            sys.path.insert(0, str(tools))
            import record  # noqa: PLC0415
            results = self.task_dir / self.task.paths.results
            for r in rows:
                record.append_result(
                    str(results), task=self.task.name, candidate=candidate.name,
                    kind="measurement", mode="decode", B=r["B"], H=r["H"],
                    state_dtype="float32", gates=r.get("gates"),
                    interface="serving-pool", metric="graphed_us_per_step",
                    value=r["candidate_us"], unit="us",
                    reference="baseline-fla-decode",
                    reference_value=r.get("fla_native_pool_us"),
                    ratio_vs_reference=r.get("speedup_vs_fla"),
                    timing_method="cuda-graph-replay", tool="bench/serving_decode.py",
                    artifact=str(out_json),
                    notes=f"mode={r.get('candidate_mode')} pool_slots={r.get('pool_slots')}")
                recorded += 1
        return StageResult("bench", ok=rc == 0, seconds=time.perf_counter() - t0,
                           summary={"rows": len(rows), "recorded": recorded,
                                    "results_csv": str(self.task_dir / self.task.paths.results)},
                           artifacts=[str(out_json)] if rows else [], log=log)

    def evidence(self, candidate: Path, dtype: str = "float32",
                 gpu: str = "4") -> StageResult:
        t0 = time.perf_counter()
        cand_name = candidate.name
        arts = []
        r1, _ = _run_module("kagent.evidence.isa", ["--candidate", str(candidate)],
                            timeout=900)
        if (candidate / "isa.json").exists():
            arts.append(str(candidate / "isa.json"))
        r2, _ = _run_module("kagent.evidence.pcs",
                            ["--candidate", str(candidate), "--gpu", gpu], timeout=900)
        if (candidate / "pcs.json").exists():
            arts.append(str(candidate / "pcs.json"))
        r3, log3 = _run_module("kagent.evidence.roofline",
                               ["--task-dir", str(self.task_dir),
                                "--candidate", cand_name, "--dtype", dtype])
        p = self.task_dir / "runs" / cand_name / "roofline.json"
        if p.exists():
            arts.append(str(p))
        ok = bool(arts) and r3 == 0
        rl = json.loads(p.read_text()) if p.exists() else {}
        return StageResult("evidence", ok=ok, seconds=time.perf_counter() - t0,
                           summary={"roofline_verdict": rl.get("verdict"),
                                    "geomean": rl.get("geomean_speedup"),
                                    "metric": rl.get("metric")},
                           artifacts=arts, log=log3)

    # ----------------------------------------------------- order guard

    def _require_verify(self, candidate: Path, stage: str, force: bool) -> None:
        rep = _verify_passed(candidate)
        if force:
            return
        if not rep or not rep.get("passed"):
            raise SystemExit(
                f"[invariant 3] {stage} refused: the FROZEN verifier has not passed "
                f"for {candidate.name}. Run 'kagent run --stages verify' first.")

    # ---------------------------------------------------------- driver

    def run(self, candidate: Path, stages: list[str], dtype: str = "float32",
            gpu: str = "4", bench_args: list[str] | None = None,
            force: bool = False) -> list[StageResult]:
        results = []
        for s in stages:
            if s in ("bench", "evidence", "report"):
                self._require_verify(candidate, s, force)
            if s == "verify":
                r = self.verify(candidate)
            elif s == "holdout":
                r = self.holdout(candidate)
            elif s == "formal":
                r = self.formal(candidate, dtype)
            elif s == "bench":
                r = self.bench(candidate, bench_args)
            elif s == "evidence":
                r = self.evidence(candidate, dtype, gpu)
            elif s == "report":
                from .report import build_report, render_markdown
                rep = build_report(self, candidate, dtype)
                md = render_markdown(rep)
                out = candidate / "agent_report.md"
                out.write_text(md)
                r = StageResult("report", ok=True, seconds=0.0,
                                summary={"report": str(out),
                                         "stages": list(rep.get("stages", {}).keys())},
                                artifacts=[str(out)])
            else:
                raise SystemExit(f"unknown stage {s!r}")
            results.append(r)
            print(f"[{r.name}] {'OK' if r.ok else 'FAIL'} ({r.seconds:.1f}s) "
                  f"{json.dumps(r.summary)[:200]}")
        return results