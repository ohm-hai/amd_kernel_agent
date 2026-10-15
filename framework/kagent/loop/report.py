"""StageReport builder + Markdown renderer (driver-agnostic).

Refuses to render a speedup claim without its artifact (framework invariant 4):
the roofline table cites the results.csv metric and the artifact files.
"""
from __future__ import annotations

import json
from pathlib import Path

from .engine import _verify_passed


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def build_report(loop, candidate: Path, dtype: str = "float32") -> dict:
    cand_name = candidate.name
    stages: dict = {}
    verify = _verify_passed(candidate) or {}
    stages["verify"] = {"passed": verify.get("passed"),
                        "gates": f"{verify.get('n_gates', 0) - verify.get('n_failed', 0)}"
                                f"/{verify.get('n_gates', 0)}",
                        "thresholds": verify.get("bar")}
    holdout = _load(candidate / "holdout_report.json") or {}
    if holdout:
        stages["holdout"] = {"gates": f"{holdout['n_gates'] - holdout['n_failed']}"
                                      f"/{holdout['n_gates']} (counts only)"}
    formal = _load(candidate / "formal_report.json") or {}
    if formal:
        stages["formal"] = {"overall": formal.get("overall"),
                            "n_chains": formal.get("n_chains"),
                            "config": formal.get("config"),
                            "rounding_points": formal.get("rounding_points"),
                            "findings": formal.get("findings")}
    cand_meta = {}
    cjsonl = loop.task_dir / loop.task.paths.candidates
    try:
        for line in cjsonl.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                if d.get("name") == candidate.name:
                    cand_meta = d
    except OSError:
        pass
    if cand_meta:
        stages["candidate"] = {k: cand_meta.get(k) for k in
                               ("interface", "state_dtype", "status", "desc")}
    roofline = _load(loop.task_dir / "runs" / cand_name / "roofline.json") or {}
    if roofline:
        stages["roofline"] = {"verdict": roofline.get("verdict"),
                              "metric": roofline.get("metric"),
                              "interface": roofline.get("interface"),
                              "state_dtype": roofline.get("state_dtype"),
                              "geomean": roofline.get("geomean_speedup"),
                              "per_B": roofline.get("per_B")}
    isa = _load(candidate / "isa.json") or {}
    if isa:
        stages["isa"] = {k: isa.get(k) for k in
                         ("math_per_lane", "data_per_lane", "math_data_ratio",
                          "issue_data_ratio", "est_waves_per_simd", "s_nop",
                          "s_waitcnt", "interpretation")}
    pcs = _load(candidate / "pcs.json") or {}
    if pcs:
        stages["pcs"] = {"n_samples": pcs.get("n_samples"),
                         "stall_reasons": pcs.get("stall_reasons"),
                         "top_unmapped_pc_offsets": pcs.get("top_unmapped_pc_offsets"),
                         "interpretation": pcs.get("interpretation")}
    return {
        "task": loop.task.name,
        "candidate": cand_name,
        "candidate_dir": str(candidate),
        "objective": loop.task.objective,
        "state_dtype": dtype,
        "stages": stages,
        "artifacts": [
            str(candidate / "verify_decode_report.json"),
            str(candidate / "formal_report.json"),
            str(candidate / "isa.json"), str(candidate / "pcs.json"),
            str(loop.task_dir / "runs" / cand_name / "roofline.json"),
            str(loop.task_dir / loop.task.paths.results),
        ],
        "next_hypothesis": "TODO(agent): one hypothesis per report, citing the "
                           "metric it attacks and by how much (guideline 04)",
    }


def render_markdown(rep: dict) -> str:
    s = rep.get("stages", {})
    lines = [f"# Agent report — {rep['task']} / {rep['candidate']}", "",
             f"Objective: {rep['objective']}", ""]
    v = s.get("verify", {})
    lines += ["## Frozen verify", "",
              f"- **{v.get('gates', 'not run')} gates passed** "
              f"({'PASS' if v.get('passed') else 'NOT PASSED'})",
              f"- thresholds: `{v.get('thresholds')}`", ""]
    if "holdout" in s:
        lines += ["## Hidden holdout", "", f"- {s['holdout']['gates']} (counts only)", ""]
    if "formal" in s:
        f_ = s["formal"]
        lines += ["## Formal verification", "",
                  f"- **{f_['overall']}** over {f_.get('n_chains', 0)} reduction chains",
                  "- rounding points (not modeled; numeric verifier's domain): "
                  f"`{f_.get('rounding_points')}`",
                  "- scope: writer search bounded by the config in the report; "
                  "exp/rsqrt are uninterpreted functions shared with the spec — "
                  "PROVED means same math modulo that abstraction, not bitwise "
                  "equality", ""]
        for fd in f_.get("findings", []):
            line = f"  - [{fd.get('verdict')}] {fd.get('check')}"
            if fd.get("witness", {}).get("spec_over_cand"):
                line += f" — witness spec/cand={fd['witness']['spec_over_cand']:.4g}"
            lines.append(line)
        lines.append("")
    if "candidate" in s:
        c = s["candidate"]
        lines += ["## Candidate metadata (from candidates.jsonl)", "",
                  f"- interface: **{c.get('interface', 'undeclared')}**; "
                  f"state dtype: {c.get('state_dtype', '?')}; status: {c.get('status', '?')}",
                  f"- desc: {c.get('desc', '')}", ""]
    if "roofline" in s:
        rl = s["roofline"]
        lines += ["## Roofline (frozen-verified basis; same metric both sides)",
                  "",
                  f"- metric `{rl.get('metric')}` on `{rl.get('interface')}`; "
                  f"state dtype `{rl.get('state_dtype')}`; "
                  f"verdict: **{rl.get('verdict')}**", "",
                  "| B | baseline µs | candidate µs | speedup | eff % | floor@baseBW µs |",
                  "|---|---|---|---|---|---|"]
        for p in rl.get("per_B", []):
            lines.append(f"| {p['B']} | {p['baseline_us']} | {p['candidate_us']} | "
                         f"{p['speedup']} | {p['candidate_efficiency_pct']} | "
                         f"{p['floor_us_at_baseline_bw']} |")
        lines.append("")
    if "isa" in s:
        i = s["isa"]
        lines += ["## ISA (gfx950)", "",
                  f"- math/lane {i.get('math_per_lane')} vs data/lane {i.get('data_per_lane')} "
                  f"(ratio {i.get('math_data_ratio')}); issue/data {i.get('issue_data_ratio')}; "
                  f"~{i.get('est_waves_per_simd')} waves/SIMD; s_nop {i.get('s_nop')}; "
                  f"s_waitcnt {i.get('s_waitcnt')}",
                  f"- **{i.get('interpretation')}**", ""]
    if "pcs" in s:
        p_ = s["pcs"]
        lines += ["## PC sampling", "",
                  f"- {p_.get('n_samples')} samples; stalls: `{p_.get('stall_reasons')}`",
                  f"- top PCs (offset, samples): `{p_.get('top_unmapped_pc_offsets')[:5]}`",
                  f"- **{p_.get('interpretation')}**", ""]
    lines += ["## Artifacts", ""]
    lines += [f"- `{a}`" for a in rep.get("artifacts", [])]
    lines += ["", "## Next hypothesis", "", rep.get("next_hypothesis", ""), ""]
    return "\n".join(lines)