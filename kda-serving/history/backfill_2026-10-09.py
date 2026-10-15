"""One-off: build results.csv / candidates.jsonl (new schema) from 2026-10-09 artifacts.

Every number below is read from a saved artifact; nothing is typed in. Rows from before
2026-10-09 stay in history/benchmark.v0.csv and history/candidates.v0.jsonl.
"""
import csv
import glob
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

TASK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK.parent / "tools"))
from record import append_result, upsert_candidate  # noqa: E402

RES, CAND = TASK / "results.csv", TASK / "candidates.jsonl"
for p in (RES, CAND):
    if p.exists():
        p.unlink()

ENV4 = json.loads((TASK / "runs/baseline-fla-decode/verify_decode_report.json").read_text())["env"]
envcols = lambda env, gpu=None: dict(physical_gpu=gpu if gpu is not None else env["physical_gpu"], rocm=env["rocm"],
                                     torch=env["torch"], fla=env["flash_linear_attention"], flydsl=env["flydsl"])

# 1. Decode verifier (pre-registered bar)
for name in ("baseline-fla-decode", "flydsl-decode-v7", "emul-bf16-state", "emul-fp16-state"):
    path = TASK / f"runs/{name}/verify_decode_report.json"
    r = json.loads(path.read_text())
    res = [x for x in r["results"] if "state_rrmse" in x]
    worst = max(res, key=lambda x: x["state_rrmse"])
    drift = next(x for x in res if x["gate"].startswith("D4"))
    sd = {"emul-bf16-state": "bf16", "emul-fp16-state": "fp16"}.get(name, "fp32")
    common = dict(task="kda-serving", candidate=name, kind="verifier", mode="decode", state_dtype=sd,
                  gates="kimi+synthetic+stress", interface="public-wrapper", timing_method="n/a",
                  tool="verifier/verify_decode.py", artifact=str(path.relative_to(TASK)), **envcols(r["env"]))
    append_result(RES, metric="gates_passed", value=r["n_gates"] - r["n_failed"], unit=f"of {r['n_gates']}", **common)
    append_result(RES, metric="state_rrmse_worst", value=worst["state_rrmse"], unit="rel-RMSE vs fp64",
                  reference="bar", reference_value=1e-2, notes=f"worst gate {worst['gate']}", **common)
    append_result(RES, metric="state_rrmse_drift8192", value=drift["state_rrmse"], unit="rel-RMSE vs fp64",
                  reference="bar", reference_value=1e-2, T=8192, **common)

# 2. Serving-interface decode (in-place state pool), graphed
for fname, gates in (("serving.json", "kimi"), ("serving_fla_synthetic.json", "synthetic")):
    path = TASK / "profile/serving-decode-v7" / fname
    s = json.loads(path.read_text())
    for row in s["rows"]:
        common = dict(task="kda-serving", kind="measurement", mode="decode", B=row["B"], H=row["H"], T=1,
                      gates=gates, interface="serving-pool", timing_method="cuda-graph-replay", state_dtype="fp32",
                      tool="bench/serving_decode.py", artifact=str(path.relative_to(TASK)), **envcols(s["env"]))
        append_result(RES, candidate="baseline-fla-decode", metric="graphed_us_per_step",
                      value=row["fla_native_pool_us"], unit="us", notes="FLA native pool, in-place", **common)
        if "candidate_us" in row:
            append_result(RES, candidate="flydsl-decode-v7", parent="flydsl-decode-v6", metric="graphed_us_per_step",
                          value=row["candidate_us"], unit="us", reference="baseline-fla-decode",
                          reference_value=row["fla_native_pool_us"], ratio_vs_reference=row["speedup_vs_fla"],
                          notes="gather-scatter wrapper (no native pool support); penalty not a kernel property",
                          **common)

# 3. TraceLens roofline (MI355X spec, KDA extension)
path = TASK / "profile/tl-decode-B128/csvs/linear_attention_fwd.csv"
for row in csv.DictReader(open(path)):
    name = row["name"].replace("kda::decode_step_", "").replace("_", "-")
    common = dict(task="kda-serving", candidate=name, kind="profile", mode="decode", B=int(row["param: B"]), H=96, T=1,
                  gates="kimi", interface="public-wrapper", timing_method="torch.profiler", state_dtype="fp32",
                  tool="TraceLens + tracelens/kda_extension.py", artifact=str(path.relative_to(TASK)), **envcols(ENV4))
    append_result(RES, metric="kernel_us_median", value=float(row["Kernel Time (µs)_median"]), unit="us", **common)
    append_result(RES, metric="achieved_tb_s", value=float(row["TB/s_median"]), unit="TB/s", **common)
    append_result(RES, metric="pct_roofline", value=float(row["Pct Roofline_median"]), unit="% of MI355X 8 TB/s",
                  **common)

# 4. rocprofv3 hardware counters (GPU 7 when it was free)
vals, dur = defaultdict(lambda: defaultdict(list)), defaultdict(list)
for f in glob.glob(str(TASK / "profile/rocprof-decode-B128/*/run_counter_collection.csv")):
    for r in csv.DictReader(open(f)):
        kn = "baseline-fla-decode" if "fused_recurrent" in r["Kernel_Name"] else (
            "flydsl-decode-v7" if "kda_decode" in r["Kernel_Name"] else None)
        if kn:
            vals[kn][r["Counter_Name"]].append(float(r["Counter_Value"]))
for r in csv.DictReader(open(TASK / "profile/rocprof-decode-B128/FETCH_SIZE/run_kernel_trace.csv")):
    kn = "baseline-fla-decode" if "fused_recurrent" in r["Kernel_Name"] else (
        "flydsl-decode-v7" if "kda_decode" in r["Kernel_Name"] else None)
    if kn:
        dur[kn].append((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3)
for kn in ("baseline-fla-decode", "flydsl-decode-v7"):
    d = {c: st.median(x) for c, x in vals[kn].items()}
    us = st.median(dur[kn])
    moved = 2 * d["FETCH_SIZE"] * 1024 + d["WRITE_SIZE"] * 1024  # FETCH_SIZE reports half on gfx950
    common = dict(task="kda-serving", candidate=kn, kind="profile", mode="decode", B=128, H=96, T=1, gates="synthetic",
                  interface="public-wrapper", timing_method="rocprofv3", state_dtype="fp32",
                  tool="rocprofv3 (one derived metric per pass)", artifact="profile/rocprof-decode-B128/",
                  **envcols(ENV4, gpu=7))
    append_result(RES, metric="kernel_us_median", value=round(us, 1), unit="us", **common)
    append_result(RES, metric="achieved_tb_s", value=round(moved / us / 1e6, 2), unit="TB/s",
                  notes="reads = 2 x FETCH_SIZE (gfx950 undercount)", **common)
    append_result(RES, metric="wait_issue_pct", value=round(100 * d["SQ_WAIT_INST_ANY"] / d["SQ_WAVE_CYCLES"], 1),
                  unit="% of wave lifetime", **common)
    append_result(RES, metric="vmem_latency_cycles", value=round(d["VmemLatency"]), unit="cycles", **common)

# 5. Candidate registry (current status; numbers live in results.csv)
reg = [
    dict(name="baseline-fla-decode", parent=None, status="baseline", language="triton", state_dtype="fp32",
         interface="public-wrapper; serving-pool via fused_recurrent_kda_fwd", code="runs/baseline-fla-decode/candidate.py",
         verifier_report="runs/baseline-fla-decode/verify_decode_report.json",
         desc="FLA 0.5.2 fused_recurrent_kda; reference for the 2x target (use its serving path)"),
    dict(name="flydsl-decode-v7", parent="flydsl-decode-v6", status="active", language="flydsl", state_dtype="fp32",
         interface="public-wrapper; raw-launch", code="runs/flydsl-decode-v7/candidate.py",
         verifier_report="runs/flydsl-decode-v7/verify_decode_report.json",
         desc="256-thread column-split, in-wave shuffle trees, 0 barriers, vector o stores",
         reason="verifier 15/15; kernel 1.45-1.70x slower than FLA; issue-bound (74% wait-for-issue)",
         next="v8: native state pool, issue-stall fixes, then fp16 state"),
    dict(name="emul-bf16-state", parent="baseline-fla-decode", status="rejected", language="triton",
         state_dtype="bf16", interface="public-wrapper", code="runs/emul-bf16-state/candidate.py",
         verifier_report="runs/emul-bf16-state/verify_decode_report.json",
         desc="accuracy emulation of bf16 state storage", reason="fails near-unity and zero-v gates (1.6%, 1.4% > 1% bar)"),
    dict(name="emul-fp16-state", parent="baseline-fla-decode", status="accuracy-only", language="triton",
         state_dtype="fp16", interface="public-wrapper", code="runs/emul-fp16-state/candidate.py",
         verifier_report="runs/emul-fp16-state/verify_decode_report.json",
         desc="accuracy emulation of fp16 state storage",
         reason="passes all 15 gates, worst 2.0e-3; fp16 is the Path B storage format"),
    dict(name="flydsl-decode-v8", parent="flydsl-decode-v7", status="planned", language="pending user decision",
         state_dtype="fp32, then fp16", interface="serving-pool (native)",
         desc="native indexed in-place pool; hoist state loads, interleave trees, DPP sum steps; then fp16 state",
         next="confirm language and 2x definition with user"),
]
for v, st_, why in ((1, "revised", "smem + barriers; correct, 0.10-0.31x"), (2, "revised", "vec4 single read; 0.22-0.50x"),
                    (3, "rejected", "coalesced-store theory wrong; regression"), (4, "revised", "FLA tiling port; ~v2"),
                    (5, "revised", "shuffle trees, 2 barriers"), (6, "revised", "column-split, 0 barriers; 1.5-1.7x kernel")):
    reg.append(dict(name=f"flydsl-decode-v{v}", parent=f"flydsl-decode-v{v - 1}" if v > 1 else "baseline-fla-decode",
                    status=st_, language="flydsl", state_dtype="fp32", interface="public-wrapper",
                    code=f"runs/flydsl-decode{'' if v == 1 else f'-v{v}'}/candidate.py", reason=why,
                    desc="lineage; details in history/candidates.v0.jsonl"))
for rec in reg:
    upsert_candidate(CAND, **rec)
print(f"wrote {sum(1 for _ in open(RES)) - 1} result rows and {len(reg)} candidates")
