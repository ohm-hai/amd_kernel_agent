"""Evidence records for kernel-lab tasks: one schema for every task.

results.csv     one row per measurement (long format). Env columns are filled from
                env_stamp() unless the caller passes them (e.g. when backfilling).
candidates.jsonl one line per candidate with its current status; history of numbers
                lives in results.csv, not here.

    from record import append_result, upsert_candidate
    append_result("kda-serving/results.csv", task="kda-serving", candidate="flydsl-decode-v8",
                  kind="measurement", mode="decode", B=128, H=96, metric="graphed_wall_us",
                  value=150.2, unit="us", interface="serving-pool", timing_method="cuda-graph-replay",
                  tool="bench/serving_decode.py", artifact="profile/<run>/serving.json")
"""
import csv
import datetime
import json
import os
from pathlib import Path

from env_stamp import env_stamp

RESULT_FIELDS = [
    "time", "task", "candidate", "parent", "kind", "mode", "B", "H", "T", "state_dtype", "gates",
    "interface", "metric", "value", "unit", "reference", "reference_value", "ratio_vs_reference",
    "timing_method", "tool", "physical_gpu", "rocm", "torch", "fla", "flydsl", "artifact", "notes",
]
REQUIRED = ["task", "candidate", "kind", "metric", "value", "unit", "tool"]
KINDS = {"measurement", "verifier", "profile", "finding"}
INTERFACES = {"public-wrapper", "serving-pool", "raw-launch", "n/a"}
TIMING_METHODS = {"cuda-graph-replay", "torch.profiler", "rocprofv3", "cuda-events", "perf_counter", "n/a"}

CANDIDATE_FIELDS = ["name", "parent", "status", "updated", "language", "state_dtype", "interface",
                    "desc", "code", "verifier_report", "key_results", "reason", "next"]
STATUSES = {"baseline", "active", "promoted", "revised", "rejected", "rejected-at-design",
            "planned", "blocked", "accuracy-only"}


def append_result(path, **row):
    missing = [f for f in REQUIRED if row.get(f) in (None, "")]
    unknown = sorted(set(row) - set(RESULT_FIELDS))
    if missing or unknown:
        raise ValueError(f"missing {missing}, unknown {unknown}")
    if row["kind"] not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")
    if row.get("interface") and row["interface"] not in INTERFACES:
        raise ValueError(f"interface must be one of {sorted(INTERFACES)}")
    if row.get("timing_method") and row["timing_method"] not in TIMING_METHODS:
        raise ValueError(f"timing_method must be one of {sorted(TIMING_METHODS)}")
    if not all(row.get(f) for f in ("physical_gpu", "rocm", "torch")):
        stamp = env_stamp()
        defaults = {"physical_gpu": stamp["physical_gpu"], "rocm": stamp["rocm"], "torch": stamp["torch"],
                    "fla": stamp["flash_linear_attention"], "flydsl": stamp["flydsl"]}
        for key, val in defaults.items():
            row.setdefault(key, val)
    row.setdefault("time", datetime.datetime.now().isoformat(timespec="seconds"))
    path = Path(path)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RESULT_FIELDS})


def upsert_candidate(path, **rec):
    unknown = sorted(set(rec) - set(CANDIDATE_FIELDS))
    if unknown or "name" not in rec:
        raise ValueError(f"need 'name'; unknown fields {unknown}")
    if rec.get("status") not in STATUSES:
        raise ValueError(f"status must be one of {sorted(STATUSES)}")
    rec.setdefault("updated", datetime.date.today().isoformat())
    path = Path(path)
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    lines = [l for l in lines if l.get("name") != rec["name"]] + [{k: rec.get(k) for k in CANDIDATE_FIELDS}]
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(l) + "\n" for l in lines))
    os.replace(tmp, path)
