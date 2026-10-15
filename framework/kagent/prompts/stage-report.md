# Stage packet — report (then STOP)

    kagent run --stages report --pack <pack> --candidate <dir>

Renders agent_report.md from the artifacts. Then:
1. State the per-stage wall-clock cost of this loop (the engine prints it) —
   the cadence decision (one candidate -> report -> user input) depends on it.
2. Fill the "Next hypothesis" section: ONE hypothesis, the metric it attacks,
   the expected magnitude, and the risk.
3. Update candidates.jsonl status (promoted / revised / rejected+reason).
4. STOP and hand the report to the user. Do not start the next version.

A report that only contains wins is a red flag (doc 05). Report failures with
the witness or the metric that failed, and what you would try next.