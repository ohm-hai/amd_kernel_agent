# Stage packet — verify (frozen evaluator)

    kagent run --stages verify --pack <pack> --candidate <dir>

- This runs the HASH-LOCKED verifier from the eval crate. You cannot edit it.
  The lock is re-checked before and after the run; tampering aborts loudly.
- Read the report: which gate failed, at which tolerance. A FAIL is a finding:
  record it in candidates.jsonl (status=rejected or active+note) with the gate
  name and the worst metric.
- If the holdout fails while the visible set passes, you have overfit the
  visible seeds. Do not tune against visible seeds again; say so in the report.
- You may NOT rerun with modified thresholds. If you believe the bar is wrong,
  STOP and ask the user (guideline 03).