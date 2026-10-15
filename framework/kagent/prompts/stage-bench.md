# Stage packet — bench

    kagent run --stages bench --pack <pack> --candidate <dir>

- Uses the task's bench script on a free GPU; every measurement lands in
  results.csv via tools/record.py with env stamps. If the GPU was busy, the
  numbers are invalid — rerun, do not argue.
- The serving interface is what counts (task.toml [target]); a gather/scatter
  wrapper around the candidate is not the serving interface — record the
  interface field honestly.
- Medians over >= 5 steps; same tool both sides. If you need a different
  metric, add it as a NEW row, never by overwriting.
- Before reporting ANY number, invariant 3 must hold (the engine enforces it).