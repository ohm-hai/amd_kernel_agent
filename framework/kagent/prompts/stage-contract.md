# Stage packet — contract (read before anything else)

You are working on task `{task}`. Read, in order:
1. the task pack: `task.toml`, `bar.toml`, `cases/` — the rules are data;
   restate the objective, the grid, the aggregation and the promotion rules in
   one sentence in your own words before proceeding.
2. `docs/guidelines/01-writing-task-specs.md` and `05-review-and-anti-reward-hacking.md`.

Hard rules you must repeat to yourself at the start of every session:
- You NEVER edit the task pack, the eval crate, the verifier, the bar, or the
  framework. If you believe a rule is wrong, STOP and ask the user.
- No performance number is reportable before the frozen verifier passes
  (invariant 3). The loop engine enforces this; do not try to route around it.
- The reference is the deployment path named in `task.toml`, not the paper path.

Deliverable of this stage: one paragraph stating objective, reference, target,
and the case classes you must not skip. If you cannot state them, re-read.