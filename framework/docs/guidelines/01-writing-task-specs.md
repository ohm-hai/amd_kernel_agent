# Guideline 01 — Writing task specs

A task pack is the contract between the user, the agent, and the evaluator. If a
rule is not in `task.toml`, the loop engine cannot enforce it, and the agent can
drift. Write specs as if a stranger will grade the work, because a frozen
evaluator — not the agent — will.

## Rules

1. **One measurable objective.** "2× FLA Triton on KDA decode" with the
   aggregation decided by the user (geomean over B ∈ {1,8,32,128}). Vague
   objectives produce vague kernels.
2. **Aggregation and minimum-ratio are explicit.** `per_point_min_ratio >= 1.0`
   prevents "wins the geomean by collapsing one point" hacks.
3. **Reference is the deployment path, not the paper path.** For kda-serving the
   reference is FLA's serving path (`fused_recurrent_kda_fwd` with
   `ssm_state_indices` + `inplace_final_state=True`), which is 8–19% faster than
   the public wrapper at B>=32. Optimizing against the wrong reference produces
   real speedups that do not ship.
4. **Promotion rules must be checkable by the loop engine.** Each rule maps to a
   stage result: frozen-verifier pass, interface check, measured ratio. A rule
   like "the kernel should be elegant" is not a rule.
5. **Constraints are data.** Language (flydsl), GPU hygiene (free-only), no
   Docker, no downloads. The loop engine refuses to run stages that violate them.
6. **Paths point at real files.** `kagent check` resolves every path before any
   stage runs; a pack with dead paths is rejected.
7. **The scope boundary is explicit in the objective.** What is out of scope
   (projections, conv, norm for KDA) prevents the agent from claiming fusion
   wins that the contract does not measure.

## Worked example: `examples/kda-serving`

- objective: 2× FLA Triton on KDA decode (Kimi K3 layer, H=96, K=V=128) on MI355X
- target: metric `graphed_wall_us`, aggregation `geomean`, grid {1,8,32,128},
  goal 2.0, per-point min ratio 1.0
- promotion: frozen verifier pass → native serving interface → faster on the
  serving path with the per-B breakdown recorded
- physics note recorded in the pack: with fp32 state the B=128 roofline floor
  is above 2× of FLA, so the target is only reachable with reduced-precision
  state — which is why the bar (03) and the drift cases (02) exist before v8.

## Anti-patterns

- Copying the baseline's assumptions into the spec (e.g. fp32 state) without
  measuring; the bar decides the state dtype, not the baseline.
- Writing the spec after the first candidate. Specs are pre-registered too.
- Hiding the tradeoff (accuracy vs speed) instead of making the bar own it.