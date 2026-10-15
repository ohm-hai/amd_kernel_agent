# Stage packet — candidate

Write ONE candidate at a time into `runs/<name>/candidate.py`:
- module-level `get_forward()` implementing the contracted interface, plus
  optional `STATE_DTYPE` / `INPLACE`;
- a `__main__` self-test that carries state across >= 3 steps and compares
  against the fp64 reference (it doubles as the IR-dump workload);
- keep the kernel shape-generic (runtime `BH` argument); a compile-time baked
  shape recompiles mid-serving (100+ ms each, TokenSpeed lesson);
- FlyDSL idioms that cost us hours before (see the MI355X skill, file 01):
  scalars pass as 1-element tensors, smem holds scalars only,
  `range_constexpr`, `ArithValue(smem[i])`, vector extf/truncf take vector
  types, lane-0 vec4 stores for masked writes.

After writing: run the candidate's self-test ONCE on a free GPU. If it prints
MISMATCH, fix it before the verifier — the frozen verifier is expensive; do
not use it as your smoke test. Record the candidate in `candidates.jsonl`
via `tools/record.py upsert_candidate` with parent + status=active.