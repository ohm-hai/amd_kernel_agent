# Guideline 04 — Evidence interpretation

Evidence artifacts (`isa.json`, `pcs.json`, `roofline.json`) exist to replace
guess-first diagnosis with measurement. The rule of the loop: **profile, then
diagnose, then plan** — never design a version from intuition alone.

## Reading `isa.json` (gfx950)

- `alu_per_lane` vs `data_per_lane` (the ALU-to-DATA-shuffle ratio): this is the
  issue-bound detector for wavefront-reduction kernels. Our v7 decode measured
  ~110 ALU vs 76 `ds_swizzle` per lane (ratio ≈ 1.4); a coalesced design should
  reach ≥ 8 before bandwidth becomes the limit. If ALU/DATA < 4 and the kernel
  is slow, suspect the reduction layout, not the memory system.
- `vgpr_count` → occupancy: ≤ 84 allows 8 waves/SIMD on gfx950; ≥ 128 halves it.
  A kernel that cannot fit the state rows in registers will spill; check
  `private_segment_fixed_size` > 0 for spills.
- `s_nop` count: compiler-inserted scheduling stalls; many `s_nop` next to
  `s_waitcnt` means the scheduler fought a dependency chain you can restructure.
- Interpretation is a *hypothesis generator*: the next version's plan must cite
  which metric it attacks and by how much.

## Reading `pcs.json` (PC sampling)

Stall-reason → next action:

| reason | meaning | next action |
|---|---|---|
| `WAITCNT` | waiting on memory (LDS/global) | overlap: more work per wait, wider loads, fewer barriers |
| `ALU_DEPENDENCY` | serial FMA chain | reassociate, split into independent lanes |
| `BARRIER_WAIT` | sync imbalance | rebalance workgroups, remove barriers |
| `NO_INSTRUCTION_AVAILABLE` | tail / occupancy | more parallel work per CU |

A dominant WAITCNT at one PC next to a buffer load of the state says the state
read is on the critical path — that is where layout changes (V-major, vector
width) pay. Samples on an "Unrecognized code object" PC are still useful: they
are aggregated per offset and mapped against the ISA dump where possible.

## Reading `roofline.json`

- `efficiency_pct` = achieved bytes/sec vs measured peak. FLA runs ~70–78% at
  B≥32 with fp32 state; v7 was 41.7%. The gap to close is issue efficiency,
  not more bandwidth.
- `floor_us` at spec bandwidth is optimistic; at *measured* bandwidth (from a
  real copy kernel or TraceLens) it is the honest bound. Use the measured one
  for feasibility claims.
- If a candidate exceeds the measured floor, the measurement is wrong; stop and
  re-verify the timing method before celebrating.

## Measurement hygiene

- Same tool for both sides (torch.profiler or rocprofv3); never compare across
  tools. CUDA-graph replay for serving-path latency; medians over ≥ 5 steps.
- `require_free_gpu()` before every measurement; the shared node invalidates
  numbers otherwise.
- A single-trial number is a hypothesis, not a result. Repeat or say so.
- Every number lands in `results.csv` via `tools/record.py` with its artifact
  path; a claim without an artifact id is not reportable.