# v8 draft — native in-place indexed state pool

Parent: flydsl-decode-v7 (formal PROVED, frozen verify 15/15, holdout 10/10,
serving-path 0.25x of FLA — wrapper-dominated).

## Diagnosis (from the evidence artifacts, guideline 04)

1. `roofline.json`: v7 serving-pool `graphed_us_per_step` = 0.25x of FLA. The
   gather (index_select) + scatter (index_copy_) around the kernel add 3 extra
   state passes; the kernel alone was measured at 42% of roofline (kernel-only
   486 us at B=128 vs 1.63 GB fp32 state traffic).
2. `isa.json`: kernel is issue-bound (math/data 2.13), NOT memory-bound — but
   the serving gap is dominated by the wrapper, not the kernel.
3. FLA serving path updates the pool in place; our contract requires the same.

## Hypothesis

Making the kernel read/write `pool[idx[b]]` directly (native `decode_pool`)
removes the extra passes and should move the serving-path ratio from 0.25x to
roughly the kernel-only parity band (~0.8-1.0x) at large B, with no kernel-math
change. The math stays exactly v7's (formal PROVED), so this version isolates
the interface fix.

## Design

- Kernel = v7 body + two changes:
  - `IDX` buffer (int32 [B]) and compile-time `H` (baked per compile, cached
    per H — H is model-fixed, so the TokenSpeed recompile trap does not apply;
    the verifier compiles H in {1, 8, 96}, deployment compiles H=96 once).
  - state addressing: `slot = IDX[b]; s_head = slot*(H*K*V) + h*(K*V)`.
  - pool passed as BOTH S_IN and S_OUT (in-place; each element is read and
    written only by its owning lane, so there is no cross-lane hazard).
- `decode_pool(q, k, v, g, beta, pool, idx)`: graph-capturable, no host sync,
  o allocated per call (lands in the CUDA-graph pool).
- `get_forward`: INPLACE=True, idx = arange(B) over initial_state.

## Expected effect and risks

- Expect `graphed_us_per_step` at B=128 to drop from 1361.7 us toward the
  kernel-only ~486 us + FLA-parity band; geomean from 0.25x toward ~0.8x.
- Risk: in-place aliasing (S_IN == S_OUT) — mitigated by the ownership
  argument above and re-checked by the frozen verifier (D2/D3 warm-state gates).
- Risk: formal checker must model the IDX indirection — handled by the
  identity-slot assumption (slot = b for the checked mapping), documented in
  the report; the arbitrary-idx case is the same per-(slot, head) math.
- Explicitly OUT of scope for v8 (next versions): issue-bound layout work
  (fewer shuffles), fp16 state (Path B — the bar already selects fp16).

## Validation commands

    cd /root/amd-workspace/kernel-lab/framework
    python3 -m kagent.loop.cli run --pack examples/kda-serving \
        --candidate ../kda-serving/runs/flydsl-decode-v8 \
        --stages verify,holdout,formal,bench,evidence,report --dtype float32

Promotion check (contract): frozen verify 15/15 at fp32 → faster than v7 on the
serving path with per-B breakdown recorded; the 2x target itself needs fp16
state (Path B) and is NOT expected from this version alone.