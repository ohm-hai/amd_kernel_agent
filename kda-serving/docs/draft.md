# Draft — Kimi K3 KDA layer serving optimization on MI355X

Recreation target: KDA² blog loop applied to Kimi K3's KDA layer (H=96, K=V=128 —
from `Primus/primus/configs/models/megatron/kimi_k3.yaml`; 69 of 93 layers are KDA).

## Baseline matrix (COMPLETE — `baseline_matrix.csv`, 50 rows, GPU 7, MI355X)

### Prefill (FLA chunk_kda, Triton, chunk 64)
- **~3.0–3.3M tokens/s, flat across T=1K→64K and B=1→8** (compute-bound; near-perfect
  B×T scaling). Peak 3.29M tok/s @ B=8 T=2K. No long-context degradation (T=64K B=1:
  2.96M tok/s, 22.1 ms).
- **Chunked prefill (8×8K with state carry) = 1.005× of monolithic 64K** — chunking
  is free; serving schedulers can chunk without perf concern.

### Decode (FLA fused_recurrent_kda, state carry)
- **Per-step latency is FLAT vs context length** — 0.048–0.050 ms/step at ctx
  1K/8K/32K (B=1..8). O(1) state confirmed empirically; the linear-attention
  advantage over MLA KV-cache reads.
- Batch scaling: B≤8 launch-bound at ~50 µs/step; B=16 68 µs (237K tok/s sweet
  spot); B=32/64/128 → 0.16/0.31/0.61 ms (~205–214K tok/s plateau, state
  bandwidth-bound).

## Environment findings (hardening the matrix itself)

1. **FLA Triton int32 overflow**: GPU memory access fault (reproduced twice) at
   B×T×H×K > 2³¹ elements (B=8, T=32K = 3.2e9). Matrix capped at 160K tokens/cell.
   Any candidate kernel must use 64-bit indexing or tile-local offsets.
2. **AITER has no KDA kernel** (v0.1.9: MLA/paged-attn/MoE only) — no AITER
   baseline; the aiter-JIT KDA kernel is a future candidate (blocked on the
   aiter↔composable_kernel JIT mismatch documented in kda-attn).
3. One outlier cell (ctx=8192 B=8: 0.095 ms vs 0.050 elsewhere) — treated as noise;
   rerun to confirm if it matters.

## Optimization targets (ranked by measured headroom)

1. **Decode, large B (≥32)**: per-step latency scales ~linearly with B — full state
   read+write dominates. Candidates: persistent-resident state (skip spill),
   fp8/bf16-state traffic reduction, fusing gate/beta into the step kernel,
   multi-step fused recurrent.
2. **Decode, small B (≤8)**: 50 µs launch-bound floor with a 3 MB state — CUDA
   graph capture or a multi-step fused kernel amortizes launches.
3. **Prefill**: 3.3M tok/s ≈ low effective TFLOP/s vs MI355X bf16 peak — targets:
   gfx950 mfma-tuned WY-representation GEMMs, fused gate-exp/L2N prologue-epilogue,
   larger BK/BV tiles.

## Next steps

1. Rerun ctx=8192 B=8 cell ×5 to confirm the outlier (1 min).
2. **ACTIVE — FlyDSL track** (see `runs/flydsl-decode*/`):
   - v1: correct (state rRMSE 3e-07), 0.10–0.31× of fla
   - v2: vec4 single-read + registers + CHUNK=32 + folded o-reduction →
     **0.22–0.50× of fla (~2× over v1)** ← FlyDSL best
   - v3: fused-o algebra + coalesced stores → REJECTED (0.30–0.31×, ~1.6×
     slower than v2; od dual-staging + store staging round-trip cost more
     than they saved — strided scalar stores were not the real bottleneck)
   - v5: shuffle_xor wavefront trees (2 barriers) — 18.7/56.5/717us
   - v6: 256-thread column-split, trees within wavefront, ZERO barriers —
     **9.1/33.2/496us = 1.49/1.72/1.60x of FLA** (gap was 2.6-3.3x in v4)
   - Conclusion so far: work distribution was the kernel gap; FlyDSL
     dispatch (~90us/call vs Triton ~38us) is the wall-clock gap at small B.
7. v7 DONE (user-approved A-then-B direction):
   - CUDA-graph bench mode (--graph): BOTH sides graphed. Wall clock:
     B=1 **1.06x — first cell win** (7us vs graphed-FLA 8us); B=2 0.86x;
     B=4-128 0.58-0.71x. Graphed-FLA B=128 = 5.1TB/s (at HBM limit).
   - hybrid layout REJECTED at design stage (345 vs 200 lane-ops)
   - masked-store attempt: GPU fault with scalar-predicate mask; lane-0
     vec4-insert store instead (perf-neutral)
   - kernel remains 1.45-1.70x of FLA across B
8. v8 plan (to reach 2x):
   a. bf16 state round-trip (user-approved Path B): halves BW floor at
      B>=32; combined with BW-efficiency tuning (store pattern) targets
      ~156us at B=128 (= 2x of graphed-FLA 311us; requires ~5.2TB/s on
      0.8GB — feasible only at FLA-level BW efficiency)
   b. small-B 2x needs kernel ~3us (from 8.6): MFMA-based reductions or
      accepting parity + winning on dispatch-free graph replay
   c. accuracy report for bf16 state mandatory (blog hack-E analog)
3. Promotion: >5% geomean over the decode grid (or prefill grid) with accuracy
   inside the baseline band.

## Commands

- Full matrix: `CUDA_VISIBLE_DEVICES=7 python3 bench/matrix.py`
- Decode only: `CUDA_VISIBLE_DEVICES=7 python3 bench/matrix.py --decode-only`
- Correctness gates: `CUDA_VISIBLE_DEVICES=7 python3 ../kda-attn/verifier/verify.py --candidate runs/<name>`