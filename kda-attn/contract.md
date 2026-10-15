# Task Contract — KDAttn forward on MI355X (AMD recreation of KDA²)

Recreation of https://nvlabs.github.io/kda/blog/2026-09-27-kda-for-kda/ on AMD hardware.
Agent = KDA workflow (kernel-lab/AGENTS.md) × TraceLens evidence engine.

## Objective

Maximize geomean speedup of the **Kimi Delta Attention (KDAttn) forward pass** over the
FLA `chunk_kda` Triton baseline (the "FlashKDA" analog) on 1× AMD MI355X (gfx950),
while passing the hardened acceptance verifier.

## Baseline

`runs/baseline-fla/` — `fla.ops.kda.chunk_kda` (Triton, bf16), FLA 0.5.2.

## Inputs / outputs

- Legal inputs: `q,k [B,T,H,K]`, `v [B,T,HV,V]`, `g [B,T,HV,K] ≤ 0` (log-decay),
  `beta [B,T,HV] ∈ (0,1)`, optional `cu_seqlens` (varlen packed layout).
- Output: `o [B,T,HV,V]`, final state `S [B,HV,K,V]`.
- Dtype: bf16 in/out. Candidates may specialize by shape but must be correct on
  **every legal input** (no distribution/shape hard-coding — see blog hacks A/B).

## Correctness requirements (hardened, blog-style)

Ground truth: fp64 recurrent reference (`ref_fp64.py`), authoritative elementwise.

| Gate | Requirement |
|---|---|
| G1 spec | outputs finite (no NaN/Inf) on all legal inputs incl. stress probes |
| G2 salts | 3 fresh random seeds × gaussian + non-gaussian (uniform, laplace) inputs |
| G3 regimes | fixed T=8192; varlen mixed (6 seqs); varlen uniform (8×1024) |
| G4 stress | deep decay (~600 bits per 64 tokens), near-unity gates, β→0/→1, boundary lengths T∈{1,2,63,64,65,127,128,129} |
| G5 tolerance | strict element-wise: max abs ≤ 5e-2 on `o`; output rel-RMSE ≤ 8e-3; final-state rel-RMSE ≤ 1e-2 at 8k. No statistical (99.9%) gates. (Adaptation notes: (a) bf16 outputs are quantization-limited near zero → relative gate is scale-aware RMSE while the absolute gate stays strictly per-element; (b) 8e-3 calibrated on the baseline's own legit precision at the T=1 boundary — real hacks (skipped normalization, hard-coded constants) land at 10%+.) |
| G6 replay | verify twice with different salts; both must pass (anti input-caching) |

## Evaluation command

```
python bench/bench.py --candidate runs/<name>            # 6-workload geomean speedup
python bench/bench.py --candidate runs/<name> --fast     # single-shape fast feedback loop
python verifier/verify.py --candidate runs/<name>        # hardened acceptance
```

## Promotion criteria

Promote when: verifier PASS ∧ geomean speedup > 1.05× over current best.
Record per-workload speedups in `benchmark.csv`; lineage in `candidates.jsonl`.
Accuracy regressions (final-state rel-RMSE worse than baseline) require explicit note.

## Measurement hygiene (blog ablation lessons)

- `--fast` first (0.9 min-class feedback), full 6-workload set only for promotion.
- Progressive synthesis: nail one fixed shape, then generalize to varlen.
- Every "too good to be true" speedup must be re-checked under G4 stress probes
  (the blog's fake 3.28–3.74× all died here).