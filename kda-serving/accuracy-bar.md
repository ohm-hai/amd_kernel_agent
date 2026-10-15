# Pre-registered accuracy bar — KDA decode candidates

Registered 2026-10-09, before any candidate was run through `verifier/verify_decode.py`
and before any reduced-precision state variant existed. Changing a threshold needs the
user's explicit approval, recorded below with a date and reason. Results never move it.

Ground truth: the fp64 token-by-token recurrence (`../kda-attn/ref_fp64.py`), fed the
same bf16 inputs as the candidate. Candidates step one token at a time and carry their
own state, in whatever storage dtype they declare (`STATE_DTYPE`, default fp32).

## Thresholds

| Quantity | Bar | Applies to |
|---|---|---|
| per-step output `o`, relative RMSE vs fp64 | ≤ 8e-3 in every checkpoint window | all gates |
| per-step output `o`, max absolute error | ≤ 5e-2 | all gates |
| final state, relative RMSE vs fp64 | ≤ 1e-2 | all gates, including 8192-step drift |
| non-finite values anywhere | fail | all gates |
| candidate modifies the `initial_state` tensor it was given | fail, unless it declares `INPLACE = True` | all gates |

Rationale: these match the prefill verifier's bars (`../kda-attn/contract.md`, G5). The
1e-2 final-state bar after 8192 steps is the one that matters for reduced-precision
state. The KDA^2 blog rejected FlashKDA's bf16 state at 3.45% final-state error after
~8K tokens, while its own kernels reached 0.2–0.3%; a bar of 1% sits between the two.

## Inputs

`../kda-serving/inputs.py`, two gate modes:

- `synthetic`: g = -|N(0,1)|. Calibration on 2026-10-09: p50 74, p90 83, p99 91 bits of
  decay per 64 tokens.
- `kimi`: FLA's KDA gate parameterization, scaled so p99 = 599 bits per 64 tokens
  (p50 75, p90 313, p99.9 803), matching the ~600 bits the KDA^2 blog measured on real
  Kimi-Linear prefills. Calibrated, not captured from the model.

## Results (reported against the bar; they do not change it)

2026-10-09, GPU 4, full suite (15 gates), reports in `runs/<name>/verify_decode_report.json`:

| Candidate | Gates | Worst state rRMSE (gate) | State rRMSE after 8192 steps |
|---|---|---|---|
| baseline-fla-decode (fp32 state) | 15/15 | 8.8e-07 (near-unity gates) | 1.2e-07 |
| flydsl-decode-v7 (fp32 state) | 15/15 | 8.9e-07 (near-unity gates) | 1.3e-07 |
| emul-bf16-state | **13/15** | **1.6e-02** (near-unity gates) | 4.4e-03 |
| emul-fp16-state | 15/15 | 2.0e-03 (near-unity gates) | 5.6e-04 |

bf16 state fails the long-memory probes (near-unity gates 1.6%, zero-input decay from
a warm state 1.4%). Its 8192-step drift with realistic gates passes only because
fast-decaying channels forget old rounding error. fp16 state passes every gate with 5×
margin at the same 2 bytes per element, so it is the Path B storage format.

## Change log

- 2026-10-09: registered.
