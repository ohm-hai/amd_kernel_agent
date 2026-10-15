# Guideline 02 — Defining verification cases

Cases are the classifier for bugs. A case that cannot fail any plausible bug is
not a case. This document defines the five classes, the rules, and how formal
verification pairs with them.

## Case classes

| class | what it does | kills |
|---|---|---|
| `numeric` | sampled correctness vs the fp64 reference on realistic inputs, multiple salts | wrong math anywhere the samples reach |
| `structural` | forces one computation path (zero v, β extremes, odd shapes, single head) | index/branch/layout bugs the samples miss |
| `drift` | long horizon (8192 steps) where tiny biases compound | precision and state-layout choices |
| `adversarial` | distribution extremes (deep decay ≈ 600 bits/64 tokens, v ×32, near-unity gates) | clamping, skipped work, range hacks |
| `holdout` | same generators, hidden seeds; counts-only reporting | overfitting to the visible set, reward hacking |

## Rules (enforced by `kagent check cases`)

1. **Every case names the bug class it kills** (`kills`) and cites the incident
   that motivated it (`incident`, >= 40 chars). The post-mortem rule: every
   incident in a campaign adds a case before the next candidate.
2. **Tolerances are references into `bar.toml`** (`tolerance_ref`), never inline
   numbers. A case without a bar reference cannot be frozen or audited.
3. **Generators are seeded and reproducible.** `generator` + `params` must
   rebuild the exact inputs; anything random outside the seed is rejected.
4. **Holdout cases are hidden.** Their values never appear in agent-visible
   reports; the loop engine reports `n_passed/n_total` only.
5. **Coverage.** The visible set must cover numeric + structural + drift +
   adversarial before any promotion; at least one hidden holdout must exist.
6. **Cheap cases first.** Per-case cost matters because the loop runs the full
   set per candidate. Expensive gates (drift) may be marked `quick=False` but
   must run before any *promotion*, not before each draft.

## Real incidents these rules come from (kda-serving)

- v1–v7 were validated with 3-step carry on synthetic gates only; the real
  Kimi gate distribution decays ~600 bits per 64 tokens and the synthetic
  `-|N(0,1)|` reaches p99 91 → the `adversarial` deep-decay case and the
  `gates="kimi"` requirement exist because of that.
- bf16 state passed the early gates and failed long memory → the `drift` class
  and the state-rrmse bar entry.
- A verifier's tolerance was once loosened after seeing the baseline fail a
  boundary case → pre-registration (guideline 03) and the `hidden` flag.
- The reference itself had two bugs (scale before L2N, per-batch initial
  state) → cases must be able to fail the *reference*, not only candidates;
  the frozen evaluator runs the baseline through the same gates.

## Pairing with formal verification

- Formal checking (kagent formal) **subsumes** most structural bugs: index-map
  bijection, reduction coverage, dropped factors (scale·β, L2N), sign errors.
  When a formal verdict is PROVED, the structural cases for those bug classes
  become redundant but are still run (cheap insurance against checker UNKNOWNs).
- Formal checking **does not** cover: precision loss (bf16/fp16 rounding
  behavior), approximate-op error (rsqrt/exp are abstracted as uninterpreted
  functions), performance, or anything the spec does not model. Numeric,
  drift, and adversarial cases remain mandatory.
- A counterexample witness (b, h, i, j, lane) is a debugging gift: it minimizes
  the failing input. Record it in the candidate report.