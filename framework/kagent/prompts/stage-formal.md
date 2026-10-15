# Stage packet — formal verification

    kagent run --stages formal --pack <pack> --candidate <dir> [--dtype fp16]

Interpret the tri-state verdict (invariant 5):
- PROVED: same math modulo the exp/rsqrt abstraction, for the addressed
  elements, with full xor-butterfly coverage on every reduction chain. It does
  NOT mean bitwise-equal — precision stays the numeric verifier's job.
- COUNTEREXAMPLE: read the witness. `spec_over_cand` near a known factor
  (scale, beta, rsqrt(q^2), rsqrt(k^2), sqrt(128), 2) names your missing or
  duplicated term. Fix the kernel, not the spec.
- UNKNOWN with the blocking pattern: the kernel uses a construct outside the
  modeled scope. Do not treat UNKNOWN as pass; either restructure the kernel
  into scope or note it and rely on the numeric gates (and say which).

Load-set mismatches are index-map bugs (wrong base, wrong stride, wrong lane
mapping) — the message lists missing/extra (buffer, index) keys; map them back
to rows/cols before touching the math.