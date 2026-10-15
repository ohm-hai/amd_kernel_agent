# Planted-bug regression pack

Each variant is v7's candidate with one targeted mutation. The checker must
produce the expected verdict (see each variant's expected.json). Run:

    python3 -m pytest framework/kagent/formal/bugs/test_formal.py -x -q

Bug classes and the campaign incidents they encode (guidelines 02):

| variant | mutation | incident it encodes | expected |
|---|---|---|---|
| bug_00_correct | none | — | PROVED |
| bug_01_no_scale_beta | drop scale from sb_qk | v3 missing scale*beta | COUNTEREXAMPLE, factor hint SCALE |
| bug_02_wrong_shuffle | first xor offset 32 -> 16 | v4 L2N/tree indexing | COUNTEREXAMPLE (butterfly) |
| bug_03_no_l2n | remove q/k normalization | v1 missing L2N | COUNTEREXAMPLE |
| bug_04_off_by_one | state row base shifted by one row | v4 base+tid indexing | COUNTEREXAMPLE |

Variants are generated from `runs/flydsl-decode-v7/candidate.py` by
`gen_bugs.py` (checked in output) so they track the real kernel.