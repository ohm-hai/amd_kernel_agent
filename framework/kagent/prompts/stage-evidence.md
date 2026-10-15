# Stage packet — evidence

    kagent run --stages evidence --pack <pack> --candidate <dir>

Produces isa.json + pcs.json + roofline.json. Read them in this order:
1. `roofline.json`: efficiency % and the honest floor at the baseline's implied
   bandwidth. If you exceed the floor, the measurement is wrong — stop.
2. `isa.json`: math/data and issue/data ratios, occupancy estimate, s_nop and
   s_waitcnt counts. math/data < 8 with a slow kernel = issue-bound: change the
   per-lane layout, not the memory system.
3. `pcs.json`: top stall reason per guideline 04's table; top PC offsets to be
   matched against the ISA dump where possible.

The next version's draft must cite WHICH number it attacks and by how much.
No number, no plan (guideline 04's diagnose-before-plan rule).