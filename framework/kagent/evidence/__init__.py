"""Evidence engine (Phase 4): machine-readable performance artifacts.

    isa.py      IR+ISA dump analysis: ALU-to-DATA-shuffle ratio, occupancy,
                scheduler stalls; the issue-bound detector for wavefront-
                reduction kernels (v7 baseline: ratio ~1.4 -> issue-bound).
    pcs.py      rocprofv3 PC sampling: stall-reason aggregation + top sampled
                PCs mapped to the ISA dump.
    roofline.py bytes-model floors and efficiency vs baseline from results.csv.

Adapters (gfx950/rocprofv3 first) are pluggable so other archs/tools slot in.
All GPU work runs under `timeout` on a free GPU (framework constraints).
"""
from .isa import analyze_isa  # noqa: F401
from .pcs import run_pcs  # noqa: F401
from .roofline import roofline_report  # noqa: F401