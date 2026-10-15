#!/usr/bin/env python3
"""Candidate: chunk32 — FLA chunk_kda with chunk_size=32 (vs default 64).

Tuning candidate: smaller intra-chunk GEMMs, more sequential steps.
Interface identical to baseline.
"""
import torch
from fla.ops.kda import chunk_kda


def get_forward():
    def fwd(q, k, v, g, beta, scale=None, cu_seqlens=None, output_final_state=False):
        kwargs = dict(output_final_state=output_final_state, use_qk_l2norm_in_kernel=True,
                      chunk_size=32)
        if cu_seqlens is not None:
            kwargs["cu_seqlens"] = cu_seqlens
        if scale is not None:
            kwargs["scale"] = scale
        o, S = chunk_kda(q, k, v, g, beta, **kwargs)
        fwd.last_final_state = S
        return o

    fwd.supports_final_state = True
    return fwd