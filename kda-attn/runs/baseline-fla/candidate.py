#!/usr/bin/env python3
"""Candidate 0 (baseline): FLA chunk_kda Triton kernel — the 'FlashKDA' analog.

Interface: get_forward() -> fwd(q, k, v, g, beta, scale=None, cu_seqlens=None) -> o
"""
import torch
from fla.ops.kda import chunk_kda


def get_forward():
    def fwd(q, k, v, g, beta, scale=None, cu_seqlens=None, output_final_state=False):
        kwargs = dict(output_final_state=output_final_state, use_qk_l2norm_in_kernel=True)
        if cu_seqlens is not None:
            kwargs["cu_seqlens"] = cu_seqlens
        if scale is not None:
            kwargs["scale"] = scale
        o, S = chunk_kda(q, k, v, g, beta, **kwargs)
        fwd.last_final_state = S
        return o

    fwd.supports_final_state = True
    return fwd