"""Accuracy emulation of a bf16-state decode kernel (Path B), not a performance candidate.

FLA computes each step in fp32; the state is rounded to bf16 between steps, which is
what a kernel that loads bf16 state, computes in fp32 and stores bf16 would do.
"""
import torch
from fla.ops.kda import fused_recurrent_kda

STATE_DTYPE = torch.bfloat16


def get_forward():
    def fwd(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False, cu_seqlens=None):
        h0 = None if initial_state is None else initial_state.float()
        o, S = fused_recurrent_kda(q, k, v, g, beta, scale=scale, initial_state=h0,
                                   use_qk_l2norm_in_kernel=True, output_final_state=True)
        fwd.last_final_state = S.to(STATE_DTYPE)
        return o

    fwd.supports_final_state = True
    return fwd
