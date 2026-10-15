"""Decode baseline: FLA fused_recurrent_kda (Triton), fp32 state, one token per call."""
import torch
from fla.ops.kda import fused_recurrent_kda

STATE_DTYPE = torch.float32


def get_forward():
    def fwd(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False, cu_seqlens=None):
        o, S = fused_recurrent_kda(q, k, v, g, beta, scale=scale, initial_state=initial_state,
                                   use_qk_l2norm_in_kernel=True, output_final_state=output_final_state)
        fwd.last_final_state = S
        return o

    fwd.supports_final_state = True
    return fwd
