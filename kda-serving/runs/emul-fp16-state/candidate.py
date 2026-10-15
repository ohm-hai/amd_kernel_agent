"""Accuracy emulation of an fp16-state decode kernel (Path B), not a performance candidate.

Same as emul-bf16-state with fp16 storage: 10 mantissa bits instead of 7, but a maximum
finite value of 65504, so overflow is the risk to watch.
"""
import torch
from fla.ops.kda import fused_recurrent_kda

STATE_DTYPE = torch.float16


def get_forward():
    def fwd(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False, cu_seqlens=None):
        h0 = None if initial_state is None else initial_state.float()
        o, S = fused_recurrent_kda(q, k, v, g, beta, scale=scale, initial_state=h0,
                                   use_qk_l2norm_in_kernel=True, output_final_state=True)
        fwd.last_final_state = S.to(STATE_DTYPE)
        return o

    fwd.supports_final_state = True
    return fwd
