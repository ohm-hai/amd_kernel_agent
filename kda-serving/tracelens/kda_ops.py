"""Register KDA decode implementations as named torch ops for profiling.

TraceLens attributes kernels to the CPU-side op that launched them and reads that op's
input shapes from the trace. Raw FLA calls and FlyDSL launches have no such op (FlyDSL
shows up as an anonymous `hipModuleLaunchKernel`), so TraceLens files them under "other"
with no shapes. Wrapping each implementation as `kda::decode_step_<impl>` fixes that;
`kda_extension.py` then maps these names to a bandwidth model.

Instrumentation rule: any decode candidate that is profiled must be called through
`decode_op(name)` and profiled with `record_shapes=True`.
"""
import importlib.util
from pathlib import Path

import torch

TASK = Path(__file__).resolve().parent.parent
_OPS = {}


def _load(name):
    spec = importlib.util.spec_from_file_location(f"cand_{name}", TASK / "runs" / name / "candidate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decode_op(name):
    """Return a callable op `kda::decode_step_<name>`: (q, k, v, g, beta, state) -> (o, new_state)."""
    if name in _OPS:
        return _OPS[name]
    fwd = _load(name).get_forward()
    op_name = "kda::decode_step_" + name.replace("-", "_")

    @torch.library.custom_op(op_name, mutates_args=())
    def op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor,
           beta: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        res = fwd(q, k, v, g, beta, initial_state=state, output_final_state=True)
        o, new_state = res if isinstance(res, tuple) else (res, fwd.last_final_state)
        return o, new_state

    _OPS[name] = op
    return op
