"""TraceLens extension: KDA decode ops (category + bandwidth model).

Pass with `TraceLens_generate_perf_report_pytorch --extension_file <this file>
--gpu_arch_platform MI355X`. Ops come from `kda_ops.decode_op()` (names
`kda::decode_step_<impl>`), profiled with `record_shapes=True`.

Model for one decode step (T=1), per batch row b and head h, K = V = 128:
  bytes = state read + state write (K*V each, in the state dtype)
        + q, k, g reads (K each), v read (V), beta read (1), o write (V), all bf16
  flops = 7*K*V   (decay K*V, k.S 2KV, beta*k(outer)u + add 2KV, q.S 2KV; norms ignored)
Checked against rocprofv3 on MI355X at B=128, H=96, fp32 state: model 1626 MB vs counters
1658 MB (FLA) and 1675 MB (FlyDSL v7).
"""
_BYTES = {"c10::BFloat16": 2, "c10::Half": 2, "float": 4, "double": 8, "c10::Float8_e4m3fn": 1}


class kda_decode_step:
    category = "linear_attention"
    bwd_category = None

    def __init__(self, event, arch=None, **_):
        args = event["args"]
        dims, types = args["Input Dims"], args.get("Input type", [])
        q, v, state = dims[0], dims[2], dims[5]
        self.B, _, self.H, self.K = q
        self.V = v[3]
        self.state_bpe = _BYTES.get(types[5] if len(types) > 5 else "float", 4)
        self.io_bpe = _BYTES.get(types[0] if types else "c10::BFloat16", 2)
        self.param_details = {"B": self.B, "H": self.H, "K": self.K, "V": self.V,
                              "state_bytes_per_elem": self.state_bpe}

    def flops(self):
        return 7 * self.B * self.H * self.K * self.V

    def bytes(self):
        bh = self.B * self.H
        state = 2 * bh * self.K * self.V * self.state_bpe
        io = bh * (3 * self.K + 2 * self.V + 1) * self.io_bpe
        return state + io

    def flops_bwd(self):
        return 0

    def bytes_bwd(self):
        return 0

    def get_maf_type(self):
        return "vector"

    def get_compute_precision(self):
        return "fp32"


_NAMES = ["baseline_fla_decode", "flydsl_decode_v7", "flydsl_decode_v6",
          "emul_bf16_state", "emul_fp16_state"]
perf_model_extension = {f"kda::decode_step_{n}": kda_decode_step for n in _NAMES}
op_category_extension = {f"kda::decode_step_{n}": "linear_attention" for n in _NAMES}
