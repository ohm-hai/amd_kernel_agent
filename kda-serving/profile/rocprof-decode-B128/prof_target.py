import importlib.util
import torch
from fla.ops.kda import fused_recurrent_kda

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

c7 = load("c7", "/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode-v7/candidate.py")
KK = VV = 128; H = 96; B = 128; dev = "cuda"
torch.manual_seed(0)
q = torch.randn(B, 1, H, KK, device=dev).bfloat16()
k = torch.randn(B, 1, H, KK, device=dev).bfloat16()
v = torch.randn(B, 1, H, VV, device=dev).bfloat16()
g = -torch.rand(B, 1, H, KK, device=dev).bfloat16().abs()
beta = torch.rand(B, 1, H, device=dev).bfloat16().sigmoid()
S = torch.zeros(B, H, KK, VV, dtype=torch.float32, device=dev)
launch = c7._compile_decode_kernel()
s_out = torch.empty_like(S); o = torch.empty(B, 1, H, VV, dtype=torch.bfloat16, device=dev)
st = torch.tensor([KK ** -0.5], dtype=torch.float32, device=dev)
qf, kf, gf = q.reshape(B*H, KK), k.reshape(B*H, KK), g.reshape(B*H, KK)
vf, bf = v.reshape(B*H, VV), beta.reshape(B*H)
for _ in range(5):
    fused_recurrent_kda(q, k, v, g, beta, initial_state=S, use_qk_l2norm_in_kernel=True, output_final_state=True)
    launch(qf, kf, gf, vf, bf, S, s_out, o, st, B*H*c7.NV, torch.cuda.current_stream())
torch.cuda.synchronize()
print("done")
