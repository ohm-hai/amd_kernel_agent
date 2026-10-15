import sys
sys.path.insert(0, "/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode")
import torch
from candidate import decode_step, KK, VV, _SCALE
from fla.ops.kda import fused_recurrent_kda

dev = "cuda"
H = 4
torch.manual_seed(7)

q = torch.zeros(1, 1, H, KK, device=dev).bfloat16(); q[..., 0] = 1.0
k = q.clone()
v = torch.zeros(1, 1, H, VV, device=dev).bfloat16(); v[..., 0] = 1.0
g = torch.zeros(1, 1, H, KK, device=dev).bfloat16()
beta = torch.ones(1, 1, H, device=dev).bfloat16()
S0 = torch.zeros(1, H, KK, VV, dtype=torch.float32, device=dev)

o_my, S_my = decode_step(q, k, g, v, beta, S0)
kw = dict(use_qk_l2norm_in_kernel=True, output_final_state=True)
o_fl, S_fl = fused_recurrent_kda(q, k, v, g, beta, **kw)
print("identity: expect o[...,0]=scale=%.4f, S[0,0]=1" % _SCALE)
print("mine: o[0]=%.4f  S[0,0]=%.4f" % (o_my[0,0,0,0].item(), S_my[0,0,0,0].item()))
print("fla : o[0]=%.4f  S[0,0]=%.4f" % (o_fl[0,0,0,0].item(), S_fl[0,0,0,0].item()))

q = torch.randn(1, 1, H, KK, device=dev).bfloat16()
k = torch.randn(1, 1, H, KK, device=dev).bfloat16()
v = torch.randn(1, 1, H, VV, device=dev).bfloat16()
g = -torch.rand(1, 1, H, KK, device=dev).bfloat16().abs()
beta = torch.rand(1, 1, H, device=dev).bfloat16().sigmoid()
o_my, S_my = decode_step(q, k, g, v, beta, S0)
o_fl, S_fl = fused_recurrent_kda(q, k, v, g, beta, **kw)
d_o = (o_my.float() - o_fl.float()).abs()
d_S = (S_my - S_fl).abs()
print("random: o max abs diff %.5f | S max abs diff %.5f" % (d_o.max().item(), d_S.max().item()))
print("mine o[0,:4]:", [round(x, 4) for x in o_my[0, 0, 0, :4].tolist()])
print("fla  o[0,:4]:", [round(x, 4) for x in o_fl[0, 0, 0, :4].tolist()])
print("mine S[0,0,:4,0]:", [round(x, 4) for x in S_my[0, 0, :4, 0].tolist()])
print("fla  S[0,0,:4,0]:", [round(x, 4) for x in S_fl[0, 0, :4, 0].tolist()])
