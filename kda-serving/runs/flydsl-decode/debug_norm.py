import sys
sys.path.insert(0, "/root/amd-workspace/kernel-lab/kda-serving/runs/flydsl-decode")
import torch
from candidate import decode_step, KK, VV, _SCALE

dev = "cuda"
H = 1
torch.manual_seed(1)
q = torch.randn(1, 1, H, KK, device=dev).bfloat16()
k = torch.randn(1, 1, H, KK, device=dev).bfloat16()
v = torch.zeros(1, 1, H, VV, device=dev).bfloat16()
g = torch.zeros(1, 1, H, KK, device=dev).bfloat16()
beta = torch.zeros(1, 1, H, device=dev).bfloat16()   # beta=0 -> S_out = S_in*exp(0) = S_in
S_in = torch.zeros(1, H, KK, VV, dtype=torch.float32, device=dev)
for i in range(VV):
    S_in[0, 0, i, i] = 1.0   # identity state -> o = scale * q_hat

o, S_out = decode_step(q, k, g, v, beta, S_in)
q_hat_kernel = (o[0, 0, 0] / _SCALE).float()
q_hat_true = (q[0, 0, 0].float() / q[0, 0, 0].float().norm())
print("kernel q_hat[:5]:", [round(x, 5) for x in q_hat_kernel[:5].tolist()])
print("true   q_hat[:5]:", [round(x, 5) for x in q_hat_true[:5].tolist()])
print("ratio (kernel/true) mean:", (q_hat_kernel / q_hat_true).mean().item(),
      "| kernel ||q_hat||:", q_hat_kernel.norm().item(), " (should be 1.0)")
print("S_out == S_in:", torch.allclose(S_out, S_in))
