#!/usr/bin/env python3
"""Generate a PyTorch profiler trace (blog Step 1) for TraceLens analysis (Step 2).

Runs a small Llama-style training loop on one GPU and exports a Chrome trace.
Usage: python gen_trace.py [gpu_id] [steps]
"""
import os
import sys

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
os.environ.setdefault("HSA_KERNARG_POOL_SIZE", "12582912")
torch.cuda.set_device(gpu_id)

# Small GPT-style model: embed -> transformer blocks -> LM head
D, V, L, H = 1024, 8192, 8, 8
blocks = nn.ModuleList(
    [
        nn.ModuleDict(
            {
                "attn": nn.MultiheadAttention(D, H, batch_first=True),
                "mlp": nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D)),
                "ln1": nn.LayerNorm(D),
                "ln2": nn.LayerNorm(D),
            }
        )
        for _ in range(L)
    ]
).cuda()
embed = nn.Embedding(V, D).cuda()
head = nn.Linear(D, V, bias=False).cuda()
params = list(blocks.parameters()) + list(embed.parameters()) + list(head.parameters())
opt = torch.optim.AdamW(params, lr=1e-4)

def step(tokens):
    x = embed(tokens)
    for b in blocks:
        a, _ = b["attn"](b["ln1"](x), b["ln1"](x), b["ln1"](x), need_weights=False)
        x = x + a
        x = x + b["mlp"](b["ln2"](x))
    return head(x)

tok = torch.randint(0, V, (16, 512), device="cuda")
for i in range(steps):
    if i == 2:  # profile steady-state steps
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        prof.start()
    logits = step(tok)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, V), tok.view(-1))
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    if i == 2:
        prof.stop()
        os.makedirs("/root/amd-workspace/traces", exist_ok=True)
        path = os.path.join("/root/amd-workspace/traces", "demo_train_trace.json")
        prof.export_chrome_trace(path)
        print(f"Trace exported: {path}")
print("done, final loss:", loss.item())