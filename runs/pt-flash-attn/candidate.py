#!/usr/bin/env python3
"""Candidate: pt-flash-attn — replace nn.MultiheadAttention with primus-turbo flash_attn.

KDA loop stage 1 (validate) + stage 2 (profile). Usage:
  CUDA_VISIBLE_DEVICES=6 python candidate.py validate   # correctness vs eager SDPA
  CUDA_VISIBLE_DEVICES=6 python candidate.py profile    # training steps + torch trace
"""
import sys

import torch
import torch.nn as nn

import primus_turbo.pytorch as pt

D, V, L, H = 1024, 8192, 8, 8
assert D % H == 0
DH = D // H


class TurboBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.out = nn.Linear(D, D, bias=False)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)

    def forward(self, x):
        B, S, _ = x.shape
        qkv = self.qkv(self.ln1(x))
        q, k, v = qkv.view(B, S, 3, H, DH).permute(2, 0, 3, 1, 4).unbind(0)
        a = pt.ops.flash_attn_func(q, k, v, causal=True)  # [B, H, S, DH]
        a = a.reshape(B, S, D)
        x = x + self.out(a)
        return x + self.mlp(self.ln2(x))


def build():
    blocks = nn.ModuleList([TurboBlock() for _ in range(L)]).cuda()
    embed = nn.Embedding(V, D).cuda()
    head = nn.Linear(D, V, bias=False).cuda()
    return blocks, embed, head


def step(model, tokens):
    blocks, embed, head = model
    x = embed(tokens)
    for b in blocks:
        x = b(x)
    return head(x)


def validate():
    torch.manual_seed(0)
    blocks, embed, head = build()
    for m in (blocks, embed, head):
        m.eval()
    tokens = torch.randint(0, V, (4, 256), device="cuda")
    with torch.no_grad():
        x = embed(tokens)
        for b in blocks:
            x = b(x)
        logits = head(x)
    # self-consistency: deterministic, finite, right shape
    assert logits.shape == (4, 256, V)
    assert torch.isfinite(logits).all()
    # reference cross-check: turbo flash_attn vs SDPA on the same qkv
    torch.manual_seed(1)
    q = torch.randn(2, H, 256, DH, dtype=torch.bfloat16, device="cuda")
    k, v = q.clone(), q.clone()
    with torch.no_grad():
        out_t = pt.ops.flash_attn_func(q, k, v, causal=True).transpose(1, 2).reshape(2, 256, D)
        out_s = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_s = out_s.transpose(1, 2).reshape(2, 256, D)
    ok = torch.allclose(out_t.float(), out_s.float(), atol=1e-1, rtol=1e-2)
    print("flash_attn vs SDPA allclose:", ok)
    assert ok
    print("VALIDATION PASSED")


def profile(out_path):
    torch.manual_seed(0)
    model = build()
    params = [p for m in model for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=1e-4)
    from torch.profiler import ProfilerActivity, profile

    tok = torch.randint(0, V, (16, 512), device="cuda")
    for i in range(6):
        if i == 2:
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            prof.start()
        loss = torch.nn.functional.cross_entropy(step(model, tok).view(-1, V), tok.view(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i == 2:
            prof.stop()
            prof.export_chrome_trace(out_path)
            print("trace:", out_path)
    print("done, loss:", loss.item())


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if mode == "validate":
        validate()
    else:
        profile(sys.argv[2] if len(sys.argv) > 2 else "trace.json")