"""Check of the ablation transforms: python baseline_cnn/test_ablation_cnn.py"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablation_cnn import ablate, IR, LI

B, T = 4, 5
x = torch.arange(B * T * 3 * 2 * 2, dtype=torch.float).reshape(B, T, 3, 2, 2)
s = ablate(x, "ir_swap"); assert torch.equal(s[:, :, IR], x[[2, 3, 0, 1], :, IR]) and torch.equal(s[:, :, LI:], x[:, :, LI:])
s = ablate(x, "li_swap"); assert torch.equal(s[:, :, LI:], x[[2, 3, 0, 1], :, LI:]) and torch.equal(s[:, :, IR], x[:, :, IR])
s = ablate(x, "ir_frozen"); assert all(torch.equal(s[:, k, IR], x[:, -1, IR]) for k in range(T)) and torch.equal(s[:, :, LI:], x[:, :, LI:])
s = ablate(x, "li_frozen"); assert all(torch.equal(s[:, k, LI:], x[:, -1, LI:]) for k in range(T)) and torch.equal(s[:, :, IR], x[:, :, IR])
assert torch.equal(ablate(x, "none"), x)
print("ok")
