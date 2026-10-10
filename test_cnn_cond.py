"""Check for CNN conditioning (tiny models, CPU, ~10 s): python test_cnn_cond.py
init_from reproduces the old model exactly; the CNN stays frozen; the new channel learns;
the evaluation loader rebuilds the conditioned model."""
import os, sys, tempfile, types, torch
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here); sys.path.insert(0, os.path.join(_here, "baseline_cnn"))
os.chdir(tempfile.mkdtemp())
sys.modules.setdefault("wandb", None)
from train import build_model
from model import load_init_weights, EDMSchedule, training_loss
from model_cnn import DeterministicCNN

torch.manual_seed(0)
C, T_in, T_out, H = 2, 3, 2, 32
arch = dict(base_channels=16, channel_mults=[1, 2], num_res_blocks=1, attn_resolutions=[8],
            emb_dim=32, img_size=[H, H])
stats = {"ir": {"mean": 0.4, "std": 0.2}, "li": {"mean": 0.1, "std": 0.3, "transform": "cbrt"}}
ch = ["ir", "li"]
cnn = DeterministicCNN(C=C, T_in=T_in, T_out=T_out, dt_min=10, ctx_channels=None, binary_li_ctx=True,
                       dropout=0.0, **{**arch, "channel_mults": (1, 2), "attn_resolutions": (8,), "img_size": H})
torch.save({"model": cnn.state_dict(), "channels": ch, "stats": stats,
            "args": dict(T_in=T_in, T_out=T_out, dt_min=10, binary_li_ctx=True, ctx_channels=None, **arch)},
           "cnn.pt")
base = dict(T_in=T_in, ctx_channels=None, binary_li_ctx=True, dropout=0.0, sigma_data=0.66, channels=ch, **arch)
old = build_model(C, T_in, T_out, 10, types.SimpleNamespace(**base))
torch.save({"ema": old.state_dict()}, "old.pt")
new = build_model(C, T_in, T_out, 10, types.SimpleNamespace(**base, cnn_cond_checkpoint="cnn.pt"), stats)
load_init_weights(new, "old.pt")
old.eval(); new.eval()

B = 4
ctx = torch.randn(B, T_in, C + 1, H, H); x = torch.randn(B, C, H, H)
sig = torch.full((B,), 0.8); mask = torch.ones(B, C); lead = torch.tensor([0, 1, 0, 1])
with torch.no_grad():
    d = (old(x, sig, ctx, mask, lead) - new(x, sig, ctx, mask, lead)).abs().max().item()
print("init reproduces old model, max|diff| =", d); assert d < 1e-5

# the extra channel really is the CNN probability, and it changes the output once its weights are non-zero
w = new.precond.unet.input_conv.weight
assert w.shape[1] == old.precond.unet.input_conv.weight.shape[1] + 1 and w[:, -1].abs().sum() == 0
with torch.no_grad():
    w[:, -1] = 0.1
    assert (old(x, sig, ctx, mask, lead) - new(x, sig, ctx, mask, lead)).abs().max() > 1e-4

# one training step: gradients reach the UNet (incl. the new column) but never the frozen CNN
new.train(); assert not new.cnn.training
batch = {"context": ctx, "target": torch.randn(B, T_out, C, H, H), "tgt_mask": torch.ones(B, T_out, C),
         "last_ctx": torch.randn(B, C, H, H)}
loss = training_loss(new, EDMSchedule(sigma_data=0.66), batch, "cpu", cfg_drop_prob=0.0, li_weight=20,
                     asym_weight=0, nbr_weight=0, spectral_weight=0, channels=ch)
loss.backward()
assert all(p.grad is None for p in new.cnn.parameters())
assert w.grad is not None and w.grad[:, -1].abs().sum() > 0
print("train step ok, loss", round(loss.item(), 4), "- CNN frozen, new channel learns")

# evaluation loader rebuilds the conditioned model from a training-style checkpoint
from select_example_cases import load_model
args_d = {**base, "T_out": T_out, "dt_min": 10, "cnn_cond_checkpoint": "cnn.pt"}
torch.save({"ema": new.state_dict(), "args": args_d, "stats": stats, "channels": ch}, "new.pt")
m, *_ = load_model(types.SimpleNamespace(checkpoint="new.pt"), torch.device("cpu"))
with torch.no_grad():
    assert torch.allclose(m(x, sig, ctx, mask, lead), new.eval()(x, sig, ctx, mask, lead))
print("evaluation loader ok")
