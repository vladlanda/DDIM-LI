"""
Deterministic CNN baseline for lightning nowcasting.

Design rationale
-----------------
Literature check (Metzl et al. 2025, LightningCast/Cintineo 2022, and the
broader satellite-lightning-CNN subfield) shows a plain deterministic CNN
is the standard baseline alongside persistence/optical-flow -- our
baseline set was missing it. This is that baseline.

Deliberately REUSES model.py's UNet class (same encoder/decoder/attention
blocks as the diffusion model) rather than writing a new architecture from
scratch. This isolates the comparison to "generative diffusion process vs.
single deterministic forward pass", not "well-engineered architecture vs.
a hastily-built one" -- a legitimate concern reviewers have about baseline
fairness.

What's stripped relative to the diffusion model's EDMPrecond wrapper:
  - No noisy-residual input (nothing to denoise -- this predicts directly).
  - No EDM preconditioning (c_skip/c_out/c_in scaling) -- not meaningful
    for a non-diffusion model.
  - A CONSTANT dummy "sigma" is still fed to UNet's sigma_emb pathway
    (harmless: sigma_emb is one additive term of three combined via
    self.emb_proj -- see model.py UNet.forward -- so a constant input
    just becomes a fixed learnable bias-like contribution the network is
    free to use or ignore; this way UNet's forward() is used completely
    unmodified).
  - Real lead_time conditioning is KEPT (same lead_emb pathway as the
    diffusion model) -- one model amortized across all lead times via
    lead-time conditioning, same design as our diffusion model, UNLIKE
    LightningCast/Metzl et al. who train a separate model per lead time.
    This is a deliberate methods choice for architectural consistency
    with our main model, worth stating explicitly in the manuscript.

Output: single-channel LI-occurrence RAW LOGITS (not sigmoid), matching
the literature convention of framing this as binary segmentation
(LightningCast, Metzl et al. BNN/AINN), while keeping training numerically
stable via BCE-with-logits (see DeterministicCNN.forward docstring for
why sigmoid-then-BCE is unsafe). Callers needing a probability (evaluation,
PR-AUC, calibration) apply torch.sigmoid() explicitly on the output.
"""
import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import UNet


def compute_in_ch_deterministic(C: int, T_in: int, ctx_channels, binary_li_ctx: bool) -> int:
    """
    Input channels for the deterministic CNN: context (T_in * C_ctx) +
    a single-channel mask (for the ONE output channel, LI), broadcast
    spatially -- NOT the diffusion model's C + T_in*C_ctx + C (no noisy-x
    term, mask sized for 1 output channel instead of C).
    """
    C_ctx_sel = len(ctx_channels) if ctx_channels else C
    li_in_ctx = (ctx_channels is None) or ("li" in ctx_channels)
    C_ctx = C_ctx_sel + 1 if (binary_li_ctx and li_in_ctx) else C_ctx_sel
    return T_in * C_ctx + 1   # +1 for the single-channel (LI) mask


class DeterministicCNN(nn.Module):
    """
    Deterministic multi-lead-time CNN wrapping model.py's UNet.

    forward(context, lead_idx) -> (B, 1, H, W) sigmoid probability map
    for LI occurrence at the given lead step.
    """

    def __init__(self, C: int, T_in: int, T_out: int, dt_min: int,
                ctx_channels, binary_li_ctx: bool, **unet_kwargs):
        super().__init__()
        self.C = C
        self.T_in = T_in
        self.T_out = T_out
        self.dt_min = dt_min
        in_ch = compute_in_ch_deterministic(C, T_in, ctx_channels, binary_li_ctx)
        self.unet = UNet(in_channels=in_ch, out_channels=1, **unet_kwargs)
        # Constant dummy sigma -- see module docstring. Registered as a
        # buffer (not a parameter) so it's fixed, not learned, and moves
        # correctly with .to(device).
        self.register_buffer("_dummy_sigma_val", torch.tensor(1.0))

    def forward(self, context: torch.Tensor, lead_idx: torch.Tensor) -> torch.Tensor:
        """
        context:  (B, T_in, C_ctx, H, W) -- same layout as the diffusion
                  model's context tensor (includes the binary-LI-presence
                  channel per frame if binary_li_ctx=True).
        lead_idx: (B,) long tensor, 0-indexed lead step.
        Returns:  (B, 1, H, W) RAW LOGITS (not sigmoid-transformed).

        Deliberately returns logits, not probabilities: training must use
        F.binary_cross_entropy_with_logits on the raw logits, not sigmoid()
        followed by F.binary_cross_entropy. The latter is numerically
        unstable -- verified empirically that at logit magnitudes entirely
        plausible mid-training (this task's ~5-6% positive rate quickly
        pushes the network toward confident negative predictions), the
        gradient through sigmoid()+binary_cross_entropy can be ~1e-10x the
        correct value, or exactly zero -- silent vanishing-gradient
        paralysis on confidently-wrong examples, not a crash, which would
        have been far harder to diagnose after a full training run than to
        catch here. Callers needing an actual probability (evaluation,
        PR-AUC, calibration) must apply torch.sigmoid() explicitly.
        """
        B, T_in, C_ctx, H, W = context.shape
        ctx_flat = context.reshape(B, T_in * C_ctx, H, W)

        # Trivial mask: LI is a REQUIRED channel (never missing in the
        # clean/regenerated dataset per this project's degeneracy audits),
        # so the mask is always 1. Kept as an explicit tensor (not just
        # omitted) so the architecture/input-construction stays structurally
        # analogous to the diffusion model's masking pathway.
        ch_mask = torch.ones(B, 1, device=context.device, dtype=context.dtype)
        mask_spatial = ch_mask[:, :, None, None].expand(B, 1, H, W)

        net_input = torch.cat([ctx_flat, mask_spatial], dim=1)

        sigma = self._dummy_sigma_val.expand(B)
        lead_time = (lead_idx.float() + 1) * self.dt_min

        return self.unet(net_input, sigma, lead_time, ch_mask)   # raw logits
