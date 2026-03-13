# METSAT Lightning Nowcasting — EDM Diffusion Model

6-hour probabilistic lightning and cloud-temperature nowcasting
using EDM-style diffusion (Karras et al. 2022) + GenCast principles.

---

## Repository structure

```
lightning_nowcast/
├── dataset.py      # METSATDataset, normalisation, DataLoader factory
├── model.py        # U-Net, EDMPrecond, MultiStepDenoiser, losses
├── train.py        # Training loop with EMA, AMP, CFG, WandB
├── evaluate.py     # CRPS, CSI, spread-skill, ensemble generation
├── infer.py        # Inference on new data → .npz ensemble + plots
└── README.md
```

---

## Data format expected

```
datasets/
├── africa_train/
│   ├── central_africa_1/
│   │   ├── {id}_{start}_{end}_ir.jpg    ← required
│   │   ├── {id}_{start}_{end}_li.jpg    ← required
│   │   ├── {id}_{start}_{end}_ch1.jpg   ← optional (BT 123)
│   │   ├── {id}_{start}_{end}_ch2.jpg   ← optional (BT 87)
│   │   └── {id}_{start}_{end}.wld       ← georeferencing
│   └── central_africa_2/
│       └── ...
└── africa_test/
    └── ...
```

Timestamps in filenames follow `%Y%m%dT%H%M%SZ` format.
Frames with exactly 10-minute spacing are grouped into sequences.

---

## Quickstart

### 1. Install dependencies
```bash
pip install torch torchvision einops pillow numpy scipy
pip install wandb          # optional logging
pip install scikit-image   # optional SSIM metric
```

### 2. Train
```bash
python train.py \
  --train_roots datasets/africa_train/central_africa_1 \
                datasets/africa_train/central_africa_2 \
                datasets/africa_train/central_africa_3 \
  --val_roots   datasets/africa_test \
  --channels ir li ch1 ch2 \
  --T_in 6 --T_out 36 \
  --batch_size 4 --epochs 200 \
  --output_dir outputs/run1 \
  --wandb_project metsat_nowcast
```

### 3. Inference
```bash
python infer.py \
  --checkpoint outputs/run1/best.pt \
  --context_dir datasets/africa_test/ \
  --n_members 20 --cfg_scale 1.5 \
  --output_dir outputs/forecasts --plot
```

---

## Architecture

### Why these choices?

#### 1. Direct multi-step prediction (not autoregressive rollout)
The model predicts all 36 future residuals **independently**, each conditioned
on the same input context but with a **different lead-time embedding**.
This completely eliminates autoregressive error accumulation — the single
biggest cause of blurring at long horizons.

Each lead step gets: `lead_time = (step + 1) × dt_min` (minutes) fed as a
sinusoidal embedding into every residual block via AdaGroupNorm. The network
learns that at +6h it should produce higher-variance, more uncertain outputs.

#### 2. Residual prediction with lead-time normalisation
We predict `y_t - y_{t-1}` rather than `y_t`. At short leads residuals are
small and tight; at long leads they grow. To keep the target distribution
roughly stationary across lead times, stats are computed per-channel
globally (not per-lead), and the EDM loss weight `λ(σ)` handles the rest.

#### 3. Classifier-free guidance (CFG)
During training, the context is randomly zeroed with probability `p=0.15`.
At inference, we compute:
```
D = D_uncond + cfg_scale × (D_cond - D_uncond)
```
with `cfg_scale=1.5`. This pushes each sample toward the conditional mode,
making individual ensemble members **sharp** rather than mean-blurred.

#### 4. Sparse LI channel treatment
The lightning index is cube-root transformed before normalisation, which
compresses the extreme dynamic range. LI also receives `3× loss weighting`
in the channel-weighted MSE. Without this, the model ignores LI (trivially
predicts zero and is right 99% of the time).

#### 5. Spectral (FFT) auxiliary loss
A loss on the magnitude spectrum of predictions vs targets directly penalises
loss of high-frequency structure. Weight 0.1 by default.

#### 6. U-Net with AdaGroupNorm conditioning
Every residual block receives the combined embedding:
```
emb = MLP(cat[σ_emb, lead_time_emb, ch_mask_emb])
```
via AdaGroupNorm: `GN(x) × (1 + scale(emb)) + shift(emb)`.
This is strictly better than the common "add embedding to feature map"
approach because it can modulate both mean and variance.

---

## Evaluation metrics

| Metric | Description |
|--------|-------------|
| CRPS | Continuous Ranked Probability Score — lower is better |
| Energy Score | Multivariate CRPS — measures ensemble calibration |
| CSI | Critical Success Index for lightning detection events |
| POD / FAR | Lightning probability of detection / false alarm rate |
| Spread-Skill | Ensemble spread / RMSE — ideal ≈ 1.0 |
| SSIM | Structural similarity — proxy for visual sharpness |

All metrics are computed **per lead-time step** so you can plot
skill degradation curves (the key diagnostic for blurriness problems).

---

## Key hyperparameters to tune

| Parameter | Default | Notes |
|-----------|---------|-------|
| `cfg_scale` | 1.5 | ↑ = sharper but less calibrated. Try 1.2–2.0 |
| `li_weight` | 3.0 | ↑ = more focus on lightning. Try 2–5 |
| `spectral_weight` | 0.1 | ↑ = sharper high-freq. Try 0.05–0.2 |
| `sigma_max` | 80.0 | May need ↑ if data variance is high |
| `n_members` | 10 (train) / 20 (infer) | ↑ = better CRPS, slower |
| `num_steps` | 20 | EDM denoising steps at inference. ↑ = better quality |
| `T_in` | 6 | 6 × 10min = 1h context |
| `T_out` | 36 | 36 × 10min = 6h forecast |

---

## Anti-blur checklist

If predictions are still blurry, check in this order:

1. **Is CSI near zero?** → Increase `li_weight` and `spectral_weight`
2. **Is spread/skill << 1?** → Model is overconfident; increase `sigma_max`
3. **Is spread/skill >> 1?** → Model is underconfident; decrease `cfg_scale`
4. **Does blur grow monotonically with lead time?** → Expected, but if severe,
   try adding per-lead-step normalisation of residual targets
5. **Are single ensemble members blurry (not just the mean)?** → The diffusion
   process is collapsing; increase `num_steps` at inference or lower `sigma_min`
6. **Does context give no information at 6h?** → Normal! Evaluate with CRPS
   against climatology baseline, not against perfect determinism
