# Methods

Status: DRAFTED (first pass). Nature-family convention: Methods follows
Discussion, written for reproducibility. Citation keys in [brackets]
refer to references.bib; all VERIFIED per citations_ledger.md before use.

---

## Study region and data

The study region comprises four test regions within Central Africa
(see Fig. 1a for exact boundaries). [Placeholder: MTG-FCI instrument
citation not yet verified -- see citations_ledger.md "not yet searched"
list. Do not finalize this paragraph until that citation is resolved.]
Input data are drawn from the MTG-FCI geostationary satellite
instrument at 4 km spatial resolution and 10-minute temporal
resolution, using two channels: an infrared window channel (ir) and a
lightning-imager-derived channel (li). The model takes a 6-hour context
window (T_in = 36 frames) and forecasts a 1-hour horizon (T_out = 6
frames at 10-minute steps).

## Preprocessing

Both channels are normalized (z-scored) prior to input. The li channel
additionally undergoes a cube-root transform prior to normalization,
chosen because the transform preserves exact zero values -- physically
meaningful for a channel that is either exactly zero (no lightning
activity) or continuously nonzero -- which matters for the thresholding
convention used consistently throughout evaluation (ground-truth
binarization is always performed in physical space, after inverting
both the z-score and the cube-root transform, using a strict
greater-than-zero threshold for the raw physical field and a small
fixed threshold, 5/255, for probability-style outputs derived from
learned models).

## Model architecture

The model follows the elucidated diffusion model (EDM) framework of
Karras et al. [karras2022edm], implemented as a UNet built from residual
blocks, conditioned on the observation context and on lead time via a
shared embedding pathway. [Placeholder: confirm exact
base_channels/channel_mults/num_res_blocks/attn_resolutions values
against the actual training configuration before finalizing -- these
should be pulled directly from configs/default.yaml or the final
checkpoint's saved args, not from memory.]

## Training

The model was trained with distributed data parallelism across two
GPUs, using exponential moving average (EMA) weights for inference,
automatic mixed precision, and a cosine annealing learning rate
schedule, for 790 epochs on a regenerated, quality-controlled version
of the training dataset (see FINDINGS.md D-section for the data-quality
issue this regeneration addressed).

## Ensemble size

Forecasts are generated as an ensemble of sampled realizations from the
diffusion process. We used 50 ensemble members for all final reported
results, a setting chosen via a controlled sensitivity sweep (10, 30,
and 50 members) that showed a clear diminishing-returns pattern: tripling
the ensemble from 10 to 30 members closed roughly half of an initially
large verification-metric gap against a directly-optimized deterministic
baseline (see below), while a further increase to 50 members closed only
a further fifth of the remainder. We attribute the closed portion of
this gap to a finite-sample bias in the empirical, ensemble-derived
probability estimate at loose classification thresholds, and confirmed
this mechanism directly by comparing each model's predicted-positive
area fraction against the true event area fraction at matched
thresholds (see Discussion).

## Baselines

**Persistence.** The most recent observed lightning field is used
unchanged as the forecast at every lead time.

**Optical flow.** Two pySTEPS-style [pulkkinen2019pysteps]
semi-Lagrangian extrapolation baselines, driven respectively by
lightning-imager-derived and infrared-derived motion fields.

**Convolutional neural network (CNN).** A deterministic UNet-style
classifier trained directly on next-frame lightning occurrence via
per-pixel binary cross-entropy, following the general architecture and
training-recipe choices of Metzl et al. [metzl2025physical] (weight
decay and learning-rate schedule matched directly to their reported
values). Two departures from their exact protocol were made
deliberately: the CNN baseline shares the diffusion model's own 6-hour
context window and single-model lead-time amortization (rather than
Metzl et al.'s ~30-minute context and separately-trained per-lead-time
models), to hold input information and model structure constant across
our own internal comparisons -- the more rigorous choice for isolating
model architecture as the controlled variable in this study specifically,
even though it diverges from their literal setup.

**LightGBM.** A gradient-boosted-tree classifier [ke2017lightgbm]
trained on 18 hand-engineered features derived from the same ir/li
channels (temporal window statistics, trend/finite-difference features,
local spatial neighbourhood aggregation, and event-recency features),
in the methodological spirit of Song et al. [song2023lightning] --
not a literal reproduction of their protocol, which uses a different
input resolution and feature set (including aerosol and reanalysis
inputs not available to this study). Class imbalance was addressed via
LightGBM's native class-weighting mechanism rather than the manual
resampling used for the CNN baseline, reflecting standard practice for
each respective model class rather than a forced consistency between
them.

## Evaluation metrics

Skill was assessed via pixel-wise precision-recall AUC (PR-AUC),
critical success index / probability of detection / false alarm ratio
at multiple probability thresholds, the Fractions Skill Score (FSS)
[roberts2008fss] at multiple spatial neighbourhood scales (12-260 km),
and the Continuous Ranked Probability Score (CRPS) with its
reliability/resolution/uncertainty decomposition
[gneiting2007scoring], together with reliability diagrams and
spread-skill relationships to assess ensemble calibration directly.

## Statistical testing

All comparisons between the model and each baseline used sequence-level
(not pixel-level) paired bootstrap confidence intervals, resampling
whole test sequences with replacement (1,000 resamples) to respect
within-scene spatial correlation in the data, following [Placeholder:
cite the bootstrap methodology source if one beyond standard practice
is being used -- check whether this needs its own citation or is
standard enough to state without one].

---

## Notes for next drafting pass

- Two explicit placeholders remain: the MTG-FCI/MSG instrument citation
  (Introduction/Methods data section) and confirmation of the exact
  architecture hyperparameters against the real training config.
- Consider whether the bootstrap methodology needs a citation (e.g. a
  standard reference on block/cluster bootstrap for correlated spatial
  data) or whether describing the procedure plainly is sufficient for
  npj's methods conventions -- not yet decided.
- All citation keys used here ([karras2022edm], [pulkkinen2019pysteps],
  [metzl2025physical], [ke2017lightgbm], [song2023lightning],
  [roberts2008fss], [gneiting2007scoring]) are VERIFIED in
  citations_ledger.md before use, per WRITING_RULES.md rules 5/6.
