# Methods

Status: NOT DRAFTED. Outline only, per MANUSCRIPT_PLAN.md Section 1.
Nature-family convention: Methods comes after Discussion, written for
reproducibility, typically compressed/smaller-type in the final layout.

## Planned subsections

### Study region and data
Central Africa, 4 test_roots regions (confirm exact names/boundaries
against configs/evaluate.yaml before drafting), MTG-FCI geostationary
satellite, ir + li channels, 4km / 10min resolution, T_in=36 (6h
context) / T_out=6 (1h forecast).

### Preprocessing
Normalization (z-score), cbrt transform for li (rationale: preserves
zero exactly, connects to the li thresholding convention used
throughout evaluation -- FINDINGS.md), packed memmap pipeline (built to
replace slow JPEG decoding, validated bit-identical to the original
loader -- see PAPER_TODO.md/git history for the validation detail if
a methods-rigor reviewer wants it).

### Model architecture
EDM/Karras et al. 2022 framework (cite karras2022edm), UNet with
ResBlocks, lead-time conditioning via a shared embedding (same
mechanism used consistently across the CNN baseline for comparability,
FINDINGS.md C7).

### Training
DDP across 2 GPUs, EMA, AMP, CosineAnnealingLR, hyperparameters,
790 epochs, regenerated/clean dataset (cross-ref the data-quality fix
in FINDINGS.md D-section if relevant to mention).

### Ensemble size
n_members=50, chosen via a documented diminishing-returns sweep
(10/30/50 -- FINDINGS.md F4). State this as a deliberate, tested choice,
not an arbitrary default -- this is itself a methodological contribution
worth stating plainly in Methods, not just Discussion.

### Baselines
- Persistence
- Optical flow x2 (pySTEPS-style semi-Lagrangian extrapolation, LI- and
  IR-derived) -- needs pySTEPS methodology citation (NOT YET VERIFIED,
  see citations_ledger.md)
- CNN: architecture/training recipe matched to Metzl et al. 2025 (cite
  metzl2025physical) where noted, with documented deliberate departures
  (context window, lead-time amortization -- FINDINGS.md C7) stated
  explicitly with their fairness rationale
- LightGBM: "in the spirit of" Song et al. 2023 (cite song2023lightning),
  feature provenance disclosed per FINDINGS.md C8 (own design, not a
  reproduction of their aerosol-informed feature set) -- state this
  explicitly in Methods, not just in supplementary documentation

### Evaluation metrics
PR-AUC, CSI/POD/FAR at multiple probability thresholds, FSS (cite
roberts2008fss) at multiple spatial scales, CRPS with reliability/
resolution/uncertainty decomposition (cite gneiting2007scoring),
calibration/reliability diagrams, spread-skill relationships.

### Statistical testing
Sequence-level (not pixel-level) paired bootstrap confidence intervals,
n_boot=1000, rationale: respects within-scene spatial correlation.

## Notes
- Every methodological choice with a "why" already documented in
  FINDINGS.md (especially the C-section) should carry that rationale
  into Methods explicitly -- reviewers respond better to stated
  reasoning than to silently-made choices they have to guess at
  (established pattern throughout this project, e.g. C7's "explaining
  the choice is a strength, not a defensive footnote").
