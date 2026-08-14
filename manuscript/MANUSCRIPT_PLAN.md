# Manuscript Plan

Target: npj Climate and Atmospheric Science (primary) / AIES (secondary)
-- see PAPER_TODO.md for the full journal-target reasoning. This file
covers rules 1 and 2 from WRITING_RULES.md: structure and figures.
Status: planning only, no prose drafted yet.

---

## 1. Manuscript structure

Nature Portfolio convention (confirmed via npj's own author guidelines,
see WRITING_RULES.md): **Results before Methods**, numbered references,
no strict word limit but written concisely.

### Title (working)
Something establishing: probabilistic/diffusion, lightning nowcasting,
Central Africa, satellite. Not finalized -- draft once Results is
written and the strongest framing is clear. Avoid overclaiming "first"
anything without checking (Dai et al. 2025 already claims a "first"
for 4h convective nowcasting at planetary scale; ours is a different,
narrower claim -- lightning-specific, regional -- so needs care not to
overlap that language).

### Abstract (~200 words, per Nature-family convention)
Structure: context (lightning hazard + gap) -> approach (diffusion
model, MTG-FCI satellite, Central Africa) -> headline result (PR-AUC
margins, growing with lead time, over 3 physical baselines and
LightGBM) -> central novel finding (E2, incoherent displacement,
independently confirmed 3 ways) -> significance. Do NOT front-load the
CNN-baseline nuance (F4) in the abstract -- that's a Discussion-level
methods contribution, would read as hedging if it's in the first
paragraph a reader sees.

### Introduction
- Lightning as a hazard (aviation, ground ops, wildfire ignition) --
  needs citation, see citations_ledger.md's "still needed" list
- Central Africa: high lightning density, understudied relative to
  CONUS/Europe -- needs citation
- Prior work, roughly in this order: physical extrapolation limits ->
  deterministic ML (Cintineo et al. 2022, Song et al. 2023, Metzl et
  al. 2025) -> generative/probabilistic ML for related tasks (GenCast,
  Dai et al. 2025, precipitation-nowcasting diffusion literature)
- Gap statement: no diffusion-based lightning-specific nowcasting
  system with this level of baseline-verification rigor; open question
  of displacement coherence at convective scale
- **Explicit differentiation paragraph vs. Dai et al. 2025** (required,
  see FINDINGS.md F3b) -- 2-3 sentences, not a footnote
- Brief statement of what this paper does and its 2 headline
  contributions (performance + the E2 mechanistic finding), reserving
  the baseline-verification-methodology contribution (F4) for
  Discussion rather than listing it as a third headline claim here

### Results
Ordered by figure, see Section 2 below for the figure-to-subsection
mapping. Roughly: model overview -> headline performance vs. physical
baselines -> the incoherent-displacement finding -> qualitative
examples -> baseline-verification/CNN+LightGBM story -> uncertainty
quantification (calibration/CRPS/spread-skill).

### Discussion
- Interpret the incoherent-displacement finding: physical implications
  for why advection-based methods (persistence, optical flow) cannot
  close the gap to ML approaches at these lead times, connects to
  Metzl et al. 2025's scale argument (F3)
- Interpret the baseline-verification story (F4): why CNN/LightGBM
  diverge, the information-access explanation, corroborated by GenCast
- Limitations: single contiguous domain (4 adjacent regions, no tested
  cross-regime generalization yet -- honest, already flagged in
  PAPER_TODO.md); ensemble-size sensitivity, settled at n_members=50
  with documented diminishing returns, not exhaustively tested beyond
  that; single-lead ensemble cost not yet benchmarked against baseline
  latency for an operational claim
- Future work: aerosol data (briefly -- flag as a compelling regional
  follow-up given Central Africa's biomass-burning aerosol regime and
  Song et al.'s aerosol-informed precedent, WITHOUT claiming results we
  don't have -- see the earlier conversation where this was explicitly
  scoped OUT of the current paper); held-out-region generalization;
  operational deployment considerations
- Broader significance: early-warning implications for the region

### Methods
- Study region and data: Central Africa, 4 test_roots, MTG-FCI,
  ir + li channels, 4km/10min, T_in=36/T_out=6
- Preprocessing: normalization, cbrt transform for li, packed pipeline
- Model architecture: EDM/Karras 2022 framework, UNet w/ ResBlocks,
  lead-time conditioning
- Training: DDP/EMA/AMP, hyperparameters, epochs, n_members=50 and why
  (cross-ref F4)
- Baselines: persistence, 2x optical flow (pySTEPS-style), CNN (cite
  Metzl et al. 2025 for the matched training recipe, C6/C7 for the
  documented departures), LightGBM (cite Song et al. 2023 for the
  "spirit of" framing, C8 for the feature-provenance disclosure)
- Evaluation metrics: PR-AUC, CSI/POD/FAR, FSS (cite Roberts & Lean
  2008), CRPS (cite Gneiting & Raftery 2007), calibration
- Statistics: sequence-level paired bootstrap CI methodology

### End matter
Data availability, Code availability (this repo), References,
Acknowledgements, Author contributions, Competing interests,
Figure legends (if not inline).

---

## 2. Figures

### Main text (curated, ~6 -- Nature-family papers keep this tight;
push everything else to Extended Data/Supplementary)

**Figure 1 -- Study overview.**
(a) Map of Central Africa showing the 4 test_roots regions (need to
confirm exact boundaries/names against configs/evaluate.yaml).
(b) Model schematic: input context window -> EDM diffusion process ->
ensemble of sampled forecasts -> probability map. User says this
"already exists" -- **need to locate the actual file** before
assuming it's ready; if it's a conceptual sketch rather than a
generated figure, it may need rebuilding to match the journal's
figure-quality bar.

**Figure 2 -- Headline performance.**
PR-AUC vs. lead time, model vs. persistence/pysteps_li/pysteps_ir,
with 95% CI shaded bands, significance markers. Direct visualization of
FINDINGS.md A1/A2/B1. This is the paper's central quantitative claim --
gets the best-real-estate figure slot.

**Figure 3 -- Displacement is incoherent, not advective.**
The best_shift PR-AUC recovery curve (E2), possibly alongside the
four-way baseline comparison (A2) as noted in FINDINGS.md's own figure
suggestion for E2. This is the central novel physical finding --
deserves its own figure, not a panel buried in Figure 2.

**Figure 4 -- Example nowcasts (qualitative).**
NOT YET BUILT -- new, per this planning session. 2-3 representative
storm cases, each showing: input IR+LI context (a few frames from the
6h history), ground truth future LI activity, several individual
ensemble member samples (demonstrating genuine sample diversity/
physical plausibility), and the resulting ensemble probability map.
This is the figure a reader uses to understand what the model's output
actually looks like -- essential for a generative-model paper, and
currently the biggest gap in the existing figure set. Case selection
matters: pick cases that illustrate (i) a clear success, (ii) a
genuinely uncertain/multimodal case where ensemble spread is
informative, and possibly (iii) a failure case, for honesty.

**Figure 5 -- Why baseline comparison needs care (the F4 story).**
Likely 2-3 panels: FSS-vs-scale comparison (diffusion vs. CNN,
existing `diffusion_vs_cnn_fss_comparison.png` as a starting point) +
the ensemble-size sensitivity result (PR-AUC gap vs. n_members,
showing the diminishing-returns curve) + a compact summary of the
5-baseline bootstrap comparison highlighting the CNN's uniquely flat
margin vs. the others' growing margins. This is the paper's
methodological contribution (F4) -- needs its own well-designed figure,
not scattered supplementary plots, given how much analysis went into
it and how directly it connects to E2.

**Figure 6 -- Genuine uncertainty quantification.**
Calibration/reliability diagram + spread-skill relationship (data
already computed by evaluate.py -- crps_decomposition, spread_skill
functions exist, confirm outputs are saved/accessible). This is the
figure that makes concrete, visually, the Discussion's claim that
CNN/LightGBM cannot provide what a genuine ensemble can -- ties
directly to the paper's core motivation for using a diffusion model at
all.

### Extended Data / Supplementary (comprehensive, lower curation bar)

- Ensemble-size (n_members=10/30/50) full sensitivity table/figure
- Full 5-baseline bootstrap CI table (the complete version of Figure 2
  /5's summary)
- Error decomposition: positional vs. existence failure at low/high
  confidence (E1)
- CNN baseline's own diagnostic plots (precision-recall, skill curves,
  calibration) -- already generated by evaluate_cnn.py
- LightGBM baseline's own diagnostic plots -- already generated by
  evaluate_lightgbm.py
- Full FSS-vs-scale for every baseline individually, not just the
  CNN/diffusion comparison
- CNN/LightGBM training curves (loss vs. epoch), if illustrative of
  the training-recipe fixes documented in C6/C8
- Data pipeline schematic (packed memmap pipeline), if reviewers are
  expected to care about reproducibility infrastructure -- likely only
  worth including if Code Availability alone doesn't satisfy a
  methods-rigor reviewer

### Still needed before this list is final
- Confirm which figures "already exist" as usable files vs. need
  rebuilding -- user mentioned "results" and "diffusion model
  structure" already exist; locate the actual files and assess
  journal-readiness (resolution, styling consistency with the
  established white-theme convention) before assuming they're final.
- Case selection for Figure 4 needs real data access (pick actual
  storm events) -- can't be done from this sandboxed environment.

---

## 3. Open questions to resolve before drafting begins

- Author list and order (needed for Abstract/title page, not blocking
  Results/Methods drafting)
- Whether the aerosol future-work paragraph needs the exploratory
  correlation check discussed earlier (still optional, not committed)
- Whether Figure 1's architecture schematic needs to be rebuilt as a
  proper journal-quality figure or already qualifies
