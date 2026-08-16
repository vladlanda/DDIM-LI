# Results

Status: DRAFTED (first pass). Numbers verified against FINDINGS.md at
time of writing -- re-check against that file if it changes before
submission. Citation keys in [brackets] refer to references.bib entries;
final numbering follows npj's numbered-reference convention once the
full manuscript's citation order is fixed (see WRITING_RULES.md).

---

## Model overview

We developed a probabilistic lightning nowcasting system based on the
elucidated diffusion model (EDM) framework [karras2022edm], trained on
infrared brightness temperature and lightning-imager observations from
the MTG-FCI geostationary instrument over four regions of Central
Africa. The model takes a 6-hour observation history (36 frames at
10-minute resolution) as context and generates a probabilistic forecast
of lightning occurrence over the following hour (six 10-minute steps),
sampled as an ensemble of 50 physically plausible realizations per
forecast (Fig. 1).

## The model substantially outperforms physical extrapolation baselines, with an advantage that grows with lead time

We compared the model's pixel-wise precision-recall AUC (PR-AUC) against
three physically-motivated baselines: persistence, and two variants of
optical-flow extrapolation (pySTEPS-style semi-Lagrangian advection
[pulkkinen2019pysteps]) driven respectively by the lightning-imager and
infrared channels. Across the six 10-minute lead-time steps tested
(+10 to +60 minutes), the model's PR-AUC ranged from 0.846 to 0.631,
compared with 0.783 to 0.391 for persistence -- the strongest of the
three physical baselines at every lead time except +60 minutes, where
the LI-derived optical-flow baseline narrowly exceeded it (0.392 versus
0.391). The model's margin over the best available physical baseline
grew from 8.1% at +10 minutes to 61.0% at +60 minutes (Fig. 2).

Using sequence-level paired bootstrap confidence intervals (1,000
resamples, resampling whole test sequences rather than individual
pixels to respect within-scene spatial correlation), every one of these
comparisons was statistically significant (95% CI excluding zero) at
every lead time tested, with the margin growing monotonically for all
three baselines (persistence: +0.063 to +0.241; LI-derived optical
flow: +0.075 to +0.239; IR-derived optical flow: +0.205 to +0.355;
Fig. 2).

## Lightning displacement at convective scale is not captured by any coherent motion field

[Placeholder -- draft after confirming exact E2 figure/data source and
best_shift methodology description; needs the positional_ceiling.csv
data description in Methods to reference correctly. Key numbers to
include once drafted: pixelwise vs. neighbourhood-relaxed vs.
best-global-shift PR-AUC recovery curve results from
diagnose_positional_ceiling.py, Fig. 3.]

## Example forecasts illustrate calibrated ensemble diversity

[Placeholder -- draft once real example cases are selected and
finalized via select_example_cases.py (Fig. 4). Needs actual case
descriptions (which storms, what makes each illustrative) that require
looking at the real generated PNGs, not available from this sandboxed
environment.]

## A directly-optimized deterministic classifier retains a narrow, lead-time-independent pointwise advantage, explained by information access rather than model class

To assess how the model compares against directly-optimized machine
learning baselines -- not just physical extrapolation -- we trained a
convolutional neural network (CNN) baseline, architecturally and in
training recipe matched to Metzl et al. [metzl2025physical], and a
LightGBM [ke2017lightgbm] baseline using hand-engineered features, in
the methodological spirit of Song et al. [song2023lightning]. Both
baselines share the CNN's or LightGBM's own directly pointwise-matched
training objective (binary cross-entropy or logistic loss,
respectively) with the pointwise verification metric used here.

The model significantly outperformed LightGBM at every lead time, with
a margin that grew from +0.018 at +10 minutes to +0.185 at +60 minutes
-- the same qualitative shape as the three physical baselines. Against
the CNN baseline, however, the model showed a small but statistically
significant pointwise disadvantage at every lead time (-0.021 to
-0.024), and unlike every other comparison, this margin remained
essentially constant across the full 50-minute range tested rather than
growing (Fig. 5b; see Discussion for interpretation of this
lead-time-independent shape).

This pointwise disadvantage against the CNN baseline was concentrated
at loose probability thresholds and did not persist under spatial
tolerance: at a strict, majority-consensus probability threshold
(p > 0.5), the model's Fractions Skill Score matched or exceeded the
CNN baseline's at spatial scales beyond approximately 130 km (Fig. 5a).
We further found that this pointwise gap was substantially, though not
entirely, attributable to the ensemble size used to construct the
model's per-pixel probability estimate: increasing the ensemble from
10 to 50 members closed roughly 60% of the original PR-AUC gap against
the CNN baseline, with a clearly diminishing-returns pattern (49% of
the gap closed increasing from 10 to 30 members, a further 21% of the
remainder closed from 30 to 50; Fig. 5b).

## The model provides calibrated, genuine forecast uncertainty

[Placeholder -- draft once the reliability diagram and CRPS/spread-skill
numbers from Figure 6 are finalized against real (not synthetic) data.
Key point to make here: this is the capability class the CNN/LightGBM
baselines structurally cannot be evaluated on, since they produce a
single deterministic output rather than a sample-based ensemble.]

---

## Notes for next drafting pass

- Two subsections above are explicit placeholders (displacement-
  incoherence and example-forecasts), pending real figure data this
  sandboxed environment cannot generate. Do not submit with placeholders
  present -- flagged clearly so this is impossible to miss.
- The uncertainty-quantification subsection is also a placeholder,
  pending confirmation of what calibration/CRPS output actually looks
  like on real (not synthetic) evaluation data.
- Once Figure 4's real cases are selected, the "Example forecasts"
  subsection needs actual descriptive prose about what's shown in each
  case (a success, an uncertain/multimodal case, and honestly a failure
  case per MANUSCRIPT_PLAN.md) -- this requires looking at the actual
  images, not just numbers.
- All citation keys used here ([karras2022edm], [pulkkinen2019pysteps],
  [metzl2025physical], [ke2017lightgbm], [song2023lightning]) are
  VERIFIED in citations_ledger.md -- confirmed before drafting, per
  WRITING_RULES.md rules 5/6.
