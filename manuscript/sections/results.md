# Results

Status: NOT DRAFTED. Outline only, per MANUSCRIPT_PLAN.md Sections 1-2.

Structure follows the figure sequence (MANUSCRIPT_PLAN.md Section 2).
Each subsection below should be drafted once its figure is finalized,
not before -- prose and figure should be written together so numbers
in text match what's plotted exactly.

## Model overview (supports Figure 1)
Brief -- full architecture detail belongs in Methods. Just enough here
to orient the reader: EDM diffusion model, MTG-FCI input, ir+li
channels, Central Africa, T_in=36 (6h context) / T_out=6 (1h forecast,
10min steps), n_members=50 ensemble.

## Headline performance vs. physical baselines (supports Figure 2)
PR-AUC 0.846->0.631 (+10 to +60min) vs. persistence/pysteps_li/
pysteps_ir, all significant (FINDINGS.md A1/A2/B1), margin GROWING
with lead time (+8.1%->+61.6% vs persistence). Use FINAL n_members=50
numbers throughout -- do not mix in earlier exploratory (n=10/30)
figures anywhere in Results.

## Displacement is incoherent, not advective (supports Figure 3)
The E2 finding -- confirmed 3 independent ways (best_shift analysis,
re-validation on final model, both optical-flow baselines independently
underperforming persistence). This is the paper's central novel
physical claim -- state it plainly and let the evidence in Figure 3
carry the weight.

## Example nowcasts (supports Figure 4)
Qualitative walkthrough of 2-3 case studies -- what the ensemble
predicts, how individual members differ, where the probability map
concentrates uncertainty. Case selection TBD (needs real data access,
not available in this sandboxed environment -- see MANUSCRIPT_PLAN.md
Section 2, "still needed").

## Baseline-verification methodology (supports Figure 5)
The F4 story: CNN/LightGBM comparison, ensemble-size sensitivity,
FSS-vs-scale. State results plainly here; save the WHY (information-
access explanation, GenCast corroboration) for Discussion. Include the
CNN's uniquely flat margin vs. the other four baselines' growing
margins as an explicit, stated empirical pattern (not just implied by
the figure).

## Uncertainty quantification (supports Figure 6)
CRPS/reliability/spread-skill results, demonstrating genuine
calibration -- the capability CNN/LightGBM structurally cannot be
evaluated on. Sets up the Discussion's argument for what a diffusion
approach actually buys beyond raw pointwise scores.

## Notes
- All numbers cited in this section must trace to a specific
  FINDINGS.md entry or a specific evaluate.py/evaluate_cnn.py/
  evaluate_lightgbm.py output file -- no numbers from memory.
- Keep interpretation minimal here; Results states what was found,
  Discussion explains why it matters.
