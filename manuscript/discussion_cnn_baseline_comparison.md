# Discussion draft: why raw pointwise metrics favored the CNN baseline

Status: first draft, unpolished, ready for the user to cut/edit/merge into
the actual manuscript. Numbers below are from FINDINGS.md F4 (final
n_members=50 comparison) — re-verify against whatever n_members setting
ends up being used for the final reported headline numbers before this
goes in the paper, since the exact figures here are from the n=10/30/50
sweep, not necessarily the final chosen setting.

Suggested placement: Discussion section, likely adjacent to (or directly
citing) the incoherent-displacement finding (E2 in FINDINGS.md — probably
your Section 4.X on motion-field analysis). Could also partly live in
Methods (the verification-metric caveat) with the interpretation in
Discussion — split as fits your section structure.

Suggested supporting figures/tables (already generated, in
baseline_cnn/ and repo root):
  - baseline_cnn_fss_vs_scale.png / diffusion_vs_cnn_fss_comparison.png
    -> a version of this at n_members=50 would be the actual paper figure
  - a small table: PR-AUC gap vs. n_members (10/30/50), showing the
    diminishing-returns curve
  - a small table: area-fraction ratio (predicted/true) at each
    probability threshold, for both models

---

## Draft prose

When benchmarked against a deterministic convolutional baseline trained
directly on next-frame lightning occurrence (following the architecture
and general training approach of Metzl et al., 2025), the diffusion
model initially appeared to *underperform* on standard pointwise
verification metrics (precision–recall AUC, critical success index) at
every lead time — a result that, taken at face value, would be difficult
to reconcile with the model's clear advantage over physically-motivated
baselines (persistence, optical-flow extrapolation; see Section X). We
investigated this discrepancy directly rather than reporting either
number without explanation, since we consider it diagnostically
informative about what pointwise verification metrics do and do not
capture for probabilistic, sample-based forecasts.

Two independent lines of evidence pointed to a common, mechanistically
coherent explanation, involving both a genuine (if modest) methodological
artifact and a real, structural difference between the two models'
training objectives.

**Spatial-tolerance sensitivity.** We recomputed the Fractions Skill
Score (FSS; Roberts & Lean, 2008) across a range of neighbourhood scales
(12–260 km) rather than relying on pixel-level (scale = 0) scoring alone.
The CNN baseline's advantage was not uniform across this range: at a
strict decision threshold (probability > 0.5, i.e. ensemble-majority
agreement), the diffusion model's relative performance improved steadily
with spatial tolerance, overtaking the CNN baseline beyond roughly
130 km. At a loose threshold (probability > 0.1), by contrast, the
diffusion model's disadvantage was essentially constant across the same
scale range. A deficit that shrinks with spatial tolerance is the
signature of a *positional* (double-penalty) effect — a forecast that is
approximately, but not exactly, right in space is penalized twice under
strict pixel matching and rewarded once tolerance is introduced. A
deficit that does *not* shrink with spatial tolerance is not primarily
positional, and points instead toward a difference in predicted event
*coverage* (how much area is flagged as positive) rather than *location*.

**Ensemble-size sensitivity.** The diffusion model's probability field is
constructed empirically, as the fraction of an ensemble of independently
sampled forecasts exceeding a physical event threshold. With a modest
ensemble size, this fraction is coarsely quantized (e.g. with 10 members,
only eleven distinct probability values are possible), and — more
importantly — a loose decision threshold such as "probability > 0.1"
corresponds to a permissive criterion ("at least one member exceeded the
event threshold"), which is expected to inflate the predicted event area
independently of true forecast skill. We tested this directly by
comparing predicted-positive-area fraction against the true event area
fraction at each threshold, and by sweeping ensemble size (10, 30, 50
members) while holding the model fixed. Both diagnostics supported the
mechanism: the diffusion model's area-coverage ratio at the loosest
threshold was disproportionately inflated relative to the CNN baseline's
(2.6× vs. 2.0× the true event area at 10 members), and simply increasing
ensemble size — with no change to the trained model — closed roughly half
of the pointwise PR-AUC gap when tripling the ensemble from 10 to 30
members, with clearly diminishing returns on a further increase to 50
(closing only a further ~20% of the remaining gap). This is consistent
with a finite-sample bias in the empirical probability estimate that
shrinks, but does not vanish, as ensemble size grows — a property of the
*verification procedure*, not of the underlying model.

**A residual, structural difference remains, and it is informative.**
Even at the largest ensemble size tested, a modest gap persisted at loose
thresholds. We attribute this to a genuine mismatch between training
objective and evaluation metric rather than a further artifact. The CNN
baseline is trained to directly minimize per-pixel binary cross-entropy —
the same quantity that pointwise threshold metrics effectively measure —
whereas the diffusion model is trained on a denoising objective over the
full predictive distribution, with no term directly optimizing pointwise
classification accuracy; its "probability" is a post-hoc construction
from ensemble samples rather than a directly learned output. Under
genuine, irreducible positional uncertainty in the underlying process —
which we establish independently in Section 4.X: lightning displacement
at convective scale is not captured by any coherent motion field — the
loss-minimizing strategy for a per-pixel classifier is to hedge,
distributing probability mass across all plausible locations. This
hedging is, by construction, rewarded by pointwise metrics, which
penalize confident errors more than diffuse uncertainty. A generative
model's individual ensemble members, by contrast, are sharp and
individually plausible realizations of the forecast distribution — the
explicit design goal of a sample-based probabilistic approach — which is
a harder target to score well under pointwise, zero-tolerance
verification even when it is arguably the more faithful representation
of genuine forecast uncertainty. This interpretation is directly
consistent with the spatial-tolerance result above: the diffusion
model's relative advantage emerges precisely where the scoring rule
stops penalizing positional hedging and starts rewarding calibrated,
majority-consensus confidence.

**Implication for evaluation practice.** We suggest that comparisons
between sample-based probabilistic forecasting models and deterministic
baselines directly optimized for a pointwise loss should report (i)
verification metrics across a range of spatial tolerances rather than at
pixel resolution alone, and (ii) sensitivity to ensemble size where
probabilities are constructed empirically from samples, rather than a
single value at a single, possibly under-powered, ensemble size. Absent
this decomposition, a probabilistic model's genuine representation of
forecast uncertainty can appear, misleadingly, as a deficit relative to
a baseline that is simply better positioned — by construction, not by
merit — to exploit a pointwise scoring rule.

---

## Notes / things to check before finalizing

- Confirm the Metzl et al. 2025 citation format/year (in review vs.
  published by the time this is submitted) — see FINDINGS.md F3.
- Add Roberts & Lean 2008 to the bibliography if not already there
  (FSS citation) — check reference list.
- The exact numbers (2.6x/2.0x area ratio, 49%/21% gap closure) are from
  the n=10/30/50 diagnostic sweep specifically for this comparison —
  re-verify they match whatever final n_members setting is used for the
  paper's actual headline numbers (see PAPER_TODO.md Phase 2 item on
  regenerating headline numbers at a larger n_members).
- Consider whether this belongs partly in Methods (verification
  procedure / metric choice, stated prospectively) vs. Discussion
  (interpretation) — currently drafted as a single Discussion block;
  may read better split.
- "Section 4.X" / "Section X" placeholders need real cross-references
  once section numbering is finalized.
