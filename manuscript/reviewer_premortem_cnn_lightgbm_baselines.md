# Reviewer pre-mortem: diffusion vs. directly-optimized ML baselines (CNN, LightGBM)

Purpose: anticipate the hardest questions a reviewer familiar with both
the nowcasting literature (Metzl et al. 2025, Cintineo et al. 2022) and
the generative-forecasting literature (GenCast/Price et al. 2024) would
raise about the fact that deterministic baselines (CNN, likely LightGBM
too) beat the diffusion model on raw pointwise PR-AUC, and draft the
preemptive answer for each. Written generally (not CNN-specific) since
LightGBM is expected to show the same pattern -- see FINDINGS.md F4 for
full technical detail; this file is the reviewer-facing framing layer.

Companion to manuscript/discussion_cnn_baseline_comparison.md, which is
the actual discussion-section prose. This file is strategy/prep, not
manuscript text.

---

## Q1. "If simple deterministic baselines beat your diffusion model at
every lead time, why should we believe the diffusion approach is the
right one for this problem at all?"

This is the single most dangerous question — answer it directly, don't
bury it.

**Preemptive answer, three parts:**
1. The diffusion model decisively beats every *physically-motivated*
   baseline (persistence, both optical-flow variants) — the actual
   competitors for an operational nowcasting system — by large, growing,
   statistically significant margins (see FINDINGS.md A1/A2/B1). The
   comparison to CNN/LightGBM is a comparison to models trained to
   directly optimize the evaluation metric itself, which is a
   structurally different, much narrower question ("can a model
   optimized FOR this metric beat one that wasn't"), not "which model
   is better for the task."
2. We show this is not an assertion but a demonstrated, testable
   mechanism: ensemble-size sensitivity (with a clean diminishing-
   returns curve) plus spatial-tolerance sensitivity (FSS vs. scale)
   plus a lead-time-shape distinction (constant CNN margin vs. growing
   margins elsewhere) — three independent lines of evidence, not one
   adjustable story.
3. The CNN/LightGBM baselines cannot produce genuinely calibrated
   ensemble forecasts at all — no CRPS, no spread-skill relationship,
   no multiple physically-plausible realizations of a single scene.
   These are the actual scientific deliverables of a probabilistic
   nowcasting system given genuinely irreducible positional uncertainty
   (E2), and they're capabilities, not just scores — a deterministic
   classifier cannot be evaluated on them because it cannot produce
   them.

## Q2. "Isn't your explanation just post-hoc rationalization for a
worse model?"

**Preemptive answer:**
- The mechanism makes falsifiable predictions, and we show the data
  matching them, not just an explanation fitted after the fact: the
  gap should shrink with ensemble size (confirmed, diminishing
  returns), shrink with spatial tolerance at strict thresholds
  (confirmed, reverses beyond ~130km at p>0.5), and NOT shrink with
  spatial tolerance at loose thresholds if it's a coverage/calibration
  issue rather than positional (confirmed — flat across all scales at
  p>0.1).
- The same qualitative phenomenon (MSE/pointwise-trained deterministic
  models blur/hedge relative to a diffusion model, presented as
  established motivation rather than a novel finding) is independently
  reported by Price et al. 2024 (GenCast, Nature) in a different
  forecasting domain, developed independently of this work. We are not
  the only group observing this.
- If we wanted to hide a weak result, the straightforward move would
  have been to simply not include the CNN/LightGBM comparison at all,
  or to only report it at a spatial tolerance/threshold favorable to
  the diffusion model. We report the full picture, including where the
  diffusion model does NOT win.

## Q3. "Your n_members sweep only goes to 50 — how do you know the gap
wouldn't keep closing with, say, 200 members?"

**Preemptive answer, be honest about the limitation:**
We did not test beyond n_members=50 due to compute cost. The
diminishing-returns pattern (49% of the original gap closed 10->30,
a further 21% of the remainder 30->50) strongly suggests further
increases would yield increasingly marginal reductions, consistent
with a finite-sample bias that shrinks but does not have to fully
vanish at any finite N -- but we cannot rule out a small additional
narrowing at much larger ensemble sizes, which were not computationally
practical to test exhaustively. State this as a limitation, don't
oversell "fully resolved."

## Q4. "Is the CNN baseline actually fair, or did you handicap it?"

**Preemptive answer:**
The CNN baseline's training recipe (weight decay, LR schedule) follows
Metzl et al. 2025's published values directly, not an arbitrary or
weakened choice (see FINDINGS.md C6). Context window and lead-time
amortization deliberately match the main model's own setup rather than
the literature's exact protocol, and we state explicitly why: holding
input information constant isolates architecture as the controlled
variable in OUR comparison, which is the more rigorous choice internally
even though it diverges from Metzl et al.'s literal setup (see C7). We
are not aware of a respect in which the baseline is under-tuned or
handicapped relative to standard practice.

## Q5. "Could the CNN's edge be an information-budget or test-set
confound rather than a real model-class difference?"

**Preemptive answer:**
No — this is directly controlled for. The CNN and diffusion model share
identical T_in context, identical T_out, identical img_size, and are
evaluated on the identical set of test sequences (required for the
paired bootstrap comparison to be valid at all — see
bootstrap_pr_auc_ci.py's validity requirement, satisfied here). The only
varying factor between the two models being compared is architecture
and training objective, which is the intended comparison.

## Q6. "Why LightGBM and not Random Forest / XGBoost / a more modern
tabular method?"

**Preemptive answer:**
Already a scoped, stated decision (PAPER_TODO.md): LightGBM only, not
also XGBoost, to avoid a redundant comparison within the same
gradient-boosted-tree family that would not add a distinct baseline
class. If a reviewer wants a specific alternative, that's a fast
follow-up experiment, not a fundamental gap.

## Q7 (anticipate once LightGBM results exist). "You have TWO baselines
now beating the diffusion model pointwise — doesn't that pattern
undermine your paper's contribution more than one baseline did?"

**Preemptive answer, THE keystone response:**
Two independent baselines showing the same qualitative pattern, fully
consistent with a mechanism predicted and confirmed in advance, is
stronger evidence FOR the explanation, not against the model. A single
anomalous result invites suspicion of a fluke or a hidden bug; a
predicted, repeated, mechanistically-explained pattern across
architecturally distinct model classes (a CNN and a gradient-boosted
tree ensemble, sharing only "directly optimized on a pointwise loss")
is closer to a confirmed regularity. Frame LightGBM's result, if it
matches the predicted pattern, explicitly as a confirmatory replication
of F4's mechanism -- not a second, separate surprise requiring a new
explanation.

**If LightGBM does NOT match the predicted pattern** (e.g. shows a
large, growing, or spatial-tolerance-robust advantage): treat this as
a genuine anomaly requiring real investigation, not something to
force-fit into the existing explanation. Update this document and
FINDINGS.md accordingly if that happens.

---

## Bottom line framing for the paper

The core contribution is not "beats every baseline on every metric
unconditionally" -- that is both a weaker and a less credible claim to
reviewers than what the data actually supports: **the diffusion model
substantially and significantly outperforms every physically-motivated
baseline, and where directly-optimized deterministic ML baselines hold
a narrow pointwise edge, that edge is small, mechanistically explained,
independently corroborated, reverses under realistic verification
conditions, and does not extend to the genuine probabilistic
capabilities (calibration, ensemble spread, CRPS) that are the actual
point of a generative approach to a problem with irreducible positional
uncertainty.** A nuanced, honestly-reported comparison is more credible
under review than an unqualified sweep, not less.
