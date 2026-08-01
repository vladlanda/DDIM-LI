# Findings Log — for Paper Structuring

Running record of every result, negative result, and methodological lesson
that could shape the paper's narrative or figures. Updated as we go.
Each entry: what we found, how solid the evidence is, and what figure (if
any) it would need. Organize/prune when we get to actual writing — this is
raw material, not manuscript prose.

Status tags: **CONFIRMED** (validated, ready to cite) / **PENDING**
(needs a re-run or further check before citing) / **CAVEAT** (real finding
but with a known limitation to state honestly).

---

## A. Headline results

### A1. Final model beats fresh persistence, margin GROWS with lead time — CONFIRMED
2-channel (ir105+li) diffusion model, 790 epochs, clean/regenerated
dataset. PR-AUC 0.819→0.589 (+10 to +60min) vs. fresh persistence
0.783→0.391. Margin: +4.6% (+10min) → **+50.9%** (+60min).
**Figure:** PR-AUC vs lead time, model vs. persistence, with margin
annotated — likely Figure 1 or 2 of the results section.

### A2. Model beats BOTH optical-flow baselines too, same growing-margin shape — CONFIRMED
Adding two pySTEPS-style extrapolation baselines (LI-derived flow,
IR-derived flow) doesn't change the story: persistence remains the best
of the three classical baselines at every lead except +60min (near-tie).
Model margin over the *best* baseline: +4.6% → **+50.3%**.
**Figure:** four-way PR-AUC comparison (model / persistence / flow_li /
flow_ir) — stronger version of A1's figure, this is probably the actual
headline figure once bootstrap CIs are added (see B, pending).

### A3. Fresh persistence ≈ old (stale-data) persistence, within 0.001 at every lead — CONFIRMED
Validates that the severe coverage bug (see D2) didn't change the *test
period's* lightning climatology, only which frames existed — so this
comparison is trustworthy. Also implies training-data completeness, not
test-period character, drove the earlier degraded numbers.
**No figure needed** — goes in methods/reproducibility as a sanity check.

---

## B. Statistical rigor (infrastructure built, not yet applied to final numbers)

### B1. Sequence-level bootstrap CI — CONFIRMED, applied to the headline comparison
All 18 comparisons (3 baselines × 6 leads) are statistically significant
(95% CI excludes zero), and critically, the margin GROWS monotonically
with lead time against all three baselines simultaneously:
  - vs persistence:  +0.036 (+10m) → +0.200 (+60m)
  - vs pysteps_li:    +0.048 (+10m) → +0.199 (+60m)
  - vs pysteps_ir:    +0.177 (+10m) → +0.315 (+60m)
This is now a statistically robust, structurally consistent result, not
just a point-estimate — the strongest form A1/A2 could take. n_boot=1000,
sequence-level (not pixel-level) paired resampling.
CI width scales with baseline reliability: model-vs-pysteps_ir has the
widest CI (±0.025 @60min vs ±0.011 for persistence), consistent with
pysteps_ir being the noisiest baseline (fits E3's physical story).
**Figure:** this is the table/figure that should anchor the Results
section — model PR-AUC with CI, alongside all three baselines with
their deltas and significance markers. Likely a combined plot (PR-AUC
vs lead, all four curves, shaded CI bands) rather than a bare table.

---

## C. Architecture / methodology lessons (mostly negative results — still valuable)

### C1. LI-in-context is the single largest driver of skill — CONFIRMED
+0.07 to +0.14 PR-AUC, larger than every other intervention tried this
entire project combined. Establishes conditioning on recent lightning
history as the dominant lever, ahead of channel selection, calibration,
or loss engineering. Worth stating explicitly early in Results as the
"floor" that everything else is incremental on top of.

### C2. Auxiliary cooling-rate prediction task — NEGATIVE RESULT, CONFIRMED, informative
Built a physically-motivated auxiliary head predicting future IR change,
hypothesizing it would inject "convective existence" information. Failed:
the target reduced algebraically to `-y_ir`, the negation of the channel
the main head already predicts — informationally redundant, not novel
information. Ablation showed it helped short lead (+0.116 @10min, where
IR residual is predictable) and hurt long lead (−0.024 @60min, where it
isn't) — exactly what the redundancy diagnosis predicts.
**General principle worth stating in methods/discussion:** any auxiliary
target that is a deterministic function of an already-predicted channel
cannot inject new information into a multi-task architecture — it can
only reweight the loss. This is a citable, generalizable lesson, not
just a dead end. Reverted; preserved on `archive/aux-cooling-redundant`.

### C3. Pre-registered channel-information tests: ir87/ir97/ir123 add nothing — CONFIRMED
Rigorous, sequence-bootstrap-CI'd, pre-registered single-hypothesis tests
(not an exploratory sweep — see C3a for why that mattered) found none of
the three auxiliary IR window channels carry EXISTENCE information beyond
ir105 level + ir105 tendency + LI history, at either instantaneous value
or ~2h tendency. ir97 sat at the random null in every single test run.
Model reduced 4→2 channels (ir105, li) on this basis.
**Figure:** could include the pre-registered test's CI table as a
supplementary table — strong evidence of methodological rigor for
reviewers, and directly rebuts the "did you even check?" question a
channel-pruning claim invites.

### C3a. Exploratory multi-feature channel sweep was methodologically unsound — METHODS LESSON
An earlier, broader version of C3 swept ~8 features × 6 leads × 2 regimes
with no multiple-comparisons correction, and had two real estimator bugs
along the way (per-sequence strata leakage inflating a "redundant"
control to 0.35 instead of ~0.05; sign-blind |AUC−0.5| misreading an
anti-correlated ir105 control as uninformative). Replaced with C3's
single pre-registered test per channel. **Methods-section lesson:** worth
a sentence on why pre-registration was adopted, since it's a genuine
strength of the methodology relative to typical ad-hoc channel ablations
in this literature (including LightningCast's, see E2).

### C4. Ensemble calibration (spread-skill) and PR-AUC are separate, non-transferable axes of skill — CONFIRMED
Lead-time-scaled churn, derived purely from VALIDATION-set spread-skill
deficit (never tuned on test), successfully corrected calibration
(spread-skill markedly improved) but produced **zero** change in PR-AUC.
**Methods/discussion point:** calibration and discrimination are
different, non-transferable properties of a probabilistic forecast — a
model can be miscalibrated but well-discriminating or vice versa, and
fixing one does not fix the other. Worth a sentence in discussion,
possibly with the reliability-diagram figure as evidence (see F, poster
work already has this figure built via `plot_pr_comparison.py`).

### C5. Data degeneracy (all-black corrupt frames) silently contaminated ~63% of training sequences — CONFIRMED, methods/reproducibility
`build_index()` used `Path.exists()`, so present-but-all-black JPEGs
(≈1.7kB) were marked valid and fed to the model as constant planes. Per-
frame black rate 0.16–1.31% across ch0/ch1/ch2 became ~63% of 42-frame
TRAINING SEQUENCES contaminated once compounded. Fixed via a size-based
degeneracy check (asymmetric: IR channels can never legitimately be
all-black; LI can, since it means "no flashes," and is never filtered).
**Belongs in Methods/Data, likely supplementary** — a genuine data-
quality contribution, not itself a headline result, but should be
disclosed since every prior baseline number in this project trained on
the contaminated data.

---

## D. Data pipeline integrity (methods/reproducibility, not scientific findings per se)

### D1. Stale index-cache silently trusted with zero staleness check — CONFIRMED, fixed
Cache WRITE was disabled but READ was not, so any leftover
`.index_cache.json` from an earlier version of this (much-rewritten)
codebase was trusted forever. Caused a silent "0 valid sequences" for one
region while a sibling region without a stale cache worked fine. Fixed:
only load a cache when explicitly requested.

### D2. Severe temporal sparsity in the original split/train data — CONFIRMED, resolved by regeneration
Two independently-checked regions showed near-identical pathology: ~60%
coverage, mean unbroken run 2.7 timesteps, longest run 22 (3.7h) across a
full year — making the required 42-consecutive-timestep sequence (T_in=36
+T_out=6) essentially impossible. Root cause never fully resolved (three
candidate explanations, none confirmed) — dataset regenerated instead.
New data: 99.2% coverage, longest run 433h. **Worth a sentence in
Methods/Data or supplementary** as a documented data-quality control step,
without necessarily dwelling on the unresolved root-cause mystery.

### D3. evaluate.py's pr_seqid save was missing — CONFIRMED, fixed before any compute spent
Code review pass (prompted deliberately before launching the expensive
model evaluation) found the seqid-tracking patch was only half-applied
to evaluate.py: accumulated in memory during the loop, never written to
the npz. persistence_baseline.py and optical_flow_baseline.py were both
correct. Would have silently discarded the data needed for the bootstrap
CI on the paper's headline comparison, on exactly the one run that's
expensive to redo. Caught by a deliberate "review before spending GPU
time" pass, not by running and failing. Fixed and verified end-to-end
with a synthetic save/load test before trusting it.
**Lesson for process, not for the paper:** worth doing one more such
review pass before any other expensive/one-shot run in this project.

### E1. Positional-vs-existence error decomposition at long lead time — CONFIRMED (on an earlier model; PENDING re-run on final model)
At high confidence threshold, +60min error is POSITIONAL (skillful at
36km neighbourhood tolerance). At low threshold, it's EXISTENCE (never
skillful at any spatial scale). Two different failure modes requiring
different fixes — this dual-regime framing is likely the paper's central
mechanistic thesis. **TODO: re-run `diagnose_fss_scale.py` /
`diagnose_prob_bins.py` on the FINAL 2-channel model** — current evidence
is from an earlier model/dataset and needs revalidating before going in
the paper as-is.
**Figure:** FSS-vs-scale curves at multiple probability thresholds,
likely a core results figure.

### E2. Displacement is incoherent, not advective — CONFIRMED, now corroborated TWO independent ways
Custom `best_shift` analysis: optimal GLOBAL translation recovers almost
nothing (+0.018–0.02 PR-AUC) — falsifies simple bulk-advection correction
at convective scale. NOW INDEPENDENTLY CONFIRMED by a completely
different, standard method: both optical-flow baselines (A2) underperform
plain persistence, meaning even a smoothly-varying, locally-estimated
motion field fails to capture whatever displacement structure exists.
**This is likely the single most citable, defensible novel claim in the
paper** — two independent methods agreeing that convective-scale
lightning displacement isn't captured by any coherent motion field,
directly contradicting the implicit assumption behind most classical
nowcasting (optical flow / advection-based) approaches.
**Figure:** best_shift PR-AUC recovery curve + the four-way baseline
comparison (A2) side by side — the paper's likely Figure 3 or 4.

### E3. IR-derived flow performs WORSE than LI-derived flow for advecting LI — CONFIRMED, counter to prediction
Predicted the opposite going in. IR-derived flow (denser, smoother field,
"should" give more reliable motion estimates) underperforms LI-derived
flow by 11–17 points of PR-AUC at every lead, and underperforms
persistence by up to 17 points. Candidate physical explanation: IR tracks
bulk cloud-top/anvil advection, which is systematically DIFFERENT from
(not just noisier than) the motion of the active convective cores that
actually produce lightning — so IR-derived flow is confidently wrong
about the right thing to track, while LI-derived flow is just noisy.
**CAVEAT:** this physical interpretation is a plausible hypothesis, not
yet independently tested — worth flagging as such in discussion rather
than stating as established fact, unless we design a direct test later.
**Figure:** part of the A2 four-way comparison; could also show
example flow fields (IR-derived vs LI-derived) overlaid on a case to
illustrate the anvil-vs-core-motion story visually.

### E4. Convective memory / cloud-top cooling saturates at ~2 hours — CONFIRMED
Cooling-rate premise diagnostic: lightning lift over baseline is 2.07–
3.73×, monotonically increasing with the cooling-integration window,
saturating at 120–180 minutes. A genuine, standalone physical finding
about this region's convective development timescale — independent of
whether we ended up using it as a model input (we didn't, see C2).
**Figure:** lift-vs-integration-window curve — clean, simple,
citable, low-cost to include even though the aux-task application (C2)
didn't pan out. Worth keeping in the paper as a physical characterization
result even without a corresponding architecture change.

---

## F. Framing / comparison caveats (important for Discussion / avoiding overclaiming)

### F1. Song et al. 2023's 0.727 PR-AUC is NOT directly comparable to ours — CAVEAT, important
Their task: 0.25°/hourly/coarse classification (LightGBM). Ours: 4km/
10min pixel-exact (diffusion). Different chance baselines (their base
rate plausibly 0.25–0.5 depending on aggregation; ours ~0.058). Comparing
raw PR-AUC numbers across these regimes is an apples-to-oranges error a
reviewer would catch immediately if we did it uncritically.
**Action for paper:** either (a) don't compare raw numbers at all and
frame the contribution differently (finer resolution, generative,
different region), or (b) if we want a number-to-number comparison,
regrid our predictions to their 0.25°/hourly protocol and report THAT
number alongside ours, with both chance baselines stated. Still on the
table as future work per PAPER_TODO.md but not started.

### F2. LightningCast's channel ablation found 1.37/6.2/8.4μm unhelpful; ours found ir87/ir97/ir123 (8.7/9.7/12.3μm) unhelpful too — related work contrast, CONFIRMED
Different satellite (GOES-ABI vs MTG-FCI), different bands, same
qualitative conclusion (narrow-band auxiliary IR channels add little
beyond the core window channel + lightning history). Worth a sentence in
Discussion/Related Work — our finding is independently consistent with
theirs despite different platforms, using a considerably more rigorous
methodology (pre-registered, bootstrap-CI'd vs. their ad-hoc ablation).

---

## G. Open items that would strengthen the paper if resolved (cross-ref PAPER_TODO.md)

- Channel-pruning result (C3) still confounded with the concurrent
  dataset regeneration (D2) — 4ch-on-clean-data control run deferred,
  logged as TODO.
- E1 needs re-running on the final model before citing as-is.
- B1 (bootstrap CIs) needs applying to the A1/A2 headline comparison.
- E3's physical interpretation (anvil vs. core motion) is a hypothesis,
  not yet tested directly — could be worth a cheap follow-up if time
  allows (e.g. compare IR-derived and LI-derived flow FIELDS directly on
  a few cases, not just downstream skill).
