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

### A1. Final model beats fresh persistence, margin GROWS with lead time — CONFIRMED, FINAL (n_members=50)
2-channel (ir105+li) diffusion model, 790 epochs, clean/regenerated
dataset, evaluated at `n_members=50` (see F4 for why this setting was
chosen — n_members=10 measurably understated performance; 10->30
closed 49% of a since-explained gap, 30->50 closed a further 21% of
the remainder, clear diminishing returns beyond this point). PR-AUC
0.846→0.631 (+10 to +60min) vs. fresh persistence 0.783→0.391.
Margin: **+8.1%** (+10min) → **+61.6%** (+60min).
**Figure:** PR-AUC vs lead time, model vs. persistence, with margin
annotated — likely Figure 1 or 2 of the results section.

### A2. Model beats BOTH optical-flow baselines too, same growing-margin shape — CONFIRMED, FINAL (n_members=50)
Adding two pySTEPS-style extrapolation baselines (LI-derived flow,
IR-derived flow) doesn't change the story: persistence remains the best
of the three classical baselines at every lead except +60min, where
pysteps_li narrowly overtakes it (0.3921 vs 0.3906 — still a near-tie).
Model margin over the *best* classical baseline at each lead:
**+8.1% → +61.0%**.
**Figure:** four-way PR-AUC comparison (model / persistence / flow_li /
flow_ir) — stronger version of A1's figure, this is the actual headline
figure now that bootstrap CIs are finalized (see B1).

### A3. Fresh persistence ≈ old (stale-data) persistence, within 0.001 at every lead — CONFIRMED
Validates that the severe coverage bug (see D2) didn't change the *test
period's* lightning climatology, only which frames existed — so this
comparison is trustworthy. Also implies training-data completeness, not
test-period character, drove the earlier degraded numbers.
**No figure needed** — goes in methods/reproducibility as a sanity check.

---

## B. Statistical rigor — FINALIZED, applied to the final (n_members=50) numbers

### B1. Sequence-level bootstrap CI — CONFIRMED, FINAL, all five comparisons significant at every lead
All 30 comparisons (5 baselines × 6 leads) are statistically significant
(95% CI excludes zero). n_boot=1000, sequence-level (not pixel-level)
paired resampling, `n_members=50` (see F4/A1 for why).
  - vs persistence:  +0.063 (+10m) → +0.241 (+60m) — margin GROWS
  - vs pysteps_li:    +0.075 (+10m) → +0.239 (+60m) — margin GROWS
  - vs pysteps_ir:    +0.205 (+10m) → +0.355 (+60m) — margin GROWS
  - vs lightgbm:      +0.018 (+10m) → +0.185 (+60m) — margin GROWS,
    same qualitative shape as the three physical baselines (see C8 for
    why: LightGBM shares the CNN's matched training objective but NOT
    its matched raw information access, and behaves accordingly)
  - vs cnn baseline:  -0.021 (+10m) → -0.024 (+60m) — margin roughly
    CONSTANT, model behind at every lead (see F4 for the full mechanistic
    explanation: ensemble-size artifact mostly resolved, small genuine
    residual from a training-objective mismatch remains)
**The CNN comparison's shape is qualitatively different from all four
others, and that difference is itself informative:** persistence,
pysteps_li, pysteps_ir, AND lightgbm all show margins widening
substantially with lead time (their skill degrades faster than the
model's does), while the model-vs-CNN gap stays within a narrow
[-0.021, -0.024] band across the entire 50-minute range tested. A
roughly lead-time-INDEPENDENT effect is consistent with a fixed
structural cause (training objective mismatch, given matched
information access) rather than a lead-time-dependent one
(positional/motion degradation, or an information-access bottleneck
that compounds with prediction difficulty like lightgbm's does) —
worth stating explicitly in the manuscript as a clean empirical
signature supporting F4's explanation. LightGBM landing with
persistence/pysteps' GROWING-margin shape rather than the CNN's flat
one, despite sharing the CNN's pointwise training objective, is
itself evidence that objective-matching alone doesn't produce the
CNN's pattern — matched information access is the operative variable
(see C8).
CI width scales with baseline reliability/nature: model-vs-pysteps_ir
has the widest CI (±0.016 @60min vs ±0.011 for persistence), consistent
with pysteps_ir being the noisiest baseline (fits E3's physical story).
model-vs-cnn's CI is the tightest of all five (±0.005 @60min, ±0.007
@10min) — expected, since the CNN baseline is deterministic (no
ensemble-sampling variance contributed from that side of the paired
comparison, unlike the other baselines being compared against a
50-member diffusion ensemble on the model side too). model-vs-lightgbm
is the second-tightest (±0.010 @60min, ±0.004 @10min) — same reason,
LightGBM is also a deterministic (non-ensemble) baseline.
**Figure:** this is the table/figure that should anchor the Results
section — model PR-AUC with CI, alongside all baselines with their
deltas and significance markers. Likely a combined plot (PR-AUC vs
lead, all curves, shaded CI bands) rather than a bare table.

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

### C6. CNN baseline's training recipe silently diverged from its own literature comparator, weakening it — CONFIRMED, fixed
While diagnosing an unexpected val_loss growth after epoch ~10-11 in
`baseline_cnn/`'s first real training run (`run1`, post-fix-#5's
lightweight-architecture correction), a direct comparison against Metzl
et al. 2025 (the paper this baseline is explicitly benchmarked against)
found two silent regularization gaps, not deliberate design choices:
  - `weight_decay` was never wired into `train_cnn.py`'s Adam optimizer
    at all. `configs/default.yaml`'s `weight_decay: 0.0` field would
    have been silently inherited even if changed, and that field's own
    justification ("EMA already regularises") is specific to the
    diffusion model — `baseline_cnn/` has no EMA, so the rationale
    never applied here. Same *class* of bug as the base_channels
    architecture-inheritance issue (item 5, `PAPER_TODO.md`), caught by
    applying the same "is this actually intentional or just inherited"
    scrutiny.
  - LR schedule was `CosineAnnealingLR` (fixed decay over `--epochs`,
    unrelated to when val_loss actually plateaus), not anything that
    responds to validation performance.
Metzl et al. 2025 report weight_decay=1e-4 (L2) and
`ReduceLROnPlateau(factor=0.1, patience=5, cooldown=3)` monitoring
val_loss explicitly "to ensure no overfitting" — i.e. their recipe is
designed against exactly the failure mode observed in `run1`.
**Fix applied** (commit on `paper` branch): `train_cnn.py` now uses
both values from Metzl et al. 2025 directly (not invented), plus a
practical (non-literature) early-stopping safeguard
(`--early_stop_patience`, default 20) to cap wall-clock once plateaued.
Verified via a synthetic scheduler/optimizer smoke test that
`weight_decay` lands on the optimizer and `ReduceLROnPlateau` correctly
drops LR on a val_loss curve shaped like `run1`'s real one, before
trusting the fix — consistent with this project's "validate before
trusting" practice. **Not yet re-run on real data as of this entry**
(`run1`'s numbers predate the fix; a fresh run, e.g. `run2`, is needed
to confirm the fix actually resolves the overfitting in practice, not
just mechanically).
**Methods-section implication:** this is worth a one-line note —
"training regularization (weight decay, LR schedule) follows Metzl et
al. 2025" — which is a stronger, more citable framing than either
silence or an arbitrary home-grown choice.

### C7. Two remaining CNN-baseline departures from the literature are being kept, deliberately — CAVEAT, worth stating explicitly
Two other differences from Metzl et al. 2025 / Cintineo et al. 2022
were reviewed alongside C6 and kept as-is, for a different reason than
C6's gaps: they serve *this paper's* internal comparison logic, not
just literature fidelity.
  - **Context window:** `T_in=36` (6h) here, vs. Metzl's BNN using only
    the last 2 observations (~15-30min). Kept because giving the CNN
    baseline the SAME context window as the main diffusion model holds
    input information constant across the comparison, isolating model
    architecture as the controlled variable — shortening it to match
    Metzl would instead confound architecture with information budget,
    the same category of confound this project has been careful to
    avoid elsewhere (see Phase 1's channel-pruning-vs-dataset-
    regeneration issue).
  - **Lead-time amortization:** one model conditioned on `lead_idx`
    across all 6 output steps, vs. Metzl/Cintineo training a SEPARATE
    network per lead time. Kept for the same reason — the main
    diffusion model is also a single model amortized across lead
    times, so this keeps that structural choice consistent across the
    paper's own model comparisons. (Already flagged honestly in
    `model_cnn.py`'s docstring before this review.)
Class-imbalance handling (oversampling via the shared
`WeightedRandomSampler`, vs. Metzl/Cintineo's undersampling) was
reviewed too and kept for the same reason: it's the same mechanism the
main model and other baselines already use, so keeping it here isolates
architecture as the controlled variable rather than introducing a
second free-floating methodological difference.
**Methods-section implication:** state these two/three departures
explicitly with the fairness rationale above, rather than leaving a
sharp reviewer to wonder why the setup doesn't match Metzl's exactly —
this is a case where explaining the choice is a strength, not a
defensive footnote.

### C8. LightGBM baseline scaffolded — feature design and consistency choices, PENDING (not yet run on real data)
Built `baseline_lightgbm/` (features.py, train_lightgbm.py,
evaluate_lightgbm.py), following the same "spirit of the literature,
not a literal reproduction" framing as the CNN baseline (F1/F3, C6/C7),
since Song et al. 2023's LightGBM operates on a completely different
resolution/feature set (0.25°/hourly, meteorological+aerosol+GLM
features) that we don't have.
**Feature provenance, stated explicitly (user asked directly; this is
the kind of thing a reviewer will ask too, so the honest answer needs
to be consistent everywhere, not just in this file):** LightGBM itself
provides none of this — it's a generic gradient-boosting library with
zero domain knowledge, every feature is our own code. The 18 features
are NOT a reproduction of Song et al.'s exact set (we lack their
aerosol/reanalysis inputs). The general categories used (sliding-window
temporal stats, trend/finite-difference features, local spatial
neighbourhood aggregation, time-since-last-event recency) are standard
practice across nowcasting ML broadly, not novel. Local spatial
aggregation for satellite-based lightning specifically has apparent
precedent in Karagiannidis et al. 2016 ("interest fields") — but that
paper has only been seen referenced secondhand (a search snippet), not
read directly, so this is a conceptual similarity only, not a
reproduction of their actual formulas. The specific 18-feature list,
9x9 window size, and lag choices are this project's own design.
**18 features** (see features.py's module docstring for full rationale):
IR temporal stats (now/mean/std/min/max over context, 30min and 2h
trends — the 2h window deliberately connects to E4's convective-memory-
saturates-at-~2h finding) and local 9x9 spatial neighbourhood stats;
LI activity/recency stats (current/recent/count/fraction active,
time-since-last-active, computed in PHYSICAL space at the same
li_event_threshold the label itself uses — using a different
"active" definition for historical-context features than for the
label would have been a real, confusing inconsistency) and local 9x9
neighbourhood stats; plus lead_idx, enabling one amortized model across
all 6 lead times (same reasoning as C7's lead-time-conditioning
argument for the CNN baseline — internal consistency across our own
baselines matters more than matching any one external protocol).
Deliberately EXCLUDES absolute pixel position — every feature is
translation-invariant, matching the CNN/diffusion model's fully-
convolutional equivariance, and preserving validity of the still-open
held-out-region generalization check (PAPER_TODO.md) — absolute
coordinates would let a tree-based model memorize training-region
geography in a way that wouldn't transfer, silently undermining that
future check.
**Class imbalance:** LightGBM's native `is_unbalance=True`, not manual
pixel oversampling — the idiomatic approach for this model class, kept
deliberately distinct from the CNN baseline's `WeightedRandomSampler`
mechanism rather than forcing consistency where the two model classes'
standard practices genuinely differ (unlike T_in/lead-amortization,
where holding structure constant was the right call — imbalance
handling isn't part of what's being compared here).
**Two real bugs caught during synthetic-data validation, before this
touched real data:** (1) both training and evaluation loop over 6 lead
times per sequence, but the initial feature-extraction API recomputed
the expensive spatial-filter features from scratch every call despite
them not depending on lead_idx — 6x redundant work, fixed by splitting
into a compute-once-per-sequence / reuse-per-lead two-step API,
verified numerically identical to the original single-step version
before trusting the refactor. (2) the initial validation-set design
used full-image (every pixel) extraction, reasoned as giving an
"unbiased early-stopping signal" — but at realistic defaults (256x256
images, T_out=6, ~hundreds of validation sequences) this would have
needed ~10-20GB of RAM for the validation matrix alone, caught by
actually computing the expected size rather than assuming the design
was fine, before it could fail on the user's machine hours into a run.
Fixed with a separate, bounded `--val_pixels_per_image_lead` (subsampled
like training, just larger, for a more precise signal without being
unbounded).
**Reuses `evaluate_cnn.py`'s `_make_plots`/`_li_to_physical` directly**
(imported, not duplicated) — required parameterizing `_make_plots` with
`filename_prefix`/`display_name` args (previously hardcoded to
"baseline_cnn"/"CNN Baseline"), since reusing it verbatim would have
mislabeled every LightGBM plot — caught by an end-to-end synthetic test
asserting on the actual output filenames, not just that files existed.
**Output schema identical to `evaluate_cnn.py`'s** (same npz/CSV
structure) specifically so `bootstrap_pr_auc_ci.py` and
`compare_diffusion_vs_cnn_fss.py`-style tooling work against it via
`--baseline lightgbm:...` with no new comparison code needed.

**RESULT (first real run, PR-AUC): the falsifiable prediction did NOT
hold — treated as a genuine, clarifying anomaly, not force-fit into F4,
per the pre-mortem's own instructions for this outcome.**
LightGBM: 0.828→0.446 (+10m→+60m). Diffusion model (n=50): 0.846→0.631.
Unlike the CNN, the diffusion model beats LightGBM at EVERY lead time,
with a margin that GROWS (+0.018→+0.185) — the same qualitative shape
as its margin over persistence/pysteps, not the CNN's flat,
LightGBM-losing pattern. LightGBM does beat all three physical
baselines by a roughly constant +0.05-0.06, so it's a genuinely useful
baseline, just not a CNN-tier one. LightGBM's gap behind the CNN itself
WIDENS with lead time (-0.039→-0.209), the opposite of the CNN's own
near-constant gap behind the diffusion model.
**Refined explanation, and it strengthens rather than undermines F4:**
the variable that actually distinguishes the CNN's near-parity result
isn't "trained on a pointwise loss" (LightGBM has that too — logloss is
BCE) — it's INFORMATION ACCESS. The CNN sees the exact same raw
T_in=36 pixel-grid context as the diffusion model, deliberately matched
for that reason (C7). LightGBM only ever sees the 18 hand-engineered
summary features (C8 above) — a severe, lossy compression of that same
raw data, incapable of recovering whatever fine-grained spatiotemporal
structure those features don't capture. That bottleneck should worsen
at longer lead times, since harder prediction problems benefit more
from rich raw information than fixed summary statistics can supply —
exactly the growing-margin pattern observed. This makes the CNN's
result MORE specific and MORE credible, not less: near-parity-with-a-
narrow-pointwise-edge requires BOTH a matched training objective AND
matched information access, not just the former. LightGBM has only the
first property and behaves like a boosted physical baseline as a
result — a sensible middle tier (physical baselines < LightGBM < CNN ≈
diffusion), not a second instance of F4's mechanism.
**Not yet run through bootstrap_pr_auc_ci.py** — point estimates only
so far; large enough margins that significance is expected, but not
confirmed with a CI yet. See PAPER_TODO.md for the exact command.
**Manuscript implication:** this refines the discussion draft and
reviewer pre-mortem's Q7 answer — LightGBM is a genuine, informative
contrast case (confirms information-access, not just objective-match,
is what matters), not a replication of the CNN's specific finding.
Update both documents accordingly (done — see their own revision notes).

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

### D4. dataset.py's make_dataloaders (JPEG path) was missing the same persistent_workers fix already applied to dataset_packed.py — CONFIRMED, fixed
`baseline_cnn/train_cnn.py`'s default launch command doesn't pass
`--use_packed`, so it exercises `dataset.py`'s plain-JPEG
`make_dataloaders`, not the memmap `dataset_packed.py` path. That
function never got the `persistent_workers=True`/`prefetch_factor`
fix that `dataset_packed.py` already carries (see the D-section intro
saga) — meaning with `num_workers>0` it re-forks all worker processes
at the START OF EVERY EPOCH, not just once at startup. A `run2`
resumption (`--num_workers 8`, W&B logging now active) hung
indefinitely between epochs with no error/crash — consistent with a
classic fork-after-threading deadlock: forking a process with live
background threads (CUDA context; newly, W&B's internal reporting
thread) can inherit a lock held mid-fork by another thread, deadlocking
the child forever. Landing exactly at the epoch boundary matches the
observed symptom. `train.py` (main model) also calls this same
function, so the gap was live there too, just apparently never
triggered in practice (real long runs there use `--use_packed`, per
project convention). **Fixed**: `persistent_workers=(num_workers>0)`
+ `prefetch_factor=4` added, matching `dataset_packed.py`'s existing,
already-battle-tested pattern exactly. **Not a certainty** — diagnosed
from code inspection and the well-known failure signature, not from a
captured stack trace of the actual hang (the process wasn't
`py-spy`-dumped before being restarted) — but consistent with all
available evidence and a strictly-safe change regardless.

### E1. Positional-vs-existence error decomposition at long lead time — CONFIRMED on final model, refined
Re-run on the final 2-channel model at both threshold extremes:
  - **prob_thr=0.5 (high confidence):** POSITIONAL at every lead 10-60min,
    skillful at just 12-20km throughout. The model's confident predictions
    are always spatially close to correct, at every lead time tested.
  - **prob_thr=0.1 (low confidence):** a clean, GRADUATED transition with
    lead time — POSITIONAL (+10/+20min) -> MIXED (+30/+40min) ->
    **EXISTENCE, never reaches skill at any scale (+50/+60min)**.
This is a sharper, more mechanistic version of the original finding: not
a binary "some leads are one type, some the other," but existence
uncertainty visibly GROWING IN and eventually dominating as lead time
increases, specifically in the low-confidence tail of the prediction.
The model's confident core stays positionally accurate throughout; it's
the uncertain/weak-signal predictions that degrade into pure existence
failure at long lead. **This graduated-onset framing is likely the
paper's central mechanistic thesis, stronger than the original two-
threshold framing.**
**Figure:** FSS-vs-scale curves at both thresholds, side by side, with
the skillful-scale-vs-lead-time trend annotated to show the graduated
transition — likely a core Results figure, possibly THE central one.

### E2. Displacement is incoherent, not advective — CONFIRMED, RE-VALIDATED on final model
Custom `best_shift` analysis: optimal GLOBAL translation recovers almost
nothing. RE-RUN on the final 2-channel model confirms this holds:
+60min exact=0.524 -> best_shift=0.539 (+0.015 only) — same signature as
originally found on the earlier model. Also independently corroborated
by both optical-flow baselines underperforming persistence (see A2/E3).
THREE independent checks now agree: custom best-shift analysis (original
model), custom best-shift analysis (final model, this re-run), and
standard optical-flow extrapolation (final model) all show no coherent
motion field captures lightning displacement at convective scale.
**This is likely the single most citable, defensible novel claim in the
paper.**
**Methodological note for the manuscript:** `diagnose_positional_ceiling.py`
runs on a 60-sequence subsample with its own ensemble settings, so its
'exact' PR-AUC (0.524 @60min) is NOT the same number as the full-test-set
bootstrap-CI'd PR-AUC (0.591 @60min, see B1) — expected, not a
discrepancy to worry about, but the bootstrap number is the one to
REPORT as the headline PR-AUC; this diagnostic's numbers are for
internal structure (exact vs pooled vs best-shift) only. State this
explicitly in methods so a careful reader doesn't find "two different
+60min PR-AUC numbers" and wonder which is authoritative.
**Figure:** best_shift PR-AUC recovery curve + the four-way baseline
comparison (A2) side by side — the paper's likely Figure 3 or 4.
**Independent corroboration from a different angle: see F4.** The
CNN-baseline verification analysis found the diffusion model's
pointwise-metric disadvantage at loose thresholds is real but shrinks
with ensemble size, while at strict thresholds/large spatial tolerance
it wins — the signature of a model representing genuine positional
uncertainty (rather than committing to a single confident location)
being penalized by pointwise scoring and rewarded once that uncertainty
is measured on its own terms. Same underlying phenomenon established
here via a completely different methodology (verification metrics vs.
motion-field analysis); worth citing together in the discussion.

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

### F3. Metzl et al. 2025 (DLR/DWD, in review) — closest comparator paper found, changes Phase 3 priorities
Near-identical study: satellite (MSG/SEVIRI) + lightning (LINET), CNN
(ResU-Net) segmentation over the Alps region. Directly relevant findings:
  - They advect lightning using a DENSER companion channel (water vapor)
    because "optical flow struggles with sparse fields" — same choice we
    made (pysteps_ir), but they never tested the direct-lightning-flow
    alternative. We tested both and found the denser-channel variant
    performs WORSE (E3) — a genuine point of novelty against this
    specific, recent, closely-related paper.
  - Their baseline set: physical extrapolation (LPNL, ≈ our pysteps_li)
    is dramatically weaker than either of their CNN variants (CSI 0.191
    vs 0.343) — consistent with our own pattern (optical flow barely
    matches persistence, model far exceeds both).
  - Their "scale argument": advection only helps once receptive_field ≲
    wind_speed×lead_time; empirically the crossover is >2h lead time.
    Below that, their advection-informed CNN barely beats their plain
    CNN. OUR ENTIRE HORIZON (60min) IS BELOW THIS THRESHOLD — gives an
    independently-derived, citable physical explanation for why optical
    flow underperforms in our results, not just an empirical observation.
**Action for paper:** cite this directly, possibly apply their scale-
argument formula to our architecture's receptive field and regional wind
speeds as a quantitative cross-check.
**Action for Phase 3:** their (and the broader subfield's, e.g.
LightningCast/BNN) convention of ALWAYS including a deterministic CNN
baseline alongside physical baselines is a real gap in our current
baseline set (persistence + 2 optical-flow variants, no trained CNN).
See PAPER_TODO.md Phase 3 for the concrete plan.

### F3b. Dai et al. 2025 (PNAS) — "DDMS", diffusion model + geostationary satellite convection nowcasting — MUST CITE, clearly differentiated
Found while researching journal targets. Kuai Dai et al., "Four-hour
thunderstorm nowcasting using a deep diffusion model for satellite
data" (DDMS), Proc. Natl. Acad. Sci. 122(51):e2517520122 (2025;
arXiv since April 2024). Genuinely close in method (diffusion model,
geostationary satellite brightness-temperature input, convective
nowcasting) — a reviewer at essentially any target journal will likely
know this paper. NOT a scoop, but MUST be cited and explicitly
differentiated in the manuscript's related-work/intro, not just
footnoted:
  - **Target variable differs:** DDMS predicts general convective cloud
    evolution (brightness-temperature fields / storm growth-decay).
    Ours predicts lightning occurrence specifically — a more targeted,
    directly hazard-relevant output.
  - **Scope differs:** DDMS is near-global/planetary (~20,000,000 km²,
    FengYun-4A), 4h lead time, 15min/4km resolution. Ours is regional
    (Central Africa), 1h lead time (T_out=6 @ 10min), 4km resolution —
    a deliberately narrower, deeper regional study, not a lesser one;
    npj Clim Atmos Sci explicitly lists "regional studies... new
    understanding of a particular locality" as a focus area (see F-
    section journal-target discussion / PAPER_TODO.md).
  - **Contribution type differs:** DDMS's abstract frames it as a
    systems/engineering contribution (broader coverage, longer lead
    time, fast inference, transferable to other satellites). No
    apparent analog to our E2 finding (a specific mechanistic claim
    about lightning displacement being incoherent, independently
    confirmed 3 ways) or to our baseline-verification-methodology
    contribution (F4 — CNN/LightGBM/ensemble-size/spatial-tolerance
    analysis, corroborated by GenCast). Only skimmed via search
    snippets/abstract, not read in full — full read needed before
    the related-work paragraph is drafted, to avoid mischaracterizing
    their method or missing a closer overlap than the abstract suggests.
**Action for paper:** cite in Introduction/related work with an
explicit 2-3 sentence differentiation paragraph along the lines above,
not just a passing mention — the closer a related paper, the more a
reviewer will scrutinize whether the differentiation is real or
hand-waved.

### F4. Why raw pointwise metrics initially favored the CNN baseline over the diffusion model — CONFIRMED, MAJOR FINDING, connects directly to E2
First real evaluation comparison: CNN baseline PR-AUC 0.867 → 0.655
vs. diffusion model 0.819 → 0.591 (+10min → +60min), a consistent
~0.05-0.065 absolute margin in the CNN's favor at every one of the 6
lead times. This directly contradicts the paper's intended headline
framing (diffusion model beats simpler baselines) and needs resolving
before any final numbers, not glossed over.
**Two hypotheses, NOT mutually exclusive, neither confirmed yet:**
  1. **Probability-granularity confound.** The CNN's PR-AUC input is a
     continuous sigmoid output. The diffusion model's, per `evaluate.py`,
     is `(ens_phys >= threshold).mean(axis=0)` with `n_members=10` —
     i.e. only 11 discrete probability levels. A coarser probability
     scale measurably lowers measured PR-AUC independent of true
     discriminative skill. Untested: whether increasing `--n_members`
     closes any of the gap.
  2. **"Double penalty" effect (the more likely, more interesting
     explanation).** Well-documented in precipitation/nowcasting
     literature (part of the standard motivation for generative over
     regression-style nowcasting models, e.g. Ravuri et al. 2021/DGMR):
     a model trained with a plain per-pixel loss under genuine
     positional uncertainty learns to hedge/blur toward wherever an
     event might plausibly be, which scores deceptively well on strict
     pointwise metrics (raw PR-AUC/CSI at zero spatial tolerance) by
     avoiding confidently-wrong exact-pixel predictions. A diffusion
     ensemble's individual members are sharp, physically-plausible
     realizations — the actual point of using one — but that is a
     harder pointwise-scoring target than a hedged/blurred field, even
     when it is the more honest representation of uncertainty. This
     connects directly to E2 (displacement is incoherent, not
     advective) — genuine, irreducible positional uncertainty is
     already established here, which is exactly the precondition for
     this effect.
**Diagnostic added, not yet run:** `evaluate_cnn.py` previously only
computed FSS at `scale=1` (pointwise, no spatial tolerance) — extended
to the same multi-scale `[1,2,4,8,16,32]` pixel neighbourhoods
`evaluate.py` already uses for the main model (`baseline_cnn_fss_vs_scale.png`,
matching layout).
**Also unverified:** both evaluations used `--config configs/evaluate.yaml`
per documented commands, so `test_roots`/`li_event_threshold`/`img_size`
*should* match — worth an explicit confirmation, not just an assumption,
before trusting either number.

**Update: FSS-vs-scale results in, via `compare_diffusion_vs_cnn_fss.py`.
Picture is MORE NUANCED than either hypothesis alone — partial support
for #2, but a large piece of the gap is scale-INVARIANT, which #2 alone
cannot explain:**
  - **p>0.5 (majority vote, matches #2's prediction):** CNN's small
    edge at 12km (-0.011) shrinks steadily and REVERSES to a diffusion
    advantage by 260km (+0.018). Textbook double-penalty signature.
  - **p>0.3:** CNN's edge shrinks with scale (-0.033 → -0.017) but
    never reverses within the tested range. Partial support.
  - **p>0.1 (loose threshold):** CNN's edge is large (~-0.15) and
    **essentially flat across every scale from 12km to 260km** — a
    scale-invariant gap is NOT what a positional/double-penalty problem
    predicts (that should shrink with spatial tolerance at every
    threshold, not just some). This points at a coverage/calibration
    difference, not a positional one, specifically at loose thresholds.
**New, more specific hypothesis (#3), motivated by the p>0.1 pattern:**
with `n_members=10`, thresholding at exactly p=0.1/0.3/0.5 corresponds
to exactly "≥1/10", "≥3/10", "≥5/10" members agreeing — p>0.1 is a
generous, union-like criterion (any single one of 10 stochastic
realizations firing counts as positive), which could inflate the
diffusion model's predicted-positive AREA well beyond the true event
area, independent of position. That inflation would hurt FSS by a
similar amount at every scale (matching the flat pattern), while p>0.5
(majority vote) is far less prone to this and behaves as #2 predicts.
**Tested directly** with `check_area_fraction_bias.py` — computes
predicted-vs-true positive-area fraction from data already saved in
both npz files (`pr_prob_t`/`pr_label_t`), no new diffusion sampling
run needed. Validated against synthetic data reproducing the
hypothesized mechanism (union-of-noisy-members inflating low-threshold
coverage far more than a calibrated classifier) before trusting it —
produces exactly the expected signature (diffusion's over-coverage
ratio roughly 2x the CNN's at the loose threshold, both converging
toward ~1x at strict thresholds). **Not yet run on real data.**
**If confirmed:** the concrete next test is re-running the diffusion
evaluation with a larger `--n_members` (e.g. 30-50) — more members
means "≥1/N" stops being dominated by rare single-member noise and
should converge toward a more genuine low end of the probability
distribution, closing the p>0.1 gap specifically if this hypothesis is
right.
**Not safe to report either model as "winning" in the paper until this
is resolved. If it holds up, the honest framing is genuinely
interesting for the paper either way: diffusion catches up/wins at
strict thresholds and larger spatial tolerance (real, literature-
grounded uncertainty representation), while the CNN's advantage at
loose thresholds may be substantially explained by ensemble-size
discretization rather than a deeper modeling deficiency** — worth
confirming with the n_members re-run before writing either claim.

**Update: n_members=30 re-run done (up from 10). Strong, mechanistically
clean confirmation of hypothesis #3:**
  - **Raw PR-AUC gap vs. CNN roughly halved at every lead time**
    (e.g. +10min: -0.048 -> -0.025; +60min: -0.064 -> -0.033) from
    tripling ensemble size alone, no model changes.
  - **FSS at scale=8 (the mid-scale CSV summary column) moved exactly
    where the mechanism predicts, and nowhere else:** p>0.1 ("≥1/N",
    the union-like criterion under test) +0.114 absolute FSS; p>0.3
    +0.016; p>0.5 (majority vote, not expected to depend on N) -0.009
    (noise-level). A criterion-specific, one-directional improvement
    exactly tracking which thresholds are theoretically sensitive to
    ensemble granularity is strong evidence for the mechanism, not
    just "the model got better."

**Update: full multi-scale + area-fraction re-check done at n_members=30
(via `compare_diffusion_vs_cnn_fss.py` and `check_area_fraction_bias.py`
against the new `plot_data.npz`). Question now RESOLVED, mechanism
understood, small residual remains:**
  - **p>0.3 and p>0.5:** CNN and diffusion now essentially tied across
    every spatial scale (deltas within ±0.01) — the earlier apparent
    CNN advantage at these thresholds is gone.
  - **p>0.1:** gap shrank from -0.153 to **-0.04, still flat across
    scale** — same signature (coverage, not positional), ~74% smaller,
    real residual remains.
  - **p>0.5 got very slightly WORSE for diffusion (not better) as
    members went 10->30, more so at larger scales** -- at first glance
    surprising, but the area-fraction data explains it as the SAME
    mechanism working in the other direction: diffusion's predicted
    area at p>0.5 dropped from 0.86x to 0.76x of true area as members
    increased. With few samples, a pixel with true probability just
    under 50% can cross "≥5/10" by pure sampling luck more easily than
    with "≥15/30" -- more members suppresses noise-driven threshold-
    crossing at BOTH ends of the range, which reads as "improvement"
    at the loose end (spurious coverage there was too high) and
    "regression" at the strict end (spurious coverage there was also
    somewhat too high, just less so). One bias-reduction mechanism,
    not two effects.
  - **Area-fraction ratios at n=30:** p=0.1 diffusion 2.28x / CNN 1.98x
    (was 2.64x/1.98x); p=0.3 diffusion 1.24x / CNN 1.12x (was
    1.37x/1.12x) -- both gaps roughly halved, consistent with the FSS
    changes.
**Conclusion:** the original "baseline beats diffusion" result is now
a well-characterized artifact of small ensemble size in threshold-based
verification, not a real modeling deficiency, with a small (-0.04),
same-mechanism residual at the loosest threshold only.
**Decision (2026-08-11): user is running n_members=50 next** (separate
`eval_ens_50` output dir) to see whether the residual p>0.1 gap
continues shrinking with diminishing returns, or plateaus -- outcome
undetermined as of this entry, will decide further action from result.
**Action for paper regardless of the n=50 outcome:** this is a
genuinely strong, citable methods point already — pointwise
verification metrics (PR-AUC, low-threshold FSS) are sensitive to
probabilistic-ensemble size in a specific, mechanistically-understood
way, distinct from genuine model skill; report ensemble-size
sensitivity alongside the headline comparison rather than a single
fixed-N number.
**Separate, higher-priority implication (see PAPER_TODO.md):** this
was discovered via the CNN-baseline diagnostic, but it applies to the
project's ESTABLISHED HEADLINE NUMBERS too (PR-AUC 0.819->0.589, and
the bootstrap CI margins vs. persistence/pysteps) — all computed at
the same `n_members=10` that just cost ~2-3 PR-AUC points for free.
Those may need regenerating at a larger n_members before being
reported as final.

**Update: n_members=50 run done. Clear diminishing returns, question
now fully settled:**
  - PR-AUC gap vs. CNN: n=10 mean -0.058 -> n=30 mean -0.029 (49%
    closed) -> n=50 mean -0.023 (a further 21% of the remainder
    closed). Tripling members (10->30) bought far more than the next
    1.67x (30->50) — the classic shape of a finite-sample bias
    shrinking with N, not a linear/open-ended effect.
  - FSS at scale=8: p>0.3 now fully resolved (gap ~0.006, noise
    level). p>0.1 keeps closing but slowing (0.539 -> 0.652 -> 0.666
    vs CNN's 0.692). p>0.5's small "regression" trend (0.747 -> 0.737
    -> 0.734) is now confirmed real and monotonic across three
    independent runs, not noise — same bias-suppression mechanism,
    just visible from the other side of the threshold range.
  - **Decision: not chasing n_members further.** Diminishing returns
    are unambiguous; further increases would cost significant compute
    for a small fraction of an already-small residual.

**FINAL SYNTHESIS (why the CNN baseline scored better, resolved):**
Two distinct, now well-quantified causes, not one:
  1. **Ensemble-size verification artifact** (explains most of the
     ORIGINAL gap). Small `n_members` makes the diffusion model's
     empirical probability coarse and biases threshold-crossing
     metrics upward at loose thresholds (union-like inflation) — a
     measurement artifact of the verification procedure, not a model
     deficiency. Now directly quantified via a controlled n_members
     sweep (10/30/50) with a mechanistic explanation (area-fraction
     analysis) and a clean diminishing-returns curve, not just an
     assumption.
  2. **A genuine, structural difference ensemble size cannot remove**
     (explains the residual). The CNN is trained with per-pixel BCE —
     literally the same quantity pointwise metrics measure. The
     diffusion model is trained on a different objective entirely
     (denoising score matching); its "probability" is a post-hoc
     ensemble construction, not a directly-optimized output. Under
     genuine positional uncertainty (E2), the Bayes-optimal strategy
     for minimizing per-pixel BCE is to hedge/blur across plausible
     locations — which pointwise metrics reward — while the diffusion
     model's sharp, physically-plausible individual realizations (the
     actual point of using a generative model) are a harder pointwise
     target even when more honest. This is why the picture flips at
     strict thresholds + large spatial tolerance, exactly where
     "sharp but uncertain" should beat "blurred but confident."
**Manuscript status:** promoted from a caveat to a genuine discussion-
section contribution — draft prose in `manuscript/discussion_cnn_baseline_comparison.md`.

**FINAL, statistically validated (not just point-estimate) confirmation
via `bootstrap_pr_auc_ci.py` at `n_members=50`, sequence-level paired
bootstrap, n_boot=1000 (see B1 for the full table):** the residual CNN
advantage IS real and statistically significant at every lead time
(-0.021 to -0.024, 95% CI excludes zero throughout, tight CI ±0.005-0.007
given the CNN's deterministic/no-sampling-variance side of the pairing)
— this is not noise that a larger bootstrap sample would wash out.
**But its SHAPE is the more important result:** unlike the model's
margin over persistence/pysteps_li/pysteps_ir, which all GROW
substantially with lead time (physical/classical extrapolation degrades
faster than the model), the model-vs-CNN gap is roughly CONSTANT across
the full 50-minute range tested. A fixed-magnitude, lead-time-
independent effect is exactly what the structural explanation (#2
above — training-objective mismatch) predicts, and is hard to explain
under a purely positional/motion-degradation account (which would be
expected to interact with lead time the way the other three
comparisons do). This distinguishes F4's mechanism empirically from
E1/E2's lead-time-dependent error growth, even though both ultimately
trace back to the same genuine positional uncertainty in the process.

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
