# Path to Submission — Checklist

Working document tracking what's needed before submission. Update status
inline as items complete. See conversation history for full rationale
behind each item.

**Target venues, revisited 2026-08-14 given the finalized results (see
FINDINGS.md F3b/F4 for the reasoning):**
1. **npj Climate and Atmospheric Science** (primary, unchanged) — direct
   precedent (Song et al. 2023, same subfield), Q1, IF ~8-9.6 (rising),
   explicit scope match: journal's own stated focus includes "regional
   studies which demonstrate new understanding of a particular
   locality" — exactly this paper's shape (Central Africa + the E2
   mechanistic finding). ~21wk avg review, APC ~$2,990-3,290, OA.
2. **NEW candidate, seriously worth adding: AMS's "Artificial
   Intelligence for the Earth Systems" (AIES).** Arguably a BETTER fit
   than the original backups for this specific paper's balance of
   contributions — AIES reviewers are specifically ML-for-geoscience
   people who would treat the baseline-verification-methodology work
   (FINDINGS.md F4: CNN/LightGBM/ensemble-size/spatial-tolerance
   analysis, corroborated by GenCast) as a first-class contribution,
   not a secondary methods detail. Direct subfield precedent: Leinonen
   et al. 2022 (deep learning lightning nowcasting, Switzerland)
   published there. AMS's "Weather and Forecasting" (where LightningCast/
   Cintineo et al. 2022 was published) is a reasonable alternative in
   the same family if AIES's ML-methods framing doesn't fit as well as
   expected once drafting starts.
3. **IEEE TGRS** — demoted from "parallel" to a secondary option.
   Reasoning: still a legitimate rigorous venue and would likely
   appreciate the statistical rigor, but its focus (general remote-
   sensing/signal-processing methodology) fits this paper's specific
   nowcasting-verification contribution less precisely than AIES does.
   Not dropped — just no longer co-equal with npj Clim Atmos Sci.
4. **Atmospheric Research** — kept as the safe fallback, unchanged.

Cross-checked against a predatory/low-quality publisher blocklist — all
clean. Two near-miss confusions to avoid: IEEE Access (flagged, pay-to-
publish mega-journal) is NOT IEEE TGRS; Scientific Reports (flagged,
Nature's no-novelty-bar mega-journal) is NOT npj Climate and Atmospheric
Science.

**IMPORTANT related-work addition (FINDINGS.md F3b):** Dai et al. 2025
(PNAS), "Four-hour thunderstorm nowcasting using a deep diffusion model
for satellite data" (DDMS) — a closely-related diffusion+geostationary-
satellite convection-nowcasting paper, found while researching journal
targets. NOT a scoop (different target variable — general convection vs.
lightning specifically; different scope — global/4h vs. regional/1h;
no analog to our E2 finding or F4's verification-methodology depth) but
MUST be cited with an explicit differentiation paragraph in the
Introduction/related work, regardless of which journal is chosen — a
reviewer at any of the above venues will likely know this paper. Only
skimmed via abstract/snippets so far — read in full before drafting
that paragraph.

**Current best model:** `outputs/nature_256_T36_ir_li_only` — 2-channel
(ir105 + li), 790 epochs, trained on the regenerated (clean) dataset,
evaluated at `n_members=50` (see Phase 2 / FINDINGS.md F4 — n_members=10
measurably understated performance, now resolved and finalized).
PR-AUC 0.846 → 0.631 (+10 to +60min), beats fresh persistence baseline
with margin growing from +8.1% to +61.6%. All five baseline comparisons
(persistence, pysteps_li, pysteps_ir, CNN, LightGBM) are sequence-level
bootstrap-CI significant at every lead — see FINDINGS.md B1.

---

## Phase 1 — Resolve the channel/data confound

- [x] 2-channel model (ir+li) trained on clean/regenerated dataset, 790
      epochs. Result: beats fresh persistence at every lead, margin grows
      with lead time. **Done.**
- [x] Fresh persistence baseline on regenerated dataset — matches old
      (stale-data) persistence numbers almost exactly (diff < 0.001 at
      every lead), confirming test-period climatology is unaffected by
      the coverage bug. **Done.**
- [ ] **DEFERRED — TODO:** 4-channel config (ir, li, ch0, ch1) retrained
      on the SAME regenerated/clean dataset, 300 epochs.
      Rationale: isolates whether the 2ch-vs-old-4ch improvement came
      from channel pruning, the data-quality fix, or both. Currently
      have 4ch@300-on-OLD-data vs 2ch@790-on-NEW-data (three confounded
      variables: channels, dataset, epochs). This run removes the
      dataset confound; epoch-count asymmetry remains but is
      interpretable (see conversation for the 3-case reading guide).
      **Owner: user (GPU). Blocks: final model selection, some framing
      in Phase 3/4, but NOT Phase 2 items below.**
- [ ] Check for an intermediate ~epoch-300 checkpoint from the existing
      2ch@790 run, if one was saved — would give a fully epoch-matched
      comparison at zero extra compute. Worth a quick look before
      prioritizing the run above.
- [x] Training-curve / convergence check for 2ch@790 — deferred by user
      decision (epoch-count argument from the 4ch@300-old vs 2ch@790-new
      near-match accepted as sufficient; not worth a dedicated compute
      run). Noted as a minor caveat for the manuscript, not a blocker.

## Phase 2 — Close reviewer-critical gaps (not blocked by Phase 1)

- [x] **n_members sensitivity — RESOLVED.** Discovered while diagnosing
      FINDINGS.md F4 (CNN baseline vs diffusion model FSS/PR-AUC): the
      original headline PR-AUC (0.819->0.589) and bootstrap CI margins
      were computed at undersized `n_members=10`. Swept 10/30/50, found
      clear diminishing returns (10->30 closed 49% of the gap vs the CNN
      baseline, 30->50 closed a further 21% of the remainder) and
      settled on **n_members=50** as final. Reran the full bootstrap CI
      comparison (`bootstrap_pr_auc_ci.py`, n_boot=1000) at this setting
      against all four baselines — new FINAL headline numbers: PR-AUC
      0.846->0.631, all 24 comparisons (4 baselines x 6 leads)
      significant. See FINDINGS.md A1/A2/B1/F4 for full detail.
- [x] Baseline beyond persistence — IMPLEMENTED. Optical-flow (pySTEPS-
      style semi-Lagrangian extrapolation) baseline, native implementation
      (Farneback dense flow + cv2.remap backward advection, no pysteps
      dependency). Two variants per literature convention:
        - `pysteps_li/run.py` (PRIMARY): flow derived from LI itself,
          advecting LI — the standard convention (flow-source == forecast
          target), matching how precip pySTEPS baselines use radar
          reflectivity for both. Expected to perform poorly given LI
          sparsity — a real, literature-consistent finding, not a bug.
        - `pysteps_ir/run.py` (SECONDARY/robustness): flow derived from
          the denser IR channel, advecting LI — gives optical flow its
          strongest reasonable chance. Direct precedent in severe-
          convection nowcasting literature (one motion field from the
          primary field, applied to advect several target fields).
      Output schema mirrors persistence_metrics.csv exactly for direct
      three-way comparability (persistence / pysteps_li / pysteps_ir).
      Core flow/advection math validated on synthetic moving-blob data:
      exact velocity recovery, zero position error after multi-step
      advection, correct stationary/k=0 edge cases.
      **Owner: user — run both variants on the test set.**
      ```
      python pysteps_li/run.py --config configs/evaluate.yaml
      python pysteps_ir/run.py --config configs/evaluate.yaml
      ```
- [x] Sequence-level bootstrap CIs — DONE. All 18 comparisons (3 baselines
      × 6 leads) significant, margin grows monotonically with lead time
      against all three baselines. See FINDINGS.md B1 for the numbers.
- [x] Re-run FSS-threshold + positional-ceiling error decomposition on
      the FINAL model — DONE. best_shift confirmed the incoherent-
      displacement finding (E2). FSS at prob_thr 0.1/0.5 confirmed and
      refined the dual-regime finding (E1): high-confidence predictions
      are positionally accurate at every lead; low-confidence predictions
      show a graduated transition into existence-dominated failure by
      +50/+60min. See FINDINGS.md E1/E2 for full detail.

## Phase 3 — Strengthen for top-of-range venues (npj / TGRS)

- [x] Deterministic CNN baseline — IMPLEMENTED (`baseline_cnn/`), ready
      to train. LIGHTWEIGHT config (~1.45M params, matching Metzl et al.
      2025's ~1.6M BNN scale — not the diffusion model's full ~22M-param
      capacity, which was the initial version but too expensive to train
      and not actually a faithful literature reproduction anyway):
      base_channels=24, channel_mults=[1,2,2,4], no attention, emb_dim=64.
      Reuses model.py's UNet class purely via hyperparameters (no new
      architecture code). Single-channel logit output, amortized across
      lead times via the same lead-time conditioning as the main model.
      Output schema matches the other baselines for bootstrap-CI
      comparability. Verified end-to-end at full 256x256 resolution and
      through the real CLI/config default path (not just hand-built args).
      A real numerical-stability bug (sigmoid+BCE instead of BCE-with-
      logits, verified empirically to cause ~1e-10x vanishing gradients
      at plausible mid-training logit magnitudes) was caught and fixed
      during a dedicated pre-training review pass — see commit history.
      **`run1` (first real training run) showed val_loss growing after
      epoch ~10-11.** Root cause: `train_cnn.py`'s training recipe had
      silently diverged from Metzl et al. 2025 (the literature baseline
      this is benchmarked against) — no `weight_decay` wired into Adam
      at all, and a fixed `CosineAnnealingLR` instead of anything that
      responds to validation performance. **Fixed**: `weight_decay=1e-4`
      and `ReduceLROnPlateau(factor=0.1, patience=5, cooldown=3)`, both
      taken directly from Metzl et al. 2025's own reported recipe, plus
      a practical early-stopping safeguard (`--early_stop_patience`).
      Also added optional W&B logging (`--wandb_project`, run name from
      `output_dir` basename, matching `train.py`'s convention for the
      main model). See FINDINGS.md C6/C7 for the full diagnosis,
      including which OTHER literature departures (context window,
      lead-time amortization) were deliberately kept, and why.
      **`run1`'s checkpoints predate this fix — re-run before trusting
      any CNN-baseline numbers for the paper.**
      **Owner: user — train, then evaluate:**
      ```
      python baseline_cnn/train_cnn.py --config configs/default.yaml \
          --epochs 150 --output_dir baseline_cnn/outputs/run2 \
          --wandb_project DDIM-LI

      python baseline_cnn/evaluate_cnn.py --config configs/evaluate.yaml \
          --checkpoint baseline_cnn/outputs/run2/best.pt \
          --output_dir baseline_cnn

      python bootstrap_pr_auc_ci.py \
          --npz outputs/nature_256_T36_ir_li_only/eval/plot_data.npz \
          --baseline cnn:baseline_cnn/baseline_cnn_pr_curves.npz \
          --label model --dt_min 10 --n_boot 1000
      ```
- [x] **Fundamental data-loading speedup: memmap-packed data pipeline.**
      Profiling traced 1h/epoch to per-file JPEG-open/decode overhead
      (~84 individual files per training sample), not model compute or
      raw disk bandwidth — confirmed by num_workers tuning plateauing
      (best 29.4 min/epoch at 6-8 workers) and regressing at 16 (I/O
      contention signature). Built `preprocess_to_memmap.py` (one-time,
      per-region) + `dataset_packed.py` as a fully separate path from
      `dataset.py` — zero risk to the existing, validated pipeline.
      Rigorously validated: field-by-field output comparison across all
      sequences of a synthetic test (including a genuinely-missing
      optional channel, to exercise the fill-value/normalization-skip
      edge case) shows zero mismatches; full `make_dataloaders` vs
      `make_dataloaders_packed` pipeline comparison shows identical
      train/val splits and EXACTLY matching density-based oversampling
      thresholds. ~4x faster even on a tiny synthetic test; real data
      (T_in=36, far more files eliminated per sample) should show more.
      **This is general infrastructure, not baseline_cnn-specific** — the
      same approach would speed up the main diffusion model's training
      too, if adopted there later.
      **Owner: user — preprocess once per region, then train:**
      ```
      python preprocess_to_memmap.py --root <region_path> \
          --channels ir li --img_size 256 256

      python baseline_cnn/train_cnn.py --config configs/default.yaml \
          --train_roots <region1> <region2> <region3> <region4> \
          --packed_dirs <region1>/_packed <region2>/_packed \
                        <region3>/_packed <region4>/_packed \
          --epochs 150 --output_dir baseline_cnn/outputs/run1
      ```

- [x] LightGBM baseline (`baseline_lightgbm/`) — COMPLETE: trained,
      evaluated, and bootstrap-CI-significant results locked in.
      Scope decision: LightGBM only (not also XGBoost —
      both are gradient-boosted trees, building both adds tuning burden
      without additional scientific insight). Trained/evaluated AT OUR
      OWN task resolution (4km/10min, pixel-exact), NOT a literal
      reproduction of Song et al.'s 0.25°/hourly protocol (that would
      reopen the base-rate/resolution non-comparability problem already
      flagged in F1). Framing: "a gradient-boosted-tree baseline in the
      methodological spirit of the most relevant prior npj publication,
      evaluated on our task" — legitimate and citable without
      overclaiming a direct number-to-number comparison with their 0.727.
      **What's built:** `features.py` (18 hand-verified, vectorized
      features: IR temporal/spatial stats, LI activity/recency/local-
      density stats, lead_idx as a feature for single-model amortization
      across lead times — same reasoning as the CNN baseline's C7),
      `train_lightgbm.py` (reuses `dataset.py`'s `make_dataloaders`
      directly — same train/val split, stats, channels as everywhere
      else; `is_unbalance=True` for class imbalance, not manual
      oversampling), `evaluate_lightgbm.py` (identical output schema to
      `evaluate_cnn.py` — same npz/CSV structure, works with
      `bootstrap_pr_auc_ci.py` and `compare_diffusion_vs_cnn_fss.py`
      unmodified via `--baseline lightgbm:...`). Reuses
      `evaluate_cnn.py`'s `_make_plots`/`_li_to_physical` directly rather
      than duplicating ~250 lines (that function was parameterized with
      `filename_prefix`/`display_name` specifically to support this
      reuse without mislabeling LightGBM's own plots as "CNN Baseline").
      Validated via synthetic data throughout (hand-verified feature
      values against constructed known scenarios, full train->evaluate
      pipeline run end-to-end with mocked data loaders) before trusting
      any of it, per this project's validate-before-trusting practice.
      Caught and fixed two real bugs during that validation: a 6x
      redundant spatial-filter computation per sequence (refactored into
      a two-step compute-once/reuse-per-lead path), and a full-image
      validation-set default that would have needed ~10-20GB of RAM
      (fixed with a separate, bounded `--val_pixels_per_image_lead`).
      **Owner: user (GPU/data access) — train, then evaluate:**
      ```
      python baseline_lightgbm/train_lightgbm.py \
          --config configs/default.yaml \
          --output_dir baseline_lightgbm/outputs/run1

      python baseline_lightgbm/evaluate_lightgbm.py \
          --config configs/evaluate.yaml \
          --model_dir baseline_lightgbm/outputs/run1 \
          --output_dir baseline_lightgbm

      python bootstrap_pr_auc_ci.py \
          --npz outputs/nature_256_T36_ir_li_only/eval_ens_50/plot_data.npz \
          --baseline persistence:outputs/persistence_baseline/persistence_pr_curves.npz \
          --baseline pysteps_li:pysteps_li/optical_flow_li_pr_curves.npz \
          --baseline pysteps_ir:pysteps_ir/optical_flow_ir_pr_curves.npz \
          --baseline cnn:baseline_cnn/baseline_cnn_pr_curves.npz \
          --baseline lightgbm:baseline_lightgbm/baseline_lightgbm_pr_curves.npz \
          --label model --dt_min 10 --n_boot 1000
      ```
      **Result (first real run, PR-AUC): diffusion model 0.846→0.631
      beats LightGBM 0.828→0.446 at EVERY lead, margin GROWING
      (+0.018→+0.185) — same shape as the physical baselines, NOT the
      CNN's flat/losing pattern. LightGBM does beat all three physical
      baselines, so it's a real, useful middle-tier baseline. See
      FINDINGS.md C8 for the full result and the information-access
      explanation (matched training objective alone, per LightGBM,
      isn't enough to reproduce the CNN's narrow edge — matched raw
      information access, which only the CNN has, is what mattered).**
      **Still to do (final significance numbers now done — see
      FINDINGS.md B1):** ~~run through `bootstrap_pr_auc_ci.py`~~ DONE.
      All 6 leads significant vs. model, growing margin +0.018→+0.185,
      same shape as physical baselines. LightGBM baseline work is now
      complete pending manuscript writeup.

- [ ] Loss-term ablation, scoped down (full loss vs. denoising-only, not
      a full factorial grid). Lower priority than the CNN/LightGBM
      baselines above — those close real reviewer-expectation gaps;
      this strengthens an already-solid methods section further.
      **Owner: user (run) + me (design).**
- [ ] Generalization check: held-out region (train on 3 of 4 regions,
      test on the 4th). **Owner: user (run) + me (design the split).**

## Phase 4 — Manuscript

- [ ] Related-work positioning: LightningCast (Cintineo 2022), Song et
      al. 2023 (npj), DDMS (PNAS 2025), SATcast (2025), Li et al. 2023.
      Can start now, doesn't depend on pending runs.
- [ ] Figures: PR-AUC/calibration panel, error-decomposition figures,
      example forecast panels. Tooling mostly exists (plot_pr_comparison.py,
      plot_forecast, diagnose_*.py).
- [ ] Methods / Results / Discussion sections.
- [ ] Data availability + reproducibility statement (MTG-FCI data access).

---

## Known confounds / honest caveats to carry into the manuscript

- Channel-pruning result (Phase 1) not yet cleanly isolated from the
  concurrent dataset regeneration — see Phase 1 TODO above.
- All four regions are geographically adjacent crops of one domain
  (Central Africa) — no tested generalization to a distinct climate
  regime yet (Phase 3).
- Ensemble inference cost (10 members × diffusion sampling) not yet
  benchmarked against baseline latency for an operational-relevance claim.
