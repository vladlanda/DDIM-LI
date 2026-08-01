# Path to Submission — Checklist

Working document tracking what's needed before submission. Update status
inline as items complete. See conversation history for full rationale
behind each item.

**Target venues:** npj Climate and Atmospheric Science (primary), IEEE TGRS
(parallel), Atmospheric Research (fallback).

**Current best model:** `outputs/nature_256_T36_ir_li_only` — 2-channel
(ir105 + li), 790 epochs, trained on the regenerated (clean) dataset.
PR-AUC 0.819 → 0.589 (+10 to +60min), beats fresh persistence baseline
with margin growing from +4.6% to +50.9%.

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
- [x] Sequence-level bootstrap CIs — INFRASTRUCTURE READY. All three
      evaluation scripts now tag pixels with their source test-sequence
      ID and save it to their npz output; `bootstrap_pr_auc_ci.py` does
      paired sequence-level (not pixel-level) bootstrap on any
      model-vs-baseline comparison. A real bug was caught and fixed
      during validation (mask-based resampling silently collapsed
      duplicate draws, understating variance) — see commit for details.
      **Still needed: re-run evaluate.py / persistence_baseline.py /
      pysteps_li / pysteps_ir once more so their npz outputs contain the
      new pr_seqid_* arrays (the CSVs you already have don't need
      re-running, only the npz side needs regenerating), then run:**
      ```
      python bootstrap_pr_auc_ci.py \
          --npz outputs/nature_256_T36_ir_li_only/eval/plot_data.npz \
          --baseline persistence:outputs/persistence_baseline_new/persistence_pr_curves.npz \
          --baseline pysteps_li:pysteps_li/optical_flow_li_pr_curves.npz \
          --baseline pysteps_ir:pysteps_ir/optical_flow_ir_pr_curves.npz \
          --label model --dt_min 10 --n_boot 1000
      ```
- [ ] Re-run FSS-threshold + positional-ceiling error decomposition on
      the FINAL model (currently only validated on an older model/data).
      Existing diagnostic scripts (diagnose_fss_scale.py,
      diagnose_positional_ceiling.py) — just need a fresh run.
      **Owner: user (run) + me (interpret).**

## Phase 3 — Strengthen for top-of-range venues (npj / TGRS)

- [ ] Loss-term ablation, scoped down (full loss vs. denoising-only,
      not a full factorial grid). **Owner: user (run) + me (design).**
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
