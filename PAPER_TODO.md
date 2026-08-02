# Path to Submission — Checklist

Working document tracking what's needed before submission. Update status
inline as items complete. See conversation history for full rationale
behind each item.

**Target venues:** npj Climate and Atmospheric Science (primary), IEEE TGRS
(parallel), Atmospheric Research (fallback). Cross-checked against a
predatory/low-quality publisher blocklist — all three clean. Note two
near-miss confusions to avoid: IEEE Access (flagged, pay-to-publish
mega-journal) is NOT IEEE TGRS (our recommendation, established subject
journal, different venue despite shared publisher); Scientific Reports
(flagged, Nature's no-novelty-bar mega-journal) is NOT npj Climate and
Atmospheric Science (our recommendation, curated Nature Portfolio
subject journal — where Song et al. 2023 was published).

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

- [ ] **NEW, higher priority: deterministic CNN baseline.** Literature
      check (see FINDINGS.md F3 — Metzl et al. 2025, the closest
      comparator paper found) shows the realistic convention in this
      specific subfield always includes a trained deep-learning baseline
      (a plain CNN, no advection/diffusion) alongside physical baselines
      — not persistence + optical-flow alone. LightningCast itself has
      also become a de facto reference point for later papers in this
      space. Concrete, scoped plan: reuse our existing UNet architecture
      (drop the diffusion machinery, single deterministic forward pass,
      plain BCE or regression loss, one model amortized across lead
      times via the existing lead-time conditioning — no need to retrain
      per-lead-time as Metzl et al. do). This is real training time, not
      a free re-run, but is the single highest-value addition to the
      baseline set for a reviewer at our target venues.
      **Owner: me (design + implement) + user (train + run).**
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
