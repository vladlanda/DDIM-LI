# Introduction

Status: NOT DRAFTED. Outline only, per MANUSCRIPT_PLAN.md Section 1.

## Planned flow (each point needs citation-verified support before drafting -- see citations_ledger.md)

1. Lightning as a hazard -- aviation, ground operations, wildfire
   ignition. NEEDS CITATION (not yet searched).
2. Central Africa: among the highest lightning flash-rate densities
   globally; understudied relative to CONUS/Europe in the ML-nowcasting
   literature. NEEDS CITATION (not yet searched, likely Christian et
   al. 2003 LIS/OTD climatology or a more recent WWLLN-based one).
3. Prior work, in this order:
   a. Physical extrapolation (persistence, optical flow) and its
      limitations under genuine motion uncertainty.
   b. Deterministic ML: Cintineo et al. 2022 (LightningCast), Song et
      al. 2023 (aerosol-informed LightGBM), Metzl et al. 2025 (ResU-Net,
      scale argument for advection).
   c. Generative/probabilistic ML for related tasks: GenCast (Price et
      al. 2024), Dai et al. 2025 (DDMS), general diffusion-nowcasting
      literature for precipitation.
4. Gap statement: no diffusion-based lightning-specific nowcasting
   system with this level of baseline-verification rigor; open
   question of displacement coherence at convective scale.
5. **Required explicit differentiation paragraph vs. Dai et al. 2025**
   (FINDINGS.md F3b) -- clarify target variable (lightning vs. general
   convection), scope (regional/1h vs. global/4h), and contribution
   type (mechanistic + verification-methodology vs. systems/
   engineering). 2-3 sentences, integrated into the prior-work
   discussion, not a bolted-on footnote.
6. Brief statement of this paper's two headline contributions:
   (i) performance vs. physical baselines, (ii) the incoherent-
   displacement finding (E2). Do not list the baseline-verification-
   methodology contribution (F4) as a third headline claim here --
   reserve it for Discussion.

## Notes
- Read Dai et al. 2025 in full before drafting point 5 (only
  abstract/snippets seen so far, per FINDINGS.md F3b).
- Confirm which journal before finalizing tone -- npj vs. AIES framing
  may differ slightly in how much the ML-methodology angle gets
  foregrounded here vs. saved for Discussion.
