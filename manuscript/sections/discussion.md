# Discussion

Status: PARTIALLY DRAFTED elsewhere -- see
`manuscript/discussion_cnn_baseline_comparison.md` (full draft prose
for the F4/baseline-verification portion, written before these rules
were established; still usable, just needs to be merged into this
file's structure and re-checked against WRITING_RULES.md rule 6 for
citation DOIs before final use). This file is the outline for the
REST of Discussion, not yet drafted.

## Planned flow

1. **Interpret the incoherent-displacement finding (E2).** Physical
   implications: why advection-based methods (persistence, optical
   flow) structurally cannot close the gap to ML approaches at these
   lead times. Connect to Metzl et al. 2025's scale argument (F3) --
   their crossover point (>2h) vs. our entire tested horizon (60min)
   being below it, as independently-derived corroboration.

2. **Interpret the baseline-verification story (F4).** Merge in
   `discussion_cnn_baseline_comparison.md`'s existing draft here. Key
   points already drafted: ensemble-size artifact (mostly resolved),
   the information-access explanation for the CNN's residual edge,
   GenCast corroboration, LightGBM as a confirmatory (not replicating)
   contrast case.

3. **Limitations**, stated plainly, not buried:
   - Single contiguous domain (4 adjacent regions) -- no tested
     generalization to a distinct climate regime yet.
   - n_members=50 chosen via a diminishing-returns sweep (10/30/50),
     not exhaustively tested beyond that -- can't rule out a small
     further narrowing of the CNN gap at much larger ensemble sizes.
   - Ensemble inference cost not yet benchmarked against baseline
     latency for an operational-relevance claim.

4. **Future work:**
   - Aerosol data (MSG/SEVIRI, discussed but explicitly scoped OUT of
     this paper) -- Central Africa's biomass-burning aerosol regime as
     a distinct setting from Song et al.'s CONUS study; frame as a
     compelling follow-up, not a claim of results we don't have.
   - Held-out-region generalization test.
   - Operational deployment considerations.

5. **Broader significance:** early-warning implications for the
   region, given the established hazard context from the Introduction.

## Notes
- Do not overclaim: this Discussion should read as confident about
  what was actually shown (E2, the baseline-verification mechanism)
  and explicitly honest about what wasn't (generalization,
  aerosol effects, exhaustive ensemble-size testing).
- Every citation used when merging in the existing draft must be
  re-verified against citations_ledger.md before this file is
  considered final -- that draft predates the DOI-verification rule.
