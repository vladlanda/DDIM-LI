# Citation Ledger

Tracks every citation used or planned for the manuscript: the claim it
supports, its DOI, and how that DOI was verified (per
`WRITING_RULES.md` rules 5 & 6). A citation may only be added to
`references.bib` and used in manuscript prose once its row here says
`VERIFIED`.

**Caveat applying to every VERIFIED row below:** verification in this
pass confirmed DOI -> title/venue/year match via `https://doi.org/<DOI>`
resolution or (where fetch was rate-limited) multiple independent
citing sources. It did NOT include a full character-by-character check
of author name lists/spelling against the original paper -- author
lists in `references.bib` were transcribed from search snippets and
should get one more pass against the actual paper page before final
submission, particularly for `price2024gencast` and `dai2025ddms`
which have long, multi-affiliation author lists.

## Verified citations (2026-08-14 verification pass)

| # | Claim it supports | Citation key | DOI | Verification method | Status |
|---|---|---|---|---|---|
| 1 | Closest comparator using LightGBM + aerosol features for lightning nowcasting; motivates LightGBM baseline framing (FINDINGS.md F1, C8) | `song2023lightning` | 10.1038/s41612-023-00451-x | Search snippet shows exact self-citation on the paper's own npj page ("npj Climate and Atmospheric Science (2023) 6:126 ; https://doi.org/10.1038/s41612-023-00451-x") | VERIFIED |
| 2 | Closest comparator CNN baseline for satellite-based lightning nowcasting; motivates CNN baseline framing (FINDINGS.md F3) | `cintineo2022lightningcast` | 10.1175/WAF-D-22-0019.1 | Multiple independent sources (AMS journal page, NOAA repository) agree on DOI/title/venue | VERIFIED |
| 3 | GenCast; independent corroboration of the blur/hedging mechanism (FINDINGS.md F4); diffusion model predicting residuals, same EDM framework | `price2024gencast` | 10.1038/s41586-024-08252-9 | User-provided URL, directly resolved earlier this session; title/venue confirmed | VERIFIED (author list needs one more pass, see caveat above) |
| 4 | Closely related diffusion+satellite convective nowcasting paper; required related-work citation (FINDINGS.md F3b) | `dai2025ddms` | 10.1073/pnas.2517520122 | Multiple independent sources (PNAS page, PubMed, HKUST research portal) agree on DOI/title/venue | VERIFIED (author list needs one more pass, see caveat above) |
| 5 | EDM/Karras diffusion framework, the basis for this project's own diffusion model architecture | `karras2022edm` | 10.48550/arXiv.2206.00364 | NeurIPS proceedings version has no standard Crossref DOI (confirmed -- NeurIPS papers generally don't). arXiv preprint DOI cross-verified via a Springer Nature-published book chapter citing the identical DOI string, plus Semantic Scholar. Direct doi.org fetch was rate-limited, not attempted successfully -- retry before final submission if possible | VERIFIED (via cross-citation, not direct resolution -- see note) |
| 6 | CRPS metric and its reliability/resolution/uncertainty decomposition, used throughout evaluate.py | `gneiting2007scoring` | 10.1198/016214506000001437 | Confirmed consistently across many independent citing sources (Taylor & Francis publisher page, multiple arXiv papers, CRAN package docs) | VERIFIED |
| 7 | Fractions Skill Score (FSS), used throughout evaluate.py/evaluate_cnn.py/evaluate_lightgbm.py for spatial-tolerance verification | `roberts2008fss` | 10.1175/2007MWR2123.1 | Confirmed consistently across many independent sources including the AMS journal's own page and multiple follow-on FSS papers (ECMWF, Mittermaier et al.) | VERIFIED |
| 8 | Closest comparator paper for the CNN baseline's exact architecture/training choices (FINDINGS.md F3, C6/C7); also strengthens AIES journal candidacy | `metzl2025physical` | 10.1175/AIES-D-25-0035.1 | Confirmed via AMS journal's own page (title, abstract, "Artificial Intelligence for the Earth Systems Volume 4 Issue 4 (2025)" match exactly) -- also resolved status update: previously recorded as "in review", now confirmed published | VERIFIED |

## Known claims still needing a citation search (not yet started)

These are claims already made in `FINDINGS.md`/session discussion that
will appear in the manuscript and need their own citation-and-DOI-
verification pass before drafting the relevant section. Listed here so
a future session doesn't have to reconstruct this list from scratch.

| Claim | Likely citation direction | Status |
|---|---|---|
| Central Africa has among the highest lightning flash-rate densities globally (motivates region choice in Introduction) | Christian et al. 2003 (LIS/OTD climatology) or a more recent WWLLN-based climatology | NOT YET SEARCHED |
| MTG-FCI / MSG-SEVIRI instrument description (Methods, data section) | EUMETSAT technical documentation or a Schmetz-et-al.-style instrument description paper (note: Metzl et al. 2025's own reference list cites Schmetz et al. 2002 for MSG -- worth checking if that's the right citation for our MTG-FCI description too, or if a separate MTG-specific citation is needed since MTG is a different generation than MSG) | NOT YET SEARCHED |
| pySTEPS / optical-flow extrapolation methodology (Methods, baselines) | Pulkkinen et al. 2019 (pySTEPS GMD paper) -- name recalled from training data, DOI NOT YET VERIFIED, do not cite without verification pass | NOT YET SEARCHED |
| LightGBM algorithm itself (Methods, LightGBM baseline) | Ke et al. 2017 (NeurIPS) -- name recalled from training data, DOI NOT YET VERIFIED, do not cite without verification pass | NOT YET SEARCHED |
| Aerosol invigoration of convection / cloud microphysics mechanism (Discussion, future-work paragraph on aerosol data, if included) | Rosenfeld et al. or Altaratz et al. review-type citations discussed conversationally this session -- NONE of these have been searched or DOI-verified, currently just names from training-data recall | NOT YET SEARCHED |
| "Double penalty" effect in spatial verification / motivation for FSS-style metrics over pointwise metrics (Discussion, F4 mechanism) | Possibly folds into the Roberts & Lean 2008 citation already verified, or may need an additional citation specifically on the double-penalty problem (e.g. work by Baldwin & Kain, or Ebert 2008/2009 review of spatial verification methods) | NOT YET SEARCHED |
| U-Net architecture (Methods, CNN baseline architecture description) | Ronneberger et al. 2015 -- name recalled from training data (also appears in Metzl et al.'s own reference list per the search results this session), DOI NOT YET VERIFIED here | NOT YET SEARCHED |
| DDPM / diffusion models foundational citation, if the Introduction wants a citation predating Karras et al. for the general diffusion-model concept | Ho et al. 2020 (DDPM) or Sohl-Dickstein et al. 2015 -- NOT YET SEARCHED |

**Process reminder:** before any of the above gets used in drafted
prose, move it through the same search -> cross-check -> doi.org
resolution -> ledger `VERIFIED` -> `references.bib` pipeline as the
8 entries above. Do not draft a sentence citing any of these from
memory alone.
