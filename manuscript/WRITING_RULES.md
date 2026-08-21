# Manuscript Writing Rules

Standing rules for the DDIM-LI manuscript, set by the user on 2026-08-14,
targeting npj Climate and Atmospheric Science (primary) / AIES
(secondary, see PAPER_TODO.md). Apply these in every future writing
session on this manuscript, not just the one where they were set.

## The rules, as given

1. Plan the manuscript structure before writing prose.
2. List all necessary figures needed — think beyond what already
   exists (results plots, model architecture) to what a reader needs,
   e.g. qualitative nowcasting examples.
3. Create a `.ipynb` notebook to generate the figures — each code
   section/cell produces one figure.
4. Content (manuscript prose) and citations live in different files,
   not interleaved.
5. Every claim we make must be checked against real references/papers
   that back it up — no unsupported assertions in a submitted
   manuscript.
6. **IMPORTANT**: every citation's DOI must be verified correct by
   resolving `https://doi.org/<DOI>` and confirming it actually
   resolves to the claimed paper (title/authors/venue match) — not
   just trusted from memory, a search snippet, or a citation someone
   else's bibliography. Example format: `https://doi.org/10.3897/rio.7.e67379`.
7. Write using the language, tone, and structural conventions actually
   used in npj publications — not generic scientific writing, and not
   AMS/AGU conventions if they differ (see "Style notes" below for the
   concrete differences that matter).

## File layout this implies

**UPDATED 2026-08-21: manuscript content is authored in LaTeX, not
Markdown** (see "Format decision" below for why). `manuscript/sections/
*.md` are now historical drafts, superseded by `manuscript/latex/
main.tex` -- their content was transcribed into main.tex already, but
if you're picking this up fresh, treat main.tex as the single source of
truth and the .md section files as reference/backup only.

```
manuscript/
  WRITING_RULES.md              <- this file
  MANUSCRIPT_PLAN.md            <- structure + full figure list (rule 1, 2)
  references.bib                <- BibTeX, one entry per citation (rule 4)
  citations_ledger.md           <- claim -> citation -> DOI verification (rule 5, 6)
  generate_figures.ipynb        <- one figure per section (rule 3)
  figures/                      <- .png outputs from the notebook
  latex/
    main.tex                    <- AUTHORITATIVE manuscript source (single file,
                                    per Springer Nature's own submission
                                    requirement -- no \input of separate
                                    section files)
    main.pdf                    <- last compiled output, committed so the
                                    formatted paper is viewable without a
                                    local LaTeX toolchain
    main.bbl                    <- compiled bibliography (regenerate via
                                    bibtex if references.bib changes)
    sn-jnl.cls, sn-nature.bst   <- Springer Nature's official template files,
                                    "sn-nature" style = the option specifically
                                    for Nature Portfolio journal submissions
  sections/                     <- SUPERSEDED, historical drafts only
    abstract.md
    introduction.md
    results.md                  <- transcribed into main.tex already
    discussion.md
    methods.md                  <- transcribed into main.tex already
  discussion_cnn_baseline_comparison.md   <- existing draft, predates these rules;
                                              not yet merged into main.tex
  reviewer_premortem_cnn_lightgbm_baselines.md   <- existing, predates these rules
```

## Format decision: LaTeX, not Markdown (added 2026-08-21)

Sections were originally drafted in Markdown, matching this project's
existing documentation (FINDINGS.md, PAPER_TODO.md, etc.). User asked
directly why not author in LaTeX given the actual npj submission target
-- reconsidered and switched, for concrete reasons, not just preference:

- npj's own submission guidelines (verified via nature.com, not
  assumed): initial submission wants a compiled PDF or Word file;
  LaTeX source is accepted (and Springer Nature explicitly supports it,
  including for the npj series specifically) at the acceptance stage.
  Authoring in LaTeX from the start produces both artifacts from one
  source -- compiles directly to the PDF needed now, and is already the
  right format for later -- rather than drafting in Markdown and having
  to convert/reformat at acceptance.
- `references.bib` already existed (built for rule 4) -- LaTeX's
  `\cite{}` + `\bibliography{}` integrates with it directly and
  automatically enforces npj's numbered-reference style (rule 7)
  without manually tracking citation order, which the Markdown
  bracket-placeholder approach could not do.
- Springer Nature provides an official template (`sn-jnl.cls`) with a
  style option built specifically for Nature Portfolio journals
  (`sn-nature`) -- confirmed via web search, not assumed, and pulled
  from the actual GitHub-mirrored template package (not hand-built),
  including the matching `sn-nature.bst` bibliography style.
- The full pipeline (pdflatex -> bibtex -> pdflatex x2) was actually
  compiled and the output visually inspected before trusting it, same
  validate-before-trusting discipline as the rest of this project --
  caught and fixed a real `sn-nature.bst` incompatibility with bare
  `@inproceedings` entries (both NeurIPS papers, karras2022edm and
  ke2017lightgbm) lacking publisher/address fields; fixed by using
  `@article` entries instead (a common, accurate convention for citing
  NeurIPS proceedings), not by inventing missing fields.

## Process for rules 5 & 6 (claim-checking + DOI verification)

This is the rule most likely to get skipped under time pressure, so it
gets an explicit process, not just a reminder:

1. While drafting any section, whenever a sentence makes a factual claim
   that isn't a novel result of THIS study, stop and add a row to
   `citations_ledger.md` before continuing: the claim (as written),
   the intended citation, and status `UNVERIFIED`.
2. Before a citation is added to `references.bib`, its DOI must be
   resolved and cross-checked:
   - Prefer resolving `https://doi.org/<DOI>` directly.
   - If direct resolution is blocked/rate-limited, cross-check via at
     least two independent sources citing the same DOI (as done for
     Karras et al. 2022 in this session) and note that in the ledger
     rather than silently treating it as fully verified.
   - Never invent or complete a DOI by pattern-matching a publisher's
     usual format (e.g. guessing a `10.1175/...` string because other
     AMS papers look like that). If a real DOI can't be found, mark
     the ledger row `NO DOI FOUND` and flag it for the user rather than
     leaving a fabricated one in the bibliography.
   - Some legitimate references have no traditional DOI (e.g. NeurIPS
     proceedings papers) — use the arXiv preprint DOI if one exists
     (`10.48550/arXiv.XXXX.XXXXX`, assigned by DataCite) and say so
     explicitly in the ledger; don't leave the DOI field blank without
     a note explaining why.
3. Once verified, update the ledger row to `VERIFIED` with the
   resolved title/venue, and add the full entry to `references.bib`.
4. A citation is not allowed into the manuscript prose until its
   ledger row says `VERIFIED`.

## Style notes for rule 7 (npj / Nature Portfolio conventions)

Confirmed via npj Clim Atmos Sci's own author guidelines (2026-08-14):

- **Numbered references**, not author-year. Cited in order of first
  appearance, superscript in text (e.g. "...as shown previously¹.").
  This differs from AMS/AGU journals (which is what most of the
  literature this project cites — Cintineo et al., Metzl et al.,
  Roberts & Lean, Gneiting & Raftery — actually uses); don't let
  AMS-style author-year citation habits leak into the drafted prose.
- **Section order**: Abstract → Introduction → Results → Discussion →
  Methods → (Data availability, Code availability, References,
  Acknowledgements, Author contributions, Competing interests). Results
  BEFORE Methods is the Nature-family convention — opposite of the
  Methods-first structure common in AMS/AGU-style geoscience papers.
  Methods is typically smaller-type/compressed, written for
  reproducibility rather than narrative flow.
- **No strict word/page limits** (npj is online-only, fully OA) but
  write concisely — this is explicit journal guidance, not just good
  practice.
- **Multi-panel figures** go on a single page, panels labeled a), b),
  c), ... — plan figure layout with this in mind from the start rather
  than retrofitting.
- **No footnotes.**
- **Only published/accepted papers or recognized preprint servers** go
  in the numbered reference list; unpublished work or personal
  communications are named in text instead.
- Tone: direct, results-forward, confident but not overclaiming —
  Nature-family abstracts front-load the finding and its significance
  in the first 1-2 sentences rather than building up through extensive
  motivation first.

## Notes

- This file records the rules; it does not enforce them automatically.
  Re-read it at the start of any future manuscript-writing session,
  since context may not carry over between sessions.
- If the user changes or adds a rule later, update this file in the
  same turn — don't let it drift out of sync with what's actually
  being followed.
