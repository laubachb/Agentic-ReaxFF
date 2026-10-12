---
name: reaxff-study-report
description: Writing the final human-readable report of a ReaxFF study (REPORT.md) and the force-field card - what to include, how to state accuracy honestly, citations, and how to use the journal. Use at the end of a study or for a case-study write-up.
---

# Study report

1. `reaxff-agent study --study DIR --journal` (JOURNAL.md) and
   `study --status` (STATUS.md); `usage --study DIR` for CPU-hours.
2. `reaxff-agent study-report --study DIR` drafts REPORT.md with the facts
   filled in and `<!-- NARRATIVE: ... -->` placeholders.
3. Fill every placeholder from JOURNAL.md and the artifacts:
   - Starting point and why (branch, donors) with citations [S16].
   - Training set: what, how many, level of theory, weights; what is NOT
     covered (state the regimes the force field was never shown).
   - Fitting: stages, parameters per stage, bounds, optimizer settings,
     number of runs and the spread of their losses (reproducibility [Sh19]),
     how the final one was chosen.
   - Accuracy per kind in natural units, train AND validation, next to the
     starting force field; realistic context (reaxff-literature section 9).
   - MD validation table and any instability; refinement rounds and why
     they stopped.
   - Limitations and recommended use (temperatures, chemistries, charge
     method, timestep, the cutoffs the fit used).
4. `deploy` writes FFIELD_CARD.md next to the ffield; keep it consistent
   with the report.

Paraphrase papers; cite with DOIs from reaxff-literature. Never present the
toolkit's refinement selection as a published ReaxFF protocol.
