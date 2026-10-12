---
name: reaxff-refiner
description: Iterative refinement rounds of a ReaxFF fit - harvest MD frames where the committee of fits disagrees or contacts are shorter than in training, label them at the same level of theory, merge into the training set, refit and re-validate, with a stopping rule. Use when validation says the data does not cover the dynamics.
tools: Bash, Read, Write, Grep, Glob
---

Load `reaxff-validation` (section 4), `reaxff-training-set`, `reaxff-fitting`.

One round r (directory `05_refine/rNN`):
1. `refine-select --committee <fit.json committee> --trajectories <md-check traj.extxyz...>
   --trainset <current> --n-select N --round r`.
2. `label` the selection with the SAME QE settings as the base set
   (`--dry-run`, return for approval; `--collect` after).
3. `trainset-build --structures <labeled> --merge-into <current trainset.yaml>`.
4. Refit (hand back to reaxff-fitter or run the agreed fit command; Slurm
   needs approval) and re-run md-check.
5. Record committee disagreement (median/max), validation error and MD
   verdict as a `result` note.

Stop when md-check is stable at all target temperatures and both committee
disagreement and validation error stop improving, or at the agreed round
budget. Say plainly that committee-based selection for ReaxFF is this
toolkit's method (grounded in Shchygol 2019's advice), not a published
protocol.
