---
name: reaxff-fitting
description: Fitting ReaxFF parameters with reaxff-agent - choosing active parameters (presets, element filters, sensitivity), bounds and constraints, optimizer and budget (CMA-ES restarts with pruning, MCFF, SOPPE polish), running on Slurm, reading fit.json/best_report.json, and what to do when a fit stalls, hits bounds, breaks structures or overfits. Use for the fitting phase.
---

# Fitting

Load `reaxff-literature` sections 4-6 alongside this.

## 1. Choose parameters (`params-select`) — the decision that matters most [L13]

Fit in stages (one preset at a time), then jointly:

| stage | preset | data that constrains it |
|---|---|---|
| charges | `charge` (chi, eta, gamma) | charges, dipoles; never charges alone [V11] |
| bonds | `bond` (or `sigma_bond` first) | dissociation curves, bond lengths, EOS |
| angles | `valence` | angle scans, molecular angles |
| coordination | `coordination` | over/under-coordinated structures, reactions |
| torsions | `torsion` | torsion scans |
| vdW / repulsion | `vdw`, `inner_wall` | dimers at short range, EOS compression, densities |
| H-bonds | `hbond` | H-bonded dimers/clusters |
| refinement | union of the above, top-N by sensitivity | everything |

- `--elements NEW` when extending a force field: published blocks stay put
  [S16, Sh19]. `--from-merge merge_report.json` touches only what
  ffield-merge copied.
- `--trainset` (default `require_data`): multi-body terms whose elements
  never appear together in the data are left out.
- Then `sensitivity --method oat` (or `hsic` for a global screen) and
  `params-select --from-sensitivity sensitivity.json --top N` with
  N <= sqrt(n_items) [P21, FH23]. `fragile` parameters (breaking
  structures on a small move) need narrow bounds.
- Bounds: default ±20% [P21]; `--bounds library --library library.json`
  for a new element (historical ranges [Sh19]); `union` when unsure.
  Constraints the start already violates are skipped and reported.

## 2. Run (`fit`)

Defaults follow [Sh19]: CMA-ES, sigma0 0.25, population 4+3 ln n,
`--restarts 4 --keep 2 --short-evals 1500 --max-evals 6000`, run 0 from
the ffield and the rest from random points in the central half of the box.
For a final fit use `--restarts 10`. Always give `--validset` (choice
among runs and early stopping `--patience-evals`). `--polish soppe` for a
last local pass (accepted only if validation improves).

Cost: one evaluation = the whole training set through LAMMPS; restrained
relaxations dominate. Measure with a short local run (`--max-evals 50
--restarts 1`), then size the job: anything over ~5 min goes to Slurm:

    reaxff-agent fit ... --machine dane --walltime-hours 1 --chunks 3 --dry-run
    (show the script; on approval rerun without --dry-run)

Chunks chain with afterany and resume from checkpoints. On Dane one node
(112 workers) did 2900 evaluations of a ~200-structure set in 3 minutes.

## 3. Read the result

- `fit.json`: `status` (done | time_limit -> resubmit with --continue),
  `runs` (loss and validation loss per run: their spread is the
  reproducibility of the fit [Sh19]), `best`, `at_bounds`,
  `warning_dominant_items`, `committee` (for refine-select).
- `best_report.json`: errors per kind and group for start vs best, train
  and validation, `top_items`, failed structures.
- Before recommending, ask `reaxff-fit-reviewer` for an independent read.

## 4. When things go wrong

| symptom | likely cause | do |
|---|---|---|
| training loss down, validation up | overfitting | fewer parameters, early stopping, more diverse data |
| runs end far apart in loss | rugged landscape | more restarts, longer short phase; report the spread |
| runs equal in loss, different parameters | degenerate minima [L13, Sh19] | choose by validation and md-check; add data where they disagree (refine-select) |
| parameters at bounds | bound set the value | widen only if physical [Sh19, MH16]; otherwise the data asks for something the form cannot give |
| `n_failed` > 0 at the start | ffield breaks a structure | inspect `errors`; a bad starting point or an invalid geometry |
| loss jumps for tiny changes | ambiguous geometry optimizations | find the items (top_items), make them rigid/single or drop them [Sh19] |
| a few items dominate | weights | fix sigmas before fitting more |
| CMA-ES stops early (`optimizer:tolx`) | converged locally | more restarts, larger sigma0 |
