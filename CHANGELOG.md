# Changelog

## 0.1.0 (2026-10-12)

First version.

- `reaxff-agent` CLI with 22 stages (setup, doctor, study, usage, ffield-info, ffield-library, ffield-merge,
  data-generate, data-import, label, trainset-build, trainset-convert, params-select, sensitivity, fit, evaluate,
  md-check, refine-select, deploy, study-report, submit, job-status); JSON contract, manifests, study journal.
- ReaxFF ffield parser/writer with stable parameter ids; round-trips every ReaxFF file shipped with LAMMPS
  22Jul2025.
- In-process LAMMPS evaluator (PyPI wheel or source build) with restrained relaxations, cell relaxation, automatic
  replication of small periodic cells, QEq/ACKS2; parallel spawned worker pool.
- Optimizers: CMA-ES (Shchygol 2019 settings), MCFF annealing, SOPPE, scipy local; restarts with pruning,
  validation-based choice, early stopping, checkpoint/resume, Slurm chunking.
- Labeling with Quantum ESPRESSO (Slurm lanes, constraints via damped dynamics, broken-symmetry spin, Löwdin
  charges), GFN-xTB (tblite) and reference ReaxFF; level-of-theory enforcement.
- Validation: evaluate (baselines, scan curves, diatomic wall and discontinuity checks), md-check (NVT->NVE,
  Furman-Wales ratio, contacts, charges, species), committee-based refinement selection.
- Claude Code layer: CLAUDE.md, 7 skills (incl. a verified ReaxFF literature digest), 7 subagents, approval hook.
- install.sh (venv, MPI detection, ffield library, env.sh, doctor); machine profiles for Dane, Stampede3, generic.

Fixed before release (independent code review): `fit --machine` into a reused directory resumed the old fit;
re-running a finished fit could overwrite the polished result; QE frame directories keyed by position could attach
one structure's DFT result to another (now keyed by name + geometry hash, checked on collect); hydrogen-bond entries
were treated as symmetric (LAMMPS reads them directionally); stale in-process worker state across pools; validation
leakage through shared structures in trainset-build splits; polish interrupted by the time limit marked as done;
non-checkpointable optimizers restarting their budget on resume; merges into legacy-imported sets; exclude_groups
ignored for geometry/charge/lattice items; 2-D slabs not replicated; stress items on non-periodic structures;
non-atomic fit state writes. Also from live testing: QEq fix ordering (random NaN), small-cell QEq energies, pw.x
constraints need damped dynamics and nosym, LAMMPS segfault on malformed ffields, default weights that let
short-range dimer forces dominate the loss.
