# Agentic-ReaxFF

A tool-calling toolkit for building **ReaxFF reactive force fields** end to
end, designed to be driven by an AI agent (Claude Code) or by hand:

    starting force field ─► reference data ─► training set ─► staged fits ─► validation ─► refinement ─► deploy
    (published / merged)    (DFT, xTB)        (weighted)      (CMA-ES ...)   (data + MD)    (MD -> DFT)

`reaxff-agent <stage>` runs one step and prints one JSON object; the
agent reads it and decides the next step. Playbooks (skills), subagents
and a literature digest with verified citations live in `.claude/`.

ReaxFF parameterization is notoriously hard: hundreds of correlated
parameters, a rugged and partly discontinuous loss, many near-equal minima
with different physics, and fits that look good on training data and fail
in MD. The defaults here encode what the parameterization literature found
(Shchygol 2019, ParAMS 2021, Larsson 2013, Furman & Wales 2019, ...; see
`.claude/skills/reaxff-literature/SKILL.md`):

- fit in **stages** (charges, bonds, angles, torsions, vdW, then jointly),
  with a **sensitivity screen** and at most ~sqrt(n) parameters at once;
- **bounds** from ±20% or from the range of published force fields;
  physical limits and constraints (η ≥ 7.2γ, radius ordering) by default;
- **CMA-ES with many short restarts, pruning, and choice on a held-out
  set**; MCFF annealing and SOPPE (van Duin's classic) for comparison and
  polishing; every run checkpointed and resumable;
- a training set of **relative energies only** (balanced), restrained
  scans, dimers for the short-range wall, forces on distorted frames, with
  one level of theory enforced;
- **validation beyond the loss**: diatomic wall/discontinuity checks,
  NVT->NVE MD with the Furman-Wales energy-conservation ratio, close
  contacts, charge runaway, species;
- **refinement rounds** that harvest MD frames where independent fits
  disagree.

## Install

```bash
git clone git@github.com:laubachb/Agentic-ReaxFF.git && cd Agentic-ReaxFF
./install.sh                                      # laptop / generic Linux
./install.sh --machine dane --account <bank>      # LLNL Dane (modules, MPI, Slurm)
source env.sh                                     # in every new shell
```

`install.sh` creates `.venv` (uv if available), installs the package with
LAMMPS (the PyPI wheel includes REAXFF/QEQ), finds an MPICH-ABI MPI library
for it, fetches the ReaxFF force fields distributed with LAMMPS, writes
`env.sh`, and runs `reaxff-agent doctor` (which checks real numerics, not
just imports). For DFT labels register or build Quantum ESPRESSO:

```bash
./install.sh --machine dane --account <bank> --qe-bin /path/to/pw.x --pseudo-dir /path/to/upf
# or: reaxff-agent setup --component quantum_espresso --machine dane      (builds pw.x + projwfc.x)
#     reaxff-agent setup --component pseudopotentials                     (downloads GBRV 1.5)
```

Optional: `setup --component lammps` (source build for large MD, OPENMP,
`--lammps-kokkos`), `setup --component jax_reaxff` (gradient-based fitting
in its own venv). New cluster: `reaxff-agent setup --init-profile my.yaml`.

## Quick tour (C/H/O, xTB labels, minutes)

```bash
reaxff-agent study --init runs/demo --name demo --goal "refit C/H/O bonds" --elements C,H,O
cd runs/demo
reaxff-agent data-generate --molecules CH4,C2H6,H2O,CH3OH --scans bond,angle --dimers true \
    --rattle '{"n": 3, "sigma": 0.08}' --output-dir 01_data/gen
reaxff-agent label --structures 01_data/gen/structures.extxyz --backend xtb --output-dir 01_data/xtb
reaxff-agent trainset-build --structures 01_data/xtb/labeled.extxyz --validation-fraction 0.2 --output-dir 02_trainset
FF=$(python -c "from agentic_reaxff import config; print(config.FFIELD_LIBRARY_DIR)")/ffield.reax.cho
reaxff-agent params-select --ffield $FF --preset bond --trainset 02_trainset/trainset.yaml --output-dir 03_fit/bond
reaxff-agent sensitivity --trainset 02_trainset/trainset.yaml --params 03_fit/bond/params.yaml --output-dir 03_fit/sens
reaxff-agent params-select --ffield $FF --from-sensitivity 03_fit/sens/sensitivity.json --top 15 \
    --trainset 02_trainset/trainset.yaml --output-dir 03_fit/top15
reaxff-agent fit --trainset 02_trainset/trainset.yaml --validset 02_trainset/validset.yaml \
    --params 03_fit/top15/params.yaml --restarts 3 --machine dane --dry-run --output-dir 03_fit/fit   # then without --dry-run
reaxff-agent evaluate --ffields 03_fit/fit/best.ffield --ffields $FF --labels fitted,published \
    --trainset 02_trainset/trainset.yaml --validset 02_trainset/validset.yaml --pair-check --plots --output-dir 04_validate/eval
reaxff-agent study --study . --status
```

With Claude Code, open the repo and ask for the goal ("refit the C/H/O
ReaxFF for methanol pyrolysis at 2000 K"); the `reaxff-study` skill plans
the study, asks you the decisions that are yours, and asks before every
Slurm submission.

## Stages

| stage | what it does |
|---|---|
| `setup`, `doctor` | install/verify MPI, LAMMPS, ffield library, QE, pseudopotentials; numerics check |
| `study`, `usage`, `study-report` | study directory, journal, dashboard, CPU-hours, report draft |
| `ffield-info` | coverage of a chemistry, sanity checks, parameter ids, diffs |
| `ffield-library` | parameter ranges from published force fields (for bounds) |
| `ffield-merge` | starting force field for a new element (donors, analog seeding, branch check) |
| `data-generate` | molecules (g2/SMILES/files), bond/angle/torsion scans, dimers, EOS, rattles, MD frames, MD boxes |
| `label` | QE DFT (Slurm, lanes, resumable, broken-symmetry spin, constraints), GFN-xTB (local), reference ReaxFF |
| `data-import` | labels from other codes (any ASE format) |
| `trainset-build` | weighted items (relative energies, forces, geometries, charges, lattice, reactions), validation split, merge |
| `trainset-convert` | classic trainset.in/geo/params (JAX-ReaxFF, standalone ReaxFF, ParAMS) both ways |
| `params-select` | active parameters by preset/element/data coverage/sensitivity, bounds, constraints |
| `sensitivity` | one-at-a-time or HSIC ranking of parameters |
| `fit` | CMA-ES / MCFF / SOPPE / Nelder-Mead / Powell; restarts, pruning, validation, polish, Slurm, resume |
| `evaluate` | errors by kind/group vs baselines, scan curves, plots, diatomic checks |
| `md-check` | NVT->NVE stability, energy conservation, contacts, charges, species |
| `refine-select` | MD frames where the committee of fits disagrees or contacts extrapolate |
| `deploy` | the ffield + FFIELD_CARD.md (provenance, accuracy, MD, LAMMPS usage, citations) |
| `submit`, `job-status` | raw Slurm submission and job verdicts |

Every stage: `--describe` (inputs/outputs), `--json-in`, `--output-dir`
(idempotent via a manifest), one JSON object on stdout.

## How the evaluation works

Every candidate force field is evaluated on every training structure by
LAMMPS (`pair_style reaxff` + `fix qeq/reaxff` or `acks2/reaxff`), in
process, one serial LAMMPS per worker; workers spread candidates x
structures over all cores of a node. Geometry targets use restrained
relaxations (`fix restrain`), crystals `fix box/relax`. The force field
written is the force field evaluated (parameters are rounded to the file's
precision first). Some LAMMPS behaviors the evaluator handles are
documented in `src/agentic_reaxff/engine/evaluator.py` (QEq fix ordering,
small periodic cells, restraint energies, malformed files).

## Status (Oct 2026)

Verified on LLNL Dane: install + doctor; parser round-trip of all LAMMPS
ReaxFF files; evaluator vs finite differences, rotation invariance and
small-cell consistency; parameter recovery (CMA-ES recovers perturbed
Chenoweth C/H/O parameters to 1e-4); an end-to-end C/H/O study with xTB
labels including a Slurm fit (112 workers, ~3 min), evaluate, md-check,
refinement selection, merge, deploy, legacy round trip; QE labeling
(relaxations, constrained scans, spin-polarized dimers) through Slurm.
Not yet exercised: the LAMMPS source build, JAX-ReaxFF installation,
QE builds from source, ACKS2 fits, multi-node jobs. 60 unit tests:
`.venv/bin/python -m pytest`.

## Layout

```
src/agentic_reaxff/   cli.py, config.py, journal.py
  io/                 ffield (parser/writer), structures, trainset, params, legacy, units
  engine/             evaluator (LAMMPS), pool (parallel), objective (loss)
  optim/              runner (budgets, checkpoints, early stopping), optimizers
  generate/           structure builders     labeling/  QE inputs/jobs/parsers
  stages/             one module per CLI stage           machines/  Slurm profiles
.claude/              skills, subagents, approval hook, settings
tests/unit/           pytest suite            install.sh, env.sh (generated)
```

License: MIT. Third-party force fields and papers are never committed;
`setup` fetches the LAMMPS-distributed force fields into `deps/`.
