# Concepts and architecture

## Stages and contracts

`reaxff-agent` is a set of independent stages. Each stage module
(`src/agentic_reaxff/stages/<name>.py`) declares `NAME`, `SUMMARY`, a JSON
`SCHEMA`, `add_arguments` and `run(args) -> dict`. The CLI wrapper
(`cli.py`) gives every stage the same contract:

- input from flags or `--json-in` (unknown keys and wrong types refused);
- exactly one JSON object on stdout; native output (LAMMPS, QE) diverted to
  `<output-dir>/<stage>.log`;
- `--output-dir` + a manifest (`.manifest-<stage>.json`): an identical
  re-run returns the previous result; different inputs in the same
  directory are refused without `--force`; input files count by content;
- the run is appended to the study journal when the output directory is
  inside a study.

Stages never call each other; the agent (or person) composes them.

## Data model

- **Structures** (`io/structures.py`): one packed `extxyz` per data set.
  Each frame has a unique `name`, a `group`, a `mode` (`single`, `relax`,
  `relax_cell`), optional `restraints` (bond/angle/dihedral targets used by
  both ReaxFF and the DFT relaxation), labels in ASE units (eV, eV/Å),
  `ref_charges`, and the level-of-theory key `lot`.
- **Training set** (`io/trainset.py`): YAML list of items — `energy`
  (balanced linear combinations), `forces`, `charges`, `distance`, `angle`,
  `dihedral`, `lattice`, `stress` — each with `sigma` (accepted error) and
  `weight`. Loss = Σ w·mean(((pred − ref)/σ)²) (ParAMS/van Duin
  convention; array items split their weight over components).
- **Parameter spec** (`io/params.py`): active parameter ids with bounds and
  linear constraints. Ids are readable and stable:
  `atom:C:r_s`, `bond:C-H:De_s`, `off:C-O:r_vdW`, `angle:H-C-H:theta_00`,
  `tors:X-C-C-X:V2`, `hbond:O-H-O:r0_hb`, `gen:p_boc1`.
- **Force field** (`io/ffield.py`): parser/writer for the van Duin format
  (general, atoms (+lg), bonds, off-diagonal, angles, torsions, H-bonds),
  names matching LAMMPS' reader.

## Evaluation

`engine/evaluator.py` runs one structure through LAMMPS in-process
(`pair_style reaxff`, `fix qeq/reaxff` or `acks2/reaxff`, `fix restrain`,
`fix box/relax`); `engine/pool.py` spreads (candidate × structure-chunk)
tasks over spawned worker processes, each with its own serial LAMMPS;
`engine/objective.py` turns per-structure results into item predictions
and the loss. Candidates are written to node-local scratch, rounded to the
file precision, so the loss reported is the loss of the file written.

## Optimization

`optim/base.py` (Runner) owns budgets, history, best-so-far files,
validation tracking and early stopping, checkpoints and time limits;
optimizers implement ask/tell in normalized coordinates
(`optim/optimizers.py`: CMA-ES, MCFF, SOPPE, scipy). The `fit` stage
orchestrates restarts → pruning → continuation → choice → polish and
can submit itself to Slurm (`stages/_remote.py`).

## Studies

A study directory (`study.json`) has phase folders, a registry of
artifacts, a journal (`journal.jsonl`, automatic + notes), a dashboard
(`study --status`) and a report draft (`study-report`).

## Claude Code layer

`.claude/skills/*` are playbooks the agent loads per task;
`.claude/agents/*` are subagents for phases; `.claude/hooks/approval_gate.py`
asks the user before any Slurm submission (including `--machine` hidden
in `--json-in`).
