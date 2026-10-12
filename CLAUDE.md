# Agentic ReaxFF

`reaxff-agent` is a tool-calling CLI for building ReaxFF reactive force
fields end to end: starting force field -> reference data (DFT/xTB) ->
weighted training set -> staged parameter fits -> validation (data + MD) ->
refinement rounds -> deployment. You (the agent) drive it: run a stage,
read its JSON, decide the next stage. No stage chains into another.

## How to drive it

- Environment: `source env.sh` (written by `./install.sh`; activates the
  venv, loads the machine modules, sets `REAXFF_ACCOUNT`). Health check:
  `reaxff-agent doctor [--machine M]`.
- One subcommand per stage; **`reaxff-agent <stage> --describe` is the
  source of truth for inputs/outputs.** Stages: `setup`, `doctor`, `study`,
  `usage`, `ffield-info`, `ffield-library`, `ffield-merge`, `data-generate`,
  `data-import`, `label`, `trainset-build`, `trainset-convert`,
  `params-select`, `sensitivity`, `fit`, `evaluate`, `md-check`,
  `refine-select`, `deploy`, `study-report`, `submit`, `job-status`.
- stdout is exactly one JSON object (errors: `{"error", "log", "log_tail"}`,
  exit 1). Native output (LAMMPS, QE) goes to `<output-dir>/<stage>.log`.
- Give every file-writing stage an `--output-dir`; identical re-runs
  short-circuit via a manifest, changed inputs at the same path are refused
  until `--force`. One directory per experiment.
- Prefer `--json-in file.json` for nested inputs; unknown keys and wrong
  types are refused (fix the key, don't drop the setting).
- Stages that submit Slurm work (`label --backend qe`, and `fit`,
  `sensitivity`, `md-check` with `--machine`) accept `--dry-run`: it renders
  the exact sbatch script. Do that first and show it; the approval hook asks
  the user before a real submission. Long local stages: run with the Bash
  tool's `run_in_background`, don't sleep-poll.
- After a submission: `reaxff-agent job-status --work-dir DIR`; a stage job
  writes its normal JSON result to `DIR/job_result.json`. Fits resume with
  `--continue` (and chained `--chunks` jobs do so automatically).
- **Every study keeps a journal** (`<study>/journal.jsonl`, automatic for
  stage runs). Add what commands cannot say, at the time:
  `reaxff-agent study --study S --note "..." --note-kind decision|finding|issue|plan|result`.
  `study --status` is the dashboard; `study --journal` renders JOURNAL.md.

## Playbooks (skills) — load before that kind of task

- `reaxff-study` — whole-study goal: phases, decisions for the user, gates
- `reaxff-training-set` — structures, DFT/xTB labeling, weights, split
- `reaxff-fitting` — parameter stages, sensitivity, bounds, CMA-ES runs, reading results
- `reaxff-validation` — evaluate, pair checks, md-check, choosing fits, refinement rounds
- `reaxff-hpc-jobs` — Slurm, QE jobs, chunking, quotas, MPI
- `reaxff-study-report` — REPORT.md and the force-field card
- `reaxff-literature` — published ReaxFF fitting practice with verified DOIs

Subagents: `reaxff-trainset-designer`, `reaxff-fitter`, `reaxff-validator`,
`reaxff-refiner`, `reaxff-job-monitor` (cheap read-only status checks),
`reaxff-fit-reviewer` (independent read before recommending a model),
`reaxff-report-writer`.

## Ground technical choices in the literature

Before choosing a preset, bound, sigma, optimizer setting, budget or
validation criterion, check `reaxff-literature` and cite it (e.g.
"Shchygol 2019, 10.1021/acs.jctc.9b00769"). Say so when you deviate.
ReaxFF fitting is hard for documented reasons — rugged loss, many
near-degenerate minima, parameter correlation, discontinuities — so:
never trust a single optimizer run; fit in stages with <= ~sqrt(n_items)
parameters; choose among runs on held-out data; validate with MD; report
realistic accuracy.

## Rules that prevent expensive or wrong results

- **One level of theory per fit**; `trainset-build` refuses mixed `lot`.
- **Energies are relative.** Energy items must be stoichiometrically
  balanced (`allow_unbalanced` only for imported legacy sets).
- **Never edit `deps/`** (built/fetched components; `reaxff-agent setup`
  recreates them) and never commit third-party force fields or papers.
- **HPC job I/O on a shared filesystem**, never /tmp (refused). Lustre
  quotas count files: no per-frame files outside label directories.
- **On Dane always a full node** (`--ntasks-per-node 112`, enforced by the
  profile). Set `REAXFF_ACCOUNT` (install.sh --account).
- **LAMMPS facts the evaluator handles** (don't work around them in your
  own scripts without the same care): charge-equilibration fixes must be
  defined after atoms exist (else random NaN energies); QEq/ACKS2 energies
  are wrong in periodic cells thinner than 10 Å (replicated automatically);
  restrained relaxations report the pair energy without the restraint;
  LAMMPS' ReaxFF reader can segfault on malformed ffields (we parse first).
- **Python scripts that use `agentic_reaxff` workers need an
  `if __name__ == "__main__":` guard** (workers are spawned).
- Units: kcal/mol, Å, e, degrees, GPa inside the toolkit; extxyz files keep
  ASE's eV-based units.
- Don't commit, push or open PRs unless asked.

## Working on this repo itself

- Tests: `.venv/bin/python -m pytest` (tests needing LAMMPS or the ffield
  library skip cleanly). Every user-visible change updates `CHANGELOG.md`
  and the stage's docstring/SCHEMA (what `--describe` shows) and `docs/`.
- A stage module in `src/agentic_reaxff/stages/` exposes `NAME`, `SUMMARY`,
  `SCHEMA`, `add_arguments`, `run` (and `SUPPORTS_DRY_RUN` /
  `USES_OUTPUT_DIR` when relevant), and is listed in `cli.STAGE_MODULE_NAMES`.
- Machine specifics live in `src/agentic_reaxff/machines/profiles/*.yaml`.
- Architecture: `docs/concepts.md`.
