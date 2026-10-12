---
name: reaxff-study
description: End-to-end plan for building a ReaxFF force field with reaxff-agent ("I need a ReaxFF model for X", "extend this ffield to element Y", "refit ReaxFF for my system") - study layout, phase order, decisions to put to the user, approval gates for Slurm work, journaling, and handoffs to the subagents. Load this first for any whole-study goal.
---

# Running a ReaxFF study

A study is one directory with `study.json`, created by
`reaxff-agent study --init DIR --name N --goal "<user's words>" --elements C,H,O`.
Layout: `01_data 02_trainset 03_fit 04_validate 05_refine 06_deploy usage`.
Every stage run with an `--output-dir` inside it is journaled
automatically; add a note at every decision, surprise, failure and
re-plan (`study --study DIR --note "..." --note-kind decision|finding|issue|plan|result`).
`study --study DIR --status` is the one-screen dashboard.

Load `reaxff-literature` before any technical choice and cite it.

## 0. Settle with the user (ask; do not assume)

1. Chemistry and use: elements, phases (gas, liquid, crystal, surface),
   temperatures/pressures, which reactions matter. This decides the training
   set and the validation MD.
2. Starting point: an existing force field to refine, or a new element to
   add? Which branch (combustion vs aqueous, [S16])? Search the library
   (`ffield-library`, `ffield-info --elements ...`) and propose candidates
   with their citations; the user chooses.
3. Level of theory and budget: QE PBE(+D3) with which pseudopotentials,
   how many node-hours. One level of theory for the whole fit.
4. Whether xTB prototyping first is acceptable (cheap shakedown of the
   training-set design before DFT).

Record each answer as a `decision` note.

## 1. Starting force field -> `start_ffield`

- Existing chemistry: copy the chosen ffield, `ffield-info --elements`
  (coverage, sanity: eta >= 7.2 gamma, inner wall / shielding, bond-order
  cutoff).
- New element: `ffield-merge --base B --add EL=donor --analog EL=OLD`; read
  `warnings` (general-parameter differences = branch mismatch [S16]) and
  `needs_fitting`.
- Register: `study --register start_ffield=PATH`.

## 2. Reference data (subagent: reaxff-trainset-designer)

`data-generate` (molecules, bond/angle/torsion scans, dimers, EOS, rattled
frames, MD boxes for validation) -> `label` (xtb for the shakedown, qe for
the real labels; qe needs approval: show `--dry-run` first) ->
`label --collect`. Register `structures`.

## 3. Training set

`trainset-build --validation-fraction 0.15` (split by group) -> read
`warnings`, `counts`, `hint`. Register `trainset`, `validset`. Re-read
`reaxff-training-set` if counts look unbalanced.

## 4. Fit (subagent: reaxff-fitter)

Stage by stage (preset per stage, sensitivity first, <= sqrt(n_items)
parameters per stage, 4-10 CMA-ES restarts, pruning, validation-based
choice) - see `reaxff-fitting`. Fits beyond a few minutes run with
`--machine` (approval; dry run first). Register `params_spec`, `fit`,
`ffield`.

## 5. Validation (subagent: reaxff-validator)

`evaluate` (candidate vs start vs published, train/validation/test,
`--pair-check`) and `md-check` at the use temperatures. Hand the result to
`reaxff-fit-reviewer` for an independent read before recommending.
Register `evaluate`, `md_check`.

## 6. Refinement (subagent: reaxff-refiner), if validation asks for it

md-check trajectories -> `refine-select` (committee from fit.json) ->
`label` -> `trainset-build --merge-into` -> `fit` -> `md-check`. Stop when
committee disagreement and validation error stop improving, or after the
round budget agreed with the user.

## 7. Deliver

`deploy --study ...` (ffield + FFIELD_CARD.md), `usage --study`,
`study-report --study`, then the reaxff-report-writer agent fills the
narrative from JOURNAL.md (`study --journal`).

## Approval gates (the hook enforces them; do not route around it)

Ask before: `label --backend qe` (unless --collect), any stage with
`--machine` (fit, sensitivity, md-check), `submit`, raw `sbatch`. Always
show the `--dry-run` output (the exact sbatch script, node count, walltime)
when asking.
