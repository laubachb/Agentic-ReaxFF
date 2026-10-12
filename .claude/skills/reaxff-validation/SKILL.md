---
name: reaxff-validation
description: Validating ReaxFF force fields with reaxff-agent - evaluate on training/validation/test sets against the starting and published force fields, diatomic short-range and smoothness checks, md-check stability criteria (Furman-Wales energy-conservation ratio, close contacts, charge runaway, species), choosing among near-equal fits, and refinement rounds (refine-select) with a stopping rule. Use after every fit and before any recommendation.
---

# Validation

Load `reaxff-literature` sections 6, 8-10 alongside this.

## 1. Numbers on data (`evaluate`)

    reaxff-agent evaluate --ffields fit/best.ffield --ffields start.ffield [--ffields published.ffield] \
        --labels fitted,start,published --trainset ... --validset ... [--testsets ...] --pair-check --plots

- Improvement must show on validation/test, not only on training.
- Report errors per kind in natural units (kcal/mol, kcal/mol/Å, Å, deg)
  and compare with realistic ReaxFF accuracy (tens of kcal/mol on hard
  reaction sets [Ba13]; ~8 kcal/mol on adsorption energies [Du24]); never
  promise "chemical accuracy".
- `curves.json` / `scans.png`: look at every dissociation curve; a fit that
  matches the minimum but not the dissociation limit is not reactive.
- `--pair-check`: `repulsive_to_rmin: false` (the wall turns over going
  inward) means atoms can collapse; `force_jumps` are bond-order cutoff
  discontinuities [FW19]. Compare with the starting force field: new
  problems are the fit's doing.

## 2. Dynamics (`md-check`)

At the temperatures the user cares about and well above (e.g. 300 K and
2000-3000 K for reactive use), on boxes from `data-generate --boxes`:

| check | threshold | meaning |
|---|---|---|
| completed / finite | must | crashes, lost atoms, NaN |
| close_contacts | none below 0.5 x covalent sum | collapse ("missing repulsion") |
| max_abs_charge | <= 2.5 e (raise for ionic systems) | QEq runaway / polarization catastrophe |
| nve_ratio | < 0.10 [FW19] | energy conservation; published force fields exceed it too, so compare with the start |
| species_final | chemically sensible | e.g. methane should not decompose at 300 K |

Run the candidate and the starting/published force field side by side. A
fit that is better on data but less stable in MD is not better.
For more than a few minutes of MD use `--machine` (approval).

## 3. Choosing among fits

Several runs reach near-equal loss with different parameters [L13, Sh19].
Choose by: validation loss, then md-check stability, then physical
sanity of the parameters (radii, energies near published values), then
simplicity (fewer parameters changed). Say which criterion decided.
Ask `reaxff-fit-reviewer` for an independent read.

## 4. Refinement rounds (`refine-select`)

When validation or MD exposes regions the data does not cover:

    md-check (trajectories) -> refine-select --committee <fit.json committee> --trajectories ... --trainset ...
    -> label (same QE settings) -> trainset-build --merge-into -> fit -> md-check

Stop when (a) md-check is stable at all target temperatures, (b) committee
disagreement (energy std per atom, force std) stops falling between rounds,
and (c) validation error stops improving; or at the round budget agreed with
the user. Record each round's numbers as a `result` note. Be explicit that
committee-driven selection for ReaxFF is this toolkit's method, grounded in
[Sh19]'s advice, not an established published protocol.
