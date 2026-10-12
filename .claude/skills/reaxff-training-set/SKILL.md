---
name: reaxff-training-set
description: Designing, generating, labeling and weighting a ReaxFF training set with reaxff-agent - which structures (scans, dimers, EOS, rattled and MD frames), DFT settings for QE (spin for dissociation, vacuum, constraints, charges), xTB shakedowns, level-of-theory discipline, sigmas/weights and the validation split. Use for the data phase of a study or when a fit's errors point at the data.
---

# Building the training set

Load `reaxff-literature` sections 2-3 alongside this.

## What to generate (`data-generate`)

For every element combination the force field must describe:

| need | recipe key | why |
|---|---|---|
| equilibrium molecules (geometries, charges) | `molecules` (g2 names, `smiles:`, files) | geometry items, charges [vD01] |
| bond dissociation, other coordinates relaxed | `scans: ["bond"]`, `bond_factors` (default 0.75-2.5 x r0) | the bond-order/bond-energy terms [vD01, C08] |
| valence-angle bending | `scans: ["angle"]` | valence terms |
| torsions about single bonds | `scans: ["torsion"]` | torsion terms |
| short-range repulsion | `dimers: true` | inner wall / shielding; too-soft repulsion is a known failure [Ca20] |
| condensed phases | `crystals` + `volume_ratios` (EOS), `relax_cell` frame | densities, cohesion [C08, S16] |
| forces off equilibrium | `rattle: {n, sigma, strain}` | force matching [DA24, Du24] |
| dynamics of the target system | `trajectories` (frames from MD) | what the FF will actually see |
| reactions / barriers | explicit `reactions` in trainset-build (balanced) | chemistry the user cares about |
| validation MD systems | `boxes` (written to boxes.extxyz, not training) | md-check |

Keep geometry optimizations unambiguous: rigid molecules, products as
separate entries, no floppy multi-conformer complexes; these caused all
of the loss noise in [Sh19].

## Labeling (`label`)

- **xtb** first for a shakedown (seconds to minutes, local): does every
  frame label, are the scans smooth, does trainset-build balance? Not for
  the final fit. Stretched open-shell frames are retried at 3000 K
  electronic temperature.
- **qe** for production. Settings (`--qe '{...}'`): pseudo_dir +
  pseudopotentials (or `setup --component pseudopotentials`), ecutwfc /
  ecutrho (GBRV USPP: 40/200 Ry nominal; check convergence for your
  elements), `vdw_corr: "dft-d3"` if dispersion matters, `vacuum` >= 10 Å
  for molecules (default 12), `kspacing` 0.25 Å^-1 for crystals,
  `spin: auto` (odd-electron systems spin-polarized; stretched bond scans
  and dimers broken-symmetry), `charges: lowdin` needs projwfc.x.
  Restrained scan frames are relaxed under the matching QE constraint.
  Always `--dry-run` first, then submit with approval; `--lanes` frames per
  node (small molecules: 6-12 lanes on 112 cores), `--chunks` for 1 h
  queues; `--collect` afterwards; unfinished frames are listed, resubmit
  the same directory to finish them.
- **data-import** for labels from elsewhere (VASP, ORCA, ...): give `--lot`.
- **One level of theory per fit.** `trainset-build` refuses mixed `lot`
  keys. Never "fix" this with `allow_mixed_lot` to save time.

## Weighting (`trainset-build`)

- Energies are relative within same-composition groups (scan, EOS, MD
  batch), reference = the group's lowest frame; EOS per atom.
- Defaults: energy sigma 2 kcal/mol, forces 5 kcal/mol/Å (rattle, md, eos
  frames only), distance 0.02 Å, angle 2°, charges 0.1 e, lattice 0.05 Å
  (classic/ParAMS practice [P21]).
- `high_energy_scale` (default 50 kcal/mol): sigma grows with height above
  the group minimum, so repulsive-wall frames inform without dominating.
- After building, check `fit`'s `warning_dominant_items` or evaluate's
  `top_items`: if 10 items carry most of the loss, fix their weights
  before fitting (the first smoke test of this toolkit had dimer forces at
  0.6 Å carrying 78% of the loss).
- `group_sigma` / `group_weight` (glob -> value) to express priorities the
  user states ("barriers matter more than EOS").
- Validation: `--validation-fraction 0.15 --split-by group` (whole scans
  held out). Refinement rounds: `--merge-into` keeps previous items and the
  previous validation set unchanged.

## Interop

`trainset-convert --to-legacy` writes trainset.in/geo/params (JAX-ReaxFF,
standalone ReaxFF, ParAMS); `--from-legacy` imports published sets (e.g.
the [Sh19] cobalt/silica/disulfide benchmarks shipped with JAX-ReaxFF).
Loss is preserved up to the format's precision (~0.1%).
