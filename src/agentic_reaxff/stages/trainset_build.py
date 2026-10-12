"""Turn labeled structures into a weighted training set (and a held-out
validation set).

Item rules (each can be tuned or switched off in the recipe):

  energy     Only differences: within every group of same-composition
             frames (a scan, an EOS, an MD batch), E_i - E_ref with ref the
             group's lowest-energy frame. ReaxFF and DFT absolute energies
             are not comparable; relative energies are what ReaxFF training
             sets have always used [vD01, C08]. EOS groups are per atom.
  reactions  explicit balanced combinations: [[name, coeff], ...]
  forces     per-frame force sets on chosen kinds (default: rattle, md,
             eos). Not on relaxed scans (their forces are ~0 by
             construction) and not on dimer curves, whose short-range
             forces (thousands of kcal/mol/Å at 0.6 Å) would swamp the loss
             while the energies already carry that information.
  geometry   bond lengths and angles of DFT-relaxed molecules (mode relax,
             kind molecule), compared after ReaxFF relaxation
  charges    reference charges where the labels have them (loosely
             weighted: population schemes disagree, and charges alone
             cannot pin EEM parameters [V11])
  lattice    a, b, c (and volume) of relaxed crystals (mode relax_cell)

Sigmas default to ParAMS-like accepted errors [P21] (energy 2 kcal/mol,
forces 5 kcal/mol/Å, distance 0.02 Å, angle 2°, charge 0.1 e, lattice
0.05 Å). `high_energy_scale` (default 50 kcal/mol) loosens sigma for frames
far above their group minimum, sigma·(1 + ΔE/scale): a frame 1000 kcal/mol
up a repulsive wall still counts, but cannot dominate the fit. ReaxFF cannot
follow DFT into every high-energy corner and should not trade the
low-energy region for it (a heuristic, cf. ParAMS' Gaussian weighting of
PES scans; a handful of items dominating the loss is the failure mode
Shchygol et al. traced their noise to [Sh19]). Set it to 0 to disable.

Validation split (`validation_fraction`, by whole group by default so a
scan is never half in each set) gives the held-out set for early stopping
and for choosing among near-equal fits [Sh19, P21].

One level of theory: frames whose `lot` differs are refused unless
`allow_mixed_lot` (don't).
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path

import numpy as np

from ..io import structures as SIO
from ..io import trainset as T
from ..io.units import EV_TO_KCALMOL

NAME = "trainset-build"
SUMMARY = "Build trainset.yaml (+ validset.yaml) from labeled structures: relative energies, forces, geometries, charges, lattice, reactions."
SCHEMA = {
    "type": "object",
    "properties": {
        "structures": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Labeled extxyz file(s)."},
        "energy_sigma": {"type": "number", "default": 2.0},
        "eos_sigma_per_atom": {"type": "number", "default": 0.5},
        "high_energy_scale": {"type": ["number", "null"], "default": 50.0, "description": "kcal/mol; sigma grows as 1 + ΔE/scale above the group minimum (0 = off)."},
        "max_relative_energy": {"type": ["number", "null"], "description": "Drop energy items more than this (kcal/mol, per atom for EOS x natoms) above the group minimum."},
        "forces_kinds": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Kinds that get force items (default rattle, md, eos)."},
        "forces_sigma": {"type": "number", "default": 5.0},
        "forces_weight": {"type": "number", "default": 1.0},
        "forces_every": {"type": "integer", "default": 1, "description": "Use every n-th eligible frame for forces."},
        "geometry": {"type": "boolean", "default": True},
        "distance_sigma": {"type": "number", "default": 0.02},
        "angle_sigma": {"type": "number", "default": 2.0},
        "charges": {"type": "boolean", "default": True},
        "charges_sigma": {"type": "number", "default": 0.1},
        "charges_kinds": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Default: molecule."},
        "lattice_sigma": {"type": "number", "default": 0.05},
        "reactions": {"type": ["array", "null"], "items": {"type": "object"}, "description": "[{terms: [[name, c],...], sigma, ref?, group?}]"},
        "group_sigma": {"type": ["object", "null"], "description": "glob over group name -> energy sigma override."},
        "group_weight": {"type": ["object", "null"], "description": "glob over group name -> weight for every item of that group."},
        "exclude_groups": {"type": ["array", "null"], "items": {"type": "string"}},
        "validation_fraction": {"type": "number", "default": 0.0},
        "split_by": {"type": "string", "enum": ["group", "frame"], "default": "group"},
        "seed": {"type": "integer", "default": 0},
        "merge_into": {"type": ["string", "null"], "description": "Existing trainset.yaml to extend (refinement rounds)."},
        "allow_mixed_lot": {"type": "boolean", "default": False},
        "evaluator": {"type": ["object", "null"], "description": "Evaluator settings stored in the training set (charge_method, ...)."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--structures", action="append", default=None)
    parser.add_argument("--energy-sigma", dest="energy_sigma", type=float, default=2.0)
    parser.add_argument("--eos-sigma-per-atom", dest="eos_sigma_per_atom", type=float, default=0.5)
    parser.add_argument("--high-energy-scale", dest="high_energy_scale", type=float, default=50.0)
    parser.add_argument("--max-relative-energy", dest="max_relative_energy", type=float, default=None)
    parser.add_argument("--forces-kinds", dest="forces_kinds", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--forces-sigma", dest="forces_sigma", type=float, default=5.0)
    parser.add_argument("--forces-weight", dest="forces_weight", type=float, default=1.0)
    parser.add_argument("--forces-every", dest="forces_every", type=int, default=1)
    parser.add_argument("--no-geometry", dest="geometry", action="store_false", default=True)
    parser.add_argument("--distance-sigma", dest="distance_sigma", type=float, default=0.02)
    parser.add_argument("--angle-sigma", dest="angle_sigma", type=float, default=2.0)
    parser.add_argument("--no-charges", dest="charges", action="store_false", default=True)
    parser.add_argument("--charges-sigma", dest="charges_sigma", type=float, default=0.1)
    parser.add_argument("--charges-kinds", dest="charges_kinds", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--lattice-sigma", dest="lattice_sigma", type=float, default=0.05)
    parser.add_argument("--reactions", type=json.loads, default=None)
    parser.add_argument("--group-sigma", dest="group_sigma", type=json.loads, default=None)
    parser.add_argument("--group-weight", dest="group_weight", type=json.loads, default=None)
    parser.add_argument("--exclude-groups", dest="exclude_groups", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--validation-fraction", dest="validation_fraction", type=float, default=0.0)
    parser.add_argument("--split-by", dest="split_by", default="group", choices=["group", "frame"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--merge-into", dest="merge_into", default=None)
    parser.add_argument("--allow-mixed-lot", dest="allow_mixed_lot", action="store_true", default=False)
    parser.add_argument("--evaluator", type=json.loads, default=None)


def _glob_get(mapping, group, default):
    for pat, v in (mapping or {}).items():
        if fnmatch.fnmatch(group, pat):
            return v
    return default


def _formula_key(s):
    return tuple(sorted(s.formula().items()))


def build_items(frames: list, a) -> tuple:
    items, notes = [], []
    frames = [s for s in frames if not any(fnmatch.fnmatch(s.group, p) for p in (a.exclude_groups or []))]
    by_group = {}
    for s in frames:
        by_group.setdefault(s.group or s.name, []).append(s)
    # ---- energies within same-composition groups
    for g, members in sorted(by_group.items()):
        withE = [s for s in members if s.energy is not None]
        subgroups = {}
        for s in withE:
            subgroups.setdefault(_formula_key(s), []).append(s)
        if len(subgroups) > 1:
            notes.append(f"group {g} mixes compositions; energies compared within each composition separately")
        for key, ss in subgroups.items():
            if len(ss) < 2:
                continue
            kind = ss[0].info.get("kind", "")
            per_atom = kind == "eos"
            ref = min(ss, key=lambda s: s.energy / (s.natoms if per_atom else 1))
            sigma0 = a.eos_sigma_per_atom if per_atom else _glob_get(a.group_sigma, g, a.energy_sigma)
            weight = _glob_get(a.group_weight, g, 1.0)
            for s in ss:
                if s is ref:
                    continue
                c = 1.0 / s.natoms if per_atom else 1.0
                dE = (s.energy - ref.energy) * c * EV_TO_KCALMOL
                if a.max_relative_energy is not None and dE * (s.natoms if per_atom else 1) > a.max_relative_energy:
                    continue
                sigma = sigma0 * (1.0 + max(dE, 0.0) / a.high_energy_scale) if a.high_energy_scale else sigma0
                items.append(T.Item(kind="energy", sigma=round(float(sigma), 6), weight=weight, group=g,
                                    id=f"E:{s.name}", terms=[[s.name, c], [ref.name, -c]]))
    # ---- reactions
    for k, r in enumerate(a.reactions or []):
        items.append(T.Item(kind="energy", sigma=float(r.get("sigma", a.energy_sigma)), weight=float(r.get("weight", 1.0)),
                            group=r.get("group", "reactions"), id=r.get("id", f"R:{k:03d}"),
                            terms=[[str(t[0]), float(t[1])] for t in r["terms"]], ref=r.get("ref")))
    # ---- forces
    fk = set(a.forces_kinds or ["rattle", "md", "eos"])
    eligible = [s for s in frames if s.forces is not None and s.info.get("kind", "md") in fk and s.mode == "single"
                and not any(fnmatch.fnmatch(s.group, p) for p in (a.exclude_groups or []))]
    for s in eligible[::max(1, a.forces_every)]:
        items.append(T.Item(kind="forces", sigma=a.forces_sigma, weight=a.forces_weight * _glob_get(a.group_weight, s.group, 1.0),
                            group=f"forces:{s.info.get('kind', 'md')}", id=f"F:{s.name}", structure=s.name))
    # ---- geometry of relaxed molecules
    if a.geometry:
        from ..generate.builders import bonds_of

        for s in frames:
            if s.mode != "relax" or s.info.get("kind") != "molecule" or s.restraints:
                continue
            atoms = SIO.to_atoms(s)
            bonds = bonds_of(atoms)
            for i, j in bonds:
                items.append(T.Item(kind="distance", sigma=a.distance_sigma, group=f"geometry:{s.group}",
                                    id=f"D:{s.name}:{i}-{j}", structure=s.name, atoms=[i, j]))
            nb = {}
            for i, j in bonds:
                nb.setdefault(i, []).append(j)
                nb.setdefault(j, []).append(i)
            for j, ns in nb.items():
                for x in range(len(ns)):
                    for y in range(x + 1, len(ns)):
                        i, k = sorted((ns[x], ns[y]))
                        items.append(T.Item(kind="angle", sigma=a.angle_sigma, group=f"geometry:{s.group}",
                                            id=f"A:{s.name}:{i}-{j}-{k}", structure=s.name, atoms=[i, j, k]))
    # ---- charges
    if a.charges:
        ck = set(a.charges_kinds or ["molecule"])
        for s in frames:
            if s.charges is not None and s.info.get("kind") in ck:
                items.append(T.Item(kind="charges", sigma=a.charges_sigma, group="charges", id=f"Q:{s.name}",
                                    structure=s.name))
    # ---- lattice
    for s in frames:
        if s.mode == "relax_cell":
            for q in ("a", "b", "c"):
                items.append(T.Item(kind="lattice", sigma=a.lattice_sigma, group=f"lattice:{s.group}",
                                    id=f"L:{s.name}:{q}", structure=s.name, quantity=q))
    return items, notes


def split(items: list, frac: float, by: str, seed: int) -> tuple:
    """Hold out whole groups (default) or a fraction of structures. Every item
    touching a held-out structure goes to validation (repeated until no
    training item shares a structure with validation), so nothing leaks."""
    if frac <= 0:
        return items, []
    rng = np.random.default_rng(seed)
    if by == "group":
        groups = sorted({it.group for it in items if it.kind == "energy" and not it.group.startswith("reactions")})
        k = max(1, int(round(frac * len(groups)))) if groups else 0
        val_groups = set(rng.choice(groups, size=k, replace=False)) if k else set()
        val_structs = {n for it in items if it.group in val_groups for n in it.structures()}
    else:
        names = sorted({n for it in items for n in it.structures()})
        k = int(round(frac * len(names)))
        val_structs = set(rng.choice(names, size=k, replace=False)) if k else set()
    while True:
        val = [it for it in items if val_structs & set(it.structures())]
        grown = val_structs | {n for it in val for n in it.structures()}
        if grown == val_structs:
            break
        val_structs = grown
    vid = {it.id for it in val}
    return [it for it in items if it.id not in vid], val


def run(args) -> dict:
    if not args.structures:
        raise ValueError("--structures (labeled extxyz) is required")
    out_dir = Path(args.output_dir or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for f in args.structures:
        frames += SIO.read(f)
    prev = prev_val = None
    new_frames = frames
    if args.merge_into:
        # refinement round: keep every previous item (training and validation) as it is,
        # add items built from the NEW frames only
        prev = T.load(args.merge_into, check=False)
        vpath = Path(args.merge_into).with_name("validset.yaml")
        prev_val = T.load(vpath, check=False) if vpath.is_file() else None
        clash = [s.name for s in frames if s.name in prev.structures]
        if clash:
            raise ValueError(f"structures already in {args.merge_into}: {clash[:5]}; give the new frames new names")
        frames = list(prev.structures.values()) + frames
    names = [s.name for s in frames]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"duplicate structure names across inputs: {dup[:5]}")
    lots = sorted({str(s.info.get("lot", "")) for s in frames if s.energy is not None})
    if len(lots) > 1 and not args.allow_mixed_lot:
        raise ValueError(f"labels from several levels of theory {lots}: one level of theory per fit (relabel, or "
                         "pass allow_mixed_lot if you really mean it)")
    unlabeled = [s.name for s in new_frames if s.energy is None]
    # previous structures are kept whatever their labels (legacy-imported sets carry refs in the items)
    keep_prev = set(prev.structures) if prev is not None else set()
    labeled = [s for s in frames if s.energy is not None or s.name in keep_prev]
    items, notes = build_items([s for s in new_frames if s.energy is not None], args)
    train, val = split(items, args.validation_fraction, args.split_by, args.seed)
    if prev is not None:
        train = prev.items + train
        val = (prev_val.items if prev_val else []) + val
    spath = out_dir / "structures.extxyz"
    SIO.write(spath, labeled)
    structs = {s.name: s for s in labeled}
    ev = args.evaluator or (prev.evaluator if prev else {})
    meta = {"lot": lots[0] if lots else None, "built_by": "reaxff-agent trainset-build"}
    ts = T.TrainingSet(items=train, structures=structs, structures_path=str(spath), evaluator=ev, meta=meta)
    problems = T.validate(ts)
    if problems["errors"]:
        raise ValueError("training set invalid: " + "; ".join(problems["errors"][:8]))
    tpath = ts.write(out_dir / "trainset.yaml", spath)
    out = {"trainset": str(tpath), "structures": str(spath), "n_items": len(train), "counts": ts.counts(),
           "n_data": ts.n_data(), "n_structures_used": len(ts.needed()), "elements": ts.elements(), "lot": meta["lot"],
           "warnings": problems["warnings"][:10] + notes[:10], "unlabeled_skipped": unlabeled[:10]}
    if val:
        vs = T.TrainingSet(items=val, structures=structs, structures_path=str(spath), evaluator=ev, meta=meta)
        vproblems = T.validate(vs)
        if vproblems["errors"]:
            raise ValueError("validation set invalid: " + "; ".join(vproblems["errors"][:8]))
        out["validset"] = str(vs.write(out_dir / "validset.yaml", spath))
        out["n_validation_items"] = len(val)
        out["validation_counts"] = vs.counts()
    out["hint"] = (f"{len(train)} items: by the ParAMS rule of thumb (n items -> at most ~sqrt(n) parameters at once), "
                   f"fit <= ~{int(np.sqrt(len(train)))} parameters per stage [P21]; use params-select presets and sensitivity")
    return out
