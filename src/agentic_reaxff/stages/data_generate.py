"""Generate unlabeled reference geometries for a ReaxFF training set.

One packed `structures.extxyz` comes out (frames named, grouped and tagged
with how ReaxFF and the DFT labeler should treat them: single point, or
relaxation holding a restrained coordinate). Labels come next from `label`.

Typical recipe (--json-in):

```json
{"molecules": ["CH4", "C2H6", "C2H4", "H2O", "CH3OH", "smiles:CC(=O)O"],
 "scans": ["bond", "angle", "torsion"],
 "dimers": true,
 "crystals": ["C:diamond:3.567"],
 "rattle": {"n": 4, "sigma": 0.08},
 "trajectories": [{"path": "md/dump.extxyz", "stride": 20}]}
```

What to include, and why, is in the `reaxff-training-set` skill: every
bond type needs a dissociation curve, every angle type a bending scan,
condensed phases an equation of state, and forces on distorted frames make
the fit far better determined than energies alone.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..generate import builders as B
from ..io import structures as SIO

NAME = "data-generate"
SUMMARY = "Generate reference geometries: molecules, bond/angle/torsion scans, dimers, EOS, rattled and MD frames (one packed extxyz)."
SCHEMA = {
    "type": "object",
    "properties": {
        "molecules": {"type": ["array", "null"], "items": {"type": "string"},
                      "description": "ASE g2 names, 'smiles:...' (RDKit) or structure files."},
        "molecule_mode": {"type": "string", "enum": ["relax", "single"], "default": "relax"},
        "scans": {"type": ["array", "null"], "items": {"type": "string", "enum": ["bond", "angle", "torsion"]}},
        "bond_factors": {"type": ["array", "null"], "items": {"type": "number"}, "description": "r/r0 values (default 0.75..2.5)."},
        "angle_deltas": {"type": ["array", "null"], "items": {"type": "number"}, "description": "degrees (default -40..40)."},
        "torsion_step": {"type": "number", "default": 30.0},
        "all_bonds": {"type": "boolean", "default": False, "description": "Scan every bond, not one per element pair per molecule."},
        "rigid_scans": {"type": "boolean", "default": False, "description": "Bond scans as rigid single points instead of restrained relaxations."},
        "dimers": {"type": ["boolean", "array", "null"], "description": "true: all element pairs present; or a list of elements."},
        "crystals": {"type": ["array", "null"], "items": {"type": "string"}, "description": "'El:structure:a' or files (CIF, POSCAR...)."},
        "volume_ratios": {"type": ["array", "null"], "items": {"type": "number"}},
        "crystal_supercell": {"type": ["array", "null"], "items": {"type": "integer"}},
        "rattle": {"type": ["object", "null"], "description": "{n, sigma, strain, of: [molecules|crystals|scans]}"},
        "trajectories": {"type": ["array", "null"], "items": {"type": "object"}, "description": "[{path, stride, max_frames, tag}]"},
        "boxes": {"type": ["array", "null"], "items": {"type": "object"},
                  "description": "MD validation boxes: [{name, molecules: {CH4: 10, O2: 20}, density (g/cc) | box (Å), seed}]; written to boxes.extxyz, not the training structures."},
        "append_to": {"type": ["string", "null"], "description": "Existing structures file to extend (names must stay unique)."},
        "seed": {"type": "integer", "default": 0},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--molecules", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--molecule-mode", dest="molecule_mode", default="relax", choices=["relax", "single"])
    parser.add_argument("--scans", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--bond-factors", dest="bond_factors", type=lambda s: [float(x) for x in s.split(",")], default=None)
    parser.add_argument("--angle-deltas", dest="angle_deltas", type=lambda s: [float(x) for x in s.split(",")], default=None)
    parser.add_argument("--torsion-step", dest="torsion_step", type=float, default=30.0)
    parser.add_argument("--all-bonds", dest="all_bonds", action="store_true", default=False)
    parser.add_argument("--rigid-scans", dest="rigid_scans", action="store_true", default=False)
    parser.add_argument("--dimers", type=lambda s: True if s.lower() in ("true", "all", "1") else [x for x in s.split(",") if x],
                        default=None)
    parser.add_argument("--crystals", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--volume-ratios", dest="volume_ratios", type=lambda s: [float(x) for x in s.split(",")], default=None)
    parser.add_argument("--crystal-supercell", dest="crystal_supercell", type=lambda s: [int(x) for x in s.split(",")], default=None)
    parser.add_argument("--rattle", type=json.loads, default=None)
    parser.add_argument("--trajectories", type=json.loads, default=None)
    parser.add_argument("--boxes", type=json.loads, default=None)
    parser.add_argument("--append-to", dest="append_to", default=None)
    parser.add_argument("--seed", type=int, default=0)


def run(args) -> dict:
    out_dir = Path(args.output_dir or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    mols = B.molecules(args.molecules or [], mode=args.molecule_mode)
    frames += mols
    scans = []
    for m in mols:
        if "bond" in (args.scans or []):
            scans += B.bond_scans(m, factors=args.bond_factors, all_bonds=args.all_bonds, restrained=not args.rigid_scans)
        if "angle" in (args.scans or []):
            scans += B.angle_scans(m, deltas=args.angle_deltas)
        if "torsion" in (args.scans or []):
            scans += B.torsion_scans(m, step=args.torsion_step)
    frames += scans
    crystals = B.eos(args.crystals or [], volume_ratios=args.volume_ratios, supercell=args.crystal_supercell)
    frames += crystals
    if args.dimers:
        els = args.dimers if isinstance(args.dimers, list) else sorted({e for s in frames for e in s.symbols})
        frames += B.dimers(els)
    if args.rattle:
        r = dict(args.rattle)
        of = r.pop("of", ["molecules", "crystals"])
        bases = (mols if "molecules" in of else []) + ([c for c in crystals if c.mode == "relax_cell"] if "crystals" in of else []) \
            + (scans if "scans" in of else [])
        frames += B.rattle(bases, n=int(r.get("n", 4)), sigma=float(r.get("sigma", 0.08)), strain=float(r.get("strain", 0.0)),
                           seed=args.seed)
    for t in args.trajectories or []:
        frames += B.from_trajectory(t["path"], stride=int(t.get("stride", 10)), max_frames=t.get("max_frames"),
                                    tag=t.get("tag"))
    boxes_path = None
    if args.boxes:
        boxes = [B.pack_box(b["molecules"], density=b.get("density"), box=b.get("box"), name=b.get("name", f"box{k}"),
                            seed=int(b.get("seed", args.seed))) for k, b in enumerate(args.boxes)]
        boxes_path = out_dir / "boxes.extxyz"
        SIO.write(boxes_path, boxes)
    if args.append_to:
        old = SIO.read(args.append_to)
        frames = old + frames
    names = [s.name for s in frames]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"duplicate structure names {dup[:5]}: give different molecules/tags, or drop the duplicates")
    if not frames and boxes_path:
        return {"boxes": str(boxes_path), "n_boxes": len(args.boxes), "next": "md-check --systems " + str(boxes_path)}
    if not frames:
        raise ValueError("nothing to generate: give molecules, crystals, dimers, trajectories or boxes")
    path = out_dir / "structures.extxyz"
    SIO.write(path, frames)
    counts, groups = {}, {}
    for s in frames:
        k = s.info.get("kind", "?")
        counts[k] = counts.get(k, 0) + 1
        groups[s.group] = groups.get(s.group, 0) + 1
    report = {"structures": str(path), "n_frames": len(frames), "by_kind": counts, "n_groups": len(groups),
              "groups": groups, "elements": sorted({e for s in frames for e in s.symbols}),
              "n_atoms_max": max(s.natoms for s in frames)}
    (out_dir / "generate_report.json").write_text(json.dumps(report, indent=1))
    if boxes_path:
        report["boxes"] = str(boxes_path)
    return {k: v for k, v in report.items() if k != "groups"} | {"report": str(out_dir / "generate_report.json"),
                                                                 "next": "label the frames: reaxff-agent label --backend qe|xtb|ffield"}
