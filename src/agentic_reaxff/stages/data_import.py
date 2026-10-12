"""Bring externally labeled structures (VASP, ORCA, Gaussian, CP2K, an
existing extxyz or ASE trajectory, ...) into the canonical packed format.

Any ASE-readable file works. Energies/forces/stresses are taken as ASE
reports them (eV, eV/Å, eV/Å^3). Each frame gets a unique `name`
(`<prefix>_<index>` unless the file carries names), a `group`, mode
`single` (or what the file says), and the level-of-theory key you give with
`--lot` (required: the toolkit refuses to mix levels of theory in one fit,
so it must know which one this is). Per-atom charges can be mapped from an
array in the file with `--charges-array`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..io import structures as SIO

NAME = "data-import"
SUMMARY = "Import externally labeled structures (any ASE-readable format) into one packed extxyz with names, groups and a level-of-theory key."
SCHEMA = {
    "type": "object",
    "required": ["files", "lot"],
    "properties": {
        "files": {"type": "array", "items": {"type": "string"}},
        "format": {"type": ["string", "null"], "description": "ASE format name if it cannot be guessed."},
        "index": {"type": "string", "default": ":", "description": "ASE index: ':' all frames, '-1' last, '::10' every 10th."},
        "lot": {"type": "string", "description": "Level-of-theory key, e.g. 'vasp-pbe-d3-520eV'. Required."},
        "prefix": {"type": ["string", "null"]},
        "group": {"type": ["string", "null"], "description": "Group label (default: per file)."},
        "kind": {"type": "string", "default": "md"},
        "mode": {"type": "string", "enum": ["single", "relax", "relax_cell"], "default": "single"},
        "charges_array": {"type": ["string", "null"], "description": "Per-atom array holding reference charges (e)."},
        "require_forces": {"type": "boolean", "default": False},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--files", action="append", default=None)
    parser.add_argument("--format", default=None)
    parser.add_argument("--index", default=":")
    parser.add_argument("--lot", default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--group", default=None)
    parser.add_argument("--kind", default="md")
    parser.add_argument("--mode", default="single", choices=["single", "relax", "relax_cell"])
    parser.add_argument("--charges-array", dest="charges_array", default=None)
    parser.add_argument("--require-forces", dest="require_forces", action="store_true", default=False)


def run(args) -> dict:
    from ase.io import read

    if not args.files:
        raise ValueError("--files is required")
    if not args.lot:
        raise ValueError("--lot (level-of-theory key) is required")
    out = []
    skipped = {}
    for fpath in args.files:
        frames = read(fpath, index=args.index, format=args.format)
        if not isinstance(frames, list):
            frames = [frames]
        stem = args.prefix or Path(fpath).stem.replace(".", "_")
        for k, a in enumerate(frames):
            name = str(a.info.get("name") or f"{stem}_{k:05d}")
            try:
                e = a.get_potential_energy()
            except Exception:  # noqa: BLE001
                skipped[name] = "no energy"
                continue
            try:
                f = a.get_forces()
            except Exception:  # noqa: BLE001
                f = None
            if args.require_forces and f is None:
                skipped[name] = "no forces"
                continue
            s = SIO.from_atoms(a, k)
            s.name = name
            s.energy = float(e)
            s.forces = None if f is None else np.asarray(f, float)
            try:
                s.stress = a.get_stress() if any(a.pbc) else None
            except Exception:  # noqa: BLE001
                s.stress = None
            if args.charges_array:
                if args.charges_array not in a.arrays:
                    raise ValueError(f"{fpath} frame {k}: no per-atom array {args.charges_array!r} (have {sorted(a.arrays)})")
                s.charges = np.asarray(a.arrays[args.charges_array], float)
            s.group = args.group or f"import_{stem}"
            s.mode = args.mode
            s.info.update({"kind": args.kind, "lot": args.lot, "source": str(fpath)})
            out.append(s)
    names = [s.name for s in out]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"duplicate names {dup[:5]}: give --prefix per file")
    if not out:
        raise ValueError(f"no labeled frames found; skipped: {dict(list(skipped.items())[:5])}")
    out_dir = Path(args.output_dir or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "labeled.extxyz"
    SIO.write(path, out)
    (out_dir / "provenance.json").write_text(json.dumps({"lot": args.lot, "backend": "import", "files": args.files,
                                                         "n_labeled": len(out), "skipped": skipped}, indent=1))
    return {"labeled": str(path), "n_labeled": len(out), "n_skipped": len(skipped), "lot": args.lot,
            "elements": sorted({e for s in out for e in s.symbols}),
            "with_forces": sum(s.forces is not None for s in out), "with_charges": sum(s.charges is not None for s in out)}
