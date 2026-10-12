"""Training structures: one packed extended-XYZ file per data set.

Every geometry the training set refers to lives in a single
`structures.extxyz` (never one file per frame: lustre file quotas), each
frame carrying in its info line:

    name        unique id the training set refers to (required)
    group       free label (e.g. "CH_bond_scan", "eos_diamond")
    mode        how ReaxFF evaluates it:
                  single      fixed geometry (energy, forces, charges)
                  relax       minimize atom positions (geometry targets,
                              relaxed energies), honouring restraints
                  relax_cell  also relax the cell (fix box/relax)
    restraints  JSON list, e.g. [["bond", 0, 1, 1.45, 2000.0]] — kind
                (bond|angle|dihedral), 0-based atoms, target (Å or deg),
                force constant (kcal/mol per Å² or rad²). Used by `relax`.
    fixed       JSON list of 0-based atom indices held fixed in `relax`
    charge      total charge (ReaxFF QEq here conserves 0: must be 0)
    lot         level-of-theory key of the label (see labeling/)

Labels use ASE conventions: `energy` (eV), `forces` (eV/Å), `stress`
(eV/Å^3, Voigt) via the frame's calculator, and per-atom `ref_charges` (e).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

MODES = ("single", "relax", "relax_cell")


@dataclass
class Structure:
    """Lightweight, picklable view of one frame for the evaluator."""
    name: str
    symbols: list
    positions: np.ndarray
    cell: np.ndarray
    pbc: tuple
    mode: str = "single"
    restraints: list = field(default_factory=list)
    fixed: list = field(default_factory=list)
    charge: float = 0.0
    group: str = ""
    energy: Optional[float] = None       # eV
    forces: Optional[np.ndarray] = None  # eV/Å
    stress: Optional[np.ndarray] = None  # eV/Å^3 (Voigt 6)
    charges: Optional[np.ndarray] = None  # e
    info: dict = field(default_factory=dict)

    @property
    def natoms(self) -> int:
        return len(self.symbols)

    @property
    def periodic(self) -> bool:
        return any(self.pbc)

    def formula(self) -> dict:
        out = {}
        for s in self.symbols:
            out[s] = out.get(s, 0) + 1
        return out


def _json_field(info, key, default):
    v = info.get(key)
    if v is None or v == "":
        return default
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError as exc:
            raise ValueError(f"frame {info.get('name')!r}: {key} is not valid JSON: {v!r}") from exc
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def from_atoms(atoms, index: int = 0) -> Structure:
    info = dict(atoms.info)
    name = str(info.get("name") or f"frame_{index:05d}")
    mode = str(info.get("mode", "single"))
    if mode not in MODES:
        raise ValueError(f"frame {name!r}: mode {mode!r} must be one of {MODES}")
    energy = forces = stress = None
    calc = atoms.calc
    if calc is not None:
        res = getattr(calc, "results", {}) or {}
        energy = res.get("energy", res.get("free_energy"))
        forces = res.get("forces")
        stress = res.get("stress")
    if energy is None and "energy" in info:
        energy = float(info["energy"])
    if forces is None and "forces" in atoms.arrays:
        forces = atoms.arrays["forces"]
    charges = atoms.arrays.get("ref_charges")
    return Structure(
        name=name,
        symbols=list(atoms.get_chemical_symbols()),
        positions=np.array(atoms.get_positions(), dtype=float),
        cell=np.array(atoms.cell.array, dtype=float),
        pbc=tuple(bool(x) for x in atoms.pbc),
        mode=mode,
        restraints=_json_field(info, "restraints", []),
        fixed=[int(i) for i in _json_field(info, "fixed", [])],
        charge=float(info.get("charge", 0.0) or 0.0),
        group=str(info.get("group", "")),
        energy=None if energy is None else float(energy),
        forces=None if forces is None else np.array(forces, dtype=float),
        stress=None if stress is None else np.array(stress, dtype=float),
        charges=None if charges is None else np.array(charges, dtype=float),
        info={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()
              if isinstance(v, (str, int, float, bool, np.ndarray)) and k not in ("energy", "free_energy", "stress")},
    )


def to_atoms(s: Structure):
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    atoms = Atoms(symbols=s.symbols, positions=s.positions, cell=s.cell, pbc=s.pbc)
    info = dict(s.info)
    info.update({"name": s.name, "mode": s.mode})
    if s.group:
        info["group"] = s.group
    if s.restraints:
        info["restraints"] = json.dumps(s.restraints)
    else:
        info.pop("restraints", None)
    if s.fixed:
        info["fixed"] = json.dumps(s.fixed)
    if s.charge:
        info["charge"] = s.charge
    for k in ("energy", "forces", "stress", "free_energy"):
        info.pop(k, None)
    atoms.info.update(info)
    if s.charges is not None:
        atoms.arrays["ref_charges"] = np.asarray(s.charges, dtype=float)
    results = {}
    if s.energy is not None:
        results["energy"] = s.energy
    if s.forces is not None:
        results["forces"] = s.forces
    if s.stress is not None:
        results["stress"] = s.stress
    if results:
        atoms.calc = SinglePointCalculator(atoms, **results)
    return atoms


def read(path) -> list:
    """All frames of an extxyz (or any ASE-readable file) as Structures.
    Names must be unique."""
    from ase.io import read as ase_read

    frames = ase_read(str(path), index=":")
    out = [from_atoms(a, i) for i, a in enumerate(frames)]
    seen = {}
    for s in out:
        if s.name in seen:
            raise ValueError(f"{path}: structure name {s.name!r} appears twice (frames {seen[s.name]} and later); "
                             "names must be unique")
        seen[s.name] = True
    return out


def read_dict(path) -> dict:
    return {s.name: s for s in read(path)}


def write(path, structures, append: bool = False) -> Path:
    from ase.io import write as ase_write

    path = Path(path)
    ase_write(str(path), [to_atoms(s) for s in structures], format="extxyz", append=append)
    return path


def scan_atoms(s: Structure) -> list:
    """The scanned atoms of a scan frame as ints (stored as "i,j" by the
    generator; ASE may read it back as an array)."""
    v = s.info.get("scan_atoms")
    if v is None or v == "":
        return []
    if isinstance(v, str):
        return [int(x) for x in v.split(",") if x.strip()]
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    return [int(v)]


def check_neutral(s: Structure) -> None:
    if abs(s.charge) > 1e-8:
        raise ValueError(f"structure {s.name!r} has total charge {s.charge}: LAMMPS fix qeq/reaxff and acks2/reaxff "
                         "conserve zero total charge, so charged structures cannot be evaluated consistently")
