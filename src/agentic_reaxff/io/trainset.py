"""The training set: what a force field is fitted to, and how much each
datum counts.

A training set is a YAML file plus the packed structures it refers to:

```yaml
version: 1
structures: structures.extxyz        # relative to this file
evaluator: {charge_method: qeq}      # optional EvalSettings overrides
items:
  - kind: energy                     # Σ c_k E_k  (a reaction / relative energy)
    terms: [[ch3_h_r1.60, 1], [ch4_eq, -1]]
    ref: 41.2                        # kcal/mol
    sigma: 2.0                       # accepted error, kcal/mol
    weight: 1.0                      # unitless importance
    group: C-H dissociation
  - kind: forces                     # per-atom forces of one structure
    structure: md_0007               # reference from structures.extxyz
    sigma: 5.0                       # kcal/mol/Å
  - kind: charges                    # reference: ref_charges or `ref`
    structure: ch4_eq
    atoms: [0]
    sigma: 0.1                       # e
  - kind: distance | angle | dihedral   # on the ReaxFF-relaxed geometry
    structure: ch4_eq
    atoms: [0, 1]
    ref: 1.090                       # default: measured on the reference geometry
    sigma: 0.02                      # Å or degrees
  - kind: lattice                    # a|b|c|alpha|beta|gamma|volume after relax_cell
    structure: diamond_eq
    quantity: a
    sigma: 0.05
  - kind: stress                     # GPa, Voigt; reference from the structure
    structure: diamond_v0.95
    sigma: 1.0
```

Loss (the van Duin / ParAMS sum of squares, [vD94, Sh19, P21]):

    L = Σ_items  w_i · mean_c ((pred_ic − ref_ic) / σ_i)²

An array item (forces, charges, stress) splits its weight over its
components, as ParAMS does, so one 100-atom force set counts like one
energy unless its weight says otherwise. Dihedral residuals are periodic.

Energy items must be stoichiometrically balanced (ReaxFF absolute energies
are not comparable with DFT total energies; only reaction/relative energies
are). `allow_unbalanced: true` on an item overrides this, deliberately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from . import structures as structures_io
from .units import EV_TO_KCALMOL, EV_PER_A3_TO_GPA

KINDS = ("energy", "forces", "charges", "distance", "angle", "dihedral", "lattice", "stress")
LATTICE_QUANTITIES = ("a", "b", "c", "alpha", "beta", "gamma", "volume")

# Default accepted errors (sigma), in toolkit units. Energies: classic
# ReaxFF practice uses 1-5 kcal/mol, ParAMS 0.002 Ha = 1.25 kcal/mol;
# distances 0.05 Å, angles 2 deg, charges 0.1 e and cell lengths 0.05 Å
# are ParAMS' extractor defaults [P21]. Forces: ParAMS lists 0.003 Ha/bohr
# (3.6 kcal/mol/Å); 5 kcal/mol/Å is used here because DFT-MD forces are
# large and ReaxFF force errors of 5-10 kcal/mol/Å are typical of good fits.
DEFAULT_SIGMA = {
    "energy": 2.0, "forces": 5.0, "charges": 0.1, "distance": 0.05, "angle": 2.0, "dihedral": 5.0,
    "lattice": 0.05, "stress": 1.0,
}
UNITS = {"energy": "kcal/mol", "forces": "kcal/mol/Å", "charges": "e", "distance": "Å", "angle": "deg",
         "dihedral": "deg", "lattice": "Å or deg or Å^3", "stress": "GPa"}

FAIL_RESIDUAL = 1.0e3  # normalized residual given to every component of an item whose structure failed


class TrainsetError(ValueError):
    pass


@dataclass
class Item:
    kind: str
    sigma: float
    weight: float = 1.0
    group: str = ""
    id: str = ""
    terms: list = field(default_factory=list)       # energy: [[name, coeff], ...]
    structure: str = ""
    atoms: Optional[list] = None
    quantity: str = ""
    ref: object = None
    allow_unbalanced: bool = False

    def structures(self) -> list:
        if self.kind == "energy":
            return [t[0] for t in self.terms]
        return [self.structure]

    def to_dict(self) -> dict:
        d = {"id": self.id, "kind": self.kind}
        if self.kind == "energy":
            d["terms"] = [[n, float(c)] for n, c in self.terms]
        else:
            d["structure"] = self.structure
        if self.atoms is not None:
            d["atoms"] = list(self.atoms)
        if self.quantity:
            d["quantity"] = self.quantity
        if self.ref is not None:
            d["ref"] = self.ref.tolist() if isinstance(self.ref, np.ndarray) else self.ref
        d["sigma"] = self.sigma
        if self.weight != 1.0:
            d["weight"] = self.weight
        if self.group:
            d["group"] = self.group
        if self.allow_unbalanced:
            d["allow_unbalanced"] = True
        return d


@dataclass
class TrainingSet:
    items: list
    structures: dict          # name -> Structure
    structures_path: Optional[str] = None
    evaluator: dict = field(default_factory=dict)
    path: Optional[str] = None
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------- queries
    def needed(self) -> dict:
        """name -> {"forces": bool, "geometry": bool} for every structure used."""
        need = {}
        for it in self.items:
            for name in it.structures():
                n = need.setdefault(name, {"forces": False, "geometry": False})
                if it.kind == "forces":
                    n["forces"] = True
                if it.kind in ("distance", "angle", "dihedral", "lattice"):
                    n["geometry"] = True
        return need

    def used_structures(self) -> list:
        return [self.structures[n] for n in self.needed()]

    def elements(self) -> list:
        els = []
        for s in self.used_structures():
            for e in s.symbols:
                if e not in els:
                    els.append(e)
        return els

    def counts(self) -> dict:
        out = {}
        for it in self.items:
            out[it.kind] = out.get(it.kind, 0) + 1
        return out

    def n_data(self) -> int:
        """Number of scalar reference values (array items count every component)."""
        n = 0
        for it in self.items:
            r = reference_values(self, it)
            n += int(np.size(r))
        return n

    def groups(self) -> list:
        return sorted({it.group or it.kind for it in self.items})

    def subset(self, item_ids) -> "TrainingSet":
        keep = set(item_ids)
        items = [it for it in self.items if it.id in keep]
        return TrainingSet(items=items, structures=self.structures, structures_path=self.structures_path,
                           evaluator=dict(self.evaluator), path=self.path, meta=dict(self.meta))

    # ---------------------------------------------------------------- I/O
    def to_dict(self, structures_rel: Optional[str] = None) -> dict:
        return {"version": 1, "structures": structures_rel or self.structures_path,
                **({"evaluator": self.evaluator} if self.evaluator else {}),
                **({"meta": self.meta} if self.meta else {}),
                "items": [it.to_dict() for it in self.items]}

    def write(self, path, structures_path=None) -> Path:
        path = Path(path)
        spath = Path(structures_path or self.structures_path)
        try:
            rel = str(spath.resolve().relative_to(path.resolve().parent))
        except ValueError:
            rel = str(spath.resolve())
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(rel), f, sort_keys=False, width=120)
        return path


def _item_from_dict(d: dict, k: int) -> Item:
    known = {"id", "kind", "terms", "structure", "atoms", "quantity", "ref", "sigma", "weight", "group",
             "allow_unbalanced"}
    unknown = set(d) - known
    if unknown:
        raise TrainsetError(f"item {k}: unknown key(s) {sorted(unknown)}; known: {sorted(known)}")
    kind = d.get("kind")
    if kind not in KINDS:
        raise TrainsetError(f"item {k}: kind {kind!r} must be one of {KINDS}")
    sigma = float(d.get("sigma", DEFAULT_SIGMA[kind]))
    if not sigma > 0:
        raise TrainsetError(f"item {k}: sigma must be > 0")
    it = Item(kind=kind, sigma=sigma, weight=float(d.get("weight", 1.0)), group=str(d.get("group", "")),
              id=str(d.get("id") or f"{kind}_{k:05d}"), terms=[[str(t[0]), float(t[1])] for t in d.get("terms", [])],
              structure=str(d.get("structure", "")), atoms=d.get("atoms"), quantity=str(d.get("quantity", "")),
              ref=d.get("ref"), allow_unbalanced=bool(d.get("allow_unbalanced", False)))
    if it.weight < 0:
        raise TrainsetError(f"item {it.id}: weight must be >= 0")
    if kind == "energy" and not it.terms:
        raise TrainsetError(f"item {it.id}: energy items need terms [[structure, coefficient], ...]")
    if kind != "energy" and not it.structure:
        raise TrainsetError(f"item {it.id}: {kind} items need a structure")
    natoms_needed = {"distance": 2, "angle": 3, "dihedral": 4}
    if kind in natoms_needed and (not it.atoms or len(it.atoms) != natoms_needed[kind]):
        raise TrainsetError(f"item {it.id}: {kind} needs atoms with {natoms_needed[kind]} indices")
    if kind == "lattice" and it.quantity not in LATTICE_QUANTITIES:
        raise TrainsetError(f"item {it.id}: lattice quantity must be one of {LATTICE_QUANTITIES}")
    return it


def load(path, *, check: bool = True) -> TrainingSet:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    unknown = set(data) - {"version", "structures", "evaluator", "items", "meta"}
    if unknown:
        raise TrainsetError(f"{path}: unknown top-level key(s) {sorted(unknown)}")
    spath = data.get("structures")
    if not spath:
        raise TrainsetError(f"{path}: `structures:` (the packed extxyz) is required")
    spath = Path(spath)
    if not spath.is_absolute():
        spath = path.parent / spath
    structs = structures_io.read_dict(spath)
    items = [_item_from_dict(d, k) for k, d in enumerate(data.get("items") or [])]
    ts = TrainingSet(items=items, structures=structs, structures_path=str(spath), evaluator=data.get("evaluator") or {},
                     path=str(path), meta=data.get("meta") or {})
    ids = [it.id for it in items]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})[:5]
        raise TrainsetError(f"{path}: duplicate item ids {dup}")
    if check:
        problems = validate(ts)
        if problems["errors"]:
            raise TrainsetError(f"{path}: " + "; ".join(problems["errors"][:10]))
    return ts


# ----------------------------------------------------------------- checks
def formula_balance(ts: TrainingSet, it: Item) -> dict:
    net = {}
    for name, c in it.terms:
        for el, n in ts.structures[name].formula().items():
            net[el] = net.get(el, 0.0) + c * n
    return {el: v for el, v in net.items() if abs(v) > 1e-8}


def validate(ts: TrainingSet) -> dict:
    """Errors (refuse to fit) and warnings (fit, but say so)."""
    errors, warnings = [], []
    for it in ts.items:
        missing = [name for name in it.structures() if name not in ts.structures]
        if missing:
            errors += [f"item {it.id}: structure {name!r} is not in {ts.structures_path}" for name in missing]
            continue
        if it.kind == "energy":
            net = formula_balance(ts, it)
            if net and not it.allow_unbalanced:
                errors.append(f"item {it.id}: energy terms are not stoichiometrically balanced (net {net}); ReaxFF and "
                              "DFT absolute energies are not comparable. Balance it or set allow_unbalanced: true")
        s = ts.structures[it.structure] if it.kind != "energy" else None
        if it.kind in ("distance", "angle", "dihedral") and s.mode == "single":
            warnings.append(f"item {it.id}: {it.kind} on {s.name}, which is evaluated without relaxation "
                            "(mode single): prediction = input geometry")
        if it.kind == "stress" and not all(s.pbc):
            errors.append(f"item {it.id}: stress needs a fully periodic structure ({s.name} has pbc {s.pbc})")
        if it.kind == "lattice" and s.mode != "relax_cell":
            warnings.append(f"item {it.id}: lattice item on {s.name}, which is not relaxed (mode {s.mode})")
        if s is not None and it.atoms is not None and any(not 0 <= int(a) < s.natoms for a in it.atoms):
            errors.append(f"item {it.id}: atom index out of range for {s.name} ({s.natoms} atoms)")
        try:
            reference_values(ts, it)
        except TrainsetError as exc:
            errors.append(str(exc))
    for s in (ts.structures[n] for n in ts.needed() if n in ts.structures):
        if abs(s.charge) > 1e-8:
            errors.append(f"structure {s.name}: total charge {s.charge}; LAMMPS QEq/ACKS2 conserve zero charge")
    return {"errors": errors, "warnings": warnings}


# --------------------------------------------------------- reference values
def _measure(kind, positions, cell, pbc, atoms):
    from ase import Atoms

    a = Atoms(positions=positions, cell=cell, pbc=pbc)
    mic = bool(any(pbc))
    if kind == "distance":
        return float(a.get_distance(atoms[0], atoms[1], mic=mic))
    if kind == "angle":
        return float(a.get_angle(atoms[0], atoms[1], atoms[2], mic=mic))
    if kind == "dihedral":
        return float(a.get_dihedral(atoms[0], atoms[1], atoms[2], atoms[3], mic=mic))
    raise ValueError(kind)


def lattice_value(cell, quantity):
    from ase.geometry import cell_to_cellpar

    cell = np.asarray(cell, float)
    if quantity == "volume":
        return float(abs(np.linalg.det(cell)))
    par = cell_to_cellpar(cell)
    return float(par["abc".index(quantity)] if quantity in "abc" else par[3 + ["alpha", "beta", "gamma"].index(quantity)])


def reference_values(ts: TrainingSet, it: Item):
    """The item's reference in toolkit units (scalar or array)."""
    if it.ref is not None and it.kind not in ("forces", "stress"):
        return np.asarray(it.ref, float) if isinstance(it.ref, (list, tuple)) else float(it.ref)
    if it.kind == "energy":
        total = 0.0
        for name, c in it.terms:
            e = ts.structures[name].energy
            if e is None:
                raise TrainsetError(f"item {it.id}: no `ref` and structure {name} has no energy label")
            total += c * e
        return total * EV_TO_KCALMOL
    s = ts.structures[it.structure]
    if it.kind == "forces":
        if it.ref is not None:
            return np.asarray(it.ref, float).reshape(-1, 3)
        if s.forces is None:
            raise TrainsetError(f"item {it.id}: structure {s.name} has no forces label")
        f = s.forces * EV_TO_KCALMOL
        return f[np.asarray(it.atoms, int)] if it.atoms is not None else f
    if it.kind == "charges":
        if s.charges is None:
            raise TrainsetError(f"item {it.id}: no `ref` and structure {s.name} has no ref_charges")
        return s.charges[np.asarray(it.atoms, int)] if it.atoms is not None else s.charges
    if it.kind in ("distance", "angle", "dihedral"):
        return _measure(it.kind, s.positions, s.cell, s.pbc, it.atoms)
    if it.kind == "lattice":
        return lattice_value(s.cell, it.quantity)
    if it.kind == "stress":
        if it.ref is not None:
            return np.asarray(it.ref, float)
        if s.stress is None:
            raise TrainsetError(f"item {it.id}: structure {s.name} has no stress label")
        st = np.asarray(s.stress, float)
        if st.shape == (3, 3):
            st = np.array([st[0, 0], st[1, 1], st[2, 2], st[1, 2], st[0, 2], st[0, 1]])
        return st * EV_PER_A3_TO_GPA
    raise TrainsetError(f"item {it.id}: unknown kind {it.kind}")


def predicted_values(ts: TrainingSet, it: Item, results: dict):
    """The item's ReaxFF prediction from per-structure results (name ->
    EvalResult-like with energy/forces/charges/positions/cell/stress), or
    None when any needed structure failed."""
    for name in it.structures():
        r = results.get(name)
        if r is None or not r.ok:
            return None
    if it.kind == "energy":
        return float(sum(c * results[name].energy for name, c in it.terms))
    r = results[it.structure]
    s = ts.structures[it.structure]
    if it.kind == "forces":
        f = np.asarray(r.forces)
        return f[np.asarray(it.atoms, int)] if it.atoms is not None else f
    if it.kind == "charges":
        q = np.asarray(r.charges)
        return q[np.asarray(it.atoms, int)] if it.atoms is not None else q
    if it.kind in ("distance", "angle", "dihedral"):
        pos = r.positions if r.positions is not None else s.positions
        cell = r.cell if r.cell is not None else s.cell
        return _measure(it.kind, pos, cell, s.pbc, it.atoms)
    if it.kind == "lattice":
        return lattice_value(r.cell if r.cell is not None else s.cell, it.quantity)
    if it.kind == "stress":
        return np.asarray(r.stress) if r.stress is not None else None
    return None


def residuals(it: Item, pred, ref) -> np.ndarray:
    """Normalized residuals (pred - ref)/sigma, flattened; periodic for dihedrals."""
    if pred is None:
        n = int(np.size(ref)) or 1
        return np.full(n, FAIL_RESIDUAL)
    d = np.atleast_1d(np.asarray(pred, float) - np.asarray(ref, float)).ravel()
    if it.kind == "dihedral":
        d = (d + 180.0) % 360.0 - 180.0
    return d / it.sigma


def item_loss(it: Item, r: np.ndarray) -> float:
    return float(it.weight * np.mean(r * r)) if r.size else 0.0


def natural_errors(it: Item, pred, ref) -> Optional[np.ndarray]:
    if pred is None:
        return None
    d = np.atleast_1d(np.asarray(pred, float) - np.asarray(ref, float)).ravel()
    if it.kind == "dihedral":
        d = (d + 180.0) % 360.0 - 180.0
    return d


def summarize_errors(ts: TrainingSet, preds: dict) -> dict:
    """Per kind and per group: n, RMSE, MAE, max |error| in natural units,
    and the loss share. `preds` maps item id -> prediction (None = failed)."""
    by = {}
    total = 0.0
    n_failed = 0
    per_item = []
    for it in ts.items:
        ref = reference_values(ts, it)
        p = preds.get(it.id)
        r = residuals(it, p, ref)
        loss = item_loss(it, r)
        total += loss
        per_item.append((loss, it.id))
        err = natural_errors(it, p, ref)
        for key in (f"kind:{it.kind}", f"group:{it.group or it.kind}"):
            b = by.setdefault(key, {"n_items": 0, "n_failed": 0, "loss": 0.0, "_e": []})
            b["n_items"] += 1
            b["loss"] += loss
            if err is None:
                b["n_failed"] += 1
            else:
                b["_e"].append(err)
        if err is None:
            n_failed += 1
    out = {"loss": total, "n_items": len(ts.items), "n_failed_items": n_failed, "kinds": {}, "groups": {}}
    for key, b in by.items():
        e = np.concatenate(b.pop("_e")) if b["_e"] else np.array([])
        kind = key.split(":", 1)[1]
        stats = {**b, "loss_fraction": b["loss"] / total if total > 0 else 0.0}
        if e.size:
            stats.update({"rmse": float(math.sqrt(np.mean(e * e))), "mae": float(np.mean(np.abs(e))),
                          "max_abs": float(np.max(np.abs(e)))})
        (out["kinds"] if key.startswith("kind:") else out["groups"])[kind] = stats
    for k, v in out["kinds"].items():
        v["units"] = UNITS.get(k, "")
    per_item.sort(reverse=True)
    out["top_items"] = [{"id": i, "loss": l} for l, i in per_item[:10]]
    out["top10_loss_fraction"] = (sum(l for l, _ in per_item[:10]) / total) if total > 0 else 0.0
    return out
