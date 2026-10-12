"""Build the reference geometries a ReaxFF training set is made of.

The classic ReaxFF training set ([vD01, C08, S16]; van Duin's 2008 training
workshop) has, for every element combination the force field must describe:

  equilibrium molecules     geometries (bond lengths/angles) and charges
  bond dissociation curves  one bond stretched, the rest relaxed under a
                            restraint, out to the dissociated limit
  angle / torsion scans     valence-angle bending, rotation about bonds
  equations of state        condensed phases compressed and expanded
  reaction energies         balanced combinations of the above

and modern fits add DFT forces on distorted and MD-sampled frames
(ParAMS [P21], De Angelis 2024 [DA24], Dumortier 2024 [Du24]). Diatomic
curves down to short distance give the inner wall / short-range repulsion
something to fit (too-soft short-range repulsion is a known ReaxFF failure
in energetic collisions; Caro 2020 [Ca20]).

Every builder returns `Structure`s with `name`, `group`, `mode` and, for
scans, the restraint that holds the scanned coordinate (the DFT labeler
applies the same coordinate constraint), plus `info` keys:

    kind        molecule | bond_scan | angle_scan | torsion_scan | eos | dimer | rattle | md
    scan_value  the scanned coordinate (Å, deg, or volume ratio)
    parent      the structure the frame derives from
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..io.structures import Structure, from_atoms


# ---------------------------------------------------------------- helpers
def _cov_radius(symbol: str) -> float:
    from ase.data import atomic_numbers, covalent_radii

    return float(covalent_radii[atomic_numbers[symbol]])


def bonds_of(atoms, scale: float = 1.25) -> list:
    """(i, j) pairs closer than scale x (sum of covalent radii)."""
    pos = atoms.get_positions()
    sym = atoms.get_chemical_symbols()
    out = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            d = atoms.get_distance(i, j, mic=any(atoms.pbc))
            if d < scale * (_cov_radius(sym[i]) + _cov_radius(sym[j])):
                out.append((i, j))
    return out


def _neighbors(n, bonds):
    nb = {i: set() for i in range(n)}
    for i, j in bonds:
        nb[i].add(j)
        nb[j].add(i)
    return nb


def side_of(n, bonds, i, j) -> Optional[set]:
    """Atoms on j's side when bond i-j is cut, or None if i-j is in a ring."""
    nb = _neighbors(n, bonds)
    seen, stack = {j}, [j]
    while stack:
        a = stack.pop()
        for b in nb[a]:
            if (a == j and b == i) or b in seen:
                continue
            if b == i:
                return None
            seen.add(b)
            stack.append(b)
    return seen


def _rotate(pos, idx, axis_point, axis, angle_deg):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    t = np.radians(angle_deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K
    out = pos.copy()
    for a in idx:
        out[a] = (pos[a] - axis_point) @ R.T + axis_point
    return out


def _struct(atoms, name, group, kind, mode="single", **info) -> Structure:
    a = atoms.copy()
    a.calc = None
    a.info = {"name": name, "group": group, "mode": mode, "kind": kind,
              **{k: v for k, v in info.items() if v is not None}}
    return from_atoms(a)


def _formula_tag(atoms) -> str:
    return atoms.get_chemical_formula(mode="hill")


# --------------------------------------------------------------- molecules
def molecule_from(spec: str):
    """ASE g2 name (CH4, C6H6, ...), 'smiles:CCO' (RDKit), or a structure file."""
    import os

    if spec.startswith("smiles:"):
        return _from_smiles(spec[len("smiles:"):])
    if os.path.exists(spec):
        from ase.io import read

        return read(spec)
    from ase.build import molecule

    try:
        return molecule(spec)
    except KeyError:
        raise ValueError(f"{spec!r} is not an ASE g2 molecule name, an existing file or 'smiles:...'") from None


def _from_smiles(smiles: str):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        raise RuntimeError("SMILES input needs RDKit: pip install -e '.[molecules]'") from None
    from ase import Atoms

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(m, randomSeed=7) != 0:
        raise ValueError(f"RDKit could not embed {smiles!r}")
    AllChem.MMFFOptimizeMolecule(m)
    conf = m.GetConformer()
    return Atoms([a.GetSymbol() for a in m.GetAtoms()], positions=conf.GetPositions())


def molecules(specs: list, *, mode: str = "relax") -> list:
    out = []
    for spec in specs:
        a = molecule_from(spec)
        a.pbc = False
        a.center(vacuum=0.0)
        tag = spec.split(":", 1)[-1] if spec.startswith("smiles:") else spec.split("/")[-1].rsplit(".", 1)[0]
        tag = tag.replace(" ", "_")
        out.append(_struct(a, f"{tag}_eq", f"mol_{tag}", "molecule", mode=mode, parent=tag))
    return out


# ----------------------------------------------------------------- scans
def bond_scans(base: Structure, *, factors=None, all_bonds: bool = False, restrained: bool = True,
               k_restraint: float = 2000.0) -> list:
    """Dissociation curve of each unique bond type (element pair) of `base`:
    r = f·r0, f in `factors` (default 0.75..2.5). The fragment on one side is
    translated; ring bonds move only the end atom. With `restrained`, the
    frame is relaxed with the bond held (mode relax + bond restraint)."""
    from ..io.structures import to_atoms

    factors = list(factors) if factors is not None else [0.75, 0.82, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5, 1.7, 1.9, 2.2, 2.5]
    atoms = to_atoms(base)
    sym = atoms.get_chemical_symbols()
    bonds = bonds_of(atoms)
    seen = set()
    out = []
    for i, j in bonds:
        key = tuple(sorted((sym[i], sym[j])))
        if key in seen and not all_bonds:
            continue
        seen.add(key)
        side = side_of(len(atoms), bonds, i, j)
        move = sorted(side) if side is not None else [j]
        if len(move) > len(atoms) / 2 and side is not None:
            i, j = j, i
            side = side_of(len(atoms), bonds, i, j)
            move = sorted(side) if side is not None else [j]
        r0 = atoms.get_distance(i, j)
        u = (atoms.positions[j] - atoms.positions[i]) / r0
        tag = f"{base.name.removesuffix('_eq')}_{sym[i]}{i}-{sym[j]}{j}"
        for f in factors:
            a = atoms.copy()
            a.positions[move] += u * (f - 1.0) * r0
            r = f * r0
            restr = [["bond", i, j, round(r, 6), k_restraint]] if restrained else None
            st = _struct(a, f"{tag}_r{r:.3f}", f"bond_{tag}", "bond_scan", mode="relax" if restrained else "single",
                         scan_value=round(r, 6), parent=base.name, scan_atoms=f"{i},{j}")
            st.restraints = restr or []
            out.append(st)
    return out


def angle_scans(base: Structure, *, deltas=None, all_angles: bool = False, k_restraint: float = 500.0) -> list:
    """Bend each unique angle type i-j-k (j central) by `deltas` degrees
    (default -40..+40), restrained relax of everything else."""
    from ..io.structures import to_atoms

    deltas = list(deltas) if deltas is not None else [-40, -30, -20, -10, 0, 10, 20, 30, 40]
    atoms = to_atoms(base)
    sym = atoms.get_chemical_symbols()
    bonds = bonds_of(atoms)
    nb = _neighbors(len(atoms), bonds)
    seen, out = set(), []
    for j in range(len(atoms)):
        for i in sorted(nb[j]):
            for k in sorted(nb[j]):
                if k <= i:
                    continue
                key = (min(sym[i], sym[k]), sym[j], max(sym[i], sym[k]))
                if key in seen and not all_angles:
                    continue
                seen.add(key)
                side = side_of(len(atoms), bonds, j, k)
                move = sorted(side) if side is not None else [k]
                if i in move:
                    move = [k]
                th0 = atoms.get_angle(i, j, k)
                normal = np.cross(atoms.positions[i] - atoms.positions[j], atoms.positions[k] - atoms.positions[j])
                if np.linalg.norm(normal) < 1e-6:  # linear: any perpendicular axis
                    v = atoms.positions[k] - atoms.positions[j]
                    normal = np.cross(v, [1, 0, 0] if abs(v[0]) < 0.9 * np.linalg.norm(v) else [0, 1, 0])
                tag = f"{base.name.removesuffix('_eq')}_{sym[i]}{i}-{sym[j]}{j}-{sym[k]}{k}"
                for d in deltas:
                    th = th0 + d
                    if not 30.0 <= th <= 180.0:
                        continue
                    a = atoms.copy()
                    a.positions = _rotate(atoms.positions, move, atoms.positions[j], normal, d)
                    real = a.get_angle(i, j, k)
                    st = _struct(a, f"{tag}_a{real:.1f}", f"angle_{tag}", "angle_scan", mode="relax",
                                 scan_value=round(real, 3), parent=base.name, scan_atoms=f"{i},{j},{k}")
                    st.restraints = [["angle", i, j, k, round(real, 4), k_restraint]]
                    out.append(st)
    return out


def torsion_scans(base: Structure, *, step: float = 30.0, k_restraint: float = 500.0) -> list:
    """Rotate about each non-ring bond j-k whose ends both have other
    neighbors, through 360 degrees in `step` increments, restrained relax."""
    from ..io.structures import to_atoms

    atoms = to_atoms(base)
    sym = atoms.get_chemical_symbols()
    bonds = bonds_of(atoms)
    nb = _neighbors(len(atoms), bonds)
    seen, out = set(), []
    for j, k in bonds:
        side = side_of(len(atoms), bonds, j, k)
        if side is None or len(nb[j]) < 2 or len(nb[k]) < 2:
            continue
        i = min(nb[j] - {k})
        l = min(nb[k] - {j})
        key = tuple(sorted((sym[j], sym[k])))
        if key in seen:
            continue
        seen.add(key)
        phi0 = atoms.get_dihedral(i, j, k, l)
        axis = atoms.positions[k] - atoms.positions[j]
        tag = f"{base.name.removesuffix('_eq')}_{sym[i]}{i}-{sym[j]}{j}-{sym[k]}{k}-{sym[l]}{l}"
        for d in np.arange(0.0, 360.0, step):
            a = atoms.copy()
            a.positions = _rotate(atoms.positions, sorted(side), atoms.positions[k], axis, d)
            phi = a.get_dihedral(i, j, k, l)
            st = _struct(a, f"{tag}_t{phi:.0f}", f"torsion_{tag}", "torsion_scan", mode="relax",
                         scan_value=round(phi, 3), parent=base.name, scan_atoms=f"{i},{j},{k},{l}")
            st.restraints = [["dihedral", i, j, k, l, round(phi, 4), k_restraint]]
            out.append(st)
    return out


def dimers(elements: list, *, distances=None) -> list:
    """Diatomic A-B curves for every element pair, from short range (inner
    wall / repulsion) to dissociation; isolated, single points."""
    from ase import Atoms

    distances = list(distances) if distances is not None else list(np.round(np.concatenate(
        [np.linspace(0.6, 1.0, 5), np.linspace(1.1, 2.4, 14), np.linspace(2.7, 4.5, 5)]), 3))
    out = []
    for x in range(len(elements)):
        for y in range(x, len(elements)):
            a, b = elements[x], elements[y]
            rmin = 0.55 * (_cov_radius(a) + _cov_radius(b))
            for d in distances:
                if d < rmin:
                    continue
                at = Atoms([a, b], positions=[[0, 0, 0], [0, 0, d]])
                out.append(_struct(at, f"dimer_{a}{b}_r{d:.3f}", f"dimer_{a}{b}", "dimer", mode="single",
                                   scan_value=float(d)))
    return out


# ------------------------------------------------------------- condensed
def bulk_from(spec: str):
    """'Si:diamond:5.43', 'Cu:fcc:3.61', 'MgO:rocksalt:4.21' or a structure file (CIF, POSCAR, ...)."""
    import os

    if os.path.exists(spec):
        from ase.io import read

        a = read(spec)
        a.pbc = True
        return a, spec.split("/")[-1].rsplit(".", 1)[0]
    parts = spec.split(":")
    from ase.build import bulk

    name = parts[0]
    crystal = parts[1] if len(parts) > 1 else None
    a = float(parts[2]) if len(parts) > 2 else None
    kw = {}
    if crystal:
        kw["crystalstructure"] = crystal
    if a:
        kw["a"] = a
    at = bulk(name, **kw)
    return at, f"{name}_{crystal or 'bulk'}"


def eos(specs: list, *, volume_ratios=None, mode: str = "single", supercell=None) -> list:
    """Isotropic volume scans V/V0 (default 0.80..1.30) of each crystal.
    mode single keeps fractional coordinates; relax relaxes them at fixed cell."""
    volume_ratios = list(volume_ratios) if volume_ratios is not None else [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]
    out = []
    for spec in specs:
        at, tag = bulk_from(spec)
        if supercell:
            at = at * tuple(supercell)
        cell0 = at.cell.array.copy()
        for v in volume_ratios:
            a = at.copy()
            a.set_cell(cell0 * v ** (1.0 / 3.0), scale_atoms=True)
            out.append(_struct(a, f"{tag}_v{v:.3f}", f"eos_{tag}", "eos", mode=mode, scan_value=float(v), parent=tag))
        out.append(_struct(at, f"{tag}_eq", f"cell_{tag}", "crystal", mode="relax_cell", parent=tag))
    return out


# ------------------------------------------------------------ distortions
def rattle(bases: list, *, n: int = 5, sigma: float = 0.1, seed: int = 0, strain: float = 0.0) -> list:
    """`n` randomly displaced copies (Gaussian, `sigma` Å) of each structure,
    with optional random cell strain (max |e| = strain) for periodic ones:
    off-equilibrium forces for force matching."""
    from ..io.structures import to_atoms

    rng = np.random.default_rng(seed)
    out = []
    for b in bases:
        atoms = to_atoms(b)
        for k in range(n):
            a = atoms.copy()
            if strain and any(a.pbc):
                e = rng.uniform(-strain, strain, (3, 3))
                e = 0.5 * (e + e.T)
                a.set_cell(a.cell.array @ (np.eye(3) + e), scale_atoms=True)
            a.positions += rng.normal(0.0, sigma, a.positions.shape)
            out.append(_struct(a, f"{b.name}_rat{k:02d}", f"rattle_{b.group or b.name}", "rattle", mode="single",
                               parent=b.name, scan_value=float(sigma)))
    return out


def from_trajectory(path: str, *, stride: int = 10, max_frames: Optional[int] = None, tag: Optional[str] = None,
                    elements_map: Optional[dict] = None) -> list:
    """Frames of an MD trajectory (any ASE-readable file, e.g. a LAMMPS dump
    from md-check), every `stride`-th, as single points."""
    from ase.io import read

    frames = read(path, index=f"::{max(1, int(stride))}")
    if max_frames:
        frames = frames[:max_frames]
    tag = tag or path.split("/")[-1].rsplit(".", 1)[0]
    out = []
    for k, a in enumerate(frames):
        if elements_map:
            a.set_chemical_symbols([elements_map.get(s, s) for s in a.get_chemical_symbols()])
        out.append(_struct(a, f"{tag}_f{k:05d}", f"md_{tag}", "md", mode="single", parent=tag))
    return out


# ------------------------------------------------------------- MD boxes
def pack_box(counts: dict, *, density: float = None, box: float = None, name: str = "box", seed: int = 0,
             min_dist: float = 2.0, max_tries: int = 2000) -> Structure:
    """Randomly oriented, non-overlapping copies of molecules in a cubic
    periodic box (by `density` in g/cm^3, or edge `box` in Å): systems for
    MD validation (Chenoweth et al. validated their C/H/O force field with
    NVT-MD of hydrocarbon/O2 mixtures [C08])."""
    from ase import Atoms
    from ase.data import atomic_masses, atomic_numbers

    rng = np.random.default_rng(seed)
    mols = []
    for spec, n in counts.items():
        m = molecule_from(spec)
        m.positions -= m.get_center_of_mass()
        mols += [m] * int(n)
    mass_amu = sum(sum(atomic_masses[atomic_numbers[s]] for s in m.get_chemical_symbols()) for m in mols)
    if box is None:
        if not density:
            raise ValueError("give density (g/cm^3) or box (Å)")
        vol = mass_amu * 1.66053906660 / density  # Å^3
        box = vol ** (1.0 / 3.0)
    symbols, positions = [], np.zeros((0, 3))
    for m in sorted(mols, key=lambda a: -len(a)):
        for _ in range(max_tries):
            R = _random_rotation(rng)
            p = m.positions @ R.T + rng.uniform(0, box, 3)
            if len(positions):
                d = p[:, None, :] - positions[None, :, :]
                d -= box * np.round(d / box)
                if np.min(np.linalg.norm(d, axis=2)) < min_dist:
                    continue
            symbols += m.get_chemical_symbols()
            positions = np.vstack([positions, p])
            break
        else:
            raise ValueError(f"could not place all molecules in a {box:.1f} Å box: lower the density or min_dist")
    a = Atoms(symbols, positions=positions % box, cell=[box] * 3, pbc=True)
    return _struct(a, name, f"mdbox_{name}", "md_box", mode="single", scan_value=float(density or 0.0))


def _random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]])
