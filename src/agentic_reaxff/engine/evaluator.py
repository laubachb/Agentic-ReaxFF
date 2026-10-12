"""Evaluate a ReaxFF force field on one structure with LAMMPS (in process).

This is the inner loop of every fit: called (structures x candidates) times,
so it keeps one serial LAMMPS instance per process and `clear`s it between
structures. What it returns is in LAMMPS `real` units and in the
structure's own (ASE) frame.

Two LAMMPS-API facts this module exists to get right:

* **Charge-equilibration fixes must be defined after the atoms exist.**
  `fix qeq/reaxff` / `acks2/reaxff` zero their solver history only for atoms
  present when the fix is created; atoms added later (create_atoms through
  the library API) start from uninitialised memory, and about half of all
  single points then return NaN or ~1e200 energies at random. Atoms are
  therefore created first, the pair style and fixes after. (Found while
  building this; tests/unit/test_evaluator.py guards it.)
* **Small periodic cells are replicated.** LAMMPS' charge equilibration
  (QEq and ACKS2) gives wrong energies when a periodic cell is thinner than
  its 10 Å cutoff (a 4 Å water cell is off by ~5 kcal/mol per molecule;
  the LAMMPS docs say the same), although the non-Coulomb terms handle
  small cells correctly. Cells are repeated until every perpendicular
  width is >= `min_periodic_width`; energies are divided by the number of
  copies and per-atom results are taken from the first copy.
* **With QEq, forces are not exactly -dE/dx.** Without charge
  equilibration LAMMPS ReaxFF forces match finite differences of the energy
  to 1e-4; with `fix qeq/reaxff` they differ by ~1% (independent of step
  size and QEq tolerance). Fits compare DFT forces with the forces LAMMPS
  MD will use, so this is left as is; it is one more reason NVE runs drift.
* **Restraint energy is not ReaxFF energy.** Restrained relaxations (bond
  scans with the other coordinates relaxed, the standard ReaxFF training
  protocol) need `fix_modify energy yes` so the minimizer sees the restraint;
  the reported energy is the pair energy only (`compute pe pair`).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..io.structures import Structure
from ..io.units import ATM_TO_GPA

CHARGE_METHODS = ("qeq", "acks2", "none")


@dataclass
class EvalSettings:
    charge_method: str = "qeq"     # qeq: fix qeq/reaxff; acks2: fix acks2/reaxff; none: fixed zero charges
    qeq_tol: float = 1.0e-8
    qeq_maxiter: int = 400
    qeq_cutoff: float = 10.0       # Å; ReaxFF's Coulomb taper radius
    # extra pair_style reaxff keywords. ReaxFF sizes its lists at setup; a
    # cell relaxation or a compressed frame can then overflow them ("Too many
    # ghost atoms"), so the default leaves generous head-room.
    pair_extra: str = "safezone 3.0 mincap 150"
    min_style: str = "cg"
    etol: float = 0.0
    # Geometry optimizations converge on the largest force component. Loose
    # convergence or low step limits add noise to the loss and overfit to the
    # limit: Shchygol et al. used 0.1 kcal/mol/Å and <= 3000 steps [Sh19];
    # Dumortier et al. found 50-200 step limits overfit [Du24].
    ftol: float = 0.1              # kcal/mol/Å, max force component (min_modify norm max) [Sh19]
    maxiter: int = 3000
    maxeval: int = 30000
    box_relax: str = "tri"         # relax_cell: iso | aniso | tri
    nonperiodic_margin: float = 12.0
    min_periodic_width: float = 10.0  # Å; replicate thinner periodic cells (QEq/ACKS2 cutoff)
    bond_k: float = 2000.0         # default restraint constants (kcal/mol/Å², kcal/mol/rad²)
    angle_k: float = 500.0
    dihedral_k: float = 500.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "EvalSettings":
        d = dict(d or {})
        unknown = set(d) - set(cls().__dict__)
        if unknown:
            raise ValueError(f"unknown evaluator setting(s) {sorted(unknown)}; known: {sorted(cls().__dict__)}")
        s = cls(**d)
        if s.charge_method not in CHARGE_METHODS:
            raise ValueError(f"charge_method must be one of {CHARGE_METHODS}")
        return s


@dataclass
class EvalResult:
    name: str
    ok: bool
    error: Optional[str] = None
    energy: Optional[float] = None          # kcal/mol (pair energy; restraints excluded)
    forces: Optional[np.ndarray] = None     # kcal/mol/Å, ASE frame
    charges: Optional[np.ndarray] = None    # e
    positions: Optional[np.ndarray] = None  # Å, ASE frame (after relax)
    cell: Optional[np.ndarray] = None       # Å, ASE frame (after relax_cell)
    stress: Optional[np.ndarray] = None     # GPa, Voigt (xx yy zz yz xz xy), sign: positive = tensile (ASE)
    min_steps: int = 0
    replicas: int = 1
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------- geometry
def lower_triangular(cell: np.ndarray):
    """(LAMMPS lower-triangular cell, rotation R) with x_lmp = x_ase @ R."""
    a, b, c = (np.asarray(v, float) for v in cell)
    ax = np.linalg.norm(a)
    ahat = a / ax
    bx = b @ ahat
    by = np.linalg.norm(b - bx * ahat)
    cx = c @ ahat
    cy = (b @ c - bx * cx) / by
    cz2 = c @ c - cx * cx - cy * cy
    cz = math.sqrt(max(cz2, 0.0))
    lmp = np.array([[ax, 0, 0], [bx, by, 0], [cx, cy, cz]])
    if np.linalg.det(cell) < 0:
        raise ValueError("left-handed cell: reorder the cell vectors to make a right-handed cell")
    rot = np.linalg.solve(np.asarray(cell, float), lmp)  # cell @ rot = lmp
    return lmp, rot


def replication(cell, pbc, min_width: float) -> tuple:
    """Copies along each periodic cell vector so every perpendicular width
    is >= min_width."""
    cell = np.array(cell, float)
    reps = [1, 1, 1]
    for d in range(3):  # 2-D/1-D conventions leave non-periodic vectors zero: use a unit normal
        if not pbc[d] and np.linalg.norm(cell[d]) < 1e-8:
            j, k = [i for i in range(3) if i != d]
            nrm = np.cross(cell[j], cell[k])
            cell[d] = nrm / np.linalg.norm(nrm) if np.linalg.norm(nrm) > 1e-12 else np.eye(3)[d]
    vol = abs(np.linalg.det(cell))
    if vol < 1e-12:
        return tuple(reps)
    for d in range(3):
        if pbc[d]:
            j, k = [i for i in range(3) if i != d]
            width = vol / np.linalg.norm(np.cross(cell[j], cell[k]))
            reps[d] = max(1, int(math.ceil(min_width / width - 1e-9)))
    return tuple(reps)


def replicate(s: Structure, reps: tuple) -> Structure:
    """The structure repeated reps[0] x reps[1] x reps[2]; the original atoms
    come first (copy k holds atoms k*n .. k*n+n-1). Restraints and fixed
    atoms are applied to every copy, so copies stay equivalent."""
    import copy as _copy

    n = s.natoms
    shifts = [(i, j, k) for i in range(reps[0]) for j in range(reps[1]) for k in range(reps[2])]
    pos = np.concatenate([s.positions + np.array(sh) @ s.cell for sh in shifts])
    out = _copy.copy(s)
    out.symbols = list(s.symbols) * len(shifts)
    out.positions = pos
    out.cell = np.asarray(s.cell) * np.array(reps)[:, None]
    out.restraints = [[r[0]] + [int(a) + c * n for a in r[1:1 + {"bond": 2, "angle": 3, "dihedral": 4}[str(r[0]).lower()]]]
                      + list(r[1 + {"bond": 2, "angle": 3, "dihedral": 4}[str(r[0]).lower()]:])
                      for c in range(len(shifts)) for r in (s.restraints or [])]
    out.fixed = [int(i) + c * n for c in range(len(shifts)) for i in (s.fixed or [])]
    return out


def _wrap(frac, pbc):
    frac = frac.copy()
    for d in range(3):
        if pbc[d]:
            frac[:, d] -= np.floor(frac[:, d])
            frac[:, d][frac[:, d] >= 1.0 - 1e-12] = 0.0
    return frac


class Evaluator:
    """One serial LAMMPS instance, reused across structures."""

    def __init__(self, settings: Optional[EvalSettings] = None):
        self.settings = settings or EvalSettings()
        self._lmp = None

    def close(self):
        if self._lmp is not None:
            try:
                self._lmp.close()
            except Exception:  # noqa: BLE001
                pass
            self._lmp = None

    def _instance(self):
        if self._lmp is None:
            from . import lmp as lmpmod

            os.environ.setdefault("OMP_NUM_THREADS", "1")
            self._lmp = lmpmod.new_instance()
        return self._lmp

    # ------------------------------------------------------------------ box
    def _box(self, s: Structure):
        """(boundary string, region command, positions in LAMMPS frame, rot or None)."""
        pbc = tuple(bool(p) for p in s.pbc)
        pos = np.asarray(s.positions, float)
        margin = self.settings.nonperiodic_margin
        if not any(pbc):
            lo = pos.min(axis=0) - margin
            hi = pos.max(axis=0) + margin
            region = f"region box block {lo[0]:.10f} {hi[0]:.10f} {lo[1]:.10f} {hi[1]:.10f} {lo[2]:.10f} {hi[2]:.10f} units box"
            # "m": shrink-wrapped but never smaller than the initial box, so
            # planar/linear molecules keep a finite thickness
            return "m m m", region, pos, None
        cell = np.asarray(s.cell, float)
        if np.any(np.linalg.norm(cell, axis=1)[list(np.where(pbc)[0])] < 1e-8):
            raise ValueError(f"{s.name}: periodic direction with a zero cell vector")
        if not all(pbc):
            # LAMMPS: a non-periodic direction must not be tilted into
            for d in range(3):
                if not pbc[d] and np.linalg.norm(cell[d]) < 1e-8:
                    cell[d] = np.eye(3)[d] * (np.ptp(pos[:, d]) + 2 * margin)
        lmpcell, rot = lower_triangular(cell)
        x = pos @ rot
        frac = np.linalg.solve(lmpcell.T, x.T).T
        frac = _wrap(frac, pbc)
        x = frac @ lmpcell
        lo = np.zeros(3)
        hi = np.array([lmpcell[0, 0], lmpcell[1, 1], lmpcell[2, 2]])
        xy, xz, yz = lmpcell[1, 0], lmpcell[2, 0], lmpcell[2, 1]
        for d in range(3):
            if not pbc[d]:
                if (d == 2 and (abs(xz) > 1e-8 or abs(yz) > 1e-8)) or (d == 1 and abs(xy) > 1e-8):
                    raise ValueError(f"{s.name}: non-periodic direction {'xyz'[d]} has tilt; make that cell vector "
                                     "orthogonal to the periodic ones (e.g. a slab with c along z)")
                lo[d] = x[:, d].min() - margin
                hi[d] = x[:, d].max() + margin
        bstr = " ".join("p" if p else "m" for p in pbc)
        region = (f"region box prism {lo[0]:.10f} {hi[0]:.10f} {lo[1]:.10f} {hi[1]:.10f} {lo[2]:.10f} {hi[2]:.10f} "
                  f"{xy:.10f} {xz:.10f} {yz:.10f} units box")
        return bstr, region, x, rot

    # ------------------------------------------------------------ evaluate
    def check_ffield(self, ffield_path: str) -> None:
        """Parse the file with our reader first: LAMMPS' ReaxFF reader
        segfaults on some malformed files, killing the whole process. Cached
        by (path, size, mtime); the worker pool skips it for the files it
        writes itself."""
        st = os.stat(ffield_path)
        key = (str(ffield_path), st.st_size, st.st_mtime_ns)
        if getattr(self, "_checked", None) == key:
            return
        from ..io import ffield as F

        F.read(ffield_path)
        self._checked = key

    def evaluate(self, ffield_path: str, s: Structure, *, ffield_elements: Optional[list] = None,
                 masses: Optional[dict] = None, want_forces: bool = True, check_ffield: bool = True) -> EvalResult:
        try:
            if check_ffield:
                self.check_ffield(ffield_path)
            return self._evaluate(ffield_path, s, masses=masses or {}, want_forces=want_forces)
        except Exception as exc:  # noqa: BLE001 - a failed evaluation is data (the optimizer penalizes it)
            lines = [ln for ln in str(exc).strip().splitlines() if ln.strip()]
            errs = [ln for ln in lines if "ERROR" in ln]
            msg = (errs[0] if errs else (lines[-1] if lines else type(exc).__name__)).replace("ERROR: ", "")
            # a LAMMPS error can leave the instance unusable
            self.close()
            return EvalResult(name=s.name, ok=False, error=msg[:500])

    def setup(self, ffield_path, s: Structure, *, masses: Optional[dict] = None, replicate_small: bool = True):
        """Build the LAMMPS system for `s` (box, atoms, pair style, charge
        fix) without running anything. Returns a dict with the instance and
        what is needed to map results back: n (atoms in LAMMPS), n_orig,
        ncopies, rot, species, structure (possibly replicated)."""
        st = self.settings
        masses = masses or {}
        if abs(s.charge) > 1e-8 and st.charge_method != "none":
            raise ValueError("charged structure: QEq/ACKS2 in LAMMPS conserve zero total charge")
        ncopies = 1
        n_orig = s.natoms
        reps = (1, 1, 1)
        if replicate_small and any(s.pbc) and st.charge_method != "none" and st.min_periodic_width > 0:
            reps = replication(s.cell, s.pbc, st.min_periodic_width)
            ncopies = reps[0] * reps[1] * reps[2]
            if ncopies > 1:
                s = replicate(s, reps)
        lmp = self._instance()
        lmp.command("clear")
        species = list(dict.fromkeys(s.symbols))
        type_of = {el: i + 1 for i, el in enumerate(species)}
        bstr, region, x, rot = self._box(s)
        lmp.commands_string("\n".join([
            "units real",
            "atom_style charge",
            "atom_modify map array sort 0 0.0",
            f"boundary {bstr}",
            region,
            f"create_box {len(species)} box",
        ] + [f"mass {type_of[el]} {masses.get(el, _mass(el)):.6f}" for el in species]))
        n = len(s.symbols)
        lmp.create_atoms(n, list(range(1, n + 1)), [type_of[e] for e in s.symbols], x.ravel().tolist())
        if lmp.get_natoms() != n:
            raise RuntimeError(f"only {lmp.get_natoms()} of {n} atoms were created (atoms outside the box?)")
        # pair + charge fixes strictly AFTER the atoms exist (see module docstring)
        checkq = "no" if st.charge_method == "none" else "yes"
        cmds = [
            f"pair_style reaxff NULL checkqeq {checkq} {st.pair_extra}".rstrip(),
            f"pair_coeff * * {ffield_path} {' '.join(species)}",
            "neighbor 2.0 nsq" if (not any(s.pbc) and n < 800) else "neighbor 2.0 bin",
            "neigh_modify every 1 delay 0 check yes",
            "compute pep all pe pair",
            "thermo_style custom step pe c_pep pxx pyy pzz pyz pxz pxy",
        ]
        if st.charge_method == "qeq":
            cmds.append(f"fix qeq all qeq/reaxff 1 0.0 {st.qeq_cutoff} {st.qeq_tol} reaxff maxiter {st.qeq_maxiter}")
        elif st.charge_method == "acks2":
            cmds.append(f"fix qeq all acks2/reaxff 1 0.0 {st.qeq_cutoff} {st.qeq_tol} reaxff maxiter {st.qeq_maxiter}")
        lmp.commands_string("\n".join(cmds))
        return {"lmp": lmp, "n": n, "n_orig": n_orig, "ncopies": ncopies, "reps": reps, "rot": rot, "species": species,
                "type_of": type_of, "structure": s}

    def _evaluate(self, ffield_path, s: Structure, *, masses, want_forces) -> EvalResult:
        ctx = self.setup(ffield_path, s, masses=masses)
        lmp, n, n_orig, ncopies, reps, rot = (ctx[k] for k in ("lmp", "n", "n_orig", "ncopies", "reps", "rot"))
        s = ctx["structure"]

        nsteps = 0
        if s.mode in ("relax", "relax_cell"):
            nsteps = self._relax(lmp, s, rot)
        else:
            lmp.command("run 0 post no")

        energy = float(lmp.extract_compute("pep", 0, 0)) / ncopies
        if not math.isfinite(energy) or abs(energy) > 1e9:
            raise FloatingPointError(f"non-finite ReaxFF energy ({energy})")
        ids = np.asarray(lmp.numpy.extract_atom("id"))[:n]
        order = np.argsort(ids)
        res = EvalResult(name=s.name, ok=True, energy=energy, min_steps=nsteps, replicas=ncopies)
        q = np.asarray(lmp.numpy.extract_atom("q"))[:n][order].astype(float)
        if not np.all(np.isfinite(q)):
            raise FloatingPointError("non-finite charges from charge equilibration")
        res.charges = q[:n_orig].copy()
        if want_forces:
            f = np.asarray(lmp.numpy.extract_atom("f"))[:n][order].astype(float)[:n_orig]
            res.forces = f @ np.linalg.inv(rot) if rot is not None else f.copy()
        if s.mode != "single":
            xl = np.asarray(lmp.numpy.extract_atom("x"))[:n][order].astype(float)
            if rot is not None:
                if s.mode == "relax_cell":
                    newlmp = _lammps_cell(lmp)
                    res.cell = (newlmp @ np.linalg.inv(rot)) / np.array(reps)[:, None] if ncopies > 1 else \
                        newlmp @ np.linalg.inv(rot)
                res.positions = (xl @ np.linalg.inv(rot))[:n_orig]
            else:
                res.positions = xl[:n_orig]
        if any(s.pbc):
            p = [lmp.get_thermo(k) for k in ("pxx", "pyy", "pzz", "pyz", "pxz", "pxy")]
            if rot is not None and all(s.pbc):
                # pressure (atm, LAMMPS frame) -> stress (GPa, ASE frame, ASE sign)
                P = np.array([[p[0], p[5], p[4]], [p[5], p[1], p[3]], [p[4], p[3], p[2]]]) * ATM_TO_GPA
                Ra = rot  # x_lmp = x_ase @ rot
                S = -(Ra @ P @ Ra.T)
                res.stress = np.array([S[0, 0], S[1, 1], S[2, 2], S[1, 2], S[0, 2], S[0, 1]])
        return res

    def _relax(self, lmp, s: Structure, rot) -> int:
        st = self.settings
        cmds = []
        for k, r in enumerate(s.restraints or []):
            kind = str(r[0]).lower()
            if kind == "bond":
                i, j, target = int(r[1]), int(r[2]), float(r[3])
                kk = float(r[4]) if len(r) > 4 else st.bond_k
                cmds.append(f"fix rs{k} all restrain bond {i + 1} {j + 1} {kk} {kk} {target}")
            elif kind == "angle":
                i, j, l, target = int(r[1]), int(r[2]), int(r[3]), float(r[4])
                kk = float(r[5]) if len(r) > 5 else st.angle_k
                cmds.append(f"fix rs{k} all restrain angle {i + 1} {j + 1} {l + 1} {kk} {kk} {target}")
            elif kind == "dihedral":
                i, j, l, m, target = int(r[1]), int(r[2]), int(r[3]), int(r[4]), float(r[5])
                kk = float(r[6]) if len(r) > 6 else st.dihedral_k
                cmds.append(f"fix rs{k} all restrain dihedral {i + 1} {j + 1} {l + 1} {m + 1} {kk} {kk} {target}")
            else:
                raise ValueError(f"{s.name}: unknown restraint kind {kind!r} (bond|angle|dihedral)")
            cmds.append(f"fix_modify rs{k} energy yes")
        if s.fixed:
            cmds.append("group frozen id " + " ".join(str(i + 1) for i in s.fixed))
            cmds.append("fix frz frozen setforce 0.0 0.0 0.0")
        if s.mode == "relax_cell":
            if not all(s.pbc):
                raise ValueError(f"{s.name}: relax_cell needs a fully periodic structure")
            cmds.append(f"fix brx all box/relax {st.box_relax} 0.0 vmax 0.001 nreset 50")
        cmds.append(f"min_style {st.min_style}")
        cmds.append("min_modify norm max")
        cmds.append(f"minimize {st.etol} {st.ftol} {st.maxiter} {st.maxeval}")
        lmp.commands_string("\n".join(cmds))
        try:
            return int(lmp.extract_global("ntimestep"))
        except Exception:  # noqa: BLE001
            return 0


def _lammps_cell(lmp) -> np.ndarray:
    boxlo, boxhi, xy, yz, xz, _, _ = lmp.extract_box()
    lx, ly, lz = (boxhi[i] - boxlo[i] for i in range(3))
    return np.array([[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]])


def _mass(symbol: str) -> float:
    from ase.data import atomic_masses, atomic_numbers

    return float(atomic_masses[atomic_numbers[symbol]])
