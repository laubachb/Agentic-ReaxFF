"""Quantum ESPRESSO labels for ReaxFF training structures.

Rendering (`render_pw_in`), the Slurm job body (`job_commands`) and parsing
(`parse_frame`). The pieces specific to ReaxFF data:

* **Isolated molecules** get a cubic box with >= `vacuum` Å of vacuum and a
  Gamma-point calculation; periodic structures a k-mesh from `kspacing`.
* **Scans hold the scanned coordinate.** A frame generated with a restraint
  (mode relax) is relaxed by pw.x (damped dynamics: pw.x does not allow
  constraints with BFGS) with the matching CONSTRAINTS entry
  (`distance`, `planar_angle`, `torsional_angle`; the target is the current
  value, which the generator set exactly), so DFT and ReaxFF compare the
  same restrained-relaxation energy profile [vD01, C08].
* **Bonds break homolytically.** Spin-restricted DFT puts dissociation
  limits far too high. With `spin: auto`, isolated systems with an odd
  electron count are spin-polarized, and stretched bond scans / dimers
  (beyond `bs_ratio` × the equilibrium length) get broken-symmetry starting
  magnetization of opposite sign on the two fragments (separate species
  labels), the standard unrestricted treatment of bond dissociation.
* **Charges** (optional): Löwdin populations from projwfc.x, converted to
  net charges. Population schemes differ a lot; ReaxFF charges are usually
  weighted loosely (sigma 0.1 e) and charge-only fits cannot pin EEM
  parameters by themselves (Verstraelen 2011 [V11]).
* **One level of theory per fit:** `lot_key` hashes every setting that
  changes the labels, and is stored on each labeled frame; `trainset-build`
  refuses to mix keys.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..io.structures import Structure


@dataclass
class QESettings:
    pseudo_dir: str = ""
    pseudopotentials: dict = field(default_factory=dict)   # element -> UPF file name (in pseudo_dir)
    ecutwfc: float = 40.0
    ecutrho: float = 200.0
    input_dft: Optional[str] = None                        # default: from the pseudopotentials
    vdw_corr: Optional[str] = None                         # e.g. "dft-d3"
    dftd3_version: int = 4                                 # 4 = D3(BJ)
    smearing: str = "mv"
    degauss: float = 0.01                                  # Ry
    kspacing: float = 0.25                                 # Å^-1 (2π included), periodic directions
    vacuum: float = 12.0                                   # Å around isolated systems
    conv_thr_per_atom: float = 1e-9                        # Ry per atom
    mixing_beta: float = 0.3
    electron_maxstep: int = 200
    spin: str = "auto"                                     # auto | none | always
    bs_ratio: float = 1.3                                  # broken symmetry beyond this x r0
    charges: str = "none"                                  # none | lowdin
    forc_conv_thr: float = 1e-4                            # Ry/bohr, relaxations
    nstep: int = 200

    def to_dict(self):
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        unknown = set(d) - set(cls().__dict__)
        if unknown:
            raise ValueError(f"unknown QE setting(s) {sorted(unknown)}; known: {sorted(cls().__dict__)}")
        return cls(**d)


def find_pseudopotentials(pseudo_dir, elements) -> dict:
    """element -> file, matching names that start with the element symbol
    followed by a non-letter (GBRV `si_pbe_v1.uspp.F.UPF`, SSSP `Si.pbe-...UPF`)."""
    files = [p.name for p in Path(pseudo_dir).iterdir() if p.suffix.lower() == ".upf"]
    out = {}
    for el in elements:
        rx = re.compile(rf"^{el}(?![a-z])", re.IGNORECASE)
        cands = sorted(f for f in files if rx.match(f) and not re.match(rf"^{el}[a-z]", f, re.IGNORECASE))
        if not cands:
            raise FileNotFoundError(f"no pseudopotential for {el} in {pseudo_dir}")
        out[el] = cands[0]
    return out


def upf_valence(path) -> float:
    text = Path(path).read_text(errors="replace")[:20000]
    m = re.search(r'z_valence\s*=\s*"\s*([0-9.Ee+-]+)', text) or re.search(r"([0-9.Ee+-]+)\s+Z valence", text)
    if not m:
        raise ValueError(f"cannot read z_valence from {path}")
    return float(m.group(1))


def lot_key(settings: QESettings) -> str:
    """Hash of everything that changes the labels (not convergence aids)."""
    d = settings.to_dict()
    for k in ("mixing_beta", "electron_maxstep", "nstep", "charges"):
        d.pop(k, None)
    hashes = {}
    for el, f in sorted(settings.pseudopotentials.items()):
        p = Path(settings.pseudo_dir) / f
        hashes[el] = hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.is_file() else f
    d["pseudopotentials"] = hashes
    return "qe-" + hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


# ------------------------------------------------------------------ geometry
def _isolated_cell(s: Structure, vacuum: float, return_shift: bool = False):
    pos = np.asarray(s.positions, float)
    extent = np.ptp(pos, axis=0) if len(pos) > 1 else np.zeros(3)
    L = float(max(extent.max() + 2 * vacuum, 2 * vacuum))
    cell = np.eye(3) * L
    shift = 0.5 * L - 0.5 * (pos.max(axis=0) + pos.min(axis=0))
    if return_shift:
        return cell, pos + shift, shift
    return cell, pos + shift


def kmesh(cell, pbc, kspacing):
    rec = 2 * np.pi * np.linalg.inv(np.asarray(cell, float)).T
    return [max(1, int(math.ceil(np.linalg.norm(rec[i]) / kspacing))) if pbc[i] else 1 for i in range(3)]


def _fragments_for_broken_symmetry(s: Structure):
    """(+ set, - set) of atom indices for a stretched bond scan or dimer, else None."""
    kind = s.info.get("kind")
    if kind == "dimer":
        return {0}, {1}
    from ..io.structures import scan_atoms

    if kind != "bond_scan" or len(scan_atoms(s)) != 2:
        return None
    i, j = scan_atoms(s)
    from ..generate.builders import bonds_of, side_of
    from ..io.structures import to_atoms

    atoms = to_atoms(s)
    bonds = [b for b in bonds_of(atoms) if set(b) != {i, j}] + [(i, j)]
    side = side_of(len(atoms), bonds, i, j)
    if side is None:
        return None
    return set(range(len(atoms))) - side, side


def _stretched(s: Structure, ratio: float) -> bool:
    kind = s.info.get("kind")
    if kind not in ("bond_scan", "dimer") or s.info.get("scan_value") is None:
        return False
    from ..generate.builders import _cov_radius

    if kind == "dimer":
        r0 = _cov_radius(s.symbols[0]) + _cov_radius(s.symbols[1])
        return float(s.info["scan_value"]) > ratio * r0
    from ..io.structures import scan_atoms

    if len(scan_atoms(s)) != 2:
        return False
    i, j = scan_atoms(s)
    r0 = _cov_radius(s.symbols[i]) + _cov_radius(s.symbols[j])
    return float(s.info["scan_value"]) > ratio * r0


# ------------------------------------------------------------------ render
def render_pw_in(s: Structure, st: QESettings, valence: dict, masses: dict) -> str:
    isolated = not any(s.pbc)
    if isolated:
        cell, pos = _isolated_cell(s, st.vacuum)
        pbc = (False, False, False)
    else:
        cell, pos = np.asarray(s.cell, float), np.asarray(s.positions, float)
        pbc = tuple(s.pbc)
    calc = {"single": "scf", "relax": "relax", "relax_cell": "vc-relax"}[s.mode]
    nel = sum(valence[e] for e in s.symbols) - s.charge
    species = list(dict.fromkeys(s.symbols))
    labels = list(s.symbols)
    mag = {}
    nspin = 1
    if st.spin == "always":
        nspin = 2
        mag = {el: 0.3 for el in species}
    elif st.spin == "auto":
        if isolated and int(round(nel)) % 2 == 1:
            nspin = 2
            mag = {el: 0.3 for el in species}
        if isolated and _stretched(s, st.bs_ratio):
            frag = _fragments_for_broken_symmetry(s)
            if frag:
                plus, minus = frag
                nspin = 2
                mag = {}
                for a in range(len(labels)):
                    lab = labels[a] + ("1" if a in plus else "2")
                    labels[a] = lab
                    mag[lab] = 0.5 if a in plus else -0.5
                species = list(dict.fromkeys(labels))
    natoms = len(s.symbols)
    lines = ["&CONTROL", f"  calculation = '{calc}'", "  restart_mode = 'from_scratch'", "  outdir = './out'",
             "  pseudo_dir = '.'", "  prefix = 'pw'", "  tprnfor = .true.", "  tstress = .true.",
             f"  forc_conv_thr = {st.forc_conv_thr}", f"  nstep = {st.nstep * (3 if s.restraints else 1)}", "  disk_io = 'low'", "/",
             "&SYSTEM", "  ibrav = 0", f"  nat = {natoms}", f"  ntyp = {len(species)}",
             f"  ecutwfc = {st.ecutwfc}", f"  ecutrho = {st.ecutrho}",
             "  occupations = 'smearing'", f"  smearing = '{st.smearing}'", f"  degauss = {st.degauss}"]
    if s.charge:
        lines.append(f"  tot_charge = {s.charge}")
    if st.input_dft:
        lines.append(f"  input_dft = '{st.input_dft}'")
    if st.vdw_corr:
        lines.append(f"  vdw_corr = '{st.vdw_corr}'")
        if "d3" in st.vdw_corr.lower():
            lines.append(f"  dftd3_version = {st.dftd3_version}")
    if nspin == 2:
        lines.append("  nspin = 2")
        for k, sp in enumerate(species, start=1):
            lines.append(f"  starting_magnetization({k}) = {mag.get(sp, 0.0)}")
    if isolated or s.restraints:
        # isolated: symmetry buys little at Gamma; constrained relaxations can break the
        # symmetry pw.x detected at the start ("checkallsym: some of the original symmetry
        # operations not satisfied", seen on a restrained CH4)
        lines.append("  nosym = .true.")
    if isolated:
        lines.append("  assume_isolated = 'none'")
    lines += ["/", "&ELECTRONS", f"  conv_thr = {max(st.conv_thr_per_atom * natoms, 1e-10):.2e}",
              f"  mixing_beta = {st.mixing_beta}", f"  electron_maxstep = {st.electron_maxstep}", "/"]
    if calc != "scf":
        # pw.x refuses CONSTRAINTS with BFGS (input.f90: "constraints only with
        # fixed-cell dynamics" when lbfgs): restrained frames use damped dynamics
        dyn = "damp" if (s.restraints and calc == "relax") else "bfgs"
        lines += ["&IONS", f"  ion_dynamics = '{dyn}'", "/"]
    if calc == "vc-relax":
        lines += ["&CELL", "  cell_dynamics = 'bfgs'", "  cell_dofree = 'all'", "/"]
    lines.append("ATOMIC_SPECIES")
    for sp in species:
        el = re.sub(r"\d+$", "", sp)
        lines.append(f"  {sp} {masses.get(el, 1.0):.4f} {st.pseudopotentials[el]}")
    lines.append("CELL_PARAMETERS angstrom")
    for v in cell:
        lines.append("  " + " ".join(f"{x:.10f}" for x in v))
    lines.append("ATOMIC_POSITIONS angstrom")
    fixed = set(s.fixed or [])
    for a, (lab, p) in enumerate(zip(labels, pos)):
        flags = " 0 0 0" if (a in fixed and calc != "scf") else ""
        lines.append(f"  {lab} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}{flags}")
    if isolated:
        lines += ["K_POINTS gamma"]
    else:
        k = kmesh(cell, pbc, st.kspacing)
        lines += ["K_POINTS automatic", f"  {k[0]} {k[1]} {k[2]} 0 0 0"]
    if calc != "scf" and s.restraints:
        lines.append("CONSTRAINTS")
        lines.append(f"  {len(s.restraints)}")
        for r in s.restraints:
            kind = str(r[0]).lower()
            if kind == "bond":
                lines.append(f"  distance {int(r[1]) + 1} {int(r[2]) + 1}")
            elif kind == "angle":
                lines.append(f"  planar_angle {int(r[1]) + 1} {int(r[2]) + 1} {int(r[3]) + 1}")
            elif kind == "dihedral":
                lines.append(f"  torsional_angle {int(r[1]) + 1} {int(r[2]) + 1} {int(r[3]) + 1} {int(r[4]) + 1}")
    return "\n".join(lines) + "\n"


def render_projwfc_in() -> str:
    return "&PROJWFC\n  prefix = 'pw'\n  outdir = './out'\n  filpdos = 'pdos'\n  lsym = .false.\n/\n"


# ------------------------------------------------------------------- parse
def frame_done(d: Path) -> bool:
    out = d / "pw.out"
    if not out.is_file():
        return False
    t = out.read_text(errors="replace")
    if "JOB DONE" not in t or "convergence NOT achieved" in t:
        return False
    if "convergence has been achieved" not in t:
        return False
    if re.search(r"calculation\s*=\s*'(vc-)?relax'", (d / "pw.in").read_text()) and \
            not any(m in t for m in ("End of BFGS Geometry Optimization", "bfgs converged", "End of damped dynamics")):
        return False
    return True


def parse_lowdin(text: str, symbols: list, valence: dict) -> Optional[np.ndarray]:
    m = list(re.finditer(r"Lowdin Charges:\s*\n(.*?)(?:\n\s*\n|Spilling)", text, re.S))
    if not m:
        return None
    block = m[-1].group(1)
    tot = [float(x) for x in re.findall(r"Atom #\s*\d+: total charge =\s*([-0-9.]+)", block)]
    if len(tot) != len(symbols):
        return None
    q = np.array([valence[e] - t for e, t in zip(symbols, tot)])
    return q - q.mean()  # Löwdin populations miss the spilled charge: restore neutrality


def parse_frame(d: Path, s: Structure, valence: dict, vacuum: float) -> Structure:
    """Labeled copy of `s` from frame dir `d` (energy eV, forces eV/Å, stress
    eV/Å^3; relaxed geometry for relax modes, mapped back to s's frame)."""
    import copy

    from ase.io import read

    atoms = read(str(d / "pw.out"), format="espresso-out", index=-1)
    syms = [re.sub(r"\d+$", "", x) for x in atoms.get_chemical_symbols()]
    if len(atoms) != s.natoms or syms != list(s.symbols):
        raise ValueError(f"{d}: pw.out has {len(atoms)} atoms {sorted(set(syms))}, structure {s.name} has {s.natoms}: "
                         "frame directory does not belong to this structure")
    out = copy.deepcopy(s)
    res = atoms.calc.results
    out.energy = float(res["energy"])
    out.forces = np.asarray(res.get("forces"), float) if res.get("forces") is not None else None
    if any(s.pbc) and res.get("stress") is not None:
        out.stress = np.asarray(res["stress"], float)
    if s.mode != "single":
        pos = atoms.get_positions()
        if not any(s.pbc):
            # undo the centering shift applied when the box was built
            _, _, shift = _isolated_cell(s, vacuum, return_shift=True)
            pos = pos - shift
        out.positions = pos
        if s.mode == "relax_cell":
            out.cell = atoms.cell.array.copy()
    pj = d / "projwfc.out"
    if pj.is_file():
        q = parse_lowdin(pj.read_text(errors="replace"), s.symbols, valence)
        if q is not None:
            out.charges = q
    return out


# ------------------------------------------------------------------- job
LEFT_S_FUNCTION = ("reaxff_left_s() { local L d=0 h=0 m=0 s=0 a b c; L=$(squeue -h -j \"$SLURM_JOB_ID\" -o %L 2>/dev/null); "
                   "case \"$L\" in ''|*INVALID*|*UNLIMITED*|*NOT_SET*) echo 999999; return;; esac; "
                   "case \"$L\" in *-*) d=${L%%-*}; L=${L#*-};; esac; IFS=: read -r a b c <<< \"$L\"; "
                   "if [ -n \"$c\" ]; then h=$a; m=$b; s=$c; else m=$a; s=$b; fi; "
                   "echo $(( 10#$d*86400 + 10#$h*3600 + 10#$m*60 + 10#$s )); }")

DONE_TEST = ("grep -qs 'JOB DONE' pw.out && grep -qs 'convergence has been achieved' pw.out && "
             "! grep -qs 'convergence NOT achieved' pw.out && "
             "{ ! grep -qs \"calculation = '.*relax'\" pw.in || grep -qsE 'End of BFGS|bfgs converged|End of damped dynamics' pw.out ; }")


def frame_command(launcher: str, ranks: int, per_node: int, *, lowdin: bool, pw_flags: str = "") -> str:
    """Shell for one frame: skip if done; else run pw.x with the job's
    remaining time as max_seconds (a frame stopped that way restarts from
    its saved state next time); projwfc.x for charges; then delete the
    wavefunction scratch (one file per rank: lustre file quota)."""
    if launcher == "srun":
        launch = f"MV2_ENABLE_AFFINITY=0 srun --exact -n {ranks} {'-N 1 ' if ranks <= per_node else ''}--cpu-bind=cores "
    else:
        launch = f"{launcher} -n {ranks} " if launcher else ""
    pw = f"OMP_NUM_THREADS=1 {launch}pw.x {pw_flags} -in pw.in > pw.out 2> pw.err".replace("  ", " ")
    pj = f" && OMP_NUM_THREADS=1 {launch}projwfc.x -in projwfc.in > projwfc.out 2> projwfc.err" if lowdin else ""
    run = ("L=$(reaxff_left_s); if [ \"$L\" -gt 420 ]; then "
           "grep -qs 'Maximum CPU time exceeded' pw.out && sed -i \"s/restart_mode = 'from_scratch'/restart_mode = 'restart'/\" pw.in; "
           "sed -i '/max_seconds/d' pw.in; sed -i \"s/^&CONTROL\\$/\\&CONTROL\\n  max_seconds = $((L - 240))/\" pw.in; "
           f"{pw}{pj}; fi")
    cleanup = f"{{ {DONE_TEST} ; }} && rm -rf out"
    return f"{{ {DONE_TEST} ; }} || {{ {run} ; }} ; {cleanup}"


def assign_lanes(costs: list, lanes: int) -> list:
    lanes = max(1, min(int(lanes), len(costs) or 1))
    out, load = [[] for _ in range(lanes)], [0.0] * lanes
    for i in sorted(range(len(costs)), key=lambda k: -costs[k]):
        j = int(np.argmin(load))
        out[j].append(i)
        load[j] += costs[i]
    return out


def frame_cost(s: Structure, st: QESettings) -> float:
    nat = s.natoms
    kp = 1 if not any(s.pbc) else int(np.prod(kmesh(s.cell, s.pbc, st.kspacing)))
    vol = (np.ptp(s.positions, axis=0) + 2 * st.vacuum).prod() if not any(s.pbc) else abs(np.linalg.det(s.cell))
    relax = {"single": 1.0, "relax": 15.0, "relax_cell": 25.0}[s.mode]
    return float(vol * nat * kp * relax)
