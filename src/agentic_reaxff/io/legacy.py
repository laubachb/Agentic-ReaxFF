"""The classic ReaxFF training format: `trainset.in`, `geo`, `params`.

Read and written so training sets move between this toolkit, van Duin's
standalone ReaxFF, JAX-ReaxFF [K22] and ParAMS' importer [P21], and so the
published benchmark sets (e.g. Shchygol et al.'s cobalt / silica /
disulfide sets [Sh19], shipped with JAX-ReaxFF) can be fitted here.

Format notes (ReaxFF user manual 2002; SCM OldReaxFF docs; JAX-ReaxFF
parser):
  trainset.in  sections CHARGE, HEATFO, GEOMETRY, FORCES, CELL PARAMETERS,
               ENERGY, each closed by END<section>; '#' and '!' comments.
               The second column ("weight") is an accuracy: error term
               ((ref - ReaxFF)/acc)^2. ENERGY lines: `acc + name /n - name /n ref`
               (explicit `+` and `/n` always written: JAX needs them).
               A reference of exactly 0 is avoided (written as 0.0001).
  geo          BIOGRF 200 / XTLGRF 200 blocks: DESCRP, RUTYPE, CRYSTX
               (a b c alpha beta gamma), fixed-column HETATM lines,
               BOND/ANGLE/TORSION RESTRAINT, END + blank line.
               Restraint energy F1·(1 - exp(-F2·Δ²)) ~ harmonic K = F1·F2.
  params       `section block item step min max`, sections 1 general,
               2 atom, 3 bond, 4 off-diagonal, 5 angle, 6 torsion, 7 hbond;
               block = entry number in that ffield section; item = parameter
               number within the entry. min/max order differs between
               sources: read either order, write min then max.

Loss equivalence: an item here contributes w·mean(((pred-ref)/σ)²); each
of its M components is written with accuracy σ·sqrt(M/w), so the classic
error sum equals this toolkit's loss.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from . import ffield as F
from . import params as P
from . import trainset as T
from .structures import Structure

HETATM_FMT = "HETATM {:5d} {:<5s} {:3s} {:1s} {:5s}{:10.5f}{:10.5f}{:10.5f}    {:<2s}{:3d}{:2d} {:8.5f}"
SECTION_CODE = {"gen": 1, "atom": 2, "bond": 3, "off": 4, "angle": 5, "tors": 6, "hbond": 7}


def _safe(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.]", "_", name)
    return s


# ------------------------------------------------------------------ write
def write_geo(structures: list, path) -> dict:
    """Write structures; returns name -> legacy name."""
    from ase.geometry import cell_to_cellpar, cellpar_to_cell

    names = {}
    used = set()
    lines = []
    for s in structures:
        ln = _safe(s.name)
        k = 1
        while ln in used:
            ln = f"{_safe(s.name)}_{k}"
            k += 1
        used.add(ln)
        names[s.name] = ln
        periodic = any(s.pbc)
        pos = np.asarray(s.positions, float)
        if periodic:
            par = cell_to_cellpar(s.cell)
            std = cellpar_to_cell(par)
            frac = np.linalg.solve(np.asarray(s.cell, float).T, pos.T).T
            pos = frac @ std
            lines.append("XTLGRF 200")
        else:
            lines.append("BIOGRF 200")
        lines.append(f"DESCRP {ln}")
        rut = {"single": "SINGLE POINT", "relax": "NORMAL RUN", "relax_cell": "CELL OPT 1"}[s.mode]
        lines.append(f"RUTYPE {rut}")
        if periodic:
            lines.append("CRYSTX " + " ".join(f"{x:10.5f}" for x in par))
        for r in s.restraints or []:
            kind = str(r[0]).lower()
            if kind == "bond":
                kk = float(r[4]) if len(r) > 4 else 2000.0
                lines.append(f"BOND RESTRAINT {int(r[1]) + 1:4d}{int(r[2]) + 1:4d}{float(r[3]):8.4f}{kk / 0.25:8.2f}{0.25:8.5f}{0.0:10.7f}")
            elif kind == "angle":
                kk = float(r[5]) if len(r) > 5 else 500.0
                lines.append(f"ANGLE RESTRAINT {int(r[1]) + 1:4d}{int(r[2]) + 1:4d}{int(r[3]) + 1:4d}{float(r[4]):8.2f}{kk:8.2f}{1.0:8.4f}{0.0:9.6f}")
            elif kind == "dihedral":
                kk = float(r[6]) if len(r) > 6 else 500.0
                lines.append(f"TORSION RESTRAINT {int(r[1]) + 1:4d}{int(r[2]) + 1:4d}{int(r[3]) + 1:4d}{int(r[4]) + 1:4d}"
                             f"{float(r[5]):8.2f}{kk / 2:8.2f}{1.0:8.4f}{0.0:9.6f}")
        lines.append("FORMAT ATOM   (a6,1x,i5,1x,a5,1x,a3,1x,a1,1x,a5,3f10.5,1x,a5,i3,i2,1x,f8.5)")
        for i, (el, p) in enumerate(zip(s.symbols, pos), start=1):
            lines.append(HETATM_FMT.format(i, el, "", "", "", p[0], p[1], p[2], el, 1, 1, 0.0))
        lines.append("END")
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n")
    return names


def _acc(sigma, weight, m):
    return sigma * math.sqrt(m / weight) if weight > 0 else 1e6


def write_trainset_in(ts: T.TrainingSet, names: dict, path) -> list:
    """Returns warnings for items the format cannot hold."""
    warn = []
    sec = {"CHARGE": [], "GEOMETRY": [], "FORCES": [], "CELL PARAMETERS": [], "ENERGY": []}

    def ref0(v):
        return 0.0001 if abs(v) < 1e-12 else v

    for it in ts.items:
        ref = T.reference_values(ts, it)
        if it.kind == "energy":
            parts = []
            for name, c in it.terms:
                n = 1.0 / abs(c)
                if abs(n - round(n)) > 1e-6:
                    warn.append(f"{it.id}: coefficient {c} is not 1/integer; skipped")
                    parts = None
                    break
                parts.append(f"{'+' if c > 0 else '-'} {names[name]} /{int(round(n))}")
            if parts is None:
                continue
            if len(parts) > 5:
                warn.append(f"{it.id}: more than 5 terms; skipped")
                continue
            sec["ENERGY"].append(f"{_acc(it.sigma, it.weight, 1):10.4f} " + " ".join(parts) + f" {ref0(float(ref)):12.4f}")
        elif it.kind == "charges":
            s = ts.structures[it.structure]
            atoms = it.atoms if it.atoms is not None else list(range(s.natoms))
            acc = _acc(it.sigma, it.weight, len(atoms))
            for a, q in zip(atoms, np.atleast_1d(ref)):
                sec["CHARGE"].append(f"{names[it.structure]} {acc:10.4f} {int(a) + 1:4d} {ref0(float(q)):10.4f}")
        elif it.kind == "forces":
            s = ts.structures[it.structure]
            atoms = it.atoms if it.atoms is not None else list(range(s.natoms))
            f = np.asarray(ref).reshape(-1, 3)
            acc = _acc(it.sigma, it.weight, 3 * len(atoms))
            for a, fv in zip(atoms, f):
                sec["FORCES"].append(f"{names[it.structure]} {acc:10.4f} {int(a) + 1:4d} " + " ".join(f"{x:12.6f}" for x in fv))
        elif it.kind in ("distance", "angle", "dihedral"):
            idx = " ".join(f"{int(a) + 1:4d}" for a in it.atoms)
            sec["GEOMETRY"].append(f"{names[it.structure]} {_acc(it.sigma, it.weight, 1):10.4f} {idx} {ref0(float(ref)):10.4f}")
        elif it.kind == "lattice":
            if it.quantity == "volume":
                warn.append(f"{it.id}: cell volume has no trainset.in form; skipped")
                continue
            sec["CELL PARAMETERS"].append(f"{names[it.structure]} {_acc(it.sigma, it.weight, 1):10.4f} {it.quantity} {float(ref):10.4f}")
        else:
            warn.append(f"{it.id}: {it.kind} items have no trainset.in form; skipped")
    out = []
    for name, lines in sec.items():
        if not lines:
            continue
        out.append(name)
        out += lines
        out.append("END" + name if name != "CELL PARAMETERS" else "ENDCELL PARAMETERS")
    Path(path).write_text("\n".join(out) + "\n")
    return warn


def param_index(ff: F.ForceField, pid: str) -> tuple:
    """(section code, block, item) of a parameter id in this ffield."""
    sec = F.section_of(pid)
    name = F.name_of(pid)
    if sec == "gen":
        container, pos = ff._locate(pid)
        return 1, pos + 1, 1
    if sec == "atom":
        el = F.elements_of(pid)[0]
        return 2, ff.index_of(el), F.ATOM_NAMES.index(name) + 1
    key = pid.split(":")[1]
    nth = 1
    if "#" in key:
        key, n = key.split("#")
        nth = int(n)
    entry = ff.find_entry(sec, key.split("-"), nth=nth)
    block = [id(e) for e in ff._entries(sec)].index(id(entry)) + 1
    return SECTION_CODE[sec], block, F.SECTION_NAMES[sec].index(name) + 1


def write_params(spec: P.ParamSpec, ff: F.ForceField, path) -> None:
    lines = []
    for p in spec.params:
        s, b, i = param_index(ff, p.id)
        step = max(abs(p.value) * 0.05, 1e-3)
        lines.append(f"{s:3d}{b:4d}{i:4d} {step:10.4f} {p.min:12.4f} {p.max:12.4f}  ! {p.id}")
    Path(path).write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------- read
def read_geo(path) -> list:
    from ase.geometry import cellpar_to_cell

    out = []
    cur = None

    def flush():
        if cur and cur["symbols"]:
            pbc = cur["cellpar"] is not None
            cell = cellpar_to_cell(cur["cellpar"]) if pbc else np.zeros((3, 3))
            mode = "single"
            rt = " ".join(cur["rutype"]).upper()
            if "SINGLE" in rt:
                mode = "single"
            elif "CELL OPT" in rt and "NO CELL OPT" not in rt:
                mode = "relax_cell"
            elif rt or not cur["rutype"]:
                m = re.search(r"MAXIT\s+(\d+)", rt)
                mode = "single" if (m and int(m.group(1)) < 5) else "relax"
            out.append(Structure(name=cur["name"], symbols=cur["symbols"], positions=np.array(cur["pos"]), cell=cell,
                                 pbc=(pbc, pbc, pbc), mode=mode, restraints=cur["restraints"], group="legacy",
                                 info={"kind": "legacy", "rutype": rt}))

    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.rstrip()
        key = line[:6].upper()
        if key in ("BIOGRF", "XTLGRF"):
            flush()
            cur = {"name": None, "symbols": [], "pos": [], "cellpar": None, "rutype": [], "restraints": []}
        elif cur is None:
            continue
        elif line.startswith("DESCRP"):
            cur["name"] = line.split()[1]
        elif line.startswith("RUTYPE"):
            cur["rutype"].append(line[6:].strip())
        elif line.startswith("CRYSTX"):
            cur["cellpar"] = [float(x) for x in line.split()[1:7]]
        elif line.startswith("HETATM") or line.startswith("ATOM"):
            el = line[13:18].strip() if len(line) > 60 else line.split()[2]
            try:
                xyz = [float(line[30:40]), float(line[40:50]), float(line[50:60])]
            except ValueError:
                xyz = [float(x) for x in line.split()[3:6]]
            el = re.sub(r"[^A-Za-z]", "", el)
            cur["symbols"].append(el[:1].upper() + el[1:2].lower())
            cur["pos"].append(xyz)
        elif line.startswith("BOND RESTRAINT"):
            t = line[14:].split()
            cur["restraints"].append(["bond", int(t[0]) - 1, int(t[1]) - 1, float(t[2]), float(t[3]) * float(t[4])])
        elif line.startswith("ANGLE RESTRAINT"):
            t = line[15:].split()
            cur["restraints"].append(["angle", int(t[0]) - 1, int(t[1]) - 1, int(t[2]) - 1, float(t[3]), float(t[4]) * float(t[5])])
        elif line.startswith("TORSION RESTRAINT"):
            t = line[17:].split()
            cur["restraints"].append(["dihedral", int(t[0]) - 1, int(t[1]) - 1, int(t[2]) - 1, int(t[3]) - 1, float(t[4]),
                                      2 * float(t[5]) * float(t[6])])
        elif line.startswith("END"):
            flush()
            cur = None
    flush()
    return out


def read_trainset_in(path, structures: dict) -> tuple:
    """(items, warnings). Forces/charges lines are grouped per structure."""
    items, warn = [], []
    section = None
    forces, charges = {}, {}
    k = 0
    for raw in Path(path).read_text(errors="replace").splitlines():
        line = re.split(r"[#!]", raw, maxsplit=1)[0].strip()
        if not line:
            continue
        up = line.upper()
        if up.startswith("END"):
            section = None
            continue
        if up in ("CHARGE", "HEATFO", "GEOMETRY", "FORCES", "CELL PARAMETERS", "ENERGY"):
            section = up
            continue
        if section is None:
            continue
        t = line.replace("/", " / ").split()
        k += 1
        if section == "ENERGY":
            acc = float(t[0])
            ref = float(t[-1])
            body = t[1:-1]
            terms, i, sign = [], 0, 1.0
            while i < len(body):
                tok = body[i]
                if tok in "+-":
                    sign = 1.0 if tok == "+" else -1.0
                    i += 1
                    continue
                name = tok
                div = 1.0
                if i + 2 < len(body) + 1 and i + 1 < len(body) and body[i + 1] == "/":
                    div = float(body[i + 2])
                    i += 3
                else:
                    i += 1
                terms.append([name, sign / div])
                sign = 1.0
            missing = [n for n, _ in terms if n not in structures]
            if missing:
                warn.append(f"ENERGY line {k}: unknown structures {missing}; skipped")
                continue
            items.append(T.Item(kind="energy", sigma=acc, terms=terms, ref=ref, id=f"E{k:05d}", group="legacy_energy",
                                allow_unbalanced=True))
        elif section == "CHARGE":
            name, acc, atom, ref = t[0], float(t[1]), int(t[2]) - 1, float(t[3])
            charges.setdefault((name, acc), []).append((atom, ref))
        elif section == "FORCES":
            name, acc, atom = t[0], float(t[1]), int(t[2]) - 1
            charges_or = [float(x) for x in t[3:6]]
            forces.setdefault((name, acc), []).append((atom, charges_or))
        elif section == "GEOMETRY":
            name, acc = t[0], float(t[1])
            idx = [int(x) - 1 for x in t[2:-1]]
            ref = float(t[-1])
            kind = {2: "distance", 3: "angle", 4: "dihedral"}.get(len(idx))
            if kind is None:
                warn.append(f"GEOMETRY line {k} ({name}): RMSG / single-atom items are not supported; skipped")
                continue
            items.append(T.Item(kind=kind, sigma=acc, structure=name, atoms=idx, ref=ref, id=f"G{k:05d}", group="legacy_geometry"))
        elif section == "CELL PARAMETERS":
            name, acc, q, ref = t[0], float(t[1]), t[2].lower(), float(t[3])
            items.append(T.Item(kind="lattice", sigma=acc, structure=name, quantity=q, ref=ref, id=f"L{k:05d}", group="legacy_cell"))
        elif section == "HEATFO":
            warn.append(f"HEATFO line {k}: heats of formation need ReaxFF heat increments; not supported, skipped")
    for (name, acc), rows in charges.items():
        items.append(T.Item(kind="charges", sigma=acc, weight=float(len(rows)), structure=name, atoms=[a for a, _ in rows],
                            ref=[q for _, q in rows], id=f"Q:{name}:{acc}", group="legacy_charges"))
    for (name, acc), rows in forces.items():
        items.append(T.Item(kind="forces", sigma=acc, weight=float(3 * len(rows)), structure=name, atoms=[a for a, _ in rows],
                            ref=[f for _, f in rows], id=f"F:{name}:{acc}", group="legacy_forces"))
    for it in items:
        for n in it.structures():
            if n not in structures:
                warn.append(f"{it.id}: structure {n} not in geo")
    items = [it for it in items if all(n in structures for n in it.structures())]
    return items, warn


def read_params(path, ff: F.ForceField) -> list:
    """Param objects from a classic params file (min/max in either order)."""
    lookup = {}
    for pid in ff.param_ids(include_frozen=True):
        try:
            lookup[param_index(ff, pid)] = pid
        except Exception:  # noqa: BLE001
            continue
    out = []
    for raw in Path(path).read_text().splitlines():
        line = re.split(r"[#!]", raw, maxsplit=1)[0].split()
        if len(line) < 6:
            continue
        s, b, i = int(line[0]), int(line[1]), int(line[2])
        lo, hi = sorted((float(line[4]), float(line[5])))
        key = (s, b, 1) if s == 1 else (s, b, i)
        pid = lookup.get(key)
        if pid is None:
            raise ValueError(f"params line {raw.strip()!r}: no such parameter in the ffield")
        v = ff.get(pid)
        out.append(P.Param(pid, v, min(lo, v), max(hi, v)))
    return out
