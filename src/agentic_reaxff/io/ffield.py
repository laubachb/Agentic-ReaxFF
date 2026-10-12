"""Read, edit and write ReaxFF force-field files (the van Duin `ffield`
format that LAMMPS `pair_style reaxff`, PuReMD, the standalone ReaxFF code,
JAX-ReaxFF and AMS all read).

Layout (see LAMMPS src/REAXFF/reaxff_ffield.cpp, which this mirrors):

    line 1                      free-text header
    n_general  ! comment        then n_general lines "value !name"
    n_atoms    ! comment        + 3 comment lines, then 4 lines per element
                                (5 with the low-gradient "lg" extension)
    n_bonds    ! comment        + 1 comment line, then 2 lines per bond
    n_offdiag  ! comment        1 line per pair
    n_angles   ! comment        1 line per triple
    n_torsions ! comment        1 line per quadruple (0 = wildcard)
    n_hbonds   ! comment        1 line per triple

Every numeric parameter has a stable string id, used by `params-select`,
the optimizers and the reports:

    gen:p_boc1                  general parameter by name (or gen:#30, 1-based)
    atom:C:r_s                  per-element
    bond:C-H:De_s               per-pair (order-independent)
    off:C-H:r_vdW               off-diagonal pair overrides
    angle:H-C-H:theta_00        j-k-l, k central (j/l order-independent)
    tors:X-C-C-X:V2             X = wildcard (index 0)
    hbond:O-H-O:p_hb1

Names follow the variable names in LAMMPS' reader, so a reader of the C++
can check them; the van Duin labels (p(boc1), De(sigma), ...) are aliases.

Values are written in the standard fixed format (`f9.4` columns, `f10.4`
for general parameters) unless `precision` asks for more digits. The
optimizers evaluate the ffield *after* rounding to the written precision,
so the cost they report is the cost of the file they write.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# --- parameter name tables (index = position in the section's value list) ---

GENERAL_NAMES = [
    "p_boc1", "p_boc2", "p_coa2", "p_trip4", "p_trip3", "kc2", "p_ovun6", "p_trip2", "p_ovun7", "p_ovun8",
    "p_trip1", "swa", "swb", "gen_unused14", "p_val7", "p_lp1", "p_val9", "p_val10", "gen_unused19", "p_pen2",
    "p_pen3", "p_pen4", "gen_unused23", "p_tor2", "p_tor3", "p_tor4", "gen_unused27", "p_cot2", "p_vdW1",
    "bo_cut100", "p_coa4", "p_ovun4", "p_ovun3", "p_val8", "acks2_bond_softness", "lg_scale", "gen_unused37",
    "gen_unused38", "p_coa3",
]
ATOM_NAMES = [
    "r_s", "valency", "mass", "r_vdw", "epsilon", "gamma", "r_pi", "valency_e",
    "alpha", "gamma_w", "valency_boc", "p_ovun5", "atom_unused13", "chi", "eta", "p_hbond",
    "r_pipi", "p_lp2", "heat_inc", "p_boc4", "p_boc3", "p_boc5", "bcut_acks2", "atom_unused24",
    "p_ovun2", "p_val3", "atom_unused27", "valency_val", "p_val5", "rcore2", "ecore2", "acore2",
]
ATOM_LG_NAMES = ["lgcij", "lgre"]
BOND_NAMES = [
    "De_s", "De_p", "De_pp", "p_be1", "p_bo5", "v13cor", "p_bo6", "p_ovun1",
    "p_be2", "p_bo3", "p_bo4", "bond_unused12", "p_bo1", "p_bo2", "ovc", "bond_unused16",
]
OFFDIAG_NAMES = ["D", "r_vdW", "alpha", "r_s", "r_p", "r_pp"]
OFFDIAG_LG_NAMES = ["lgcij"]
ANGLE_NAMES = ["theta_00", "p_val1", "p_val2", "p_coa1", "p_val7", "p_pen1", "p_val4"]
TORSION_NAMES = ["V1", "V2", "V3", "p_tor1", "p_cot1", "tors_unused6", "tors_unused7"]
HBOND_NAMES = ["r0_hb", "p_hb1", "p_hb2", "p_hb3"]

SECTION_NAMES = {
    "gen": GENERAL_NAMES, "atom": ATOM_NAMES, "bond": BOND_NAMES, "off": OFFDIAG_NAMES,
    "angle": ANGLE_NAMES, "tors": TORSION_NAMES, "hbond": HBOND_NAMES,
}

# van Duin's labels -> our names (the comments in most published ffields)
ALIASES = {
    "atom": {"ro(sigma)": "r_s", "val": "valency", "rvdw": "r_vdw", "dij": "epsilon", "ro(pi)": "r_pi",
             "val(e)": "valency_e", "alfa": "alpha", "gamma(w)": "gamma_w", "val(angle)": "valency_boc",
             "p(ovun5)": "p_ovun5", "chieem": "chi", "etaeem": "eta", "ro(pipi)": "r_pipi", "p(lp2)": "p_lp2",
             "heat increment": "heat_inc", "p(boc4)": "p_boc4", "p(boc3)": "p_boc3", "p(boc5)": "p_boc5",
             "p(ovun2)": "p_ovun2", "p(val3)": "p_val3", "val(boc)": "valency_val", "p(val5)": "p_val5"},
    "bond": {"de(sigma)": "De_s", "de(pi)": "De_p", "de(pipi)": "De_pp", "p(be1)": "p_be1", "p(bo5)": "p_bo5",
             "13corr": "v13cor", "p(bo6)": "p_bo6", "p(ovun1)": "p_ovun1", "p(be2)": "p_be2", "p(bo3)": "p_bo3",
             "p(bo4)": "p_bo4", "p(bo1)": "p_bo1", "p(bo2)": "p_bo2"},
    "off": {"dij": "D", "rvdw": "r_vdW", "ro(sigma)": "r_s", "ro(pi)": "r_p", "ro(pipi)": "r_pp"},
}

# Parameters that are structural, integer-valued or unused: never optimized
# (`params-select` refuses them unless forced).
FROZEN_ALWAYS = {
    "gen": {"swa", "swb", "bo_cut100", "lg_scale", "gen_unused14", "gen_unused19", "gen_unused23", "gen_unused27",
            "gen_unused37", "gen_unused38"},
    "atom": {"valency", "mass", "valency_e", "valency_boc", "valency_val", "p_hbond", "heat_inc", "atom_unused13",
             "atom_unused24", "atom_unused27"},
    "bond": {"bond_unused12", "bond_unused16"},
    "off": set(),
    "angle": set(),
    "tors": {"tors_unused6", "tors_unused7"},
    "hbond": set(),
}

WILDCARD = "X"


class FFieldError(ValueError):
    pass


@dataclass
class Entry:
    """One multi-body entry: 1-based ffield atom indices (0 = wildcard in
    torsions) and its parameter values."""
    idx: tuple
    values: list
    lg: Optional[list] = None  # off-diagonal lg extension value(s)


@dataclass
class AtomEntry:
    symbol: str
    values: list  # 32
    lg: Optional[list] = None  # [lgcij, lgre] with the lg extension


@dataclass
class ForceField:
    header: str
    general: list
    general_comments: list
    atoms: list
    bonds: list
    offdiag: list
    angles: list
    torsions: list
    hbonds: list
    section_comments: dict = field(default_factory=dict)
    lg: bool = False
    source: Optional[str] = None

    # ---------------------------------------------------------------- basics
    @property
    def elements(self) -> list:
        return [a.symbol for a in self.atoms]

    def index_of(self, symbol: str) -> int:
        """1-based ffield index of an element (case-insensitive)."""
        for i, a in enumerate(self.atoms, start=1):
            if a.symbol.lower() == symbol.lower():
                return i
        raise FFieldError(f"element {symbol!r} is not in this ffield (has {self.elements})")

    def symbol_of(self, idx: int) -> str:
        if idx == 0:
            return WILDCARD
        if 1 <= idx <= len(self.atoms):
            return self.atoms[idx - 1].symbol
        return f"?{idx}"

    def is_orphan(self, entry: "Entry") -> bool:
        """An entry naming an element index beyond the atom section. LAMMPS
        skips these silently; they are kept for a faithful rewrite but have
        no parameter ids."""
        return any(i < 0 or i > len(self.atoms) for i in entry.idx)

    def copy(self) -> "ForceField":
        return copy.deepcopy(self)

    def mass(self, symbol: str) -> float:
        return float(self.atoms[self.index_of(symbol) - 1].values[2])

    # ------------------------------------------------------------- parameter ids
    def _key(self, section: str, idx: tuple) -> str:
        return "-".join(self.symbol_of(i) for i in idx)

    def _entries(self, section: str) -> list:
        return {"bond": self.bonds, "off": self.offdiag, "angle": self.angles, "tors": self.torsions,
                "hbond": self.hbonds}[section]

    def param_ids(self, sections: Optional[Iterable[str]] = None, include_frozen: bool = False) -> list:
        """Every parameter id in file order."""
        sections = list(sections) if sections else list(SECTION_NAMES)
        out = []
        seen = {}
        for sec in sections:
            names = SECTION_NAMES[sec]
            if sec == "gen":
                for i, _ in enumerate(self.general):
                    name = names[i] if i < len(names) else f"gen_extra{i + 1}"
                    if include_frozen or name not in FROZEN_ALWAYS["gen"]:
                        out.append(f"gen:{name}")
            elif sec == "atom":
                for a in self.atoms:
                    for name in names:
                        if include_frozen or name not in FROZEN_ALWAYS["atom"]:
                            out.append(f"atom:{a.symbol}:{name}")
            else:
                for e in self._entries(sec):
                    if self.is_orphan(e):
                        continue
                    key = self._key(sec, e.idx)
                    n = seen.get((sec, key), 0) + 1
                    seen[(sec, key)] = n
                    suffix = "" if n == 1 else f"#{n}"
                    for name in names[:len(e.values)]:
                        if include_frozen or name not in FROZEN_ALWAYS[sec]:
                            out.append(f"{sec}:{key}{suffix}:{name}")
        return out

    def _locate(self, pid: str):
        """(container list, position) for a parameter id."""
        parts = pid.split(":")
        sec = parts[0]
        if sec not in SECTION_NAMES:
            raise FFieldError(f"bad parameter id {pid!r}: section must be one of {list(SECTION_NAMES)}")
        if sec == "gen":
            if len(parts) != 2:
                raise FFieldError(f"bad parameter id {pid!r}; expected gen:<name> or gen:#<n>")
            name = parts[1]
            if name.startswith("#"):
                pos = int(name[1:]) - 1
            elif name in GENERAL_NAMES:
                pos = GENERAL_NAMES.index(name)
            else:
                raise FFieldError(f"unknown general parameter {name!r}")
            if not 0 <= pos < len(self.general):
                raise FFieldError(f"{pid}: this ffield has {len(self.general)} general parameters")
            return self.general, pos
        if len(parts) != 3:
            raise FFieldError(f"bad parameter id {pid!r}; expected {sec}:<elements>:<name>")
        _, key, name = parts
        name = ALIASES.get(sec, {}).get(name.lower(), name)
        if sec == "atom":
            atom = self.atoms[self.index_of(key) - 1]
            if name in ATOM_NAMES:
                return atom.values, ATOM_NAMES.index(name)
            if name in ATOM_LG_NAMES and atom.lg is not None:
                return atom.lg, ATOM_LG_NAMES.index(name)
            raise FFieldError(f"unknown atom parameter {name!r}")
        nth = 1
        if "#" in key:
            key, n = key.split("#", 1)
            nth = int(n)
        names = SECTION_NAMES[sec]
        if name not in names and not (sec == "off" and name in OFFDIAG_LG_NAMES):
            raise FFieldError(f"unknown {sec} parameter {name!r}; known: {names}")
        entry = self.find_entry(sec, key.split("-"), nth=nth)
        if entry is None:
            raise FFieldError(f"{pid}: no {sec} entry {key} in this ffield")
        if sec == "off" and name in OFFDIAG_LG_NAMES:
            return entry.lg, 0
        pos = names.index(name)
        while len(entry.values) <= pos:
            entry.values.append(0.0)
        return entry.values, pos

    def find_entry(self, section: str, symbols: list, nth: int = 1) -> Optional[Entry]:
        """The nth entry whose element key matches `symbols`, or None. Bonds,
        off-diagonal pairs, angles (j-k-l == l-k-j) and torsions are
        symmetric in LAMMPS; hydrogen bonds are directional (hbp[j][k][m] is
        not symmetrized), so X-H-Y and Y-H-X are different entries."""
        want = [0 if s.upper() == WILDCARD else self.index_of(s) for s in symbols]
        n = 0
        for e in self._entries(section):
            idx = list(e.idx)
            if idx == want or (section != "hbond" and idx == want[::-1]):
                n += 1
                if n == nth:
                    return e
        return None

    def get(self, pid: str) -> float:
        container, pos = self._locate(pid)
        return float(container[pos])

    def set(self, pid: str, value: float) -> None:
        container, pos = self._locate(pid)
        container[pos] = float(value)

    def get_many(self, pids: list) -> list:
        return [self.get(p) for p in pids]

    def set_many(self, pids: list, values) -> None:
        for p, v in zip(pids, values):
            self.set(p, v)

    # --------------------------------------------------------------- structure
    def has_interaction(self, section: str, symbols: list) -> bool:
        return self.find_entry(section, symbols) is not None

    def subset(self, elements: list) -> "ForceField":
        """A copy restricted to `elements` (reindexed), for a smaller file
        that LAMMPS reads faster and that is easier to inspect."""
        keep = [self.index_of(e) for e in elements]
        remap = {old: new for new, old in enumerate(keep, start=1)}
        remap[0] = 0
        ff = self.copy()
        ff.atoms = [copy.deepcopy(self.atoms[i - 1]) for i in keep]

        def _filter(entries):
            out = []
            for e in entries:
                if all(i in remap for i in e.idx):
                    out.append(Entry(tuple(remap[i] for i in e.idx), list(e.values), copy.deepcopy(e.lg)))
            return out

        ff.bonds, ff.offdiag = _filter(self.bonds), _filter(self.offdiag)
        ff.angles, ff.torsions, ff.hbonds = _filter(self.angles), _filter(self.torsions), _filter(self.hbonds)
        return ff

    def summary(self) -> dict:
        return {
            "elements": self.elements,
            "n_general": len(self.general),
            "n_bonds": len(self.bonds),
            "n_offdiag": len(self.offdiag),
            "n_angles": len(self.angles),
            "n_torsions": len(self.torsions),
            "n_hbonds": len(self.hbonds),
            "lg": self.lg,
            "header": self.header.strip()[:200],
            "bond_order_cutoff": self.general[29] / 100.0 if len(self.general) > 29 else None,
        }

    # -------------------------------------------------------------------- I/O
    def to_string(self, precision: int = 4) -> str:
        return write_string(self, precision=precision)

    def write(self, path, precision: int = 4) -> Path:
        path = Path(path)
        path.write_text(self.to_string(precision=precision))
        return path


# ------------------------------------------------------------------- parsing
_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?$")


def _floats(tokens) -> list:
    out = []
    for t in tokens:
        if t.startswith("!"):
            break
        if not _NUM.match(t):
            break
        out.append(float(t.replace("d", "e").replace("D", "e")))
    return out


def _count(line: str, what: str, lineno: int) -> int:
    tok = line.split()
    if not tok:
        raise FFieldError(f"line {lineno}: expected the number of {what}, found a blank line")
    try:
        return int(float(tok[0]))
    except ValueError:
        raise FFieldError(f"line {lineno}: expected the number of {what}, found {line.strip()!r}") from None


def parse_string(text: str, source: Optional[str] = None) -> ForceField:
    lines = text.splitlines()
    pos = 0

    def nxt():
        nonlocal pos
        if pos >= len(lines):
            raise FFieldError(f"unexpected end of file after line {pos}")
        line = lines[pos]
        pos += 1
        return line

    header = nxt()
    gen_line = nxt()
    n_gen = _count(gen_line, "general parameters", pos)
    comments = {"general": gen_line[gen_line.find("!"):] if "!" in gen_line else "! Number of general parameters"}
    general, general_comments = [], []
    for _ in range(n_gen):
        line = nxt()
        vals = _floats(line.split())
        if not vals:
            raise FFieldError(f"line {pos}: expected a general parameter value, found {line.strip()!r}")
        general.append(vals[0])
        general_comments.append(line[line.find("!") + 1:].strip() if "!" in line else "")

    atom_line = nxt()
    n_atoms = _count(atom_line, "atom types", pos)
    comments["atom"] = [atom_line] + [nxt() for _ in range(3)]
    atoms = []
    lg = False
    for a in range(n_atoms):
        first = nxt().split()
        if not first:
            raise FFieldError(f"line {pos}: blank line where element {a + 1} should start")
        symbol = first[0]
        vals = _floats(first[1:])
        for _ in range(3):
            vals += _floats(nxt().split())
        if len(vals) < 32:
            raise FFieldError(f"element {symbol} (line {pos}): expected 32 values over 4 lines, got {len(vals)}")
        entry = AtomEntry(symbol=symbol, values=vals[:32])
        # lg extension: a fifth, all-numeric line with 2 values
        if pos < len(lines):
            peek = lines[pos].split()
            nums = _floats(peek)
            if peek and len(nums) == 2 and len(nums) == len([t for t in peek if not t.startswith("!")]):
                entry.lg = nums
                lg = True
                pos += 1
        atoms.append(entry)

    bond_line = nxt()
    n_bonds = _count(bond_line, "bonds", pos)
    comments["bond"] = [bond_line, nxt()]
    bonds = []
    for _ in range(n_bonds):
        t1 = nxt().split()
        t2 = nxt().split()
        idx = (int(t1[0]), int(t1[1]))
        vals = _floats(t1[2:])
        vals2 = _floats(t2)
        if len(vals) < 8 or len(vals2) < 6:
            raise FFieldError(f"bond {idx} (line {pos}): expected 8 + 6..8 values, got {len(vals)} + {len(vals2)}")
        vals2 = (vals2 + [0.0] * 8)[:8]
        bonds.append(Entry(idx, vals[:8] + vals2))

    def _simple(n_idx, n_vals, what):
        line = nxt()
        n = _count(line, what, pos)
        comments[what] = line
        out = []
        for _ in range(n):
            t = nxt().split()
            try:
                idx = tuple(int(x) for x in t[:n_idx])
            except ValueError:
                raise FFieldError(f"line {pos}: bad {what} entry {' '.join(t)!r}") from None
            vals = _floats(t[n_idx:])
            if len(vals) < n_vals:
                raise FFieldError(f"line {pos}: {what} entry {idx} has {len(vals)} values, expected {n_vals}")
            extra = vals[n_vals:] or None
            out.append(Entry(idx, vals[:n_vals], extra))
        return out

    offdiag = _simple(2, 6, "off")
    angles = _simple(3, 7, "angle")
    torsions = _simple(4, 7, "tors")
    hbonds = _simple(3, 4, "hbond") if pos < len(lines) and lines[pos].strip() else []
    if not lg:
        for e in offdiag + angles + torsions + hbonds:
            e.lg = None

    return ForceField(header=header, general=general, general_comments=general_comments, atoms=atoms, bonds=bonds,
                      offdiag=offdiag, angles=angles, torsions=torsions, hbonds=hbonds, section_comments=comments,
                      lg=lg, source=source)


def read(path) -> ForceField:
    path = Path(path)
    return parse_string(path.read_text(errors="replace"), source=str(path))


# ------------------------------------------------------------------- writing
def _fmt(v: float, width: int, precision: int) -> str:
    w = max(width, precision + 5)
    s = f"{v:{w}.{precision}f}"
    if s == f"{-0.0:{w}.{precision}f}":
        s = f"{0.0:{w}.{precision}f}"
    return s if s.startswith(" ") else " " + s  # always whitespace-separated


def _row(values, precision, width=9) -> str:
    return "".join(_fmt(v, width, precision) for v in values)


def write_string(ff: ForceField, precision: int = 4) -> str:
    c = ff.section_comments
    out = [ff.header.rstrip()]
    out.append(f"{len(ff.general):3d}       {c.get('general', '! Number of general parameters')}".rstrip())
    for i, v in enumerate(ff.general):
        name = ff.general_comments[i] if i < len(ff.general_comments) and ff.general_comments[i] else (
            GENERAL_NAMES[i] if i < len(GENERAL_NAMES) else "")
        out.append(f"{_fmt(v, 10, precision)} !{name}")
    atom_c = c.get("atom") or ["", "            alfa;gamma(w);Val(angle);p(ovun5);n.u.;chiEEM;etaEEM;n.u.",
                               "            ro(pipi);p(lp2);Heat increment;p(boc4);p(boc3);p(boc5),n.u.;n.u.",
                               "            p(ovun2);p(val3);n.u.;Val(boc);p(val5);n.u.;n.u.;n.u."]
    out.append(f"{len(ff.atoms):3d}    ! Nr of atoms; atomID;ro(sigma); Val;atom mass;Rvdw;Dij;gamma;ro(pi);Val(e)")
    out.extend(line.rstrip() for line in atom_c[1:4])
    for a in ff.atoms:
        out.append(f" {a.symbol:<2s}" + _row(a.values[0:8], precision))
        for k in (8, 16, 24):
            out.append("   " + _row(a.values[k:k + 8], precision))
        if ff.lg and a.lg is not None:
            out.append("   " + _row(a.lg, precision))
    bond_c = c.get("bond") or ["", "                      p(be2);p(bo3);p(bo4);n.u.;p(bo1);p(bo2)"]
    out.append(f"{len(ff.bonds):3d}      ! Nr of bonds; at1;at2;De(sigma);De(pi);De(pipi);p(be1);p(bo5);13corr;n.u.;"
               "p(bo6),p(ovun1)")
    out.append(bond_c[1].rstrip())
    for e in ff.bonds:
        out.append(f"{e.idx[0]:3d}{e.idx[1]:3d}" + _row(e.values[0:8], precision))
        out.append("      " + _row(e.values[8:16], precision))
    out.append(f"{len(ff.offdiag):3d}    ! Nr of off-diagonal terms; at1;at2;Dij;RvdW;alfa;ro(sigma);ro(pi);ro(pipi)")
    for e in ff.offdiag:
        out.append(f"{e.idx[0]:3d}{e.idx[1]:3d}" + _row(e.values + (e.lg or [] if ff.lg else []), precision))
    out.append(f"{len(ff.angles):3d}    ! Nr of angles;at1;at2;at3;Thetao,o;ka;kb;pv1;pv2")
    for e in ff.angles:
        out.append("".join(f"{i:3d}" for i in e.idx) + _row(e.values, precision))
    out.append(f"{len(ff.torsions):3d}    ! Nr of torsions;at1;at2;at3;at4;;V1;V2;V3;V2(BO);vconj;n.u;n")
    for e in ff.torsions:
        out.append("".join(f"{i:3d}" for i in e.idx) + _row(e.values, precision))
    out.append(f"{len(ff.hbonds):3d}    ! Nr of hydrogen bonds;at1;at2;at3;Rhb;Dehb;vhb1")
    for e in ff.hbonds:
        out.append("".join(f"{i:3d}" for i in e.idx) + _row(e.values, precision))
    return "\n".join(out) + "\n"


def round_to_precision(values, precision: int = 4):
    """What a value becomes after a write/read round trip."""
    return [round(float(v), precision) for v in values]


def section_of(pid: str) -> str:
    return pid.split(":", 1)[0]


def name_of(pid: str) -> str:
    return pid.rsplit(":", 1)[-1]


def elements_of(pid: str) -> list:
    """Elements a parameter id refers to ([] for general parameters)."""
    parts = pid.split(":")
    if parts[0] == "gen":
        return []
    key = parts[1].split("#", 1)[0]
    return [s for s in key.split("-") if s != WILDCARD]
