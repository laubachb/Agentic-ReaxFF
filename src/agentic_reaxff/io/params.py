"""The active-parameter specification: which ffield parameters a fit may
change, within which bounds, under which constraints.

```yaml
ffield: start.ffield            # the template every candidate is written from
precision: 4                    # decimals written (and evaluated)
params:
  - {id: "bond:C-H:De_s", value: 167.2, min: 133.8, max: 200.6}
  - {id: "atom:C:r_s", value: 1.3825, min: 1.106, max: 1.659}
constraints:                    # linear inequalities "lhs >= rhs", checked per candidate
  - "atom:C:r_s >= atom:C:r_pi"
```

Optimizers work in normalized coordinates z = (x - min)/(max - min) in
[0, 1]^n, as Shchygol et al. [Sh19] and ParAMS [P21] do, so one step size
means the same thing for every parameter.

Presets encode the classic sequential protocol (charges -> bonds ->
valence angles -> torsions -> vdW/inner wall -> global refinement;
van Duin 2001 [vD01], Chenoweth 2008 [C08], Senftle 2016 [S16]):
`params-select --preset bond --elements Si` activates the bond-order and
bond-energy parameters of every interaction involving Si.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from . import ffield as F

# preset -> {section: [parameter names]}
PRESETS = {
    "charge": {"atom": ["chi", "eta", "gamma"]},
    "bond": {"atom": ["r_s", "r_pi", "r_pipi"],
             "bond": ["De_s", "De_p", "De_pp", "p_be1", "p_be2", "p_bo1", "p_bo2", "p_bo3", "p_bo4", "p_bo5", "p_bo6"],
             "off": ["r_s", "r_p", "r_pp"]},
    "sigma_bond": {"atom": ["r_s"], "bond": ["De_s", "p_be1", "p_be2", "p_bo1", "p_bo2"], "off": ["r_s"]},
    "coordination": {"atom": ["p_ovun2", "p_ovun5", "p_lp2", "p_boc3", "p_boc4", "p_boc5"], "bond": ["p_ovun1"]},
    "valence": {"angle": ["theta_00", "p_val1", "p_val2", "p_val4", "p_val7", "p_pen1", "p_coa1"],
                "atom": ["p_val3", "p_val5"]},
    "torsion": {"tors": ["V1", "V2", "V3", "p_tor1", "p_cot1"]},
    "vdw": {"atom": ["r_vdw", "epsilon", "alpha", "gamma_w"], "off": ["D", "r_vdW", "alpha"]},
    "hbond": {"hbond": ["r0_hb", "p_hb1", "p_hb2", "p_hb3"]},
    "inner_wall": {"atom": ["rcore2", "ecore2", "acore2"]},
}
PRESETS["all"] = {}
for _p in ("charge", "bond", "coordination", "valence", "torsion", "vdw", "hbond"):
    for _sec, _names in PRESETS[_p].items():
        PRESETS["all"].setdefault(_sec, [])
        PRESETS["all"][_sec] += [n for n in _names if n not in PRESETS["all"][_sec]]

# switches stored as numbers: never optimized
FLAG_PARAMS = {("bond", "v13cor"), ("bond", "ovc")}

# Hard physical limits by parameter name (applied after any other bound rule).
# Positive quantities stay positive; bond-order exponents stay >= 1 (ParAMS'
# default constraint [P21]); angles stay in [0, 180].
_POSITIVE = {("atom", n) for n in ("r_s", "r_vdw", "epsilon", "gamma", "alpha", "gamma_w", "eta", "rcore2", "ecore2",
                                   "acore2")} | {("bond", "De_s"), ("bond", "De_p"), ("bond", "De_pp")} | \
            {("off", n) for n in ("D", "r_vdW", "alpha", "r_s")} | {("hbond", "r0_hb")}
_EXPONENT = {("bond", "p_bo2"), ("bond", "p_bo4"), ("bond", "p_bo6"), ("angle", "p_val4")}
# pi radii <= 0 switch pi bonding off for that element/pair: keep them off
_PI_SWITCH = {("atom", "r_pi"), ("atom", "r_pipi"), ("off", "r_p"), ("off", "r_pp")}


class ParamError(ValueError):
    pass


@dataclass
class Param:
    id: str
    value: float
    min: float
    max: float

    def to_dict(self) -> dict:
        return {"id": self.id, "value": round(self.value, 8), "min": round(self.min, 8), "max": round(self.max, 8)}


@dataclass
class ParamSpec:
    ffield: str
    params: list
    precision: int = 4
    constraints: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def ids(self) -> list:
        return [p.id for p in self.params]

    @property
    def lo(self) -> np.ndarray:
        return np.array([p.min for p in self.params], float)

    @property
    def hi(self) -> np.ndarray:
        return np.array([p.max for p in self.params], float)

    @property
    def x0(self) -> np.ndarray:
        return np.array([p.value for p in self.params], float)

    def to_z(self, x) -> np.ndarray:
        return (np.asarray(x, float) - self.lo) / (self.hi - self.lo)

    def to_x(self, z) -> np.ndarray:
        z = np.clip(np.asarray(z, float), 0.0, 1.0)
        return self.lo + z * (self.hi - self.lo)

    def round_x(self, x) -> np.ndarray:
        """What x becomes once written to an ffield (and read back)."""
        return np.round(np.asarray(x, float), self.precision)

    def apply(self, ff: F.ForceField, x) -> F.ForceField:
        out = ff.copy()
        out.set_many(self.ids, self.round_x(x))
        return out

    def violation(self, x) -> float:
        """Sum of constraint violations (0 = feasible)."""
        if not self.constraints:
            return 0.0
        vals = dict(zip(self.ids, np.asarray(x, float)))
        tot = 0.0
        for c in self.constraints:
            lhs, rhs = c["lhs"], c["rhs"]
            a = vals.get(lhs, c.get("lhs_value"))
            b = vals.get(rhs, c.get("rhs_value"))
            if a is None or b is None:
                continue
            b = c.get("coef", 1.0) * b
            if a < b:
                tot += b - a
        return tot

    def to_dict(self) -> dict:
        return {"ffield": self.ffield, "precision": self.precision,
                "params": [p.to_dict() for p in self.params],
                "constraints": [f"{c['lhs']} >= " + (f"{c['coef']:g} * " if c.get("coef", 1.0) != 1.0 else "") + c["rhs"]
                                for c in self.constraints],
                **({"meta": self.meta} if self.meta else {})}

    def write(self, path) -> Path:
        path = Path(path)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, width=120)
        return path


_CONSTRAINT = re.compile(r"^\s*(\S+)\s*>=\s*(?:([-+0-9.eE]+)\s*\*\s*)?(\S+)\s*$")


def parse_constraints(lines, ff: Optional[F.ForceField] = None) -> list:
    out = []
    for line in lines or []:
        m = _CONSTRAINT.match(line)
        if not m:
            raise ParamError(f"constraint {line!r}: expected 'param_id >= param_id' or 'param_id >= c * param_id'")
        c = {"lhs": m.group(1), "rhs": m.group(3), "coef": float(m.group(2)) if m.group(2) else 1.0}
        if ff is not None:
            c["lhs_value"] = ff.get(c["lhs"])
            c["rhs_value"] = ff.get(c["rhs"])
        out.append(c)
    return out


def load(path, ffield_override: Optional[str] = None) -> ParamSpec:
    path = Path(path)
    d = yaml.safe_load(path.read_text()) or {}
    unknown = set(d) - {"ffield", "precision", "params", "constraints", "meta"}
    if unknown:
        raise ParamError(f"{path}: unknown key(s) {sorted(unknown)}")
    ffp = ffield_override or d.get("ffield")
    if not ffp:
        raise ParamError(f"{path}: `ffield:` (template force field) is required")
    ffp = Path(ffp)
    if not ffp.is_absolute():
        ffp = (path.parent / ffp)
    ff = F.read(ffp)
    params = []
    for k, p in enumerate(d.get("params") or []):
        pid = p.get("id")
        if not pid:
            raise ParamError(f"{path}: params[{k}] has no id")
        value = float(p.get("value", ff.get(pid)))
        lo, hi = float(p["min"]), float(p["max"])
        if not lo < hi:
            raise ParamError(f"{path}: {pid}: min {lo} must be < max {hi}")
        if not lo <= value <= hi:
            raise ParamError(f"{path}: {pid}: value {value} outside [{lo}, {hi}]")
        params.append(Param(pid, value, lo, hi))
    ids = [p.id for p in params]
    if len(set(ids)) != len(ids):
        raise ParamError(f"{path}: duplicate parameter ids")
    if not params:
        raise ParamError(f"{path}: no active parameters")
    return ParamSpec(ffield=str(ffp.resolve()), params=params, precision=int(d.get("precision", 4)),
                     constraints=parse_constraints(d.get("constraints"), ff), meta=d.get("meta") or {})


# ------------------------------------------------------------ selection
def hard_limits(pid: str, x0: float):
    """(lo, hi) physical limits, or None if the parameter must stay fixed."""
    sec, name = F.section_of(pid), F.name_of(pid)
    if (sec, name) in FLAG_PARAMS or name in F.FROZEN_ALWAYS.get(sec, set()):
        return None
    if (sec, name) in _PI_SWITCH and x0 <= 0:
        return None
    lo, hi = -math.inf, math.inf
    if (sec, name) in _POSITIVE or ((sec, name) in _PI_SWITCH):
        lo = 1e-4
    if (sec, name) in _EXPONENT:
        lo = 1.0
    if sec == "angle" and name == "theta_00":
        lo, hi = 0.0, 180.0
    if (sec, name) == ("atom", "gamma_w") and x0 > 0.5:
        lo = 0.5 + 1e-4  # LAMMPS switches vdW shielding on only for gamma_w > 0.5
    return lo, hi


def select(ff: F.ForceField, *, presets=None, include=None, exclude=None, elements=None, only_elements=None,
           data_elements=None, include_zero: bool = False) -> list:
    """Parameter ids matching the rules, in file order.

    presets         names from PRESETS (union)
    include         extra ids or regexes (`re.fullmatch` on the id)
    exclude         ids or regexes removed at the end
    elements        keep parameters that involve at least one of these elements
    only_elements   keep parameters whose elements are all in this set
    data_elements   list of element sets present together in the training
                    data; multi-body parameters are kept only when some
                    structure contains all their elements (no data = no fit)
    include_zero    keep parameters whose current value is 0 (off by default:
                    a zero is usually a switched-off term, ParAMS keeps them [P21])
    """
    wanted = {}
    for p in presets or []:
        if p not in PRESETS:
            raise ParamError(f"unknown preset {p!r}; known: {sorted(PRESETS)}")
        for sec, names in PRESETS[p].items():
            wanted.setdefault(sec, set()).update(names)
    ids = []
    all_ids = ff.param_ids()
    for pid in all_ids:
        sec, name = F.section_of(pid), F.name_of(pid)
        if name in wanted.get(sec, ()):
            ids.append(pid)
    for pat in include or []:
        rx = re.compile(pat)
        ids += [pid for pid in all_ids if (pid == pat or rx.fullmatch(pid)) and pid not in ids]
    if elements:
        els = set(elements)
        ids = [p for p in ids if F.section_of(p) == "gen" or els & set(F.elements_of(p))]
    if only_elements:
        els = set(only_elements)
        ids = [p for p in ids if set(F.elements_of(p)) <= els]
    if data_elements is not None:
        sets = [set(s) for s in data_elements]
        ids = [p for p in ids if F.section_of(p) in ("gen", "atom") or any(set(F.elements_of(p)) <= s for s in sets)
               or (F.section_of(p) == "atom" and any(set(F.elements_of(p)) <= s for s in sets))]
    for pat in exclude or []:
        rx = re.compile(pat)
        ids = [p for p in ids if not (p == pat or rx.fullmatch(p))]
    out = []
    for pid in ids:
        x0 = ff.get(pid)
        if not include_zero and x0 == 0.0:
            continue
        if hard_limits(pid, x0) is None:
            continue
        out.append(pid)
    return list(dict.fromkeys(out))


def make_bounds(ff: F.ForceField, ids: list, *, rule: str = "relative", scale: float = 0.2,
                library: Optional[dict] = None, library_margin: float = 0.1, overrides: Optional[dict] = None) -> list:
    """Bounds for each id.

    relative  x0 ± scale·|x0| (ParAMS' bounds_scale = 1 + scale [P21]); zero -> ±1
    library   the range of published values of that parameter (same name,
              same elements first, else same name any elements), widened by
              library_margin of the range and always containing x0 — the
              "historical interval" idea of Shchygol 2019 [Sh19]
    union     the wider of the two
    """
    out = []
    for pid in ids:
        x0 = ff.get(pid)
        rel = (x0 - scale * abs(x0), x0 + scale * abs(x0)) if x0 != 0 else (-1.0, 1.0)
        lo, hi = rel
        if rule in ("library", "union"):
            from ..stages.ffield_library import lookup

            lib = lookup(library or {}, pid)
            if lib:
                w = lib["max"] - lib["min"]
                pad = library_margin * (w if w > 0 else max(abs(x0), 1.0))
                llo, lhi = min(lib["min"] - pad, x0), max(lib["max"] + pad, x0)
                lo, hi = (llo, lhi) if rule == "library" else (min(lo, llo), max(hi, lhi))
        elif rule != "relative":
            raise ParamError(f"unknown bounds rule {rule!r} (relative|library|union)")
        if overrides and pid in overrides:
            lo, hi = overrides[pid]
        lim = hard_limits(pid, x0)
        if lim:
            lo, hi = max(lo, lim[0]), min(hi, lim[1])
        if not lo < hi:
            span = max(abs(x0) * 0.01, 1e-3)
            lo, hi = x0 - span, x0 + span
        lo, hi = min(lo, x0), max(hi, x0)
        out.append(Param(pid, x0, float(lo), float(hi)))
    return out


def default_constraints(ff: F.ForceField, ids: list) -> list:
    """r_s >= r_pi >= r_pipi for every element whose radii are active and
    switched on (ParAMS' default bond-radius ordering [P21]); and, when EEM
    parameters are active, eta >= 7.2 gamma, below which the shielded
    Coulomb matrix loses positive definiteness at short range and charges
    run away (the "polarization catastrophe" bound in the SCM ReaxFF manual).

    A constraint the starting force field already violates is left out (and
    listed by `skipped_constraints`): published force fields do not all obey
    the ordering (Chenoweth 2008 C has r_pi < r_pipi), and a fit should not
    be forced away from a validated set by a rule of thumb."""
    return [c for c in _candidate_constraints(ff, ids) if _holds(ff, c)]


def skipped_constraints(ff: F.ForceField, ids: list) -> list:
    return [c for c in _candidate_constraints(ff, ids) if not _holds(ff, c)]


def _holds(ff, c) -> bool:
    m = _CONSTRAINT.match(c)
    coef = float(m.group(2)) if m.group(2) else 1.0
    return ff.get(m.group(1)) >= coef * ff.get(m.group(3)) - 1e-12


def _candidate_constraints(ff: F.ForceField, ids: list) -> list:
    active = set(ids)
    out = []
    for a in ff.atoms:
        eta, gam = f"atom:{a.symbol}:eta", f"atom:{a.symbol}:gamma"
        if eta in active or gam in active:
            out.append(f"{eta} >= 7.2 * {gam}")
        rs, rp, rpp = (f"atom:{a.symbol}:{n}" for n in ("r_s", "r_pi", "r_pipi"))
        if (rs in active or rp in active) and ff.get(rp) > 0:
            out.append(f"{rs} >= {rp}")
        if (rp in active or rpp in active) and ff.get(rp) > 0 and ff.get(rpp) > 0:
            out.append(f"{rp} >= {rpp}")
    return out
