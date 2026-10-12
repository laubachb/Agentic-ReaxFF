"""Inspect a ReaxFF force field: what it contains, what it lacks for a
chemistry, which parameter ids exist, sanity checks, and differences
between two files.

Checks reported (none modify the file):

- interactions missing for the requested elements: bonds, off-diagonal
  pairs, angles, torsions (a missing bond means no covalent bonding between
  those elements; a missing angle or torsion contributes no energy)
- EEM parameters below the polarization-catastrophe bound eta >= 7.2 gamma
  (SCM ReaxFF manual), which lets charges run away at short distance
- elements with neither vdW shielding (gamma_w > 0.5) nor an inner wall
  (rcore2, acore2 > 0.01): LAMMPS warns about these; nothing keeps such
  atoms apart at very short range except the bond terms
- the bond-order cutoff (general parameter 30 / 100), which with LAMMPS'
  thb_cutoff controls where interactions switch off discontinuously
  (Furman & Wales 2019 [FW19])
"""

from __future__ import annotations

import itertools
import re

from ..io import ffield as F

NAME = "ffield-info"
SUMMARY = "Summarize a ffield: elements, sections, missing interactions for a chemistry, sanity checks, parameter ids, diff."
USES_OUTPUT_DIR = False
SCHEMA = {
    "type": "object",
    "required": ["ffield"],
    "properties": {
        "ffield": {"type": "string"},
        "elements": {"type": ["array", "null"], "items": {"type": "string"},
                     "description": "Check coverage for this chemistry (default: every element in the file)."},
        "list_params": {"type": ["string", "null"],
                        "description": "Regex over parameter ids to list with values (e.g. 'bond:C-H:.*' or 'atom:Si:.*')."},
        "diff": {"type": ["string", "null"], "description": "Another ffield: report every parameter that differs."},
        "diff_tol": {"type": "number", "default": 1e-6},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--ffield", default=None)
    parser.add_argument("--elements", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--list-params", dest="list_params", default=None)
    parser.add_argument("--diff", default=None)
    parser.add_argument("--diff-tol", dest="diff_tol", type=float, default=1e-6)


def coverage(ff: F.ForceField, elements: list) -> dict:
    els = [e for e in elements]
    missing = {"bond": [], "off": [], "angle": [], "tors": []}
    for a, b in itertools.combinations_with_replacement(els, 2):
        if not ff.has_interaction("bond", [a, b]):
            missing["bond"].append(f"{a}-{b}")
        if a != b and not ff.has_interaction("off", [a, b]):
            missing["off"].append(f"{a}-{b}")
    for j, k, l in itertools.product(els, repeat=3):
        if j <= l and not ff.has_interaction("angle", [j, k, l]):
            missing["angle"].append(f"{j}-{k}-{l}")
    for j, k in itertools.product(els, repeat=2):
        if j <= k:
            has = any(ff.find_entry("tors", [x, j, k, y]) is not None
                      for x in els + ["X"] for y in els + ["X"])
            if not has:
                missing["tors"].append(f"*-{j}-{k}-*")
    return missing


def sanity(ff: F.ForceField) -> list:
    issues = []
    for a in ff.atoms:
        v = dict(zip(F.ATOM_NAMES, a.values))
        if v["eta"] < 7.2 * v["gamma"]:
            issues.append({"check": "eem_polarization", "element": a.symbol,
                           "detail": f"eta {v['eta']:.4f} < 7.2*gamma {7.2 * v['gamma']:.4f}: charges can run away at short range"})
        shield = v["gamma_w"] > 0.5
        wall = v["rcore2"] > 0.01 and v["acore2"] > 0.01
        if not shield and not wall:
            issues.append({"check": "short_range_repulsion", "element": a.symbol,
                           "detail": "no vdW shielding (gamma_w <= 0.5) and no inner wall (rcore2/acore2 <= 0.01)"})
    bo = ff.general[29] / 100.0 if len(ff.general) > 29 else None
    if bo is not None and bo > 0.001 + 1e-12:
        issues.append({"check": "bond_order_cutoff", "detail": f"bond-order cutoff {bo:g} > 0.001: larger cutoffs make "
                       "bonds switch off more abruptly (energy drift in NVE, noisy geometry optimizations) [FW19]"})
    return issues


def run(args) -> dict:
    ff = F.read(args.ffield)
    out = {"ffield": args.ffield, **ff.summary()}
    els = args.elements or ff.elements
    unknown = [e for e in els if e.lower() not in [x.lower() for x in ff.elements]]
    if unknown:
        out["elements_not_in_ffield"] = unknown
        els = [e for e in els if e not in unknown]
    out["missing_interactions"] = coverage(ff, els)
    out["sanity"] = sanity(ff)
    out["n_param_ids"] = len(ff.param_ids())
    if args.list_params:
        rx = re.compile(args.list_params)
        out["params"] = {p: ff.get(p) for p in ff.param_ids(include_frozen=True) if rx.fullmatch(p)}
    if args.diff:
        other = F.read(args.diff)
        common = set(ff.param_ids(include_frozen=True)) & set(other.param_ids(include_frozen=True))
        diffs = {}
        for p in ff.param_ids(include_frozen=True):
            if p in common:
                a, b = ff.get(p), other.get(p)
                if abs(a - b) > args.diff_tol:
                    diffs[p] = {"this": a, "other": b, "rel": (b - a) / abs(a) if a else None}
        out["diff"] = {"n_differ": len(diffs), "params": diffs,
                       "only_here": sorted(set(ff.param_ids(include_frozen=True)) - common)[:200],
                       "only_there": sorted(set(other.param_ids(include_frozen=True)) - common)[:200]}
    return out
