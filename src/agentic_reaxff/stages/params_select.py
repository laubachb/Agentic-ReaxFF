"""Choose which parameters a fit may change, and their bounds.

The single most consequential choice in ReaxFF fitting [L13]. Defaults
follow the literature:

  * fit in stages, not everything at once: charges -> bonds -> valence
    angles -> torsions -> vdW / coordination -> joint refinement ([vD01,
    C08, S16]; presets below), and no more than ~sqrt(n_items) parameters
    at a time (ParAMS rule of thumb [P21]);
  * activate only parameters the training data can see: multi-body terms
    whose elements never occur together in a training structure are left
    out (`--require-data`, default on), as ParAMS' ZnS example freezes
    "anything without training data" [P21];
  * when extending a force field, touch only the new element's terms
    (`--elements Si`), keeping published blocks intact for compatibility
    with the rest of the branch [S16, Sh19];
  * never optimize structural switches, valencies, masses, unused slots,
    taper radii or the bond-order cutoff; keep zero-valued (switched-off)
    terms at zero unless `include_zero`;
  * bounds: ±20% of the start value by default (ParAMS bounds_scale 1.2
    [P21]); `--bounds library` uses the range of published values from
    `ffield-library` (Shchygol's historical intervals [Sh19]); `union`
    takes the wider. Hard physical limits (positive energies/radii,
    exponents >= 1, 0 <= theta_00 <= 180) always apply;
  * default constraints: r_s >= r_pi >= r_pipi [P21] and, for EEM
    parameters, eta >= 7.2 gamma (polarization-catastrophe bound).

`--from-sensitivity sensitivity.json --top N` keeps the N most influential
parameters (ParAMS' sensitivity-first workflow, Freitas Gustavo 2023 [FH23]).
`--from-merge merge_report.json` activates the interactions ffield-merge
copied from donors/analogs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..io import ffield as F
from ..io import params as P
from ..io import trainset as T

NAME = "params-select"
SUMMARY = "Select active ffield parameters (presets per fitting stage, element filters, data coverage) with literature-based bounds and constraints."
SCHEMA = {
    "type": "object",
    "required": ["ffield"],
    "properties": {
        "ffield": {"type": "string"},
        "preset": {"type": ["array", "null"], "items": {"type": "string", "enum": sorted(P.PRESETS)}},
        "include": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Extra ids or regexes."},
        "exclude": {"type": ["array", "null"], "items": {"type": "string"}},
        "elements": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Keep parameters involving any of these."},
        "only_elements": {"type": ["array", "null"], "items": {"type": "string"}},
        "trainset": {"type": ["string", "null"], "description": "Used for --require-data and the sqrt(n) check."},
        "require_data": {"type": "boolean", "default": True},
        "include_zero": {"type": "boolean", "default": False},
        "bounds": {"type": "string", "enum": ["relative", "library", "union"], "default": "relative"},
        "scale": {"type": "number", "default": 0.2, "description": "relative bounds: ±scale·|x0|."},
        "library": {"type": ["string", "null"], "description": "library.json from ffield-library."},
        "from_sensitivity": {"type": ["string", "null"]},
        "top": {"type": ["integer", "null"]},
        "from_merge": {"type": ["string", "null"], "description": "merge_report.json: activate the preset names only on merged/seeded interactions."},
        "precision": {"type": "integer", "default": 4},
        "constraints": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Extra 'a >= b' / 'a >= c * b' constraints."},
        "no_default_constraints": {"type": "boolean", "default": False},
        "bounds_override": {"type": ["object", "null"], "description": "param id -> [min, max]."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--ffield", default=None)
    parser.add_argument("--preset", action="append", default=None, choices=sorted(P.PRESETS))
    parser.add_argument("--include", action="append", default=None)
    parser.add_argument("--exclude", action="append", default=None)
    parser.add_argument("--elements", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--only-elements", dest="only_elements", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--trainset", default=None)
    parser.add_argument("--no-require-data", dest="require_data", action="store_false", default=True)
    parser.add_argument("--include-zero", dest="include_zero", action="store_true", default=False)
    parser.add_argument("--bounds", default="relative", choices=["relative", "library", "union"])
    parser.add_argument("--scale", type=float, default=0.2)
    parser.add_argument("--library", default=None)
    parser.add_argument("--from-sensitivity", dest="from_sensitivity", default=None)
    parser.add_argument("--top", type=int, default=None)
    parser.add_argument("--from-merge", dest="from_merge", default=None)
    parser.add_argument("--precision", type=int, default=4)
    parser.add_argument("--constraint", dest="constraints", action="append", default=None)
    parser.add_argument("--no-default-constraints", dest="no_default_constraints", action="store_true", default=False)
    parser.add_argument("--bounds-override", dest="bounds_override", type=json.loads, default=None)


def _data_element_sets(ts: T.TrainingSet) -> list:
    return [set(s.symbols) for s in ts.used_structures()]


def run(args) -> dict:
    ff = F.read(args.ffield)
    ts = T.load(args.trainset) if args.trainset else None
    warnings = []
    if args.from_sensitivity:
        sens = json.loads(Path(args.from_sensitivity).read_text())
        ranked = [r["id"] for r in sens["ranking"] if r.get("influential", True)]
        ids = ranked[: args.top] if args.top else ranked
        ids = [p for p in ids if P.hard_limits(p, ff.get(p)) is not None]
    else:
        if not (args.preset or args.include):
            raise ValueError("give --preset (charge, bond, valence, torsion, vdw, coordination, hbond, inner_wall, "
                             "sigma_bond, all) and/or --include, or --from-sensitivity")
        ids = P.select(ff, presets=args.preset, include=args.include, exclude=None, elements=args.elements,
                       only_elements=args.only_elements,
                       data_elements=_data_element_sets(ts) if (ts is not None and args.require_data) else None,
                       include_zero=args.include_zero)
    if args.from_merge:
        rep = json.loads(Path(args.from_merge).read_text())
        merged = set(rep.get("needs_fitting") or [])
        keep = []
        for p in ids:
            sec = F.section_of(p)
            if sec in ("gen",):
                continue
            if sec == "atom":
                el = F.elements_of(p)[0]
                if rep.get("provenance", {}).get(f"atom:{el}", "base") != "base":
                    keep.append(p)
                continue
            key = p.rsplit(":", 1)[0].split("#", 1)[0]
            if key in merged or f"{sec}:{'-'.join(reversed(key.split(':', 1)[1].split('-')))}" in merged:
                keep.append(p)
        ids = keep
    if args.exclude:
        import re

        for pat in args.exclude:
            rx = re.compile(pat)
            ids = [p for p in ids if not (p == pat or rx.fullmatch(p))]
    if not ids:
        raise ValueError("no parameters matched: check the preset/elements, or --no-require-data if the training set "
                         "does not yet cover these interactions")
    library = json.loads(Path(args.library).read_text()) if args.library else None
    if args.bounds != "relative" and library is None:
        raise ValueError("--bounds library/union needs --library library.json (ffield-library)")
    overrides = {k: tuple(v) for k, v in (args.bounds_override or {}).items()}
    params = P.make_bounds(ff, ids, rule=args.bounds, scale=args.scale, library=library, overrides=overrides)
    cons = [] if args.no_default_constraints else P.default_constraints(ff, ids)
    skipped = [] if args.no_default_constraints else P.skipped_constraints(ff, ids)
    if skipped:
        warnings.append(f"default constraints the starting force field already violates were left out: {skipped}")
    cons += list(args.constraints or [])
    spec = P.ParamSpec(ffield=str(Path(args.ffield).resolve()), params=params, precision=args.precision,
                       constraints=P.parse_constraints(cons, ff),
                       meta={"preset": args.preset, "elements": args.elements, "bounds": args.bounds, "scale": args.scale})
    viol = spec.violation(spec.x0)
    if viol > 0:
        warnings.append(f"the starting force field violates the constraints by {viol:.4g}: the fit starts from an "
                        "infeasible point (check the constraint list)")
    out_dir = Path(args.output_dir or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = spec.write(out_dir / "params.yaml")
    by_section = {}
    for p in ids:
        by_section[F.section_of(p)] = by_section.get(F.section_of(p), 0) + 1
    out = {"params": str(path), "n_active": len(ids), "by_section": by_section, "constraints": spec.to_dict()["constraints"],
           "ids": ids[:60] + ([f"... +{len(ids) - 60}"] if len(ids) > 60 else [])}
    if ts is not None:
        n_items = len(ts.items)
        limit = int(np.sqrt(n_items))
        out["n_training_items"] = n_items
        if len(ids) > limit:
            warnings.append(f"{len(ids)} active parameters for {n_items} items exceeds ~sqrt(n) = {limit} (ParAMS rule of "
                            "thumb [P21]): split into stages, or rank with `sensitivity` and keep the top ones")
    out["warnings"] = warnings
    return out
