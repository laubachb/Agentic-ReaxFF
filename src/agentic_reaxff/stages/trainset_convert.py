"""Convert training sets between this toolkit and the classic ReaxFF format
(trainset.in + geo + params), used by van Duin's standalone ReaxFF,
JAX-ReaxFF and ParAMS' importer.

  --to-legacy    trainset.yaml (+ params.yaml) -> trainset.in, geo, params
  --from-legacy  trainset.in + geo (+ params + ffield) -> trainset.yaml,
                 structures.extxyz (+ params.yaml)

Imported energy items keep their published combinations (allow_unbalanced:
legacy sets may use heats-of-formation-like combinations). Unsupported
item types (HEATFO, RMSG, single-atom displacement) are reported, not
silently dropped.
"""

from __future__ import annotations

from pathlib import Path

from ..io import ffield as F
from ..io import legacy as L
from ..io import params as P
from ..io import structures as SIO
from ..io import trainset as T

NAME = "trainset-convert"
SUMMARY = "Convert training sets to/from the classic ReaxFF format (trainset.in, geo, params) for JAX-ReaxFF/standalone ReaxFF/ParAMS."
SCHEMA = {
    "type": "object",
    "properties": {
        "to_legacy": {"type": ["string", "null"], "description": "trainset.yaml to export."},
        "params": {"type": ["string", "null"], "description": "params.yaml to export as a classic params file (needs the ffield it names)."},
        "from_legacy": {"type": ["string", "null"], "description": "Directory with trainset.in and geo (and optionally params, ffield)."},
        "trainset_in": {"type": ["string", "null"]},
        "geo": {"type": ["string", "null"]},
        "legacy_params": {"type": ["string", "null"]},
        "ffield": {"type": ["string", "null"]},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--to-legacy", dest="to_legacy", default=None)
    parser.add_argument("--params", default=None)
    parser.add_argument("--from-legacy", dest="from_legacy", default=None)
    parser.add_argument("--trainset-in", dest="trainset_in", default=None)
    parser.add_argument("--geo", default=None)
    parser.add_argument("--legacy-params", dest="legacy_params", default=None)
    parser.add_argument("--ffield", default=None)


def run(args) -> dict:
    out = Path(args.output_dir or ".")
    out.mkdir(parents=True, exist_ok=True)
    if args.to_legacy:
        ts = T.load(args.to_legacy)
        names = L.write_geo(ts.used_structures(), out / "geo")
        warn = L.write_trainset_in(ts, names, out / "trainset.in")
        res = {"trainset_in": str(out / "trainset.in"), "geo": str(out / "geo"), "n_structures": len(names),
               "warnings": warn[:20], "n_warnings": len(warn)}
        if args.params:
            spec = P.load(args.params)
            ff = F.read(spec.ffield)
            L.write_params(spec, ff, out / "params")
            ff.write(out / "ffield")
            res.update({"params": str(out / "params"), "ffield": str(out / "ffield")})
        res["note"] = ("JAX-ReaxFF: jaxreaxff --init_FF ffield --params params --geo geo --train_file trainset.in "
                       "(it ignores CELL PARAMETERS and cell optimization; EEM only)")
        return res
    src = Path(args.from_legacy) if args.from_legacy else None
    tin = Path(args.trainset_in) if args.trainset_in else (src / "trainset.in" if src else None)
    geo = Path(args.geo) if args.geo else (src / "geo" if src else None)
    if not tin or not geo or not tin.is_file() or not geo.is_file():
        raise ValueError("give --to-legacy, or --from-legacy DIR (with trainset.in and geo) / --trainset-in + --geo")
    structs = L.read_geo(geo)
    sd = {s.name: s for s in structs}
    items, warn = L.read_trainset_in(tin, sd)
    used = {n for it in items for n in it.structures()}
    keep = [s for s in structs if s.name in used]
    spath = out / "structures.extxyz"
    SIO.write(spath, keep)
    ts = T.TrainingSet(items=items, structures={s.name: s for s in keep}, structures_path=str(spath),
                       meta={"imported_from": str(tin)})
    tpath = ts.write(out / "trainset.yaml", spath)
    res = {"trainset": str(tpath), "structures": str(spath), "n_items": len(items), "counts": ts.counts(),
           "n_structures": len(keep), "warnings": warn[:20], "n_warnings": len(warn)}
    ffp = Path(args.ffield) if args.ffield else (src / "ffield" if src and (src / "ffield").is_file() else None)
    lp = Path(args.legacy_params) if args.legacy_params else (src / "params" if src and (src / "params").is_file() else None)
    if ffp and lp:
        ff = F.read(ffp)
        params = L.read_params(lp, ff)
        spec = P.ParamSpec(ffield=str(ffp.resolve()), params=params, precision=4,
                           constraints=P.parse_constraints(P.default_constraints(ff, [p.id for p in params]), ff))
        res["params"] = str(spec.write(out / "params.yaml"))
        res["n_params"] = len(params)
    return res
