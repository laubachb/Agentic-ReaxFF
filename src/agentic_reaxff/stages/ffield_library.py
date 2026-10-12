"""Index a library of published force fields: parameter ranges for bounds,
element donors for new elements.

Shchygol et al. bounded each parameter by an interval and built "educated"
starting guesses from the median of historical values of the same parameter
(same elements first, then any elements) [Sh19]; Dumortier et al. used the
historical range widened to include ±20% around the start [Du24]; Larsson
et al. found the choice of ranges mattered more than any optimizer setting
[L13]. This stage produces those statistics from whatever ffields you have:

    library.json
      params:   "bond:C-H:De_s"  -> {min, max, median, n, sources}
                "*:bond:De_s"    -> same, over every element combination
      elements: "Si" -> [files defining Si, with their header lines]

`params-select --bounds library --library library.json` uses it. Element
keys are order-normalized, so C-H and H-C pool together. Files that fail to
parse are listed, not fatal. `setup --component ffield_library` fetches the
ReaxFF files distributed with LAMMPS into deps/ffields/ as a starting set;
add your group's ffields with --dirs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import config
from ..io import ffield as F

NAME = "ffield-library"
SUMMARY = "Index a directory of published ffields: per-parameter value ranges (for bounds) and element donors."
SCHEMA = {
    "type": "object",
    "properties": {
        "dirs": {"type": ["array", "null"], "items": {"type": "string"},
                 "description": "Directories (searched recursively) and/or ffield files. Default: deps/ffields."},
        "pattern": {"type": "string", "default": "*ffield*", "description": "Glob for files inside directories."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--dirs", action="append", default=None)
    parser.add_argument("--pattern", default="*ffield*")


def _norm_key(pid: str) -> str:
    sec, rest = pid.split(":", 1)
    if sec in ("gen", "atom"):
        return pid
    key, name = rest.rsplit(":", 1)
    key = key.split("#", 1)[0]
    parts = key.split("-")
    if sec in ("bond", "off"):
        parts = sorted(parts)
    elif sec in ("angle", "hbond"):
        if parts[0] > parts[2]:
            parts = parts[::-1]
    elif sec == "tors":
        if parts[::-1] < parts:
            parts = parts[::-1]
    return f"{sec}:{'-'.join(parts)}:{name}"


def build(files: list) -> dict:
    vals, srcs, elements, bad = {}, {}, {}, []
    for f in files:
        try:
            ff = F.read(f)
        except Exception as exc:  # noqa: BLE001
            bad.append({"file": str(f), "error": str(exc)[:200]})
            continue
        for a in ff.atoms:
            elements.setdefault(a.symbol, []).append({"file": str(f), "header": ff.header.strip()[:200]})
        for pid in ff.param_ids():
            v = ff.get(pid)
            for key in (_norm_key(pid), f"*:{F.section_of(pid)}:{F.name_of(pid)}"):
                vals.setdefault(key, []).append(v)
                srcs.setdefault(key, set()).add(Path(f).name)
    params = {}
    for key, v in vals.items():
        arr = np.asarray(v, float)
        nz = arr[arr != 0.0]
        params[key] = {"min": float(arr.min()), "max": float(arr.max()), "median": float(np.median(arr)),
                       "n": int(arr.size), "n_nonzero": int(nz.size), "sources": sorted(srcs[key])[:20]}
    return {"n_files": len(files) - len(bad), "files": [str(f) for f in files], "unreadable": bad,
            "elements": elements, "params": params}


def lookup(library: dict, pid: str):
    """Stats for a parameter id: exact element key first, else any elements."""
    p = library.get("params", {})
    return p.get(_norm_key(pid)) or p.get(f"*:{F.section_of(pid)}:{F.name_of(pid)}")


def run(args) -> dict:
    roots = args.dirs or [str(config.resolve_component("ffield_library"))]
    files = []
    for r in roots:
        p = Path(r)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files += sorted(x for x in p.rglob(args.pattern) if x.is_file())
        else:
            raise FileNotFoundError(r)
    files = list(dict.fromkeys(files))
    if not files:
        raise ValueError(f"no files matching {args.pattern!r} under {roots}")
    lib = build(files)
    out_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "library.json"
    path.write_text(json.dumps(lib, indent=1))
    return {"library": str(path), "n_files": lib["n_files"], "unreadable": lib["unreadable"],
            "elements": {e: len(v) for e, v in sorted(lib["elements"].items())}, "n_param_keys": len(lib["params"])}
