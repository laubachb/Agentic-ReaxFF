"""Assemble a starting force field for a new chemistry from published ones.

ReaxFF parameter sets are not freely combinable: they come in "branches"
(e.g. the combustion branch built on Chenoweth 2008 and the aqueous branch,
which re-fits O/H to get liquid water right), and within one branch only the
bond and angle terms between newly combined elements need fitting, while
crossing branches needs much more refitting [S16]. So:

  1. `--base` supplies the general parameters and its elements (keep the
     elements you need with --elements);
  2. each `--add EL=donor.ffield` copies element EL's atom block and every
     interaction among the final elements that the base lacks;
  3. `--analog NEW=OLD` fills interactions still missing for NEW by copying
     the corresponding OLD ones (e.g. Si=C gives Si-H from C-H), the usual
     way to seed a new element before fitting (cf. the ParAMS ZnS example,
     which seeds S blocks from O [P21]);
  4. everything copied across files or from analogs is listed in
     `needs_fitting`, which `params-select --from-merge` can activate.

The report says which general parameters differ between base and donors:
many differences mean the donor is from another branch and its blocks were
fitted against different general parameters.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from ..io import ffield as F
from . import ffield_info

NAME = "ffield-merge"
SUMMARY = "Build a starting ffield for a chemistry from a base ffield + element donors + analog seeding; reports what needs fitting."
SCHEMA = {
    "type": "object",
    "required": ["base"],
    "properties": {
        "base": {"type": "string", "description": "Base ffield (its general parameters are kept)."},
        "elements": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Final element list (default: base elements + added ones)."},
        "add": {"type": ["array", "null"], "items": {"type": "string"}, "description": "EL=donor_ffield entries."},
        "analog": {"type": ["array", "null"], "items": {"type": "string"}, "description": "NEW=OLD entries: seed missing NEW interactions from OLD ones."},
        "header": {"type": ["string", "null"]},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--base", default=None)
    parser.add_argument("--elements", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--add", action="append", default=None, help="EL=donor_ffield (repeatable)")
    parser.add_argument("--analog", action="append", default=None, help="NEW=OLD (repeatable)")
    parser.add_argument("--header", default=None)


def _sym_key(ff, e):
    return tuple(ff.symbol_of(i) for i in e.idx)


def _has(out: F.ForceField, sec: str, syms) -> bool:
    try:
        return out.find_entry(sec, list(syms)) is not None
    except F.FFieldError:
        return False


def _add_entry(out: F.ForceField, sec: str, syms, values, lg=None):
    idx = tuple(0 if s == F.WILDCARD else out.index_of(s) for s in syms)
    out._entries(sec).append(F.Entry(idx, list(values), copy.deepcopy(lg)))


def run(args) -> dict:
    base = F.read(args.base)
    adds = {}
    for item in args.add or []:
        el, _, path = item.partition("=")
        if not path:
            raise ValueError(f"--add {item!r}: expected EL=path")
        adds[el] = F.read(path)
    analogs = {}
    for item in args.analog or []:
        new, _, old = item.partition("=")
        if not old:
            raise ValueError(f"--analog {item!r}: expected NEW=OLD")
        analogs[new] = old
    final = args.elements or (base.elements + [e for e in adds if e not in base.elements])
    keep_base = [e for e in final if e in base.elements]
    out = base.subset(keep_base)
    provenance = {}
    for e in out.elements:
        provenance[f"atom:{e}"] = "base"

    # 1-2: donor atom blocks + their interactions
    for el, donor in adds.items():
        if el in out.elements:
            raise ValueError(f"{el} is already in the base ffield; drop it from --elements to take it from the donor")
        if el not in donor.elements:
            raise ValueError(f"donor for {el} does not contain {el} (has {donor.elements})")
        out.atoms.append(copy.deepcopy(donor.atoms[donor.index_of(el) - 1]))
        provenance[f"atom:{el}"] = f"donor:{Path(donor.source).name}"
    order = [e for e in final if e in out.elements]
    missing_el = [e for e in final if e not in out.elements]
    if missing_el:
        raise ValueError(f"no source for elements {missing_el}: add them with --add EL=donor.ffield")
    for el, donor in adds.items():
        for sec in ("bond", "off", "angle", "tors", "hbond"):
            for e in donor._entries(sec):
                if donor.is_orphan(e):
                    continue
                syms = _sym_key(donor, e)
                if el not in syms or any(s not in order and s != F.WILDCARD for s in syms):
                    continue
                if _has(out, sec, syms):
                    continue
                _add_entry(out, sec, syms, e.values, e.lg)
                provenance[f"{sec}:{'-'.join(syms)}"] = f"donor:{Path(donor.source).name}"

    # 3: analog seeding
    for new, old in analogs.items():
        if new not in out.elements or old not in out.elements:
            raise ValueError(f"--analog {new}={old}: both must be in the final ffield")
        for sec in ("bond", "off", "angle", "tors", "hbond"):
            for e in list(out._entries(sec)):
                syms = _sym_key(out, e)
                if old not in syms:
                    continue
                # every way of replacing occurrences of OLD by NEW (at least one)
                pos = [i for i, s in enumerate(syms) if s == old]
                for mask in range(1, 2 ** len(pos)):
                    new_syms = list(syms)
                    for b, i in enumerate(pos):
                        if mask >> b & 1:
                            new_syms[i] = new
                    if not _has(out, sec, new_syms):
                        _add_entry(out, sec, new_syms, e.values, e.lg)
                        provenance[f"{sec}:{'-'.join(new_syms)}"] = f"analog:{'-'.join(syms)}"

    # general-parameter compatibility
    gen_diff = {}
    for el, donor in adds.items():
        d = []
        for i, (a, b) in enumerate(zip(base.general, donor.general)):
            if abs(a - b) > 1e-6:
                name = F.GENERAL_NAMES[i] if i < len(F.GENERAL_NAMES) else f"#{i + 1}"
                d.append({"param": name, "base": a, "donor": b})
        gen_diff[el] = d
    if args.header:
        out.header = args.header
    else:
        out.header = (f"Merged by reaxff-agent ffield-merge: base {Path(args.base).name}"
                      + "".join(f"; {el} from {Path(d.source).name}" for el, d in adds.items())
                      + "".join(f"; {n} seeded from {o}" for n, o in analogs.items()))
    out_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ffield.merged"
    out.write(path)
    out2 = F.read(path)  # round trip: what LAMMPS will read
    needs = sorted(k for k, v in provenance.items() if v != "base" and not k.startswith("atom:"))
    warnings = []
    for el, d in gen_diff.items():
        if d:
            warnings.append(f"{el}'s donor differs from the base in {len(d)} general parameter(s) "
                            f"({', '.join(x['param'] for x in d[:6])}{'...' if len(d) > 6 else ''}): likely another "
                            "branch; its blocks were fitted with different general parameters [S16]")
    report = {"ffield": str(path), "elements": out2.elements, "provenance": provenance, "needs_fitting": needs,
              "general_parameter_differences": gen_diff, "warnings": warnings,
              "missing_interactions": ffield_info.coverage(out2, out2.elements), "sanity": ffield_info.sanity(out2)}
    (out_dir / "merge_report.json").write_text(json.dumps(report, indent=1))
    return {k: report[k] for k in ("ffield", "elements", "needs_fitting", "warnings", "missing_interactions", "sanity")} | {
        "report": str(out_dir / "merge_report.json")}
