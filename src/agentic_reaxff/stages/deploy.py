"""Package the chosen force field for use: the ffield plus a card that says
where it came from, what it was fitted to, how good it is, how it was
validated, how to run it, and what to cite.

LAMMPS usage in the card reproduces the settings the fit was evaluated with
(charge equilibration, tolerance, the bond-order cutoff in the file): a
ReaxFF force field is only defined together with those (cutoffs are part of
the force field [FW19]; validate other engines against the original [S16]).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from ..io import ffield as F
from . import ffield_info

NAME = "deploy"
SUMMARY = "Package a fitted ffield with a provenance/accuracy/validation card and LAMMPS usage."
SCHEMA = {
    "type": "object",
    "required": ["ffield"],
    "properties": {
        "ffield": {"type": "string"},
        "name": {"type": ["string", "null"]},
        "fit": {"type": ["string", "null"], "description": "fit output dir (fit.json, best_report.json)."},
        "evaluate": {"type": ["string", "null"], "description": "evaluate output dir."},
        "md_check": {"type": ["string", "null"], "description": "md-check output dir."},
        "trainset": {"type": ["string", "null"]},
        "start_ffield": {"type": ["string", "null"]},
        "charge_method": {"type": "string", "default": "qeq"},
        "cite": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Extra citations (the starting force field's papers)."},
    },
}


def add_arguments(parser) -> None:
    for k in ("ffield", "name", "fit", "evaluate", "md_check", "trainset", "start_ffield"):
        parser.add_argument(f"--{k.replace('_', '-')}", dest=k, default=None)
    parser.add_argument("--charge-method", dest="charge_method", default="qeq")
    parser.add_argument("--cite", action="append", default=None)


def _j(path, name):
    if not path:
        return None
    p = Path(path)
    p = p / name if p.is_dir() else p
    return json.loads(p.read_text()) if p.is_file() else None


def run(args) -> dict:
    ff = F.read(args.ffield)
    els = ff.elements
    name = args.name or f"ffield.reax.{''.join(els)}"
    out = Path(args.output_dir or ".") / name.replace("ffield.reax.", "")
    out.mkdir(parents=True, exist_ok=True)
    dest = out / name
    shutil.copy2(args.ffield, dest)
    fit = _j(args.fit, "fit.json")
    rep = _j(args.fit, "best_report.json")
    ev = _j(args.evaluate, "evaluate.json")
    md = _j(args.md_check, "md_check.json")
    sanity = ffield_info.sanity(ff)
    qfix = {"qeq": "fix qeq all qeq/reaxff 1 0.0 10.0 1.0e-8 reaxff maxiter 400",
            "acks2": "fix qeq all acks2/reaxff 1 0.0 10.0 1.0e-8 reaxff maxiter 400",
            "none": "# no charge equilibration (fit used fixed zero charges)"}[args.charge_method]
    L = [f"# {name}", "", f"Elements: {', '.join(els)}", "",
         "Built with Agentic-ReaxFF (`reaxff-agent`).", "", "## Use in LAMMPS", "", "```",
         "units real", "atom_style charge",
         "pair_style reaxff NULL checkqeq yes safezone 3.0 mincap 150",
         f"pair_coeff * * {name} {' '.join(els)}   # map to your atom types", qfix,
         "timestep 0.25   # fs; 0.1 fs for high-temperature reactive runs", "```", "",
         f"Bond-order cutoff in the file: {ff.general[29] / 100:g}. The fit used LAMMPS defaults for the control "
         "settings (no control file: nbrhood_cutoff 5.0, hbond_cutoff 7.5, thb_cutoff 0.001); change them and you "
         "change the force field.", "", "Define the charge-equilibration fix after the atoms exist (after read_data / "
         "create_atoms): the solver history of atoms added later is not initialized.", ""]
    if fit:
        L += ["## Fit", "", f"- runs: {fit.get('n_runs')}, evaluations: {fit.get('n_evals_total')}, chosen by "
              f"{fit.get('chosen_by')} loss", f"- start loss {fit.get('start_loss'):.4g} -> {fit.get('best', {}).get('loss'):.4g}"
              if fit.get("best") else "- (no best recorded)"]
        if fit.get("at_bounds"):
            L.append(f"- parameters at their bounds: {', '.join(fit['at_bounds'][:10])}")
        L.append("")
    if rep:
        L += ["## Accuracy", "", "| set | kind | n items | RMSE | MAE | max |", "|---|---|---|---|---|---|"]
        for setname in ("train", "validation"):
            if setname in rep:
                for k, v in rep[setname]["kinds"].items():
                    if "rmse" in v:
                        L.append(f"| {setname} | {k} ({v.get('units', '')}) | {v['n_items']} | {v['rmse']:.3f} | {v['mae']:.3f} | {v['max_abs']:.3f} |")
        L.append("")
    if ev:
        L += ["## Comparison", "", "```json", json.dumps([{k: v for k, v in e.items() if k in ("label",)} | {
            s: {"loss": e[s]["loss"]} for s in ev.get("sets", []) if s in e} for e in ev["ffields"]], indent=1), "```", ""]
    if md:
        L += ["## MD validation", "", f"Verdict: **{md.get('verdict')}**", ""]
        for r in md.get("runs", [])[:30]:
            L.append(f"- {r['system']} @ {r['T']:.0f} K: {'stable' if r['stable'] else 'UNSTABLE'}"
                     + (f" ({'; '.join(r['flags'])})" if r.get("flags") else "")
                     + (f", NVE ratio {r['nve_ratio']:.3f}" if r.get("nve_ratio") is not None else ""))
        L.append("")
    if sanity:
        L += ["## Checks", ""] + [f"- {s['check']}: {s.get('element', '')} {s['detail']}" for s in sanity] + [""]
    L += ["## Cite", "", "- the force field(s) this one started from:"] + [f"  - {c}" for c in (args.cite or [])] + [
        "- ReaxFF: van Duin, Dasgupta, Lorant, Goddard, J. Phys. Chem. A 105, 9396 (2001), 10.1021/jp004368u",
        "- LAMMPS ReaxFF: Aktulga, Fogarty, Pandit, Grama, Parallel Comput. 38, 245 (2012), 10.1016/j.parco.2011.08.005",
        "- CMA-ES for ReaxFF (if fitted with it): Shchygol et al., JCTC 15, 6799 (2019), 10.1021/acs.jctc.9b00769", ""]
    (out / "FFIELD_CARD.md").write_text("\n".join(L))
    meta = {"ffield": str(dest), "elements": els, "start_ffield": args.start_ffield, "trainset": args.trainset,
            "fit": args.fit, "evaluate": args.evaluate, "md_check": args.md_check, "charge_method": args.charge_method}
    (out / "provenance.json").write_text(json.dumps(meta, indent=1))
    return {"deployed": str(out), "ffield": str(dest), "card": str(out / "FFIELD_CARD.md"),
            "md_verdict": md.get("verdict") if md else None, "sanity_issues": len(sanity)}
