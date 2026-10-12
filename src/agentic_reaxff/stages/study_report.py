"""Draft REPORT.md for a study from its registered artifacts and journal.

Facts (data counts, level of theory, parameters fitted, losses, errors per
kind, validation and MD verdicts, CPU-hours) are filled in from the
artifacts; the reasoning is left as `<!-- NARRATIVE: ... -->` placeholders
for the reaxff-report-writer agent, which writes them from JOURNAL.md.
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import journal
from . import study as study_stage

NAME = "study-report"
SUMMARY = "Draft REPORT.md for a study from its artifacts and journal (facts filled in, narrative placeholders left)."
USES_OUTPUT_DIR = False
SCHEMA = {"type": "object", "required": ["study"], "properties": {"study": {"type": "string"}}}


def add_arguments(parser) -> None:
    parser.add_argument("--study", default=None)


def _read(p, name=None):
    if not p:
        return None
    p = Path(p)
    if p.is_dir() and name:
        p = p / name
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def run(args) -> dict:
    root = study_stage.find_study(args.study)
    if root is None:
        raise ValueError(f"no study at or above {args.study}")
    data = study_stage.load(root)
    arts = {k: study_stage.artifact(root, k) for k in study_stage.KNOWN_ARTIFACTS}
    L = [f"# {data.get('name')}", "", f"**Goal.** {data.get('goal') or ''}", "",
         "<!-- NARRATIVE: summary (what was built, how good it is, what it is not good for) -->", "", "## Starting point", "",
         f"- start force field: `{arts.get('start_ffield')}`", "<!-- NARRATIVE: why this branch / donors (cite) -->", ""]
    ts = None
    if arts.get("trainset"):
        try:
            from ..io import trainset as T

            ts = T.load(arts["trainset"])
        except Exception:  # noqa: BLE001
            ts = None
    L += ["## Reference data and training set", ""]
    if ts:
        L += [f"- {len(ts.items)} items ({ts.counts()}), {len(ts.needed())} structures, elements {ts.elements()}",
              f"- level of theory: `{(ts.meta or {}).get('lot')}`"]
    L += ["<!-- NARRATIVE: what the training set covers and why (cite reaxff-literature) -->", "", "## Fit", ""]
    fit = _read(arts.get("fit"), "fit.json")
    if fit and fit.get("best"):
        b = fit["best"]
        L += [f"- {fit['n_runs']} runs, {fit['n_evals_total']} evaluations; chosen by {fit['chosen_by']} loss",
              f"- loss {fit['start_loss']:.4g} -> {b['loss']:.4g}" + (f" (validation {b['val_loss']:.4g})" if b.get("val_loss") else "")]
        for k, v in (b.get("kinds") or {}).items():
            if v.get("rmse") is not None:
                L.append(f"- {k}: RMSE {v['rmse']:.3f} {v.get('units', '')}")
    L += ["<!-- NARRATIVE: fitting stages, parameters, bounds, what changed (cite) -->", "", "## Validation", ""]
    md = _read(arts.get("md_check"), "md_check.json")
    if md:
        L.append(f"- MD: **{md.get('verdict')}**")
    L += ["<!-- NARRATIVE: validation and comparison with the starting / published force field -->", "",
          "## Limitations", "", "<!-- NARRATIVE: known failures, untested regimes -->", ""]
    ent = journal.read(root)
    L += ["## Record", "", f"{sum(e.get('type') == 'stage' for e in ent)} stage runs and "
          f"{sum(e.get('type') == 'note' for e in ent)} notes in `journal.jsonl` (render: `reaxff-agent study --journal`)."]
    path = root / "REPORT.md"
    path.write_text("\n".join(L) + "\n")
    return {"report": str(path), "placeholders": sum("NARRATIVE" in x for x in L)}
