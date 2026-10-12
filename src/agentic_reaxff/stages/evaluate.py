"""Evaluate force fields on training / validation / test sets, side by side.

For each ffield and each set: loss, per-kind and per-group RMSE/MAE/max
error in natural units, failed structures; per-item predictions
(items.csv); scan curves (curves.json: reference vs every ffield along each
bond/angle/torsion/dimer/EOS group); optional parity and scan plots.

`--pair-check` adds a fine diatomic scan of every element pair with every
ffield (no reference data needed), reporting for each pair:

  * short-range repulsion: does the energy keep rising as the atoms are
    pushed together down to `pair_rmin` (and how high is it there)? A force
    field whose diatomic curve turns over at short range lets atoms collapse
    at high temperature or in collisions (too-soft short-range repulsion is documented for ReaxFF in
    energetic deposition, Caro 2020 [Ca20]); inner-wall parameters
    (rcore2/ecore2/acore2) or vdW shielding fix it;
  * discontinuities: jumps in energy or force along the curve, which come
    from ReaxFF's hard bond-order cutoffs and cause NVE energy drift and
    noisy geometry optimizations (Furman & Wales 2019 [FW19]).

Compare a fitted ffield with its starting point and, where one exists, the
published force field for the same chemistry: improvement must show on the
validation and test sets, not only on training.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..engine.evaluator import EvalSettings
from ..engine.objective import Objective
from ..io import ffield as F
from ..io import params as P
from ..io import trainset as T

NAME = "evaluate"
SUMMARY = "Errors of one or more ffields on training/validation/test sets, scan curves, parity plots, diatomic short-range and smoothness checks."
SCHEMA = {
    "type": "object",
    "required": ["ffields"],
    "properties": {
        "ffields": {"type": "array", "items": {"type": "string"}, "description": "First = the candidate; others = baselines."},
        "labels": {"type": ["array", "null"], "items": {"type": "string"}},
        "trainset": {"type": ["string", "null"]},
        "validset": {"type": ["string", "null"]},
        "testsets": {"type": ["array", "null"], "items": {"type": "string"}},
        "pair_check": {"type": "boolean", "default": False},
        "pair_elements": {"type": ["array", "null"], "items": {"type": "string"}},
        "pair_rmin": {"type": "number", "default": 0.5},
        "pair_rmax": {"type": "number", "default": 5.0},
        "plots": {"type": "boolean", "default": False},
        "workers": {"type": ["integer", "null"]},
        "charge_method": {"type": ["string", "null"]},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--ffields", action="append", default=None)
    parser.add_argument("--labels", type=lambda s: s.split(","), default=None)
    parser.add_argument("--trainset", default=None)
    parser.add_argument("--validset", default=None)
    parser.add_argument("--testsets", action="append", default=None)
    parser.add_argument("--pair-check", dest="pair_check", action="store_true", default=False)
    parser.add_argument("--pair-elements", dest="pair_elements", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--pair-rmin", dest="pair_rmin", type=float, default=0.5)
    parser.add_argument("--pair-rmax", dest="pair_rmax", type=float, default=5.0)
    parser.add_argument("--plots", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--charge-method", dest="charge_method", default=None)


def _dummy_spec(ffield_path):
    ff = F.read(ffield_path)
    pid = ff.param_ids(["bond"])[0] if ff.bonds else ff.param_ids(["atom"])[0]
    v = ff.get(pid)
    return P.ParamSpec(ffield=str(Path(ffield_path).resolve()), params=[P.Param(pid, v, v - 1, v + 1)], precision=6)


def evaluate_set(ffield_path, ts, workers, settings) -> tuple:
    spec = _dummy_spec(ffield_path)
    with Objective(ts, spec, settings=settings, n_workers=workers) as obj:
        # precision 6 + the file's own value: the file is evaluated as written
        ev = obj.evaluate_x([spec.x0], details=True)[0]
    summ = T.summarize_errors(ts, ev.preds)
    summ["n_failed_structures"] = ev.n_failed
    summ["errors"] = ev.errors
    return summ, ev.preds


def scan_curves(ts, preds_by_ff: dict) -> dict:
    """group -> {x, ref, <ffield label>: pred} for energy items that carry a scan value."""
    curves = {}
    for it in ts.items:
        if it.kind != "energy" or len(it.terms) != 2:
            continue
        s = ts.structures[it.terms[0][0]]
        x = s.info.get("scan_value")
        if x is None:
            continue
        c = curves.setdefault(it.group, {"kind": s.info.get("kind"), "x": [], "ref": [], "items": []})
        c["x"].append(float(x))
        c["ref"].append(float(T.reference_values(ts, it)))
        c["items"].append(it.id)
    for g, c in curves.items():
        order = np.argsort(c["x"])
        c["x"] = [c["x"][i] for i in order]
        c["ref"] = [c["ref"][i] for i in order]
        ids = [c["items"][i] for i in order]
        for lab, preds in preds_by_ff.items():
            c[lab] = [None if preds.get(i) is None else float(preds[i]) for i in ids]
        del c["items"]
    return curves


def pair_check(ffield_path, elements, rmin, rmax, settings) -> dict:
    from ase import Atoms

    from ..engine.evaluator import Evaluator
    from ..io.structures import from_atoms

    ff = F.read(ffield_path)
    masses = {a.symbol: a.values[2] for a in ff.atoms}
    ev = Evaluator(settings)
    rs = np.round(np.arange(rmin, rmax + 1e-9, 0.01), 4)
    out = {}
    for x in range(len(elements)):
        for y in range(x, len(elements)):
            a, b = elements[x], elements[y]
            E, Fz = [], []
            for r in rs:
                s = from_atoms(Atoms([a, b], positions=[[0, 0, 0], [0, 0, r]]))
                res = ev.evaluate(ffield_path, s, masses=masses)
                E.append(res.energy if res.ok else np.nan)
                Fz.append(res.forces[1, 2] if res.ok else np.nan)
            E, Fz = np.array(E), np.array(Fz)
            ok = np.isfinite(E)
            info = {"n_failed": int((~ok).sum())}
            if ok.sum() > 10:
                Ein = E[ok] - E[ok][-1]
                r_ok = rs[ok]
                k = int(np.argmin(Ein))
                info.update({"r_min": float(r_ok[k]), "well_depth": float(-Ein[k]), "E_at_rmin_scan": float(Ein[0])})
                inner = Ein[: k + 1]
                # the wall must keep rising all the way in: a turnover (energy falling again as
                # atoms are pushed closer) is what lets them collapse. E at the inner limit is
                # reported, not judged: for H-H it is legitimately still below the dissociated
                # limit at 0.5 Å.
                info["repulsive_to_rmin"] = bool(np.all(np.diff(inner) <= 1e-6))
                # numerical force from energy vs analytic: their disagreement localizes jumps
                dE = np.diff(E[ok]) / np.diff(r_ok)
                # a discontinuity is an isolated spike in |ΔF| between neighbouring grid points
                # (a steep but smooth wall has large ΔF everywhere, not a spike)
                d = np.abs(np.diff(Fz[ok]))
                idx = [i for i in range(1, len(d) - 1) if d[i] > 10 * max(d[i - 1], d[i + 1]) + 0.5]
                info["force_jumps"] = [{"r": float(r_ok[i]), "dF": float(d[i])} for i in idx[:5]]
                info["energy_force_mismatch_max"] = float(np.max(np.abs(-dE - 0.5 * (Fz[ok][1:] + Fz[ok][:-1])))) if dE.size else 0.0
            out[f"{a}-{b}"] = info
    ev.close()
    return out


def _plots(out_dir, curves, results, labels):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    groups = list(curves)[:24]
    if groups:
        ncol = 4
        nrow = int(np.ceil(len(groups) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
        for ax, g in zip(axs.ravel(), groups):
            c = curves[g]
            ax.plot(c["x"], c["ref"], "ko", ms=3, label="reference")
            for lab in labels:
                y = [np.nan if v is None else v for v in c[lab]]
                ax.plot(c["x"], y, "-", label=lab)
            ax.set_title(g[:40], fontsize=8)
            ax.set_ylabel("ΔE (kcal/mol)", fontsize=8)
        for ax in axs.ravel()[len(groups):]:
            ax.axis("off")
        axs.ravel()[0].legend(fontsize=7)
        fig.tight_layout()
        p = out_dir / "scans.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        files.append(str(p))
    return files


def run(args) -> dict:
    if not args.ffields:
        raise ValueError("--ffields is required")
    labels = args.labels or [Path(f).parent.name + "/" + Path(f).name if Path(f).parent.name else Path(f).name
                             for f in args.ffields]
    if len(set(labels)) != len(labels):
        labels = [f"ff{k}:{l}" for k, l in enumerate(labels)]
    out_dir = Path(args.output_dir or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    sets = {}
    if args.trainset:
        sets["train"] = T.load(args.trainset)
    if args.validset:
        sets["validation"] = T.load(args.validset)
    for k, t in enumerate(args.testsets or []):
        sets[f"test{k}" if len(args.testsets) > 1 else "test"] = T.load(t)
    result = {"ffields": [], "sets": list(sets)}
    rows = []
    all_preds = {}
    for f, lab in zip(args.ffields, labels):
        entry = {"ffield": f, "label": lab}
        for name, ts in sets.items():
            settings = EvalSettings.from_dict(ts.evaluator)
            if args.charge_method:
                settings.charge_method = args.charge_method
            summ, preds = evaluate_set(f, ts, args.workers, settings)
            entry[name] = summ
            all_preds.setdefault(name, {})[lab] = preds
            for it in ts.items:
                ref = T.reference_values(ts, it)
                p = preds.get(it.id)
                err = T.natural_errors(it, p, ref)
                rows.append({"ffield": lab, "set": name, "id": it.id, "kind": it.kind, "group": it.group,
                             "ref": float(np.mean(ref)) if np.size(ref) == 1 else "array",
                             "pred": (float(p) if np.size(p) == 1 else "array") if p is not None else "FAILED",
                             "abs_err": float(np.sqrt(np.mean(err ** 2))) if err is not None else "",
                             "sigma": it.sigma})
        if args.pair_check:
            els = args.pair_elements or (sets["train"].elements() if "train" in sets else F.read(f).elements)
            settings = EvalSettings.from_dict(sets["train"].evaluator if "train" in sets else {})
            entry["pair_check"] = pair_check(f, els, args.pair_rmin, args.pair_rmax, settings)
        result["ffields"].append(entry)
    with open(out_dir / "items.csv", "w", newline="") as fh:
        if rows:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    curves = {name: scan_curves(sets[name], all_preds[name]) for name in sets}
    (out_dir / "curves.json").write_text(json.dumps(curves, indent=1))
    (out_dir / "evaluate.json").write_text(json.dumps(result, indent=1, default=str))
    summary = []
    for e in result["ffields"]:
        line = {"label": e["label"]}
        for name in sets:
            line[name] = {"loss": round(e[name]["loss"], 4), "n_failed_structures": e[name]["n_failed_structures"],
                          **{k: round(v["rmse"], 4) for k, v in e[name]["kinds"].items() if "rmse" in v}}
        if "pair_check" in e:
            bad = [p for p, i in e["pair_check"].items() if not i.get("repulsive_to_rmin", True) or i.get("force_jumps")]
            line["pair_check_flags"] = bad
        summary.append(line)
    out = {"evaluate": str(out_dir / "evaluate.json"), "items_csv": str(out_dir / "items.csv"),
           "curves": str(out_dir / "curves.json"), "summary": summary}
    if args.plots and "train" in curves:
        try:
            out["plots"] = _plots(out_dir, curves["train"], result, labels)
        except ImportError:
            out["plots"] = "matplotlib not installed (pip install -e '.[plots]')"
    return out
