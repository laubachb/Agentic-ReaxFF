"""Pick new training frames from MD: where the force field is uncertain or
extrapolating.

One refinement round is: md-check (trajectories with the current fit) ->
refine-select -> label -> trainset-build --merge-into -> fit -> md-check.
It is the standard way to make a fit hold up in the dynamics it will be
used for (training on MD-sampled DFT frames: De Angelis 2024 [DA24],
Dumortier 2024 [Du24]; the same loop with DFT relabeling of ReaxFF-MD
frames seeds Yoo et al.'s CHNO network [Yo21]). No published ReaxFF work
runs it with an uncertainty-driven selection, so the selection rules here
are this toolkit's, built on two literature observations:

  committee  independent fits reach near-equal loss with different
             parameters [L13, Sh19]; where they disagree the data does not
             constrain the force field, and Shchygol et al. recommend adding
             training data exactly there [Sh19]. Disagreement = spread of
             energy per atom and of forces across the committee (the
             `committee` list in fit.json).
  contacts   frames with a pair distance shorter than anything the training
             set contains are extrapolations in the region where ReaxFF most
             often fails (collapse, missing repulsion).

Frames are scored, then picked greedily with diversity (no two frames
closer than `min_gap` in the same trajectory) up to `n_select`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..engine.evaluator import EvalSettings
from ..engine.pool import EvalPool
from ..io import ffield as F
from ..io import structures as SIO
from ..io import trainset as T

NAME = "refine-select"
SUMMARY = "Select MD frames for DFT labeling by committee disagreement and short-contact extrapolation (one refinement round)."
SCHEMA = {
    "type": "object",
    "required": ["committee", "trajectories"],
    "properties": {
        "committee": {"type": "array", "items": {"type": "string"}, "description": "ffields of independent fits (fit.json 'committee'), >= 2."},
        "trajectories": {"type": "array", "items": {"type": "string"}, "description": "extxyz trajectories (md-check traj.extxyz) or frame sets."},
        "trainset": {"type": ["string", "null"], "description": "Current training set: its shortest pair distances define 'extrapolating'."},
        "n_select": {"type": "integer", "default": 40},
        "stride": {"type": "integer", "default": 1},
        "min_gap": {"type": "integer", "default": 3, "description": "Minimum frame separation within one trajectory."},
        "contact_weight": {"type": "number", "default": 1.0},
        "round": {"type": "integer", "default": 1},
        "workers": {"type": ["integer", "null"]},
        "charge_method": {"type": "string", "default": "qeq"},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--committee", action="append", default=None)
    parser.add_argument("--trajectories", action="append", default=None)
    parser.add_argument("--trainset", default=None)
    parser.add_argument("--n-select", dest="n_select", type=int, default=40)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--min-gap", dest="min_gap", type=int, default=3)
    parser.add_argument("--contact-weight", dest="contact_weight", type=float, default=1.0)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--charge-method", dest="charge_method", default="qeq")


def _min_pair_distances(s) -> dict:
    from ase import Atoms

    a = Atoms(s.symbols, positions=s.positions, cell=s.cell, pbc=s.pbc)
    if len(a) < 2:
        return {}
    D = a.get_all_distances(mic=any(a.pbc))
    np.fill_diagonal(D, np.inf)
    arr = np.array(s.symbols)
    out = {}
    for x in set(s.symbols):
        for y in set(s.symbols):
            if x <= y:
                m = D[np.ix_(arr == x, arr == y)]
                if m.size and np.isfinite(m).any():
                    out[f"{x}-{y}"] = float(m.min())
    return out


def run(args) -> dict:
    if not args.committee or len(args.committee) < 2:
        raise ValueError("--committee needs >= 2 ffields (independent fits, e.g. fit.json 'committee')")
    frames = []
    for k, tpath in enumerate(args.trajectories or []):
        for j, s in enumerate(SIO.read(tpath)[:: max(1, args.stride)]):
            s.name = f"r{args.round:02d}_t{k:02d}_f{j:05d}"
            s.mode = "single"
            s.restraints, s.fixed = [], []
            s.group = f"refine_r{args.round:02d}"
            s.info = {"kind": "md", "source": str(tpath), "traj_index": k, "frame_index": j}
            s.energy = s.forces = s.charges = s.stress = None
            frames.append(s)
    if not frames:
        raise ValueError("no frames in --trajectories")
    ref = F.read(args.committee[0])
    masses = {a.symbol: a.values[2] for a in ref.atoms}
    needs = {s.name: {"forces": True, "geometry": False} for s in frames}
    with EvalPool(frames, EvalSettings(charge_method=args.charge_method), needs, masses, n_workers=args.workers) as pool:
        res = pool.evaluate([Path(f).read_text() for f in args.committee])
    e_std, f_std, failed = [], [], []
    for s in frames:
        rs = [r[s.name] for r in res]
        if not all(r.ok for r in rs):
            failed.append(s.name)
            e_std.append(np.inf)
            f_std.append(np.inf)
            continue
        e = np.array([r.energy / s.natoms for r in rs])
        f = np.stack([r.forces for r in rs])
        e_std.append(float(e.std()))
        f_std.append(float(np.sqrt(np.mean(f.var(axis=0)))))
    e_std, f_std = np.array(e_std), np.array(f_std)
    # extrapolation in contact distance relative to the training set
    train_min = {}
    if args.trainset:
        ts = T.load(args.trainset)
        for s in ts.used_structures():
            for k, v in _min_pair_distances(s).items():
                train_min[k] = min(train_min.get(k, np.inf), v)
    contact = np.zeros(len(frames))
    contact_pairs = []
    for i, s in enumerate(frames):
        worst = 0.0
        pairs = []
        for k, v in _min_pair_distances(s).items():
            if k in train_min and v < train_min[k]:
                worst = max(worst, (train_min[k] - v) / train_min[k])
                pairs.append(k)
        contact[i] = worst
        contact_pairs.append(pairs)

    def _norm(x):
        x = np.where(np.isfinite(x), x, np.nan)
        hi = np.nanpercentile(x, 95) if np.any(np.isfinite(x)) else 1.0
        return np.nan_to_num(x / (hi if hi > 0 else 1.0), nan=10.0)

    score = _norm(e_std) + _norm(f_std) + args.contact_weight * 10.0 * contact
    order = np.argsort(-score)
    chosen, taken = [], {}
    for i in order:
        s = frames[i]
        k, j = s.info["traj_index"], s.info["frame_index"]
        if any(abs(j - jj) < args.min_gap for jj in taken.get(k, [])):
            continue
        chosen.append(i)
        taken.setdefault(k, []).append(j)
        if len(chosen) >= args.n_select:
            break
    sel = [frames[i] for i in chosen]
    for i, s in zip(chosen, sel):
        s.info.update({"committee_e_std": float(e_std[i]) if np.isfinite(e_std[i]) else -1.0,
                       "committee_f_std": float(f_std[i]) if np.isfinite(f_std[i]) else -1.0,
                       "contact_extrapolation": float(contact[i])})
    out = Path(args.output_dir or ".")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "selected.extxyz"
    SIO.write(path, sel)
    fin = np.isfinite(e_std)
    report = {"selected": str(path), "n_candidates": len(frames), "n_selected": len(sel),
              "n_committee_failures": len(failed),
              "committee_e_std_per_atom": {"median": float(np.median(e_std[fin])) if fin.any() else None,
                                           "max": float(e_std[fin].max()) if fin.any() else None},
              "committee_f_std": {"median": float(np.median(f_std[fin])) if fin.any() else None,
                                  "max": float(f_std[fin].max()) if fin.any() else None},
              "n_contact_extrapolating": int((contact > 0).sum()),
              "contact_pairs": sorted({p for ps in contact_pairs for p in ps}),
              "next": f"label --structures {path} --backend qe ... then trainset-build --merge-into <trainset.yaml>"}
    (out / "refine_select.json").write_text(json.dumps(report, indent=1))
    return report
