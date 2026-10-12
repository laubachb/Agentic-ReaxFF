"""MD validation of candidate force fields: do they survive dynamics?

A low training loss says nothing about stability: well-fitting ReaxFF
parameter sets can fail to keep a crystal stable at 2000 K (Nakata & Bai
[NB19]), and overfitting shows up as "low training error but unphysical MD"
[Sh19]. For every (ffield, system, temperature) this runs, in LAMMPS,

    velocity -> NVT (Nose-Hoover) for `nvt_ps` -> NVE for `nve_ps`

and reports:

  completed        no LAMMPS error, NaN or lost atoms
  nve_drift        total-energy drift (kcal/mol/ps per atom) in the NVE part
  nve_ratio        sqrt(<dE_tot^2>/<dE_kin^2>): Furman & Wales call < 10%
                   acceptable; ReaxFF's hard bond-order cutoffs push it up
                   (69-152% for published force fields in their tests) [FW19]
  min_distance     closest approach per element pair over the trajectory;
                   flagged below `contact_factor` x (sum of covalent radii):
                   atoms collapsing onto each other
  max_abs_charge   largest |q|; flagged above `max_charge` (charge runaway /
                   polarization catastrophe)
  species          molecules in the final frame (covalent connectivity), to
                   compare with what the chemistry should do

Systems: any periodic or molecular structures (e.g. `data-generate
--boxes`). Trajectories are written as one extxyz per run (frames every
`dump_every` steps) for `refine-select`. Many runs execute in parallel, one
LAMMPS process each; `--machine` runs the whole set on a compute node.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np

from ..io import ffield as F
from ..io import structures as SIO
from . import _remote

NAME = "md-check"
SUMMARY = "MD stability of ffields: NVT->NVE runs per system and temperature; drift, Furman-Wales ratio, close contacts, charges, species."
SUPPORTS_DRY_RUN = True
SCHEMA = {
    "type": "object",
    "required": ["ffields", "systems"],
    "properties": {
        "ffields": {"type": "array", "items": {"type": "string"}},
        "labels": {"type": ["array", "null"], "items": {"type": "string"}},
        "systems": {"type": "string", "description": "extxyz with the systems to simulate (all frames, or --select)."},
        "select": {"type": ["array", "null"], "items": {"type": "string"}},
        "temperatures": {"type": "array", "items": {"type": "number"}, "default": [300, 1500]},
        "timestep_fs": {"type": "number", "default": 0.25},
        "nvt_ps": {"type": "number", "default": 5.0},
        "nve_ps": {"type": "number", "default": 2.0},
        "thermo_every": {"type": "integer", "default": 20},
        "dump_every": {"type": "integer", "default": 200},
        "seed": {"type": "integer", "default": 12345},
        "charge_method": {"type": "string", "default": "qeq"},
        "contact_factor": {"type": "number", "default": 0.5},
        "max_charge": {"type": "number", "default": 2.5},
        "max_ratio": {"type": "number", "default": 0.1, "description": "Furman-Wales energy-fluctuation ratio threshold."},
        "workers": {"type": ["integer", "null"]},
        **_remote.MACHINE_SCHEMA,
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--ffields", action="append", default=None)
    parser.add_argument("--labels", type=lambda s: s.split(","), default=None)
    parser.add_argument("--systems", default=None)
    parser.add_argument("--select", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--temperatures", type=lambda s: [float(x) for x in s.split(",")], default=[300.0, 1500.0])
    parser.add_argument("--timestep-fs", dest="timestep_fs", type=float, default=0.25)
    parser.add_argument("--nvt-ps", dest="nvt_ps", type=float, default=5.0)
    parser.add_argument("--nve-ps", dest="nve_ps", type=float, default=2.0)
    parser.add_argument("--thermo-every", dest="thermo_every", type=int, default=20)
    parser.add_argument("--dump-every", dest="dump_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--charge-method", dest="charge_method", default="qeq")
    parser.add_argument("--contact-factor", dest="contact_factor", type=float, default=0.5)
    parser.add_argument("--max-charge", dest="max_charge", type=float, default=2.5)
    parser.add_argument("--max-ratio", dest="max_ratio", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=None)
    _remote.add_machine_arguments(parser, walltime=1.0)


def _parse_thermo(log_text: str) -> list:
    """Rows of every thermo block as dicts."""
    rows, header = [], None
    for line in log_text.splitlines():
        tok = line.split()
        if tok[:1] == ["Step"]:
            header = tok
            continue
        if header and len(tok) == len(header):
            try:
                rows.append(dict(zip(header, map(float, tok))))
            except ValueError:
                header = None
        elif header and tok[:1] == ["Loop"]:
            header = None
    return rows


def _species(symbols, positions, cell, pbc):
    from ase import Atoms

    from ..generate.builders import _cov_radius

    a = Atoms(symbols, positions=positions, cell=cell, pbc=pbc)
    n = len(a)
    D = a.get_all_distances(mic=any(pbc))
    rc = np.array([_cov_radius(s) for s in symbols])
    adj = D < 1.2 * (rc[:, None] + rc[None, :])
    np.fill_diagonal(adj, False)
    seen = np.zeros(n, bool)
    counts = {}
    for i in range(n):
        if seen[i]:
            continue
        stack, comp = [i], []
        seen[i] = True
        while stack:
            j = stack.pop()
            comp.append(j)
            for k in np.where(adj[j] & ~seen)[0]:
                seen[k] = True
                stack.append(k)
        f = Atoms([symbols[c] for c in comp]).get_chemical_formula(mode="hill")
        counts[f] = counts.get(f, 0) + 1
    return dict(sorted(counts.items(), key=lambda t: -t[1]))


def _task(t):
    """One MD run; returns its result dict (never raises)."""
    os.environ["OMP_NUM_THREADS"] = "1"
    from ..engine.evaluator import EvalSettings, Evaluator
    from ..generate.builders import _cov_radius

    d = Path(t["dir"])
    d.mkdir(parents=True, exist_ok=True)
    res = {"ffield": t["label"], "system": t["system"], "T": t["T"], "dir": str(d)}
    s = t["structure"]
    t0 = time.time()
    ev = Evaluator(EvalSettings(charge_method=t["charge_method"]))
    try:
        ctx = ev.setup(t["ffield"], s, masses=t["masses"])
        lmp, n = ctx["lmp"], ctx["n"]
        res["n_atoms"] = n
        dt = t["timestep_fs"]
        n_nvt = int(round(t["nvt_ps"] * 1000 / dt))
        n_nve = int(round(t["nve_ps"] * 1000 / dt))
        lmp.commands_string("\n".join([
            f"log {d / 'log.lammps'}",
            f"velocity all create {t['T']} {t['seed']} mom yes rot yes dist gaussian",
            f"timestep {dt}",
            "thermo_style custom step temp pe ke etotal press",
            f"thermo {t['thermo_every']}",
            "thermo_modify flush yes lost error",
            f"dump trj all custom {t['dump_every']} {d / 'dump.lammpstrj'} id type x y z q",
            "dump_modify trj sort id",
            f"fix tstat all nvt temp {t['T']} {t['T']} {100 * dt}",
            f"run {n_nvt}",
            "unfix tstat",
            "fix integ all nve",
            f"run {n_nve}",
        ]))
        res["completed"] = True
    except Exception as exc:  # noqa: BLE001
        msg = [ln for ln in str(exc).splitlines() if "ERROR" in ln]
        res["completed"] = False
        res["error"] = (msg[0] if msg else str(exc))[:300]
    finally:
        try:
            ev.close()
        except Exception:  # noqa: BLE001
            pass
    res["wall_s"] = round(time.time() - t0, 1)
    log = d / "log.lammps"
    rows = _parse_thermo(log.read_text(errors="replace")) if log.is_file() else []
    if rows:
        n = res.get("n_atoms", s.natoms)
        steps = np.array([r["Step"] for r in rows])
        n_nvt_steps = int(round(t["nvt_ps"] * 1000 / t["timestep_fs"]))
        nve = [r for r in rows if r["Step"] >= n_nvt_steps]
        nvt = [r for r in rows if r["Step"] < n_nvt_steps]
        if nvt:
            res["T_mean_nvt"] = float(np.mean([r["Temp"] for r in nvt[len(nvt) // 2:]]))
        if len(nve) > 5:
            tt = np.array([r["Step"] for r in nve]) * t["timestep_fs"] / 1000.0
            et = np.array([r["TotEng"] for r in nve])
            ke = np.array([r["KinEng"] for r in nve])
            slope = np.polyfit(tt, et, 1)[0]
            res["nve_drift_per_atom"] = float(slope / n)
            res["nve_ratio"] = float(np.std(et) / max(np.std(ke), 1e-12))
        res["finite"] = bool(np.all(np.isfinite([r["PotEng"] for r in rows])))
        res["n_thermo"] = len(steps)
    # trajectory -> extxyz, analysis
    dump = d / "dump.lammpstrj"
    if dump.is_file():
        try:
            from ase.io import read, write

            frames = read(str(dump), format="lammps-dump-text", index=":")
            types = ctx["species"] if res.get("n_atoms") else list(dict.fromkeys(s.symbols))
            sym_of = {i + 1: el for i, el in enumerate(types)}
            mind, qmax = {}, 0.0
            for fr in frames:
                syms = [sym_of[int(x)] for x in fr.get_atomic_numbers()]
                fr.set_chemical_symbols(syms)
                q = fr.arrays.get("q", fr.arrays.get("initial_charges"))  # ASE's dump reader names it initial_charges
                if q is not None and len(q):
                    qmax = max(qmax, float(np.max(np.abs(q))))
                D = fr.get_all_distances(mic=any(fr.pbc))
                np.fill_diagonal(D, np.inf)
                arr = np.array(syms)
                for a in set(syms):
                    for b in set(syms):
                        if a > b:
                            continue
                        m = D[np.ix_(arr == a, arr == b)]
                        if m.size:
                            key = f"{a}-{b}"
                            mind[key] = min(mind.get(key, np.inf), float(m.min()))
            res["min_distance"] = mind
            res["close_contacts"] = {k: v for k, v in mind.items()
                                     if v < t["contact_factor"] * (_cov_radius(k.split("-")[0]) + _cov_radius(k.split("-")[1]))}
            res["max_abs_charge"] = qmax
            last = frames[-1]
            res["species_final"] = dict(list(_species(last.get_chemical_symbols(), last.positions, last.cell,
                                                      last.pbc).items())[:12])
            for fr in frames:
                fr.calc = None
            write(str(d / "traj.extxyz"), frames, format="extxyz")
            res["traj"] = str(d / "traj.extxyz")
            res["n_frames"] = len(frames)
            dump.unlink()
        except Exception as exc:  # noqa: BLE001
            res["traj_error"] = str(exc)[:200]
    # verdict
    flags = []
    if not res.get("completed"):
        flags.append("crashed")
    if res.get("finite") is False:
        flags.append("non-finite energy")
    if res.get("close_contacts"):
        flags.append("close contacts " + ",".join(res["close_contacts"]))
    if res.get("max_abs_charge", 0) > t["max_charge"]:
        flags.append(f"|q| up to {res['max_abs_charge']:.2f}")
    if res.get("nve_ratio") is not None and res["nve_ratio"] > t["max_ratio"]:
        flags.append(f"NVE fluctuation ratio {res['nve_ratio']:.2f} > {t['max_ratio']}")
    res["flags"] = flags
    res["stable"] = bool(res.get("completed") and res.get("finite", True) and not res.get("close_contacts")
                         and res.get("max_abs_charge", 0) <= t["max_charge"])
    (d / "result.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def run(args) -> dict:
    if getattr(args, "machine", None):
        return _remote.submit(args, NAME)
    if getattr(args, "dry_run", False):
        raise ValueError("--dry-run applies with --machine")
    if not args.ffields or not args.systems:
        raise ValueError("--ffields and --systems are required")
    out = Path(args.output_dir or ".")
    out.mkdir(parents=True, exist_ok=True)
    systems = SIO.read(args.systems)
    if args.select:
        systems = [s for s in systems if s.name in set(args.select)]
    if not systems:
        raise ValueError("no systems selected")
    labels = args.labels or [f"ff{k}" for k in range(len(args.ffields))]
    tasks = []
    for f, lab in zip(args.ffields, labels):
        ff = F.read(f)
        masses = {a.symbol: a.values[2] for a in ff.atoms}
        for s in systems:
            missing = set(s.symbols) - set(ff.elements)
            if missing:
                raise ValueError(f"{f} lacks elements {sorted(missing)} needed by system {s.name}")
            for T in args.temperatures:
                tasks.append({"ffield": str(Path(f).resolve()), "label": lab, "system": s.name, "structure": s, "T": float(T),
                              "masses": masses, "dir": str(out / "runs" / re.sub(r"[^\w.-]", "_", f"{lab}__{s.name}__{int(T)}K")),
                              "timestep_fs": args.timestep_fs, "nvt_ps": args.nvt_ps, "nve_ps": args.nve_ps,
                              "thermo_every": args.thermo_every, "dump_every": args.dump_every, "seed": args.seed,
                              "charge_method": args.charge_method, "contact_factor": args.contact_factor,
                              "max_charge": args.max_charge, "max_ratio": args.max_ratio})
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    from ..engine.pool import default_workers

    nw = max(1, min(args.workers or default_workers(), len(tasks)))
    if nw == 1:
        results = [_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(nw, mp_context=mp.get_context("spawn")) as ex:
            results = list(ex.map(_task, tasks, chunksize=1))
    per_ff = {}
    for r in results:
        p = per_ff.setdefault(r["ffield"], {"runs": 0, "stable": 0, "flags": []})
        p["runs"] += 1
        p["stable"] += int(r["stable"])
        p["flags"] += [f"{r['system']}@{int(r['T'])}K: {fl}" for fl in r["flags"]]
    for p in per_ff.values():
        p["verdict"] = "stable" if p["stable"] == p["runs"] else "unstable"
    verdict = "stable" if all(p["verdict"] == "stable" for p in per_ff.values()) else "unstable"
    summary = {"verdict": verdict, "per_ffield": per_ff,
               "runs": [{k: r.get(k) for k in ("ffield", "system", "T", "stable", "flags", "nve_ratio", "nve_drift_per_atom",
                                                "max_abs_charge", "species_final", "traj", "wall_s")} for r in results]}
    (out / "md_check.json").write_text(json.dumps(summary, indent=1, default=str))
    return {"md_check": str(out / "md_check.json"), "verdict": verdict, "per_ffield": per_ff,
            "trajectories": [r.get("traj") for r in results if r.get("traj")][:20]}
