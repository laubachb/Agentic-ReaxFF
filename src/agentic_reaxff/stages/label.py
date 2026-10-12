"""Label reference structures with energies, forces, stresses and charges.

Backends (the level of theory is recorded on every frame as `lot`, and
`trainset-build` refuses to mix levels — one level of theory per fit):

  qe      Quantum ESPRESSO DFT, the production backend. Prepares one
          directory per frame (pw.in [+ projwfc.in]), submits ONE Slurm job
          (or a chain of 1 h chunks) that runs frames in parallel lanes and
          resumes cleanly, then `--collect` parses the finished frames.
          Restrained scan frames are relaxed under the matching QE
          constraint; stretched bonds get broken-symmetry spin (see
          labeling/qe.py).
  xtb     GFN2-xTB / GFN1-xTB through tblite, locally in parallel: minutes
          instead of node-hours, for designing and debugging a training set
          before paying for DFT. Not a substitute for DFT labels. Frames
          whose SCF fails are retried with 3000 K electronic temperature
          (open-shell / stretched cases); periodic boxes are slow.
  ffield  an existing ReaxFF force field (LAMMPS): for recovery tests
          (perturb a published ffield, refit to its own labels) and for
          fitting a reduced force field to a reference ReaxFF.

Relaxation frames (mode relax / relax_cell) come back with the relaxed
geometry, so geometry items compare relaxed with relaxed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from .. import config, hpc, machines
from ..io import structures as SIO
from ..io.units import KCALMOL_TO_EV
from ..labeling import qe as QE

NAME = "label"
SUMMARY = "Label structures with DFT (Quantum ESPRESSO via Slurm), GFN-xTB (local) or a reference ReaxFF ffield."
SUPPORTS_DRY_RUN = True
SCHEMA = {
    "type": "object",
    "properties": {
        "structures": {"type": ["string", "null"], "description": "Packed extxyz from data-generate."},
        "backend": {"type": "string", "enum": ["qe", "xtb", "ffield"], "default": "qe"},
        "select": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Only these structure names (default all)."},
        "ffield": {"type": ["string", "null"], "description": "ffield backend: the reference force field."},
        "charge_method": {"type": "string", "default": "qeq", "description": "ffield backend: qeq | acks2 | none."},
        "xtb_method": {"type": "string", "default": "GFN2-xTB"},
        "fmax": {"type": "number", "default": 0.01, "description": "xtb relaxations: force tolerance (eV/Å)."},
        "workers": {"type": ["integer", "null"], "description": "Local backends: parallel processes (default: all CPUs available)."},
        "qe": {"type": ["object", "null"], "description": "QE settings (labeling.qe.QESettings): pseudo_dir, pseudopotentials, ecutwfc, ecutrho, vdw_corr, kspacing, vacuum, spin, charges, ..."},
        "machine": {"type": ["string", "null"], "description": "qe backend: machine profile to submit to."},
        "queue": {"type": "string", "default": "debug"},
        "nodes": {"type": "integer", "default": 1},
        "ntasks_per_node": {"type": ["integer", "null"]},
        "lanes": {"type": "integer", "default": 4, "description": "qe: frames run concurrently per job, each on ntasks/lanes ranks."},
        "walltime_hours": {"type": "number", "default": 1.0},
        "chunks": {"type": "integer", "default": 1, "description": "qe: chained jobs (afterany); finished frames are skipped."},
        "pw_flags": {"type": "string", "default": ""},
        "collect": {"type": ["string", "null"], "description": "qe: parse a prepared/finished label directory into labeled.extxyz."},
        "run_local": {"type": "boolean", "default": False, "description": "qe: run frames here with plain pw.x (workstation; no Slurm)."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--structures", default=None)
    parser.add_argument("--backend", default="qe", choices=["qe", "xtb", "ffield"])
    parser.add_argument("--select", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--ffield", default=None)
    parser.add_argument("--charge-method", dest="charge_method", default="qeq")
    parser.add_argument("--xtb-method", dest="xtb_method", default="GFN2-xTB")
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--qe", type=json.loads, default=None)
    parser.add_argument("--machine", default=None)
    parser.add_argument("--queue", default="debug")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--ntasks-per-node", dest="ntasks_per_node", type=int, default=None)
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--walltime-hours", dest="walltime_hours", type=float, default=1.0)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--pw-flags", dest="pw_flags", default="")
    parser.add_argument("--collect", default=None)
    parser.add_argument("--run-local", dest="run_local", action="store_true", default=False)


# ------------------------------------------------------------------ common
def _load(args):
    if not args.structures:
        raise ValueError("--structures is required")
    frames = SIO.read(args.structures)
    if args.select:
        want = set(args.select)
        frames = [s for s in frames if s.name in want]
        missing = want - {s.name for s in frames}
        if missing:
            raise ValueError(f"--select names not found: {sorted(missing)[:5]}")
    if not frames:
        raise ValueError("no structures to label")
    return frames


def _finish(out_dir: Path, labeled: list, failed: dict, lot: str, extra: dict) -> dict:
    for s in labeled:
        s.info["lot"] = lot
    path = out_dir / "labeled.extxyz"
    if labeled:
        SIO.write(path, labeled)
    prov = {"lot": lot, "n_labeled": len(labeled), "n_failed": len(failed), "failed": failed, **extra}
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    return {"labeled": str(path) if labeled else None, "lot": lot, "n_labeled": len(labeled), "n_failed": len(failed),
            "failed": dict(list(failed.items())[:10]), "provenance": str(out_dir / "provenance.json")}


# ------------------------------------------------------------------ ffield
def _label_ffield(args, frames, out_dir):
    from ..engine.evaluator import EvalSettings
    from ..engine.pool import EvalPool
    from ..io import ffield as F

    if not args.ffield:
        raise ValueError("--backend ffield needs --ffield")
    ff = F.read(args.ffield)
    masses = {a.symbol: a.values[2] for a in ff.atoms}
    needs = {s.name: {"forces": True, "geometry": True} for s in frames}
    with EvalPool(frames, EvalSettings(charge_method=args.charge_method), needs, masses, n_workers=args.workers) as pool:
        res = pool.evaluate([Path(args.ffield).read_text()])[0]
    labeled, failed = [], {}
    for s in frames:
        r = res[s.name]
        if not r.ok:
            failed[s.name] = r.error
            continue
        s.energy = r.energy * KCALMOL_TO_EV
        s.forces = r.forces * KCALMOL_TO_EV
        s.charges = r.charges
        if r.stress is not None:
            s.stress = r.stress / 160.21766208  # GPa -> eV/Å^3
        if r.positions is not None:
            s.positions = r.positions
        if r.cell is not None:
            s.cell = r.cell
        labeled.append(s)
    lot = "reaxff-" + hashlib.sha256(Path(args.ffield).read_bytes()).hexdigest()[:12] + f"-{args.charge_method}"
    return _finish(out_dir, labeled, failed, lot, {"backend": "ffield", "ffield": str(Path(args.ffield).resolve())})


# --------------------------------------------------------------------- xtb
def _xtb_one(task):
    s, method, fmax = task
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        from ase.constraints import FixAtoms, FixInternals
        from ase.optimize import BFGS
        from tblite.ase import TBLite

        atoms = SIO.to_atoms(s)
        atoms.calc = TBLite(method=method, verbosity=0, charge=float(s.charge))
        # Stretched bonds, O2 and other open-shell cases often fail the closed-shell
        # SCF at the default 300 K electronic temperature; retry with Fermi smearing.
        try:
            atoms.get_potential_energy()
        except Exception:  # noqa: BLE001
            atoms.calc = TBLite(method=method, verbosity=0, charge=float(s.charge), electronic_temperature=3000.0,
                                max_iterations=500)
        if s.mode in ("relax", "relax_cell"):
            cons = []
            bonds, angles, dihedrals = [], [], []
            for r in s.restraints or []:
                k = str(r[0]).lower()
                if k == "bond":
                    bonds.append([float(r[3]), [int(r[1]), int(r[2])]])
                elif k == "angle":
                    angles.append([float(r[4]), [int(r[1]), int(r[2]), int(r[3])]])
                elif k == "dihedral":
                    dihedrals.append([float(r[5]), [int(r[1]), int(r[2]), int(r[3]), int(r[4])]])
            if bonds or angles or dihedrals:
                cons.append(FixInternals(bonds=bonds, angles_deg=angles, dihedrals_deg=dihedrals))
            if s.fixed:
                cons.append(FixAtoms(indices=s.fixed))
            atoms.set_constraint(cons)
            target = atoms
            if s.mode == "relax_cell":
                from ase.filters import FrechetCellFilter

                target = FrechetCellFilter(atoms)
            BFGS(target, logfile=None).run(fmax=fmax, steps=500)
            atoms.set_constraint()
        e = float(atoms.get_potential_energy())
        f = atoms.get_forces()
        out = SIO.from_atoms(atoms)
        out.info, out.group, out.mode, out.restraints, out.fixed = s.info, s.group, s.mode, s.restraints, s.fixed
        out.name = s.name
        out.energy, out.forces = e, f
        try:
            out.charges = np.asarray(atoms.calc.get_charges(), float)
        except Exception:  # noqa: BLE001
            out.charges = None
        if any(s.pbc):
            try:
                out.stress = atoms.get_stress()
            except Exception:  # noqa: BLE001
                pass
        return s.name, out, None
    except Exception as exc:  # noqa: BLE001
        return s.name, None, f"{type(exc).__name__}: {exc}"[:300]


def _label_xtb(args, frames, out_dir):
    try:
        import tblite  # noqa: F401
    except ImportError:
        raise RuntimeError("the xtb backend needs tblite: pip install -e '.[xtb]'") from None
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    from ..engine.pool import default_workers

    n = max(1, min(args.workers or default_workers(), len(frames)))
    tasks = [(s, args.xtb_method, args.fmax) for s in frames]
    if n == 1:
        results = list(map(_xtb_one, tasks))
    else:
        with ProcessPoolExecutor(n, mp_context=mp.get_context("spawn")) as ex:
            results = list(ex.map(_xtb_one, tasks, chunksize=1))
    labeled, failed = [], {}
    for name, s, err in results:
        if err:
            failed[name] = err
        else:
            labeled.append(s)
    import tblite as _t

    lot = f"xtb-{args.xtb_method}"
    return _finish(out_dir, labeled, failed, lot, {"backend": "xtb", "method": args.xtb_method, "workers": n})


# ---------------------------------------------------------------------- qe
_QE_MANIFEST = "label_manifest.json"


def _qe_prepare(args, frames, out_dir):
    st = QE.QESettings.from_dict(args.qe or {})
    if not st.pseudo_dir:
        p = config.resolve_component("pseudo_dir", required=False)
        if not p:
            raise ValueError("qe settings need pseudo_dir (or run `reaxff-agent setup --component pseudopotentials`)")
        st.pseudo_dir = str(p)
    els = sorted({e for s in frames for e in s.symbols})
    if not st.pseudopotentials:
        st.pseudopotentials = QE.find_pseudopotentials(st.pseudo_dir, els)
    missing = [e for e in els if e not in st.pseudopotentials]
    if missing:
        raise ValueError(f"no pseudopotential given for {missing}")
    pseudo_local = out_dir / "pseudo"
    pseudo_local.mkdir(exist_ok=True)
    for el in els:
        src = Path(st.pseudo_dir) / st.pseudopotentials[el]
        if not (pseudo_local / src.name).exists():
            shutil.copy2(src, pseudo_local / src.name)
    valence = {el: QE.upf_valence(pseudo_local / st.pseudopotentials[el]) for el in els}
    from ase.data import atomic_masses, atomic_numbers

    masses = {el: float(atomic_masses[atomic_numbers[el]]) for el in els}
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    rows = []
    import hashlib
    import re as _re

    for k, s in enumerate(frames):
        # keyed by structure (name + geometry), never by list position: a rerun with --select or new frames
        # must not pick up another structure's finished calculation
        geo = hashlib.sha1((" ".join(s.symbols) + np.array2string(np.round(s.positions, 5)) +
                            np.array2string(np.round(s.cell, 5))).encode()).hexdigest()[:8]
        d = frames_dir / f"{_re.sub(r'[^A-Za-z0-9_.-]', '_', s.name)[:80]}__{geo}"
        d.mkdir(exist_ok=True)
        text = QE.render_pw_in(s, st, valence, masses).replace("pseudo_dir = '.'", f"pseudo_dir = '{pseudo_local.resolve()}'")
        if not (d / "pw.in").is_file() or not QE.frame_done(d):
            (d / "pw.in").write_text(text)
        if st.charges == "lowdin":
            (d / "projwfc.in").write_text(QE.render_projwfc_in())
        rows.append({"dir": str(d), "name": s.name, "cost": QE.frame_cost(s, st)})
    SIO.write(out_dir / "input_structures.extxyz", frames)
    manifest = {"settings": st.to_dict(), "lot": QE.lot_key(st), "valence": valence, "frames": rows,
                "structures": str((out_dir / "input_structures.extxyz").resolve())}
    (out_dir / _QE_MANIFEST).write_text(json.dumps(manifest, indent=1))
    return st, manifest


def _qe_submit(args, out_dir, manifest):
    st = QE.QESettings.from_dict(manifest["settings"])
    todo = [r for r in manifest["frames"] if not QE.frame_done(Path(r["dir"]))]
    if not todo:
        return {"status": "all frames already finished", "next": f"reaxff-agent label --backend qe --collect {out_dir}"}
    dry = bool(getattr(args, "dry_run", False))
    if args.run_local:
        if dry:
            raise ValueError("--run-local has no dry run")
        pw = config.resolve_component("qe_pw_bin")
        import subprocess

        for r in todo:
            d = Path(r["dir"])
            with open(d / "pw.out", "w") as fo, open(d / "pw.err", "w") as fe:
                subprocess.run([str(pw), "-in", "pw.in"], cwd=d, stdout=fo, stderr=fe)
            if st.charges == "lowdin":
                pj = Path(pw).parent / "projwfc.x"
                with open(d / "projwfc.out", "w") as fo:
                    subprocess.run([str(pj), "-in", "projwfc.in"], cwd=d, stdout=fo, stderr=subprocess.STDOUT)
            if QE.frame_done(d):
                shutil.rmtree(d / "out", ignore_errors=True)
        return {"status": "ran locally", "n_frames": len(todo), "next": f"reaxff-agent label --backend qe --collect {out_dir}"}
    if not args.machine:
        raise ValueError("--backend qe needs --machine (Slurm submission) or --run-local")
    profile = machines.load_profile(args.machine)
    per_node = int(args.ntasks_per_node or profile.default_ntasks_per_node)
    total = per_node * int(args.nodes)
    lanes = max(1, min(int(args.lanes), len(todo)))
    ranks = max(1, total // lanes)
    cmd = QE.frame_command(profile.launcher, ranks, per_node, lowdin=st.charges == "lowdin", pw_flags=args.pw_flags)
    pw_bin = config.resolve_component("qe_pw_bin", required=not dry)
    body = [f"export PATH={Path(pw_bin).parent if pw_bin else '<pw.x dir>'}:$PATH", QE.LEFT_S_FUNCTION]
    for lane in QE.assign_lanes([r["cost"] for r in todo], lanes):
        seq = " ; ".join(f"cd {Path(todo[i]['dir']).resolve()} && {{ {cmd} ; }}" for i in lane)
        body.append(f"( {seq} ) &")
    body.append("wait")
    handles = hpc.submit_chain(profile, n_jobs=int(args.chunks), job_name="label", commands=body, work_dir=out_dir,
                               nodes=int(args.nodes), ntasks_per_node=per_node, walltime_hours=float(args.walltime_hours),
                               queue=args.queue, dry_run=dry)
    return {"job_ids": [h.job_id for h in handles], "job_files": [str(h.job_file) for h in handles], "dry_run": dry,
            "n_frames_to_run": len(todo), "lanes": lanes, "ranks_per_lane": ranks, "lot": manifest["lot"],
            "next": f"reaxff-agent job-status --work-dir {out_dir}; then reaxff-agent label --backend qe --collect {out_dir}"}


def _qe_collect(path):
    out_dir = Path(path)
    manifest = json.loads((out_dir / _QE_MANIFEST).read_text())
    st = QE.QESettings.from_dict(manifest["settings"])
    frames = {s.name: s for s in SIO.read(manifest["structures"])}
    labeled, failed = [], {}
    for r in manifest["frames"]:
        d = Path(r["dir"])
        if not QE.frame_done(d):
            out = d / "pw.out"
            tail = out.read_text(errors="replace").splitlines()[-3:] if out.is_file() else ["not run"]
            failed[r["name"]] = " | ".join(t.strip() for t in tail)[:300]
            continue
        try:
            labeled.append(QE.parse_frame(d, frames[r["name"]], manifest["valence"], st.vacuum))
        except Exception as exc:  # noqa: BLE001
            failed[r["name"]] = f"parse error: {exc}"[:300]
    res = _finish(out_dir, labeled, failed, manifest["lot"], {"backend": "qe", "settings": manifest["settings"]})
    if failed:
        res["next"] = (f"{len(failed)} frame(s) unfinished: resubmit (finished frames are skipped) with the same --output-dir, "
                       "or drop them; SCF failures on stretched bonds usually need more electron_maxstep/lower mixing_beta")
    return res


def run(args) -> dict:
    if args.collect:
        return _qe_collect(args.collect)
    out_dir = Path(args.output_dir or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = _load(args)
    if getattr(args, "dry_run", False) and args.backend != "qe":
        raise ValueError(f"the {args.backend} backend runs locally and has no dry run")
    t0 = time.time()
    if args.backend == "ffield":
        res = _label_ffield(args, frames, out_dir)
    elif args.backend == "xtb":
        res = _label_xtb(args, frames, out_dir)
    else:
        st, manifest = _qe_prepare(args, frames, out_dir)
        res = _qe_submit(args, out_dir, manifest)
        res["label_dir"] = str(out_dir)
        if args.run_local:
            res.update(_qe_collect(out_dir))
    res["wall_s"] = round(time.time() - t0, 1)
    return res
