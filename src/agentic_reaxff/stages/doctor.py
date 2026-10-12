"""Check that this installation can run a study, before a study finds out.

Each check reports ok / warn / fail, what it saw and, when not ok, the
command that fixes it. Includes real numerics: one ReaxFF single point of
CH4 with the Chenoweth 2008 C/H/O force field (from the ffield library)
against a stored reference, and the same through spawned worker processes
(the path every fit uses). With `--machine`: account, scratch, sbatch,
partitions and (on Lustre) the file-count quota.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .. import config

NAME = "doctor"
SUMMARY = "Check the installation (LAMMPS+MPI numerics, workers, QE, pseudopotentials, ffield library, machine profile) and how to fix problems."
USES_OUTPUT_DIR = False
SCHEMA = {
    "type": "object",
    "properties": {
        "machine": {"type": ["string", "null"]},
        "quick": {"type": "boolean", "default": False, "description": "Skip the numerical and worker checks."},
    },
}

# CH4 (ase.build.molecule geometry), ffield.reax.cho, fix qeq/reaxff tol 1e-8
_CH4_REF = -463.1717274896
_CH4_Q_C = -0.272112


def add_arguments(parser) -> None:
    parser.add_argument("--machine", default=None)
    parser.add_argument("--quick", action="store_true", default=False)


def _c(name, status, detail, fix=None):
    d = {"check": name, "status": status, "detail": detail}
    if fix and status != "ok":
        d["fix"] = fix
    return d


def _cho_ffield():
    lib = config.resolve_component("ffield_library", required=False)
    for p in ([Path(lib) / "ffield.reax.cho"] if lib else []) + [config.REPO_ROOT / "tests" / "data" / "ffield.reax.cho"]:
        if p.is_file():
            return p
    return None


def _numerics(checks):
    from ase.build import molecule

    from ..engine.evaluator import Evaluator
    from ..io.structures import from_atoms

    ff = _cho_ffield()
    if ff is None:
        checks.append(_c("reaxff_numerics", "warn", "no ffield.reax.cho to test with",
                         "reaxff-agent setup --component ffield_library"))
        return
    m = molecule("CH4")
    m.info["name"] = "ch4"
    r = Evaluator().evaluate(str(ff), from_atoms(m))
    if not r.ok:
        checks.append(_c("reaxff_numerics", "fail", f"single point failed: {r.error}", "reaxff-agent setup --component mpi"))
        return
    err = abs(r.energy - _CH4_REF)
    checks.append(_c("reaxff_numerics", "ok" if err < 1e-3 and abs(r.charges[0] - _CH4_Q_C) < 1e-3 else "fail",
                     f"CH4 energy {r.energy:.6f} kcal/mol (reference {_CH4_REF}), q(C) {r.charges[0]:.5f}",
                     "LAMMPS gives different numbers than the validated build: check `setup --status` and the LAMMPS version"))
    # through spawned workers
    try:
        from ..engine.evaluator import EvalSettings
        from ..engine.pool import EvalPool

        s = from_atoms(m)
        with EvalPool([s], EvalSettings(), {"ch4": {"forces": True}}, {}, n_workers=2) as pool:
            res = pool.evaluate([ff.read_text(), ff.read_text()])
        ok = all(abs(x["ch4"].energy - _CH4_REF) < 1e-3 for x in res)
        checks.append(_c("worker_pool", "ok" if ok else "fail", "2 spawned LAMMPS workers agree" if ok else str(res)[:200]))
    except Exception as exc:  # noqa: BLE001
        checks.append(_c("worker_pool", "fail", str(exc)[:300], "check MPI: reaxff-agent setup --component mpi"))


def run(args) -> dict:
    checks = []
    checks.append(_c("python", "ok" if sys.version_info >= (3, 10) else "fail", sys.version.split()[0],
                     "use Python >= 3.10 (./install.sh --python python3.12)"))
    for mod in ("numpy", "scipy", "ase", "cma", "yaml"):
        try:
            __import__(mod)
            checks.append(_c(f"import:{mod}", "ok", "importable"))
        except ImportError:
            checks.append(_c(f"import:{mod}", "fail", "missing", "pip install -e ."))
    mpi = config.resolve_component("mpi_lib", required=False)
    checks.append(_c("mpi_lib", "ok" if mpi and Path(mpi).exists() else "warn", str(mpi),
                     "reaxff-agent setup --component mpi --machine <m>"))
    try:
        from ..engine import lmp as lmpmod

        caps = lmpmod.capabilities()
        good = caps["packages"]["REAXFF"] and caps["packages"]["QEQ"] and caps["fix_styles"]["restrain"]
        checks.append(_c("lammps", "ok" if good else "fail",
                         f"version {caps['version']}, REAXFF {caps['packages']['REAXFF']}, QEQ {caps['packages']['QEQ']}, "
                         f"acks2 {caps['fix_styles']['acks2/reaxff']}, restrain {caps['fix_styles']['restrain']}, "
                         f"module {caps['module']}", "reaxff-agent setup --component lammps --machine <m>"))
    except Exception as exc:  # noqa: BLE001
        checks.append(_c("lammps", "fail", str(exc)[:300], "./install.sh, then reaxff-agent setup --machine <m>"))
    if not args.quick and all(c["status"] == "ok" for c in checks if c["check"] == "lammps"):
        _numerics(checks)
    lib = config.resolve_component("ffield_library", required=False)
    checks.append(_c("ffield_library", "ok" if lib and Path(lib).is_dir() else "warn", str(lib),
                     "reaxff-agent setup --component ffield_library"))
    for comp, fix in (("qe_pw_bin", "reaxff-agent setup --component quantum_espresso --qe-bin /path/to/pw.x (or build)"),
                      ("pseudo_dir", "reaxff-agent setup --component pseudopotentials [--pseudo-dir DIR]")):
        p = config.resolve_component(comp, required=False)
        checks.append(_c(comp, "ok" if p and Path(p).exists() else "warn",
                         str(p) if p else "not installed (needed only for DFT labeling)", fix))
    try:
        import tblite  # noqa: F401

        checks.append(_c("tblite", "ok", "xtb labeling backend available"))
    except ImportError:
        checks.append(_c("tblite", "warn", "xtb backend unavailable", "pip install -e '.[xtb]'"))
    if args.machine:
        from .. import machines

        try:
            prof = machines.load_profile(args.machine)
            checks.append(_c("profile", "ok", f"{prof.name} ({prof.source_path})"))
            checks.append(_c("account", "ok" if prof.account else "fail", prof.account or "unset",
                             "export REAXFF_ACCOUNT=<your Slurm bank>"))
            scratch = (prof.filesystem or {}).get("scratch_root")
            ok = bool(scratch) and Path(scratch).is_dir() and os.access(scratch, os.W_OK)
            checks.append(_c("scratch", "ok" if ok else "fail", str(scratch), "set filesystem.scratch_root (shared, writable)"))
            checks.append(_c("sbatch", "ok" if shutil.which("sbatch") else "fail", str(shutil.which("sbatch")),
                             "run on a Slurm login node"))
            if shutil.which("sinfo"):
                parts = subprocess.run(["sinfo", "-h", "-o", "%P"], capture_output=True, text=True).stdout.split()
                parts = [p.rstrip("*") for p in parts]
                for q, name in prof.partitions.items():
                    checks.append(_c(f"partition:{q}", "ok" if name in parts else "warn", name,
                                     f"edit partitions.{q} in the profile (have {parts[:8]})"))
            if scratch and shutil.which("lfs"):
                q = subprocess.run(["lfs", "quota", "-u", os.environ.get("USER", ""), scratch], capture_output=True, text=True).stdout
                # lfs wraps the numbers onto the next line when the path is long
                body = q.split("grace", 2)[-1].split() if q.count("grace") >= 2 else []
                vals = [t for t in body if t.rstrip("*").isdigit() or t == "-"]
                if len(vals) >= 6:
                    try:
                        files, soft = int(vals[4].rstrip("*")), int(vals[5].rstrip("*"))
                        frac = files / soft if soft else 0
                        checks.append(_c("file_quota", "ok" if frac < 0.9 else "warn", f"{files}/{soft} files ({frac:.0%})",
                                         "archive many-file directories (tar) before large labeling jobs"))
                    except ValueError:
                        pass
        except Exception as exc:  # noqa: BLE001
            checks.append(_c("profile", "fail", str(exc)[:300]))
    worst = "fail" if any(c["status"] == "fail" for c in checks) else "warn" if any(c["status"] == "warn" for c in checks) else "ok"
    return {"status": worst, "checks": checks, "fixes": [c["fix"] for c in checks if c.get("fix")]}
