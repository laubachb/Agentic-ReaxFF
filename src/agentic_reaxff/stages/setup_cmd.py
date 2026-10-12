"""`reaxff-agent setup`: install what pip cannot, for a machine profile.

Default components (fast, needed by every study): `mpi` (the MPI library
the LAMMPS wheel loads, verified by a real ReaxFF single point),
`lammps_wheel` (checks REAXFF/QEQ/fix restrain) and `ffield_library` (the
ReaxFF force fields distributed with LAMMPS). On demand: `quantum_espresso`
(register --qe-bin, or build), `pseudopotentials` (register --pseudo-dir, or
download GBRV), `lammps` (source build for large MD), `jax_reaxff`
(separate venv). `./install.sh` runs the defaults for you.
"""

from __future__ import annotations

from .. import config, machines
from ..setup import components as C

NAME = "setup"
SUMMARY = "Install/verify components for a machine: MPI for the LAMMPS wheel, ffield library, QE, pseudopotentials, LAMMPS source build, JAX-ReaxFF."
USES_OUTPUT_DIR = False
SCHEMA = {
    "type": "object",
    "properties": {
        "component": {"type": ["array", "null"], "items": {"type": "string", "enum": C.COMPONENTS + ["all"]},
                      "description": f"Default: {C.DEFAULT}."},
        "machine": {"type": ["string", "null"], "description": "Machine profile name or YAML path (default: generic)."},
        "force": {"type": "boolean"},
        "status": {"type": "boolean", "default": False},
        "qe_bin": {"type": ["string", "null"], "description": "Register an existing pw.x (or its bin directory) instead of building QE."},
        "pseudo_dir": {"type": ["string", "null"], "description": "Register an existing pseudopotential directory."},
        "lammps_kokkos": {"type": "boolean", "default": False},
        "jobs": {"type": "integer", "default": 8, "description": "Parallel make jobs for source builds."},
        "init_profile": {"type": ["string", "null"], "description": "Write a machine profile YAML detected from Slurm and exit."},
        "account": {"type": ["string", "null"]},
        "debug_partition": {"type": ["string", "null"]},
        "batch_partition": {"type": ["string", "null"]},
        "cores_per_node": {"type": ["integer", "null"]},
        "scratch": {"type": ["string", "null"]},
        "modules": {"type": ["array", "null"], "items": {"type": "string"}},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--component", action="append", default=None, choices=C.COMPONENTS + ["all"])
    parser.add_argument("--machine", default=None)
    parser.add_argument("--status", action="store_true", default=False)
    parser.add_argument("--qe-bin", dest="qe_bin", default=None)
    parser.add_argument("--pseudo-dir", dest="pseudo_dir", default=None)
    parser.add_argument("--lammps-kokkos", dest="lammps_kokkos", action="store_true", default=False)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--init-profile", dest="init_profile", default=None)
    parser.add_argument("--account", default=None)
    parser.add_argument("--debug-partition", dest="debug_partition", default=None)
    parser.add_argument("--batch-partition", dest="batch_partition", default=None)
    parser.add_argument("--cores-per-node", dest="cores_per_node", type=int, default=None)
    parser.add_argument("--scratch", default=None)
    parser.add_argument("--modules", type=lambda s: [x for x in s.split(",") if x], default=None)


def run(args) -> dict:
    if args.status:
        return {"deps_dir": str(config.DEPS_DIR), "installed": C.status()}
    if args.init_profile:
        from ..setup import init_profile

        return init_profile.write(args.init_profile, account=args.account, debug_partition=args.debug_partition,
                                  batch_partition=args.batch_partition, cores_per_node=args.cores_per_node,
                                  scratch=args.scratch, modules=args.modules, hosttype="none")
    comps = args.component or list(C.DEFAULT)
    if "all" in comps:
        comps = list(C.COMPONENTS)
    profile = machines.load_profile(args.machine or "generic")
    res = C.run_setup(comps, profile, force=bool(getattr(args, "force", False)), qe_bin=args.qe_bin,
                      pseudo_dir=args.pseudo_dir, kokkos=args.lammps_kokkos, jobs=args.jobs)
    failed = [c for c, r in res.items() if r.get("status") == "failed"]
    out = {"machine": profile.name, "results": res, "failed": failed, "installed_json": str(config.INSTALLED_JSON)}
    if not failed:
        out["next"] = "reaxff-agent doctor"
    return out
