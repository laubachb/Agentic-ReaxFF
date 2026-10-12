"""Load the LAMMPS python module reliably.

The PyPI `lammps` wheel is built against the MPICH ABI and dlopens
`libmpi.so.12` at import. Rather than asking users to set LD_LIBRARY_PATH
before Python starts, we preload the MPI library recorded by `reaxff-agent
setup --component mpi` (mvapich2/cray-mpich/Intel MPI on clusters, the pip
`mpich` wheel elsewhere) with RTLD_GLOBAL; the dynamic loader then reuses it
when liblammps asks for that soname. A source-built LAMMPS (setup
--component lammps) takes precedence over the wheel.

Every evaluation runs LAMMPS serially inside one process (MPI singleton);
parallelism comes from many worker processes, never from MPI ranks.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys
from pathlib import Path
from typing import Optional

from .. import config

_loaded = None
_mpi_handle = None


def _candidate_mpi_libs() -> list:
    """Places an MPICH-ABI libmpi.so.12 commonly lives, in preference order."""
    out = []
    p = config.resolve_component("mpi_lib", required=False)
    if p:
        out.append(str(p))
    # pip `mpich` wheel installs into <venv>/lib
    out += sorted(glob.glob(os.path.join(sys.prefix, "lib", "libmpi.so.12*")))
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if d:
            out += sorted(glob.glob(os.path.join(d, "libmpi.so.12*")))
    return out


def preload_mpi() -> Optional[str]:
    global _mpi_handle
    if _mpi_handle is not None:
        return _mpi_handle[0]
    for path in _candidate_mpi_libs():
        try:
            handle = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        _mpi_handle = (path, handle)
        return path
    return None


def lammps_module():
    """The imported `lammps` python package (source build first, else wheel)."""
    global _loaded
    if _loaded is not None:
        return _loaded
    src = config.resolve_component("lammps_python", required=False)
    if src and Path(src).is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
        lib = config.resolve_component("lammps_lib", required=False)
        if lib:
            os.environ.setdefault("LAMMPS_SHARED_LIB", str(lib))
    preload_mpi()
    try:
        import lammps  # noqa: F401
    except ImportError as exc:
        raise config.ComponentNotInstalled(
            "the LAMMPS python module is not importable; run ./install.sh (pip installs the `lammps` wheel) "
            f"or `reaxff-agent setup --component lammps --machine <m>` ({exc})") from exc
    _loaded = lammps
    return lammps


def new_instance(screen: Optional[str] = None, log: Optional[str] = None):
    """A fresh serial LAMMPS instance with screen/log output suppressed."""
    lammps = lammps_module()
    args = ["-nocite", "-log", log or "none", "-screen", screen or "none"]
    try:
        return lammps.lammps(cmdargs=args)
    except OSError as exc:
        if "libmpi" in str(exc):
            raise config.ComponentNotInstalled(
                f"LAMMPS could not load an MPI library ({exc}). Run `reaxff-agent setup --component mpi --machine <m>` "
                "to record your cluster's MPICH-ABI libmpi.so.12, or `pip install mpich` into this environment.") from exc
        raise


def capabilities() -> dict:
    """Version and the packages/styles the evaluator needs."""
    lmp = new_instance()
    try:
        need = {"REAXFF": lmp.has_package("REAXFF"), "QEQ": lmp.has_package("QEQ")}
        styles = {s: lmp.has_style("fix", s) for s in ("qeq/reaxff", "acks2/reaxff", "restrain", "box/relax")}
        return {"version": lmp.version(), "mpi": bool(lmp.has_mpi_support), "packages": need, "fix_styles": styles,
                "openmp": lmp.has_package("OPENMP"), "kokkos": lmp.has_package("KOKKOS"),
                "mpi_lib": _mpi_handle[0] if _mpi_handle else None,
                "module": str(Path(lammps_module().__file__).parent)}
    finally:
        lmp.close()
