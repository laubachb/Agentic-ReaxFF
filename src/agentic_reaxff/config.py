"""Where installed components live and how stages find them.

The package runs from an editable install inside the Agentic-ReaxFF repo
(`./install.sh` does this). Built or fetched dependencies live under
`deps/` (gitignored) and are recorded in `deps/installed.json` by
`reaxff-agent setup`; every stage resolves them through
:func:`resolve_component`, which honours an environment-variable override
first. Set `AGENTIC_REAXFF_ROOT` to point at a different checkout (tests).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


def _find_repo_root() -> Path:
    env = os.environ.get("AGENTIC_REAXFF_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "agentic_reaxff" / "config.py").is_file():
            return parent
    # installed non-editable: keep state in the user's home instead
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "agentic-reaxff"


REPO_ROOT = _find_repo_root()
DEPS_DIR = Path(os.environ.get("AGENTIC_REAXFF_DEPS", str(REPO_ROOT / "deps")))
INSTALLED_JSON = DEPS_DIR / "installed.json"
FFIELD_LIBRARY_DIR = DEPS_DIR / "ffields"
PACKAGE_DATA = Path(__file__).resolve().parent / "data"


class ComponentNotInstalled(RuntimeError):
    """A stage needs something `reaxff-agent setup` has not installed."""


# component -> (env var override, description, setup command that provides it)
_COMPONENTS = {
    "mpi_lib": ("AGENTIC_REAXFF_MPI_LIB", "MPICH-ABI libmpi.so.12 that the LAMMPS python module loads",
                "reaxff-agent setup --component mpi --machine <m>"),
    "lammps_python": ("AGENTIC_REAXFF_LAMMPS_PYTHON", "directory holding a source-built `lammps` python package "
                      "(else the PyPI wheel is used)", "reaxff-agent setup --component lammps --machine <m>"),
    "lammps_lib": ("AGENTIC_REAXFF_LAMMPS_LIB", "liblammps.so of a source-built LAMMPS",
                   "reaxff-agent setup --component lammps --machine <m>"),
    "lmp_bin": ("AGENTIC_REAXFF_LMP_BIN", "LAMMPS executable for MD (source build, or the wheel's lmp)",
                "reaxff-agent setup --component lammps_wheel"),
    "qe_pw_bin": ("AGENTIC_REAXFF_QE_BIN", "pw.x (Quantum ESPRESSO)",
                  "reaxff-agent setup --component quantum_espresso --machine <m>  (or --qe-bin /path/to/pw.x)"),
    "qe_projwfc_bin": ("AGENTIC_REAXFF_PROJWFC_BIN", "projwfc.x (Loewdin charges)",
                       "reaxff-agent setup --component quantum_espresso --machine <m>"),
    "pseudo_dir": ("AGENTIC_REAXFF_PSEUDO_DIR", "directory of pseudopotentials for pw.x",
                   "reaxff-agent setup --component pseudopotentials  (or --pseudo-dir DIR)"),
    "ffield_library": ("AGENTIC_REAXFF_FFIELD_LIBRARY", "directory of published ffields (bounds + starting points)",
                       "reaxff-agent setup --component ffield_library"),
    "jax_reaxff": ("AGENTIC_REAXFF_JAX_PYTHON", "python interpreter with JAX-ReaxFF installed",
                   "reaxff-agent setup --component jax_reaxff"),
}


def known_components() -> list:
    return sorted(_COMPONENTS)


def describe_component(name: str) -> dict:
    env, desc, fix = _COMPONENTS[name]
    return {"env": env, "description": desc, "install": fix}


def installed_components() -> dict:
    if not INSTALLED_JSON.is_file():
        return {}
    try:
        return json.loads(INSTALLED_JSON.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def write_installed_component(component: str, *, path: str, machine: str = "", version: str = "", **extra) -> None:
    if component not in _COMPONENTS:
        raise ValueError(f"unknown component {component!r}; known: {known_components()}")
    data = installed_components()
    data[component] = {"path": str(path), "machine": machine, "version": version, **extra}
    DEPS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = INSTALLED_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(INSTALLED_JSON)


def forget_component(component: str) -> None:
    data = installed_components()
    if data.pop(component, None) is not None:
        INSTALLED_JSON.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def resolve_component(component: str, *, required: bool = True) -> Optional[Path]:
    """Env var override, then deps/installed.json. Raises
    ComponentNotInstalled (with the command that installs it) when missing
    and `required`."""
    if component not in _COMPONENTS:
        raise ValueError(f"unknown component {component!r}; known: {known_components()}")
    env_var, description, fix = _COMPONENTS[component]
    val = os.environ.get(env_var)
    if val:
        return Path(val)
    entry = installed_components().get(component)
    if entry and entry.get("path"):
        return Path(entry["path"])
    if required:
        raise ComponentNotInstalled(f"{description} is not installed. Install it with `{fix}`, or set {env_var}.")
    return None
