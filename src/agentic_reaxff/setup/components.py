"""What `reaxff-agent setup` installs, one function per component.

Every component records itself in deps/installed.json (machine, path,
version) and is idempotent: an installed component is skipped unless
`force`. Nothing here is needed for a pure-Python checkout to import; it is
what turns the checkout into a working ReaxFF-fitting installation.

  mpi              find an MPICH-ABI libmpi.so.12 (cluster MPI: mvapich2,
                   cray-mpich, Intel MPI; else the pip `mpich` wheel) that the
                   LAMMPS wheel can load; verified by a LAMMPS single point
                   in a subprocess
  lammps_wheel     verify the PyPI LAMMPS (REAXFF, QEQ, fix restrain) and
                   record its `lmp` executable
  lammps           optional source build (pinned stable tag, cmake, shared
                   library + python module installed into this env, OPENMP;
                   KOKKOS with `--lammps-kokkos`), for large MD on clusters
  quantum_espresso pw.x + projwfc.x: register an existing build (--qe-bin)
                   or clone a pinned release and build `make pw pp`
  pseudopotentials register a directory (--pseudo-dir) or download GBRV 1.5
                   PBE ultrasoft potentials
  ffield_library   the ReaxFF force fields distributed with LAMMPS (pinned
                   tag) into deps/ffields/, for ffield-library / ffield-merge
  jax_reaxff       JAX-ReaxFF in its own virtual environment (it pins
                   jax<=0.4.30 and scipy<=1.12, which must not leak into
                   this one)
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from .. import config

LAMMPS_TAG = "stable_22Jul2025_update4"
LAMMPS_REPO = "https://github.com/lammps/lammps.git"
QE_REPO = "https://gitlab.com/QEF/q-e.git"
QE_TAG = "qe-7.3.1"
GBRV_URL = "https://www.physics.rutgers.edu/gbrv/all_pbe_uspp.tar.gz"
JAX_REAXFF_REPO = "https://github.com/cagrikymk/JAX-ReaxFF.git"

COMPONENTS = ["mpi", "lammps_wheel", "ffield_library", "pseudopotentials", "quantum_espresso", "lammps", "jax_reaxff"]
DEFAULT = ["mpi", "lammps_wheel", "ffield_library"]


def _sh(script: str, *, cwd=None, log: Path, modules=None, timeout=None) -> subprocess.CompletedProcess:
    """Run in a login shell (so `module` works), log everything, raise on failure."""
    log.parent.mkdir(parents=True, exist_ok=True)
    pre = ("module load " + " ".join(modules) + " >/dev/null 2>&1\n") if modules else ""
    full = pre + script
    proc = subprocess.run(["bash", "-lc", full], cwd=str(cwd) if cwd else None, capture_output=True, text=True,
                          timeout=timeout)
    log.write_text(f"$ {full}\n\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).splitlines()[-15:])
        raise RuntimeError(f"command failed (exit {proc.returncode}); log {log}\n{tail}")
    return proc


def _logs() -> Path:
    d = config.DEPS_DIR / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------- mpi
_PROBE = r"""
import sys, ctypes
ctypes.CDLL(sys.argv[1], mode=ctypes.RTLD_GLOBAL)
from lammps import lammps
l = lammps(cmdargs=['-nocite','-log','none','-screen','none'])
ok = l.has_package('REAXFF') and l.has_style('fix','qeq/reaxff')
l.close()
print('OK' if ok else 'NOREAXFF')
"""


def _probe_mpi(path: str) -> bool:
    try:
        p = subprocess.run([sys.executable, "-c", _PROBE, path], capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return False
    return p.returncode == 0 and "OK" in p.stdout


def _mpi_candidates(profile) -> list:
    out = []
    if profile is not None and profile.mpi_lib:
        out.append(profile.mpi_lib)
    if shutil.which("mpicc"):
        try:
            show = subprocess.run(["mpicc", "-show"], capture_output=True, text=True, timeout=30).stdout
            for tok in show.split():
                if tok.startswith("-L"):
                    out += sorted(glob.glob(os.path.join(tok[2:], "libmpi.so.12*")))
        except (OSError, subprocess.TimeoutExpired):
            pass
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if d:
            out += sorted(glob.glob(os.path.join(d, "libmpi.so.12*")))
    out += sorted(glob.glob(os.path.join(sys.prefix, "lib", "libmpi.so.12*")))
    return list(dict.fromkeys(p for p in out if p))


def install_mpi(profile, force=False, **kw) -> dict:
    tried = []
    for cand in _mpi_candidates(profile):
        if not Path(cand).exists():
            continue
        tried.append(cand)
        if _probe_mpi(cand):
            config.write_installed_component("mpi_lib", path=cand, machine=profile.name if profile else "")
            return {"status": "ok", "mpi_lib": cand}
    # fall back to the pip MPICH wheel
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "mpich"], check=False, capture_output=True)
    for cand in sorted(glob.glob(os.path.join(sys.prefix, "lib", "libmpi.so.12*"))):
        tried.append(cand)
        if _probe_mpi(cand):
            config.write_installed_component("mpi_lib", path=cand, machine=profile.name if profile else "", version="pip-mpich")
            return {"status": "ok", "mpi_lib": cand, "note": "using the pip mpich wheel"}
    raise RuntimeError(f"no MPICH-ABI libmpi.so.12 made the LAMMPS wheel work (tried {tried}). OpenMPI systems: "
                       "`pip install mpich` then rerun, or build LAMMPS from source (--component lammps)")


# ------------------------------------------------------------ lammps wheel
def install_lammps_wheel(profile, force=False, **kw) -> dict:
    from ..engine import lmp as lmpmod

    caps = lmpmod.capabilities()
    missing = [k for k, v in {**caps["packages"], **caps["fix_styles"]}.items() if not v and k != "acks2/reaxff"]
    if missing:
        raise RuntimeError(f"the LAMMPS python module lacks {missing}; build from source: --component lammps")
    lmp_exe = shutil.which("lmp") or str(Path(sys.executable).parent / "lmp")
    if Path(lmp_exe).exists():
        config.write_installed_component("lmp_bin", path=lmp_exe, machine=profile.name if profile else "",
                                         version=str(caps["version"]))
    return {"status": "ok", **caps, "lmp_bin": lmp_exe if Path(lmp_exe).exists() else None}


# ---------------------------------------------------------- lammps source
def install_lammps(profile, force=False, kokkos=False, jobs=8, **kw) -> dict:
    src = config.DEPS_DIR / "lammps-src"
    build = src / "build-reaxff"
    log = _logs()
    mods = profile.modules if profile else []
    if force and src.exists():
        shutil.rmtree(src)
    if not src.exists():
        _sh(f"git clone --depth 1 --branch {LAMMPS_TAG} {LAMMPS_REPO} {src}", log=log / "lammps_clone.log", timeout=1800)
    pk = ["REAXFF", "QEQ", "MOLECULE", "EXTRA-MOLECULE", "EXTRA-FIX", "EXTRA-COMPUTE", "OPENMP"]
    flags = " ".join(f"-D PKG_{p}=yes" for p in pk)
    if kokkos:
        flags += " -D PKG_KOKKOS=yes -D Kokkos_ENABLE_OPENMP=yes"
    extra = " ".join((profile.lammps or {}).get("cmake_args", [])) if profile else ""
    _sh(f"cmake -S cmake -B {build} -D BUILD_SHARED_LIBS=yes -D BUILD_MPI=yes -D LAMMPS_EXCEPTIONS=yes "
        f"-D CMAKE_BUILD_TYPE=Release {flags} {extra} -D PYTHON_EXECUTABLE={sys.executable}",
        cwd=src, log=log / "lammps_cmake.log", modules=mods, timeout=1800)
    _sh(f"cmake --build {build} -j {jobs}", cwd=src, log=log / "lammps_build.log", modules=mods, timeout=4 * 3600)
    _sh(f"cmake --build {build} --target install-python", cwd=src, log=log / "lammps_install_python.log", modules=mods,
        timeout=1800)
    lib = next(iter(sorted(build.glob("liblammps.so*"))), None)
    lmp_bin = build / "lmp"
    config.write_installed_component("lammps_lib", path=str(lib), machine=profile.name if profile else "", version=LAMMPS_TAG)
    config.write_installed_component("lmp_bin", path=str(lmp_bin), machine=profile.name if profile else "", version=LAMMPS_TAG)
    return {"status": "built", "lmp_bin": str(lmp_bin), "lammps_lib": str(lib), "packages": pk + (["KOKKOS"] if kokkos else []),
            "note": "the python module was installed into this environment, replacing the PyPI wheel"}


# --------------------------------------------------------- quantum espresso
def install_quantum_espresso(profile, force=False, qe_bin=None, jobs=8, **kw) -> dict:
    if qe_bin:
        pw = Path(qe_bin)
        if pw.is_dir():
            pw = pw / "pw.x"
        if not pw.is_file():
            raise FileNotFoundError(f"{pw} does not exist")
        config.write_installed_component("qe_pw_bin", path=str(pw), machine=profile.name if profile else "", version="registered")
        pj = pw.parent / "projwfc.x"
        if pj.is_file():
            config.write_installed_component("qe_projwfc_bin", path=str(pj), machine=profile.name if profile else "")
        return {"status": "registered", "qe_pw_bin": str(pw), "projwfc": str(pj) if pj.is_file() else None,
                "note": None if pj.is_file() else "no projwfc.x next to pw.x: Löwdin charges unavailable (QE `make pp`)"}
    qe = config.DEPS_DIR / "quantum-espresso"
    log = _logs()
    mods = profile.modules if profile else []
    if force and qe.exists():
        shutil.rmtree(qe)
    if not qe.exists():
        _sh(f"git clone --depth 1 --branch {QE_TAG} {QE_REPO} {qe}", log=log / "qe_clone.log", timeout=1800)
    extra = " ".join((profile.qe or {}).get("configure_extra_args", [])) if profile else ""
    _sh(f"./configure {extra} && make -j {jobs} pw pp", cwd=qe, log=log / "qe_build.log", modules=mods, timeout=4 * 3600)
    pw = qe / "bin" / "pw.x"
    if not pw.is_file():
        raise RuntimeError(f"QE build finished but {pw} is missing; see {log / 'qe_build.log'}")
    config.write_installed_component("qe_pw_bin", path=str(pw), machine=profile.name if profile else "", version=QE_TAG)
    config.write_installed_component("qe_projwfc_bin", path=str(qe / "bin" / "projwfc.x"),
                                     machine=profile.name if profile else "", version=QE_TAG)
    return {"status": "built", "qe_pw_bin": str(pw), "version": QE_TAG}


# ---------------------------------------------------------- pseudopotentials
def install_pseudopotentials(profile, force=False, pseudo_dir=None, **kw) -> dict:
    if pseudo_dir:
        d = Path(pseudo_dir)
        n = len(list(d.glob("*.[Uu][Pp][Ff]")))
        if not n:
            raise FileNotFoundError(f"no .UPF files in {d}")
        config.write_installed_component("pseudo_dir", path=str(d.resolve()), version="registered")
        return {"status": "registered", "pseudo_dir": str(d.resolve()), "n_upf": n}
    d = config.DEPS_DIR / "pseudo" / "gbrv_1.5_pbe_uspp"
    if force and d.exists():
        shutil.rmtree(d)
    if not d.exists():
        d.mkdir(parents=True)
        tgz = d.parent / "all_pbe_uspp.tar.gz"
        urllib.request.urlretrieve(GBRV_URL, tgz)
        with tarfile.open(tgz) as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.lower().endswith(".upf"):
                    m.name = Path(m.name).name
                    tf.extract(m, d)
        tgz.unlink()
    n = len(list(d.glob("*.[Uu][Pp][Ff]")))
    config.write_installed_component("pseudo_dir", path=str(d), version="GBRV-1.5-PBE-USPP")
    return {"status": "ok", "pseudo_dir": str(d), "n_upf": n,
            "note": "GBRV 1.5 USPP: 40/200 Ry nominal cutoffs; check convergence for your systems"}


# ----------------------------------------------------------- ffield library
def install_ffield_library(profile, force=False, **kw) -> dict:
    d = config.FFIELD_LIBRARY_DIR
    d.mkdir(parents=True, exist_ok=True)
    api = f"https://api.github.com/repos/lammps/lammps/contents/potentials?ref={LAMMPS_TAG}"
    with urllib.request.urlopen(api, timeout=60) as r:
        listing = json.load(r)
    names = [e["name"] for e in listing if "reax" in e["name"].lower() and e["name"].startswith("ffield")]
    got = []
    for n in names:
        dest = d / n
        if dest.exists() and not force:
            got.append(n)
            continue
        url = f"https://raw.githubusercontent.com/lammps/lammps/{LAMMPS_TAG}/potentials/{n}"
        urllib.request.urlretrieve(url, dest)
        got.append(n)
    (d / "SOURCE.txt").write_text(f"ReaxFF force fields distributed with LAMMPS {LAMMPS_TAG} (potentials/).\n"
                                  "Cite the paper named in each file's header when you use it. Add your own ffields here "
                                  "or pass --dirs to ffield-library.\n")
    config.write_installed_component("ffield_library", path=str(d), version=LAMMPS_TAG)
    return {"status": "ok", "ffield_library": str(d), "n_files": len(got), "files": got}


# --------------------------------------------------------------- jax-reaxff
def install_jax_reaxff(profile, force=False, python=None, **kw) -> dict:
    venv = config.DEPS_DIR / "jax-reaxff-venv"
    src = config.DEPS_DIR / "JAX-ReaxFF"
    log = _logs()
    if force:
        shutil.rmtree(venv, ignore_errors=True)
        shutil.rmtree(src, ignore_errors=True)
    if not src.exists():
        _sh(f"git clone --depth 1 {JAX_REAXFF_REPO} {src}", log=log / "jax_clone.log", timeout=900)
    py = python or shutil.which("python3.11") or shutil.which("python3.10") or sys.executable
    if not venv.exists():
        if shutil.which("uv"):
            _sh(f"uv venv -q -p {py} {venv}", log=log / "jax_venv.log")
        else:
            _sh(f"{py} -m venv {venv}", log=log / "jax_venv.log")
    vpy = venv / "bin" / "python"
    pip = f"uv pip install -q -p {vpy}" if shutil.which("uv") else f"{vpy} -m pip install -q"
    _sh(f"{pip} {src}", log=log / "jax_install.log", timeout=3600)
    exe = venv / "bin" / "jaxreaxff"
    config.write_installed_component("jax_reaxff", path=str(vpy), version="git", exe=str(exe))
    return {"status": "ok", "python": str(vpy), "jaxreaxff": str(exe),
            "note": "GPL-3.0; EEM only, no lg, no cell optimization. Use trainset-convert --to-legacy to feed it."}


INSTALLERS = {
    "mpi": install_mpi,
    "lammps_wheel": install_lammps_wheel,
    "lammps": install_lammps,
    "quantum_espresso": install_quantum_espresso,
    "pseudopotentials": install_pseudopotentials,
    "ffield_library": install_ffield_library,
    "jax_reaxff": install_jax_reaxff,
}

_KEYS = {"mpi": ["mpi_lib"], "lammps_wheel": ["lmp_bin"], "lammps": ["lammps_lib", "lmp_bin"],
         "quantum_espresso": ["qe_pw_bin", "qe_projwfc_bin"], "pseudopotentials": ["pseudo_dir"],
         "ffield_library": ["ffield_library"], "jax_reaxff": ["jax_reaxff"]}


def status() -> dict:
    inst = config.installed_components()
    return {c: {k: inst.get(k) for k in keys} for c, keys in _KEYS.items()}


def run_setup(components, profile, *, force=False, **kw) -> dict:
    out = {}
    inst = config.installed_components()
    for c in components:
        if c not in INSTALLERS:
            raise ValueError(f"unknown component {c!r}; known: {COMPONENTS}")
        keys = _KEYS[c]
        if not force and c not in ("lammps_wheel",) and all(k in inst and Path(inst[k]["path"]).exists() for k in keys) \
                and not (c == "quantum_espresso" and kw.get("qe_bin")) and not (c == "pseudopotentials" and kw.get("pseudo_dir")):
            out[c] = {"status": "already installed", **{k: inst[k]["path"] for k in keys}}
            continue
        try:
            out[c] = INSTALLERS[c](profile, force=force, **kw)
        except Exception as exc:  # noqa: BLE001
            out[c] = {"status": "failed", "error": str(exc)[:1500]}
    return out
