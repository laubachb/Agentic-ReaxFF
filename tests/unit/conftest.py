"""Shared fixtures. Tests that need LAMMPS or the ReaxFF library skip cleanly
when `reaxff-agent setup` has not installed them."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("REAXFF_AGENT_ALLOW_LOCAL_JOB_DIRS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def _lammps_ok() -> bool:
    try:
        from agentic_reaxff.engine import lmp

        caps = lmp.capabilities()
        return bool(caps["packages"]["REAXFF"])
    except Exception:  # noqa: BLE001
        return False


@pytest.fixture(scope="session")
def lammps_available():
    if not _lammps_ok():
        pytest.skip("LAMMPS with REAXFF not available (./install.sh, reaxff-agent setup)")
    return True


@pytest.fixture(scope="session")
def cho_ffield(lammps_available):
    from agentic_reaxff import config

    lib = config.resolve_component("ffield_library", required=False)
    p = Path(lib) / "ffield.reax.cho" if lib else None
    if not p or not p.is_file():
        pytest.skip("ffield.reax.cho not installed (reaxff-agent setup --component ffield_library)")
    return str(p)


def make_ffield(elements=("C", "H")):
    """A small synthetic force field (numbers are arbitrary, not physical)."""
    from agentic_reaxff.io import ffield as F

    n = len(elements)
    atoms = [F.AtomEntry(symbol=e, values=[1.0 + 0.1 * k + 0.01 * i for i in range(32)]) for k, e in enumerate(elements)]
    for a in atoms:
        a.values[F.ATOM_NAMES.index("eta")] = 8.0
        a.values[F.ATOM_NAMES.index("gamma")] = 0.9
        a.values[F.ATOM_NAMES.index("gamma_w")] = 2.0
        a.values[F.ATOM_NAMES.index("r_pi")] = 1.2
        a.values[F.ATOM_NAMES.index("r_pipi")] = 1.1
        a.values[F.ATOM_NAMES.index("r_s")] = 1.4
    bonds = [F.Entry((i, j), [100.0 + i + j] + [0.5] * 15) for i in range(1, n + 1) for j in range(i, n + 1)]
    off = [F.Entry((1, 2), [0.1, 1.8, 10.0, 1.1, 1.0, 0.9])] if n > 1 else []
    angles = [F.Entry((2, 1, 2), [70.0, 20.0, 2.0, 0.0, 0.1, 0.0, 1.05])] if n > 1 else []
    tors = [F.Entry((0, 1, 1, 0), [0.0, 30.0, 0.5, -5.0, -2.0, 0.0, 0.0])]
    hb = []
    gen = [float(i) for i in range(39)]
    gen[29] = 0.1
    return F.ForceField(header="synthetic test force field", general=gen, general_comments=[""] * 39, atoms=atoms,
                        bonds=bonds, offdiag=off, angles=angles, torsions=tors, hbonds=hb)
