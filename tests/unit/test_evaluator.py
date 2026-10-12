"""LAMMPS evaluator numerics (skip without LAMMPS / the ffield library)."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from agentic_reaxff.engine.evaluator import EvalSettings, Evaluator
from agentic_reaxff.io.structures import from_atoms

CH4_REF = -463.1717274896  # kcal/mol, ase CH4, ffield.reax.cho, QEq tol 1e-8


def _s(atoms, name="x", **info):
    atoms = atoms.copy()
    atoms.info.update({"name": name, **info})
    return from_atoms(atoms)


@pytest.fixture(scope="module")
def ev(lammps_available):
    e = Evaluator()
    yield e
    e.close()


def test_reference_single_point(ev, cho_ffield):
    r = ev.evaluate(cho_ffield, _s(molecule("CH4")))
    assert r.ok and r.energy == pytest.approx(CH4_REF, abs=1e-4)
    assert r.charges.sum() == pytest.approx(0.0, abs=1e-6)


def test_qeq_fix_after_atoms_no_random_nan(ev, cho_ffield):
    """Regression: defining fix qeq/reaxff before create_atoms gave NaN or
    ~1e200 energies in about half of all single points."""
    base = molecule("CH4")
    energies = []
    for k in range(40):
        a = base.copy()
        a.positions *= 1.0 + 0.012 * k
        r = ev.evaluate(cho_ffield, _s(a))
        assert r.ok, r.error
        energies.append(r.energy)
    assert np.all(np.isfinite(energies)) and max(abs(e) for e in energies) < 1e4


def test_forces_match_finite_differences(cho_ffield):
    """Without charge equilibration LAMMPS ReaxFF forces are the exact
    energy gradient. With fix qeq/reaxff they differ from -dE/dx by ~1%
    (independent of step and QEq tolerance: a property of the LAMMPS
    implementation, measured while building this). Fits compare against
    the forces MD actually uses, so this is recorded, not corrected."""
    m = molecule("CH3OH")
    h = 1e-4
    for method, rel in (("none", 1e-3), ("qeq", 0.03)):
        ev = Evaluator(EvalSettings(charge_method=method))
        r = ev.evaluate(cho_ffield, _s(m))
        for atom, d in ((1, 0), (3, 2)):
            p, q = m.copy(), m.copy()
            p.positions[atom, d] += h
            q.positions[atom, d] -= h
            fd = -(ev.evaluate(cho_ffield, _s(p)).energy - ev.evaluate(cho_ffield, _s(q)).energy) / (2 * h)
            assert r.forces[atom, d] == pytest.approx(fd, rel=rel, abs=1e-3), method
        ev.close()


def test_rotation_invariance(ev, cho_ffield):
    m = molecule("C2H6")
    r1 = ev.evaluate(cho_ffield, _s(m))
    m2 = m.copy()
    m2.rotate(33, "x")
    m2.rotate(71, "z")
    r2 = ev.evaluate(cho_ffield, _s(m2))
    assert r1.energy == pytest.approx(r2.energy, abs=1e-6)


def test_small_periodic_cells_are_replicated(ev, cho_ffield):
    """QEq energies are wrong in cells thinner than the 10 Å cutoff; the
    evaluator replicates them, so 1x1x1 and 2x2x2 agree per copy."""
    w = molecule("H2O")
    w.set_cell([[4.0, 0, 0], [0.3, 4.4, 0], [0.2, 0.1, 3.8]])
    w.pbc = True
    w.center()
    r1 = ev.evaluate(cho_ffield, _s(w))
    r2 = ev.evaluate(cho_ffield, _s(w * (2, 2, 2)))
    assert r1.replicas > 1
    assert r1.energy == pytest.approx(r2.energy / 8, abs=1e-3)
    assert r1.forces.shape == (3, 3)


def test_flat_molecules_evaluate(ev, cho_ffield):
    for name in ("C2H4", "C2H2", "H2O"):
        r = ev.evaluate(cho_ffield, _s(molecule(name)))
        assert r.ok, (name, r.error)


def test_restrained_relaxation_holds_the_bond(ev, cho_ffield):
    s = _s(molecule("CH4"), mode="relax")
    s.restraints = [["bond", 0, 1, 1.6, 4000.0]]
    r = ev.evaluate(cho_ffield, s)
    assert r.ok
    assert np.linalg.norm(r.positions[1] - r.positions[0]) == pytest.approx(1.6, abs=0.03)
    free = ev.evaluate(cho_ffield, _s(molecule("CH4"), mode="relax"))
    assert r.energy > free.energy + 10.0          # stretched: higher energy, restraint energy excluded


def test_failures_are_reported_not_raised(ev, tmp_path):
    bad = tmp_path / "ffield"
    bad.write_text("not a force field\n")
    r = ev.evaluate(str(bad), _s(molecule("CH4")))
    assert not r.ok and r.error
