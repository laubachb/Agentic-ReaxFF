import numpy as np
import pytest

from agentic_reaxff.io import params as P

from conftest import make_ffield


def test_preset_selection_respects_frozen_and_zero():
    ff = make_ffield()
    ff.set("bond:C-H:De_pp", 0.0)
    ids = P.select(ff, presets=["bond"])
    assert "bond:C-H:De_s" in ids
    assert "bond:C-H:De_pp" not in ids          # zero-valued: switched off, left alone
    assert not any(i.endswith(":mass") or i.endswith(":valency") for i in ids)
    assert "bond:C-H:De_pp" in P.select(ff, presets=["bond"], include_zero=True)


def test_pi_radius_switched_off_stays_off():
    ff = make_ffield()
    ff.set("atom:H:r_pi", -0.1)
    assert "atom:H:r_pi" not in P.select(ff, presets=["bond"])


def test_element_and_data_filters():
    ff = make_ffield(("C", "H", "O"))
    ids = P.select(ff, presets=["bond"], elements=["O"])
    assert ids and all("O" in i for i in ids if not i.startswith("gen"))
    ids = P.select(ff, presets=["bond"], data_elements=[{"C", "H"}])
    assert not any("O" in i.split(":")[1] for i in ids if i.startswith("bond"))


def test_relative_bounds_and_hard_limits():
    ff = make_ffield()
    ff.set("bond:C-H:p_bo2", 1.1)
    ps = P.make_bounds(ff, ["bond:C-H:De_s", "bond:C-H:p_bo2"], rule="relative", scale=0.2)
    de, bo2 = ps
    assert de.min == pytest.approx(0.8 * de.value) and de.max == pytest.approx(1.2 * de.value)
    assert bo2.min == pytest.approx(1.0)        # exponent >= 1 [P21]


def test_library_bounds_contain_start():
    ff = make_ffield()
    lib = {"params": {"*:bond:De_s": {"min": 50.0, "max": 60.0}}}
    p = P.make_bounds(ff, ["bond:C-H:De_s"], rule="library", library=lib)[0]
    assert p.min <= p.value <= p.max and p.min < 50.0


def test_constraints_with_coefficient_and_violation():
    ff = make_ffield()
    cons = P.parse_constraints(["atom:C:eta >= 7.2 * atom:C:gamma"], ff)
    spec = P.ParamSpec(ffield="x", params=[P.Param("atom:C:eta", 8.0, 1.0, 20.0), P.Param("atom:C:gamma", 0.9, 0.1, 2.0)],
                       constraints=cons)
    assert spec.violation([8.0, 0.9]) == 0.0
    assert spec.violation([6.0, 1.0]) == pytest.approx(1.2)


def test_default_constraints_skip_ones_the_start_violates():
    ff = make_ffield()
    ff.set("atom:C:r_pi", 1.0)
    ff.set("atom:C:r_pipi", 1.2)                 # like Chenoweth 2008 carbon
    ids = ["atom:C:r_s", "atom:C:r_pi", "atom:C:r_pipi"]
    assert "atom:C:r_pi >= atom:C:r_pipi" not in P.default_constraints(ff, ids)
    assert "atom:C:r_pi >= atom:C:r_pipi" in P.skipped_constraints(ff, ids)
    assert "atom:C:r_s >= atom:C:r_pi" in P.default_constraints(ff, ids)


def test_normalized_coordinates_roundtrip():
    spec = P.ParamSpec(ffield="x", params=[P.Param("a", 1.0, 0.0, 2.0), P.Param("b", -3.0, -4.0, -1.0)])
    x = np.array([1.5, -2.0])
    assert np.allclose(spec.to_x(spec.to_z(x)), x)
    assert np.allclose(spec.to_x([1.5, -0.5]), [2.0, -4.0])   # z outside [0,1] is clipped to the box
