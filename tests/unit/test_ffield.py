import numpy as np
import pytest

from agentic_reaxff.io import ffield as F

from conftest import make_ffield


def test_roundtrip_values_and_ids():
    ff = make_ffield(("C", "H", "O"))
    text = ff.to_string()
    ff2 = F.parse_string(text)
    ids = ff.param_ids(include_frozen=True)
    assert ids == ff2.param_ids(include_frozen=True)
    assert np.allclose(ff.get_many(ids), ff2.get_many(ids), atol=5e-5)
    assert ff2.elements == ["C", "H", "O"]


def test_param_ids_get_set_and_aliases():
    ff = make_ffield()
    ff.set("bond:H-C:De_s", 123.4567)          # order-independent pair key
    assert ff.get("bond:C-H:De_s") == pytest.approx(123.4567)
    assert ff.get("bond:C-H:de(sigma)") == pytest.approx(123.4567)  # van Duin label alias
    ff.set("gen:p_boc1", 42.0)
    assert ff.get("gen:#1") == 42.0
    assert ff.get("tors:X-C-C-X:V2") == 30.0
    with pytest.raises(F.FFieldError):
        ff.get("bond:C-Zz:De_s")


def test_frozen_parameters_are_not_listed():
    ff = make_ffield()
    ids = ff.param_ids()
    assert "atom:C:mass" not in ids and "atom:C:valency" not in ids and "gen:bo_cut100" not in ids
    assert "atom:C:mass" in ff.param_ids(include_frozen=True)


def test_precision_rounding_is_what_is_written():
    ff = make_ffield()
    ff.set("atom:C:r_s", 1.234567891)
    assert F.parse_string(ff.to_string()).get("atom:C:r_s") == pytest.approx(1.2346)
    assert F.parse_string(ff.to_string(precision=6)).get("atom:C:r_s") == pytest.approx(1.234568)


def test_wide_numbers_stay_whitespace_separated():
    ff = make_ffield()
    ff.set("bond:C-C:De_s", -12345.678)
    ff2 = F.parse_string(ff.to_string())
    assert ff2.get("bond:C-C:De_s") == pytest.approx(-12345.678, abs=1e-3)


def test_subset_reindexes():
    ff = make_ffield(("C", "H", "O"))
    sub = ff.subset(["H", "O"])
    assert sub.elements == ["H", "O"]
    assert sub.find_entry("bond", ["H", "O"]) is not None
    assert all(max(e.idx) <= 2 for e in sub.bonds)


def test_orphan_entries_kept_but_not_addressable():
    ff = make_ffield()
    ff.torsions.append(F.Entry((1, 5, 5, 1), [1.0] * 7))   # index 5 > 2 elements: LAMMPS skips it
    ids = ff.param_ids()
    assert not any("?5" in i for i in ids)
    assert len(F.parse_string(ff.to_string()).torsions) == len(ff.torsions)


def test_lammps_shipped_ffields_roundtrip(cho_ffield):
    from pathlib import Path

    for p in sorted(Path(cho_ffield).parent.glob("ffield*")):
        ff = F.read(p)
        ids = ff.param_ids(include_frozen=True)
        ff2 = F.parse_string(ff.to_string())
        assert np.allclose(ff.get_many(ids), ff2.get_many(ids)), p.name
