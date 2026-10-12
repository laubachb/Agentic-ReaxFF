import numpy as np
import pytest
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from agentic_reaxff.io import structures as SIO
from agentic_reaxff.io import trainset as T
from agentic_reaxff.io.units import EV_TO_KCALMOL


def _frames(tmp_path):
    out = []
    for k, d in enumerate([0.74, 0.9, 1.2]):
        a = Atoms("H2", positions=[[0, 0, 0], [0, 0, d]])
        a.calc = SinglePointCalculator(a, energy=-1.0 + 0.5 * k, forces=np.full((2, 3), 0.1 * k))
        a.info.update({"name": f"h2_{k}", "group": "h2", "mode": "single"})
        a.arrays["ref_charges"] = np.zeros(2)
        out.append(a)
    w = Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    w.calc = SinglePointCalculator(w, energy=-5.0)
    w.info.update({"name": "h2o", "group": "w", "mode": "relax"})
    out.append(w)
    from ase.io import write

    write(tmp_path / "s.extxyz", out)
    return tmp_path / "s.extxyz"


def _ts(tmp_path, items):
    spath = _frames(tmp_path)
    yaml.safe_dump({"version": 1, "structures": str(spath), "items": items}, open(tmp_path / "t.yaml", "w"))
    return tmp_path / "t.yaml"


def test_energy_item_reference_in_kcalmol(tmp_path):
    ts = T.load(_ts(tmp_path, [{"kind": "energy", "terms": [["h2_1", 1], ["h2_0", -1]], "sigma": 1.0}]))
    assert T.reference_values(ts, ts.items[0]) == pytest.approx(0.5 * EV_TO_KCALMOL)


def test_unbalanced_energy_refused(tmp_path):
    with pytest.raises(T.TrainsetError, match="balanced"):
        T.load(_ts(tmp_path, [{"kind": "energy", "terms": [["h2o", 1], ["h2_0", -1]]}]))
    ts = T.load(_ts(tmp_path, [{"kind": "energy", "terms": [["h2o", 1], ["h2_0", -1]], "allow_unbalanced": True}]))
    assert len(ts.items) == 1


def test_unknown_key_and_structure_refused(tmp_path):
    with pytest.raises(T.TrainsetError, match="unknown key"):
        T.load(_ts(tmp_path, [{"kind": "energy", "terms": [["h2_1", 1], ["h2_0", -1]], "sigmaa": 1}]))
    with pytest.raises(T.TrainsetError, match="not in"):
        T.load(_ts(tmp_path, [{"kind": "forces", "structure": "nope"}]))


def test_dihedral_residual_is_periodic():
    it = T.Item(kind="dihedral", sigma=1.0, structure="x", atoms=[0, 1, 2, 3])
    assert T.residuals(it, 179.0, -179.0)[0] == pytest.approx(-2.0)


def test_array_item_weight_is_split_over_components():
    it = T.Item(kind="forces", sigma=2.0, weight=3.0, structure="x")
    r = T.residuals(it, np.ones((2, 3)) * 2.0, np.zeros((2, 3)))
    assert T.item_loss(it, r) == pytest.approx(3.0 * 1.0)   # w * mean((2/2)^2)


def test_failed_prediction_gets_large_residual():
    it = T.Item(kind="energy", sigma=1.0, terms=[["a", 1]])
    assert T.residuals(it, None, 0.0)[0] == T.FAIL_RESIDUAL


def test_summary_kinds_groups_and_top_items(tmp_path):
    ts = T.load(_ts(tmp_path, [{"kind": "energy", "terms": [["h2_1", 1], ["h2_0", -1]], "sigma": 1.0, "group": "g"},
                               {"kind": "forces", "structure": "h2_2", "sigma": 1.0}]))
    preds = {ts.items[0].id: T.reference_values(ts, ts.items[0]) + 2.0, ts.items[1].id: None}
    s = T.summarize_errors(ts, preds)
    assert s["kinds"]["energy"]["rmse"] == pytest.approx(2.0)
    assert s["n_failed_items"] == 1
    assert s["top_items"][0]["id"] == ts.items[1].id


def test_structures_roundtrip_keeps_restraints_and_labels(tmp_path):
    a = Atoms("CH4", positions=np.random.default_rng(0).normal(size=(5, 3)))
    s = SIO.from_atoms(a)
    s.name, s.mode, s.restraints, s.energy = "m", "relax", [["bond", 0, 1, 1.2, 2000.0]], -3.0
    s.forces = np.zeros((5, 3))
    SIO.write(tmp_path / "x.extxyz", [s])
    s2 = SIO.read(tmp_path / "x.extxyz")[0]
    assert s2.restraints == [["bond", 0, 1, 1.2, 2000.0]] and s2.mode == "relax" and s2.energy == -3.0
