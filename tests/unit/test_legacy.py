import numpy as np
import pytest
from ase import Atoms

from agentic_reaxff.io import legacy as L
from agentic_reaxff.io import params as P
from agentic_reaxff.io import trainset as T
from agentic_reaxff.io.structures import Structure, from_atoms

from conftest import make_ffield


def _structs():
    a = from_atoms(Atoms("CH4", positions=[[0, 0, 0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63], [-0.63, 0.63, -0.63],
                                            [0.63, -0.63, -0.63]]))
    a.name, a.mode, a.restraints = "ch4", "relax", [["bond", 0, 1, 1.2, 2000.0]]
    a.energy = -1.0
    a.forces = np.ones((5, 3)) * 0.01
    b = from_atoms(Atoms("CH4", positions=a.positions * 1.05))
    b.name, b.energy = "ch4-big", -0.5
    c = from_atoms(Atoms("C2", positions=[[0, 0, 0], [0.9, 0.9, 0.9]], cell=[[0, 1.78, 1.78], [1.78, 0, 1.78], [1.78, 1.78, 0]],
                         pbc=True))
    c.name, c.mode, c.energy = "diamond", "relax_cell", -2.0
    return [a, b, c]


def test_geo_and_trainset_roundtrip_preserves_loss_terms(tmp_path):
    ss = _structs()
    ts = T.TrainingSet(items=[
        T.Item(kind="energy", sigma=2.0, terms=[["ch4-big", 1], ["ch4", -1]], id="e1"),
        T.Item(kind="forces", sigma=5.0, weight=2.0, structure="ch4", id="f1"),
        T.Item(kind="distance", sigma=0.02, structure="ch4", atoms=[0, 2], id="d1"),
        T.Item(kind="lattice", sigma=0.05, structure="diamond", quantity="a", id="l1"),
    ], structures={s.name: s for s in ss}, structures_path="x")
    names = L.write_geo(ss, tmp_path / "geo")
    assert names["ch4-big"] == "ch4_big"           # '-' is an operator in trainset.in
    warn = L.write_trainset_in(ts, names, tmp_path / "trainset.in")
    assert warn == []
    back = {s.name: s for s in L.read_geo(tmp_path / "geo")}
    assert back["ch4"].mode == "relax" and back["diamond"].mode == "relax_cell"
    assert back["ch4"].restraints[0][:4] == ["bond", 0, 1, 1.2]
    assert back["ch4"].restraints[0][4] == pytest.approx(2000.0)
    assert abs(np.linalg.det(back["diamond"].cell)) == pytest.approx(abs(np.linalg.det(ss[2].cell)), rel=1e-4)
    items, warn = L.read_trainset_in(tmp_path / "trainset.in", back)
    kinds = sorted(it.kind for it in items)
    assert kinds == ["distance", "energy", "forces", "lattice"]
    f = next(it for it in items if it.kind == "forces")
    # sigma*sqrt(M/w) per component, regrouped with weight = M: same loss contribution
    orig = T.Item(kind="forces", sigma=5.0, weight=2.0, structure="ch4")
    r0 = T.residuals(orig, np.zeros((5, 3)), np.ones((5, 3)))
    r1 = T.residuals(f, np.zeros((5, 3)), np.ones((5, 3)))
    assert T.item_loss(f, r1) == pytest.approx(T.item_loss(orig, r0), rel=1e-3)


def test_params_file_indices():
    ff = make_ffield(("C", "H"))
    s, b, i = L.param_index(ff, "atom:H:chi")
    assert (s, b, i) == (2, 2, 14)
    s, b, i = L.param_index(ff, "bond:C-H:p_bo2")
    assert s == 3 and i == 14 and ff.bonds[b - 1].idx == (1, 2)
    assert L.param_index(ff, "gen:p_boc2")[:2] == (1, 2)


def test_params_file_roundtrip(tmp_path):
    ff = make_ffield(("C", "H"))
    ff.write(tmp_path / "ffield")
    spec = P.ParamSpec(ffield=str(tmp_path / "ffield"), params=P.make_bounds(ff, ["atom:C:r_s", "bond:C-H:De_s"]))
    L.write_params(spec, ff, tmp_path / "params")
    back = L.read_params(tmp_path / "params", ff)
    assert [p.id for p in back] == ["atom:C:r_s", "bond:C-H:De_s"]
    assert back[0].min == pytest.approx(spec.params[0].min, abs=1e-4)
