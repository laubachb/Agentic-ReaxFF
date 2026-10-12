"""Regressions for bugs found in the first code review."""

import argparse

import numpy as np
import pytest

from agentic_reaxff.io import ffield as F
from agentic_reaxff.io import trainset as T

from conftest import make_ffield


def test_hbond_entries_are_directional():
    ff = make_ffield(("C", "H", "O"))
    ff.hbonds = [F.Entry((3, 2, 1), [1.1, -1.0, 2.0, 3.0]), F.Entry((1, 2, 3), [2.2, -2.0, 2.0, 3.0])]
    assert ff.get("hbond:O-H-C:r0_hb") == pytest.approx(1.1)
    assert ff.get("hbond:C-H-O:r0_hb") == pytest.approx(2.2)
    ff.set("hbond:C-H-O:r0_hb", 9.0)
    assert ff.hbonds[0].values[0] == pytest.approx(1.1)
    ids = [i for i in ff.param_ids(["hbond"]) if i.endswith("r0_hb")]
    assert sorted(ids) == ["hbond:C-H-O:r0_hb", "hbond:O-H-C:r0_hb"]


def test_split_never_shares_structures():
    from agentic_reaxff.stages.trainset_build import split

    items = []
    for g in range(6):
        names = [f"g{g}_{k}" for k in range(4)]
        items += [T.Item(kind="energy", sigma=1, group=f"g{g}", id=f"E{g}{k}", terms=[[n, 1], [names[0], -1]])
                  for k, n in enumerate(names[1:])]
        items.append(T.Item(kind="forces", sigma=1, group="forces", id=f"F{g}", structure=names[1]))
    items.append(T.Item(kind="energy", sigma=1, group="reactions", id="R", terms=[["g0_1", 1], ["g5_1", -1]]))
    for by in ("group", "frame"):
        for seed in range(5):
            tr, va = split(items, 0.3, by, seed)
            s_tr = {n for it in tr for n in it.structures()}
            s_va = {n for it in va for n in it.structures()}
            assert not (s_tr & s_va) and len(tr) + len(va) == len(items)


def test_slab_with_zero_vector_is_replicated():
    from agentic_reaxff.engine.evaluator import replication

    cell = np.array([[2.5, 0, 0], [0, 2.5, 0], [0, 0, 0]])
    reps = replication(cell, (True, True, False), 10.0)
    assert reps[0] >= 4 and reps[1] >= 4 and reps[2] == 1


def test_fit_fingerprint_tracks_inputs(tmp_path):
    from agentic_reaxff.stages.fit import fingerprint

    for name, text in (("t.yaml", "a"), ("p.yaml", "b")):
        (tmp_path / name).write_text(text)
    a = argparse.Namespace(trainset=str(tmp_path / "t.yaml"), validset=None, params=str(tmp_path / "p.yaml"),
                           optimizer="cmaes", restarts=4, keep=2, short_evals=10, max_evals=20, starts="mixed", sigma0=0.25,
                           popsize=None, mcff_iter=1, soppe_step=0.05, polish=None, polish_evals=1, patience_evals=None,
                           stall_evals=None, seed=0)
    f1 = fingerprint(a)
    (tmp_path / "t.yaml").write_text("changed")
    assert fingerprint(a) != f1
    a.restarts = 5
    assert fingerprint(a) != f1


def test_in_process_pool_does_not_reuse_stale_state():
    from agentic_reaxff.engine import pool as P
    from agentic_reaxff.engine.evaluator import EvalSettings

    p1 = P.EvalPool([], EvalSettings(), {}, {}, n_workers=1)
    p1._ensure()
    owner1 = P._W.get("owner")
    p1.close()
    p2 = P.EvalPool([], EvalSettings(), {}, {}, n_workers=1)
    p2._ensure()
    assert P._W.get("owner") != owner1
    p2.close()
