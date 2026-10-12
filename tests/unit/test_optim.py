"""Optimizers and the runner on an analytic objective (no LAMMPS)."""

import numpy as np
import pytest

from agentic_reaxff.io.params import Param, ParamSpec
from agentic_reaxff.optim.base import Budget, Runner
from agentic_reaxff.optim.optimizers import CMAES, MCFF, SOPPE, ScipyLocal, default_popsize


class _Ev:
    def __init__(self, loss, x):
        self.loss, self.x, self.n_failed, self.violation = loss, x, 0, 0.0


class FakeObjective:
    """Rosenbrock-ish bowl in physical units with a minimum at x*."""

    def __init__(self, n=4, validation=False):
        self.spec = ParamSpec(ffield="x", params=[Param(f"p{i}", 0.0, -2.0, 2.0) for i in range(n)])
        self.target = np.linspace(-1, 1, n) * 0.7
        self.validation = True if validation else None
        self.n = 0

    def _f(self, x):
        return float(np.sum((x - self.target) ** 2) * 100.0)

    def evaluate_z(self, Z, **kw):
        return self.evaluate_x([self.spec.to_x(z) for z in Z])

    def evaluate_x(self, X, validation=False, **kw):
        self.n += len(X)
        return [_Ev(self._f(np.asarray(x)) * (1.1 if validation else 1.0), np.asarray(x)) for x in X]

    def write_ffield(self, x, path):
        open(path, "w").write(" ".join(map(str, x)))


def test_default_popsize():
    assert default_popsize(15) == 4 + int(np.floor(3 * np.log(15)))


@pytest.mark.parametrize("make,evals,tol", [
    (lambda z0, spec: CMAES(z0, sigma0=0.25, seed=1), 3000, 1e-3),
    (lambda z0, spec: MCFF(z0, n_iter=3000, seed=1, replicas=4), 13000, 2.0),
    (lambda z0, spec: SOPPE(z0, spec.to_x, step=0.2, n_sweeps=40), 3000, 1e-2),
    (lambda z0, spec: ScipyLocal(z0, method="nelder-mead", max_evals=3000), 3000, 1e-3),
])
def test_optimizers_reduce_loss(tmp_path, make, evals, tol):
    obj = FakeObjective()
    z0 = np.full(4, 0.5)
    st = Runner(obj, make(z0, obj.spec), tmp_path, budget=Budget(max_evals=evals), log=lambda m: None).run()
    assert st.best_loss < tol
    assert (tmp_path / "best.ffield").exists() and (tmp_path / "history.jsonl").exists()


def test_checkpoint_resume_continues(tmp_path):
    obj = FakeObjective()
    r = Runner(obj, CMAES(np.full(4, 0.5), seed=3), tmp_path, budget=Budget(max_evals=200), log=lambda m: None)
    st = r.run()
    assert st.status == "max_evals"
    ck = Runner.load_checkpoint(tmp_path)
    r2 = Runner(obj, ck["optimizer"], tmp_path, budget=Budget(max_evals=1500), state=ck["state"], log=lambda m: None)
    st2 = r2.run()
    assert st2.n_evals > st.n_evals and st2.best_loss <= st.best_loss


def test_time_limit_stops_cleanly(tmp_path):
    obj = FakeObjective()
    st = Runner(obj, MCFF(np.full(4, 0.5), n_iter=10**9), tmp_path, budget=Budget(max_evals=10**9, time_limit_s=0.5),
                log=lambda m: None).run()
    assert st.status == "time_limit" and (tmp_path / "checkpoint.pkl").exists()


def test_validation_tracking_and_early_stopping(tmp_path):
    obj = FakeObjective(validation=True)
    st = Runner(obj, CMAES(np.full(4, 0.5)), tmp_path, budget=Budget(max_evals=5000, patience_evals=50),
                log=lambda m: None).run()
    assert (tmp_path / "best_val.ffield").exists()
    assert st.best_val_loss < np.inf
