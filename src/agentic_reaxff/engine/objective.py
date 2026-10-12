"""The loss a fit minimizes: active parameters in, weighted sum of squares out.

    candidate z in [0,1]^n  ->  x = lo + z (hi - lo)  ->  rounded to the
    written precision  ->  ffield text  ->  LAMMPS on every training
    structure  ->  item predictions  ->  L = Σ w·mean(((pred-ref)/σ)²)

Constraint violations (e.g. r_s < r_pi) and failed structures (NaN, LAMMPS
error, QEq not converging) are not skipped: a failed structure gives every
item that uses it a residual of FAIL_RESIDUAL σ, so candidates are ranked
by how many structures they break before how well they fit. CMA-ES only
uses ranks, so the huge numbers do not distort it [Sh19 used the largest
float for out-of-bounds points].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..io import ffield as F
from ..io import trainset as T
from ..io.params import ParamSpec
from .evaluator import EvalSettings
from .pool import EvalPool

CONSTRAINT_PENALTY = 1.0e10


@dataclass
class Evaluation:
    loss: float
    x: np.ndarray
    n_failed: int = 0
    violation: float = 0.0
    failed: list = field(default_factory=list)
    errors: dict = field(default_factory=dict)   # structure -> error message (first few)
    preds: Optional[dict] = None                 # item id -> prediction (when details requested)
    results: Optional[dict] = None               # structure -> EvalResult (when details requested)


class Objective:
    def __init__(self, trainset: T.TrainingSet, spec: ParamSpec, *, settings: Optional[EvalSettings] = None,
                 n_workers: Optional[int] = None, validation: Optional[T.TrainingSet] = None):
        self.ts = trainset
        self.spec = spec
        self.template = F.read(spec.ffield)
        missing = [e for e in trainset.elements() if e.lower() not in [x.lower() for x in self.template.elements]]
        if missing:
            raise ValueError(f"training structures contain {missing}, which the ffield {spec.ffield} does not define")
        self.settings = settings or EvalSettings.from_dict(trainset.evaluator)
        self.validation = validation
        structs = trainset.used_structures()
        needs = trainset.needed()
        if validation is not None:
            for name, n in validation.needed().items():
                cur = needs.setdefault(name, {"forces": False, "geometry": False})
                cur["forces"] |= n["forces"]
                cur["geometry"] |= n["geometry"]
            have = {s.name for s in structs}
            structs += [s for s in validation.used_structures() if s.name not in have]
        masses = {a.symbol: a.values[2] for a in self.template.atoms}
        self.pool = EvalPool(structs, self.settings, needs, masses, n_workers=n_workers)
        self.refs = {it.id: T.reference_values(trainset, it) for it in trainset.items}
        self.val_refs = {it.id: T.reference_values(validation, it) for it in validation.items} if validation else {}
        self.train_names = list(trainset.needed())
        self.val_names = list(validation.needed()) if validation else []
        self.n_evals = 0

    @property
    def n_params(self) -> int:
        return len(self.spec.params)

    def close(self):
        self.pool.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------ helpers
    def ffield_text(self, x) -> str:
        return self.spec.apply(self.template, x).to_string(precision=self.spec.precision)

    def write_ffield(self, x, path) -> None:
        self.spec.apply(self.template, x).write(path, precision=self.spec.precision)

    def _loss(self, ts, refs, results, *, details):
        preds = {}
        loss = 0.0
        for it in ts.items:
            p = T.predicted_values(ts, it, results)
            r = T.residuals(it, p, refs[it.id])
            loss += T.item_loss(it, r)
            if details:
                preds[it.id] = p
        return loss, preds

    # ------------------------------------------------------------ evaluate
    def evaluate_x(self, X, *, details: bool = False, validation: bool = False) -> list:
        """Evaluate parameter vectors (physical units). With validation=True
        the validation set is evaluated instead of the training set."""
        X = [self.spec.round_x(x) for x in X]
        out: list = [None] * len(X)
        todo = []
        for i, x in enumerate(X):
            v = self.spec.violation(x)
            if v > 0:
                out[i] = Evaluation(loss=CONSTRAINT_PENALTY * (1.0 + v), x=x, violation=v)
            else:
                todo.append(i)
        if todo:
            names = self.val_names if validation else self.train_names
            texts = [self.ffield_text(X[i]) for i in todo]
            all_res = self.pool.evaluate(texts, names)
            ts = self.validation if validation else self.ts
            refs = self.val_refs if validation else self.refs
            for i, res in zip(todo, all_res):
                loss, preds = self._loss(ts, refs, res, details=details)
                failed = [n for n, r in res.items() if not r.ok]
                errs = {n: res[n].error for n in failed[:5]}
                out[i] = Evaluation(loss=loss, x=X[i], n_failed=len(failed), failed=failed, errors=errs,
                                    preds=preds if details else None, results=res if details else None)
        self.n_evals += len(X)
        return out

    def evaluate_z(self, Z, **kw) -> list:
        return self.evaluate_x([self.spec.to_x(z) for z in Z], **kw)

    def report(self, x, *, validation: bool = False) -> dict:
        """Per-kind and per-group error summary for one parameter vector."""
        ev = self.evaluate_x([x], details=True, validation=validation)[0]
        ts = self.validation if validation else self.ts
        summ = T.summarize_errors(ts, ev.preds or {})
        summ["n_failed_structures"] = ev.n_failed
        summ["failed_structures"] = ev.failed[:20]
        summ["errors"] = ev.errors
        return summ
