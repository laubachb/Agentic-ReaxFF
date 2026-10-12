"""The optimization runner shared by every optimizer.

An optimizer exposes ask/tell in normalized coordinates z in [0,1]^n:

    zs = opt.ask()            # candidates for this step
    opt.tell(zs, losses)      # their losses
    opt.stop()                # None, or why the optimizer itself is done

The runner owns everything else, identically for all optimizers:
  - evaluation through the Objective (parallel over candidates x structures)
  - history.jsonl (one line per step: losses, failures, timing, validation)
  - the best candidate: best.json + best.ffield, rewritten on improvement
  - validation-set early stopping (ParAMS-style patience in evaluations [P21],
    recommended by Shchygol et al. [Sh19] against overfitting)
  - checkpoint.pkl every step, so `--continue` resumes exactly (the job
    can be cut into 1 h chunks on a debug queue)
  - budgets: max evaluations, wall-time limit, target loss, stall limit
"""

from __future__ import annotations

import json
import math
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..engine.objective import Objective


@dataclass
class Budget:
    max_evals: int = 5000
    time_limit_s: Optional[float] = None
    target_loss: Optional[float] = None
    stall_evals: Optional[int] = None        # stop after this many evaluations without improvement
    validate_every: int = 1                  # optimizer steps between validation evaluations
    patience_evals: Optional[int] = None     # early stopping: evals without validation improvement


@dataclass
class RunState:
    n_evals: int = 0
    n_steps: int = 0
    best_loss: float = math.inf
    best_x: Optional[np.ndarray] = None
    best_evals: int = 0
    best_val_loss: float = math.inf
    best_val_x: Optional[np.ndarray] = None
    best_val_evals: int = 0
    last_val_x: Optional[np.ndarray] = None
    elapsed_s: float = 0.0
    status: str = "running"
    history_tail: list = field(default_factory=list)


class Optimizer:
    name = "base"

    def ask(self) -> list:
        raise NotImplementedError

    def tell(self, zs: list, losses: list) -> None:
        raise NotImplementedError

    def stop(self) -> Optional[str]:
        return None

    def describe(self) -> dict:
        return {"optimizer": self.name}


class Runner:
    def __init__(self, objective: Objective, optimizer: Optimizer, outdir, *, budget: Budget,
                 state: Optional[RunState] = None, log=print):
        self.obj = objective
        self.opt = optimizer
        self.out = Path(outdir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.budget = budget
        self.state = state or RunState()
        self.log = log

    # ---------------------------------------------------------- persistence
    def checkpoint(self) -> None:
        tmp = self.out / "checkpoint.pkl.tmp"
        payload = {"optimizer": self.opt, "state": self.state, "budget": self.budget,
                   "ids": self.obj.spec.ids, "lo": self.obj.spec.lo, "hi": self.obj.spec.hi}
        try:
            blob = pickle.dumps(payload)
        except TypeError:  # optimizer without resumable state (scipy): keep the run state only
            payload["optimizer"] = None
            blob = pickle.dumps(payload)
        tmp.write_bytes(blob)
        tmp.replace(self.out / "checkpoint.pkl")

    @staticmethod
    def load_checkpoint(outdir) -> Optional[dict]:
        p = Path(outdir) / "checkpoint.pkl"
        if not p.is_file():
            return None
        with open(p, "rb") as f:
            return pickle.load(f)

    def _write_best(self, x, loss, which="best") -> None:
        self.obj.write_ffield(x, self.out / f"{which}.ffield.tmp")
        os.replace(self.out / f"{which}.ffield.tmp", self.out / f"{which}.ffield")
        info = {"loss": loss, "n_evals": self.state.n_evals, "params": dict(zip(self.obj.spec.ids, map(float, x)))}
        tmp = self.out / f"{which}.json.tmp"
        tmp.write_text(json.dumps(info, indent=1))
        tmp.replace(self.out / f"{which}.json")

    def _history(self, rec: dict) -> None:
        with open(self.out / "history.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ------------------------------------------------------------------ run
    def run(self) -> RunState:
        st, b = self.state, self.budget
        t_start = time.time() - st.elapsed_s
        session_start = time.time()
        last_step_s = 0.0
        st.status = "running"
        while True:
            reason = self._should_stop(session_start, last_step_s)
            if reason:
                st.status = reason
                break
            t0 = time.time()
            zs = self.opt.ask()
            if not zs:
                st.status = f"optimizer:{self.opt.stop() or 'done'}"
                break
            evs = self.obj.evaluate_z(zs)
            losses = [e.loss for e in evs]
            self.opt.tell(zs, losses)
            st.n_evals += len(zs)
            st.n_steps += 1
            k = int(np.argmin(losses))
            improved = losses[k] < st.best_loss
            if improved:
                st.best_loss, st.best_x, st.best_evals = float(losses[k]), evs[k].x.copy(), st.n_evals
                self._write_best(st.best_x, st.best_loss)
            rec = {"step": st.n_steps, "n_evals": st.n_evals, "best": st.best_loss, "step_min": float(losses[k]),
                   "step_median": float(np.median(losses)), "n_failed_min": int(min(e.n_failed for e in evs)),
                   "frac_failed": float(np.mean([e.n_failed > 0 for e in evs])), "t_step": round(time.time() - t0, 3)}
            rec.update(self.opt.step_info() if hasattr(self.opt, "step_info") else {})
            if self.obj.validation is not None and st.best_x is not None and \
                    (st.n_steps % max(1, b.validate_every) == 0) and \
                    (st.last_val_x is None or not np.array_equal(st.last_val_x, st.best_x)):
                v = self.obj.evaluate_x([st.best_x], validation=True)[0]
                st.last_val_x = st.best_x.copy()
                rec["val"] = v.loss
                if v.loss < st.best_val_loss:
                    st.best_val_loss, st.best_val_x, st.best_val_evals = v.loss, st.best_x.copy(), st.n_evals
                    self._write_best(st.best_val_x, v.loss, which="best_val")
            st.elapsed_s = time.time() - t_start
            last_step_s = time.time() - t0
            self._history(rec)
            st.history_tail = (st.history_tail + [rec])[-5:]
            if st.n_steps % 10 == 0 or improved:
                self.log(f"step {st.n_steps} evals {st.n_evals} best {st.best_loss:.6g} step_min {losses[k]:.6g}")
            self.checkpoint()
        st.elapsed_s = time.time() - t_start
        self.checkpoint()
        return st

    def _should_stop(self, session_start, last_step_s) -> Optional[str]:
        st, b = self.state, self.budget
        if st.n_evals >= b.max_evals:
            return "max_evals"
        if b.target_loss is not None and st.best_loss <= b.target_loss:
            return "target_loss"
        if b.stall_evals and st.best_x is not None and st.n_evals - st.best_evals >= b.stall_evals:
            return "stalled"
        if b.patience_evals and st.best_val_x is not None and st.n_evals - st.best_val_evals >= b.patience_evals:
            return "early_stopped"
        if b.time_limit_s is not None:
            # stop if the next step would probably not finish in time
            if time.time() - session_start + 1.5 * last_step_s >= b.time_limit_s:
                return "time_limit"
        reason = self.opt.stop()
        if reason:
            return f"optimizer:{reason}"
        return None
