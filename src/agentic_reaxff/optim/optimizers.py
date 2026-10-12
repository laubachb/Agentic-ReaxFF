"""Optimizers for ReaxFF parameters, all in normalized coordinates z in [0,1]^n.

cmaes   Covariance Matrix Adaptation ES (pycma), the recommended default.
        Shchygol et al. [Sh19] found it reached the lowest loss in 2 of 3
        benchmark sets; ParAMS [P21] uses it by default. Defaults follow them:
        population 4 + floor(3 ln n) [Sh19, P21], sigma0 = 0.25 of each
        range ([Sh19]: initial std = range/4; ParAMS' ReaxFF example used 0.3
        with ±20% bounds). Many short runs from different seeds, continuing
        only the best, beat one long run [Sh19] (`fit --restarts/--keep`).

mcff    Metropolis Monte Carlo with simulated annealing, after Iype et al.
        [I13] with the settings Shchygol et al. used [Sh19]: each move
        perturbs 10% of the parameters uniformly within ±s% of their range
        (s starts at 1, x/÷1.05 to hold acceptance in 50-70%), beta raised
        geometrically from sqrt(n/(2·C1·L0)) to sqrt(n/(2·C2)), C1=1, C2=5.
        With several workers, `replicas` moves are proposed per step and the
        best is put to the Metropolis test (the SCM MCFF `replic` option).

soppe   van Duin's successive one-parameter parabolic extrapolation [vD94],
        the method most published ReaxFF force fields were fitted with [S16]:
        for each parameter evaluate p-δ, p+δ (δ relative to |p|), fit a
        parabola through the three losses, move to its minimum (extrapolation
        limited, clamped to bounds) if that lowers the loss, else keep the best
        of the three. Cheap and transparent, but local and blind to parameter
        correlation [S16, L13]: use it to polish, not to explore.

nelder-mead / powell
        scipy's derivative-free local methods (bounded), for a final polish.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .base import Optimizer


# ----------------------------------------------------------------- CMA-ES
def default_popsize(n: int) -> int:
    return 4 + int(math.floor(3 * math.log(max(n, 1))))


class CMAES(Optimizer):
    name = "cmaes"

    def __init__(self, z0, *, sigma0: float = 0.25, popsize: Optional[int] = None, seed: int = 0,
                 tolx: float = 1e-5, tolfun: float = 1e-11, max_evals: Optional[int] = None):
        import cma

        n = len(z0)
        self.popsize = int(popsize or default_popsize(n))
        opts = {"bounds": [0.0, 1.0], "popsize": self.popsize, "seed": int(seed) + 1, "verbose": -9,
                "tolx": tolx, "tolfun": tolfun, "CMA_diagonal": False}
        if max_evals:
            opts["maxfevals"] = int(max_evals)
        if n == 1:
            # pycma needs n >= 2; pad with a dummy dimension
            self._pad = True
            z0 = list(z0) + [0.5]
        else:
            self._pad = False
        self.es = cma.CMAEvolutionStrategy(list(np.clip(z0, 0, 1)), float(sigma0), opts)
        self._last = None

    def ask(self):
        zs = self.es.ask()
        self._last = zs
        return [np.asarray(z[:-1] if self._pad else z) for z in zs]

    def tell(self, zs, losses):
        self.es.tell(self._last, list(map(float, losses)))

    def stop(self):
        s = self.es.stop()
        s = {k: v for k, v in s.items() if k not in ("maxfevals",)}
        return ",".join(sorted(s)) if s else None

    def step_info(self):
        return {"sigma": float(self.es.sigma)}

    def describe(self):
        return {"optimizer": self.name, "popsize": self.popsize, "sigma": float(self.es.sigma)}


# ------------------------------------------------------------------- MCFF
class MCFF(Optimizer):
    name = "mcff"

    def __init__(self, z0, *, n_iter: int = 3000, frac_moved: float = 0.1, step_pct: float = 1.0,
                 c1: float = 1.0, c2: float = 5.0, replicas: int = 1, seed: int = 0,
                 acc_lo: float = 0.5, acc_hi: float = 0.7):
        self.rng = np.random.default_rng(seed)
        self.z = np.clip(np.asarray(z0, float), 0, 1)
        self.n = len(self.z)
        self.n_iter, self.frac, self.s = int(n_iter), float(frac_moved), float(step_pct)
        self.c1, self.c2 = c1, c2
        self.replicas = max(1, int(replicas))
        self.loss = None
        self.beta = None
        self.beta_factor = None
        self.it = 0
        self.acc_window = []
        self.acc_lo, self.acc_hi = acc_lo, acc_hi
        self._first = True

    def _propose(self):
        z = self.z.copy()
        k = max(1, int(round(self.frac * self.n)))
        idx = self.rng.choice(self.n, size=k, replace=False)
        z[idx] += self.rng.uniform(-1, 1, size=k) * self.s / 100.0
        # reflect into the box
        z = np.abs(z)
        z = np.where(z > 1, 2 - z, z)
        return np.clip(z, 0, 1)

    def ask(self):
        if self._first:
            return [self.z.copy()]
        return [self._propose() for _ in range(self.replicas)]

    def tell(self, zs, losses):
        losses = np.asarray(losses, float)
        if self._first:
            self._first = False
            self.loss = float(losses[0])
            l0 = max(self.loss, 1e-12)
            self.beta = math.sqrt(self.n / (2 * self.c1 * l0))
            beta_end = math.sqrt(self.n / (2 * self.c2))
            self.beta_factor = (beta_end / self.beta) ** (1.0 / max(self.n_iter, 1))
            return
        k = int(np.argmin(losses))
        d = losses[k] - self.loss
        accept = d <= 0 or self.rng.random() < math.exp(-self.beta * d)
        if accept:
            self.z, self.loss = np.asarray(zs[k], float), float(losses[k])
        self.acc_window = (self.acc_window + [accept])[-50:]
        if len(self.acc_window) >= 20:
            rate = float(np.mean(self.acc_window))
            if rate > self.acc_hi:
                self.s *= 1.05
            elif rate < self.acc_lo:
                self.s /= 1.05
            self.s = float(np.clip(self.s, 1e-3, 50.0))
        self.beta *= self.beta_factor
        self.it += 1

    def stop(self):
        return "n_iter" if self.it >= self.n_iter else None

    def step_info(self):
        return {"beta": self.beta, "step_pct": self.s, "acceptance": float(np.mean(self.acc_window)) if self.acc_window else None}


# ------------------------------------------------------------------ SOPPE
class SOPPE(Optimizer):
    name = "soppe"

    def __init__(self, z0, x_of_z, *, step: float = 0.05, max_extrapolation: float = 2.0, n_sweeps: int = 3,
                 shrink: float = 0.5, min_step: float = 1e-4, order: Optional[list] = None):
        """step: relative change δ = step·|x| (classic ReaxFF params-file
        convention; 0.005-0.1 typical); converted to z units per parameter."""
        self.z = np.clip(np.asarray(z0, float), 0, 1)
        self.n = len(self.z)
        self.x_of_z = x_of_z
        self.step = float(step)
        self.max_extra = float(max_extrapolation)
        self.n_sweeps, self.shrink, self.min_step = n_sweeps, shrink, min_step
        self.order = list(order) if order is not None else list(range(self.n))
        self.pos = 0
        self.sweep = 0
        self.phase = "init"
        self.loss = None
        self.improved_in_sweep = False
        self._trial = None
        self._dz = None

    def _dz_for(self, i):
        x = self.x_of_z(self.z)
        span = self.x_of_z(np.ones(self.n))[i] - self.x_of_z(np.zeros(self.n))[i]
        dz = self.step * max(abs(x[i]), 1e-3) / span
        return float(np.clip(dz, 1e-6, 0.25))

    def ask(self):
        if self.phase == "init":
            return [self.z.copy()]
        i = self.order[self.pos]
        if self.phase == "probe":
            self._dz = self._dz_for(i)
            zm, zp = self.z.copy(), self.z.copy()
            zm[i] = max(0.0, self.z[i] - self._dz)
            zp[i] = min(1.0, self.z[i] + self._dz)
            return [zm, zp]
        return [self._trial]

    def tell(self, zs, losses):
        if self.phase == "init":
            self.loss = float(losses[0])
            self.phase = "probe"
            return
        i = self.order[self.pos]
        if self.phase == "probe":
            (zm, zp), (lm, lp) = zs, losses
            l0 = self.loss
            pts = [(zm[i], lm), (self.z[i], l0), (zp[i], lp)]
            best_z, best_l = min(pts, key=lambda t: t[1])
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            trial = None
            if np.all(np.isfinite(ys)) and len(set(np.round(xs, 12))) == 3:
                a, b, _ = np.polyfit(xs, ys, 2)
                if a > 0:
                    zmin = -b / (2 * a)
                    lo = self.z[i] - self.max_extra * self._dz
                    hi = self.z[i] + self.max_extra * self._dz
                    trial = float(np.clip(zmin, max(0.0, lo), min(1.0, hi)))
            self._best3 = (best_z, best_l)
            if trial is not None and abs(trial - best_z) > 1e-12:
                self._trial = self.z.copy()
                self._trial[i] = trial
                self.phase = "trial"
                return
            self._accept(i, best_z, best_l)
            return
        # trial phase
        best_z, best_l = self._best3
        if losses[0] < best_l:
            best_z, best_l = zs[0][i], float(losses[0])
        self._accept(i, best_z, best_l)

    def _accept(self, i, zi, li):
        if li < self.loss:
            self.z[i] = zi
            self.loss = li
            self.improved_in_sweep = True
        self.phase = "probe"
        self.pos += 1
        if self.pos >= len(self.order):
            self.pos = 0
            self.sweep += 1
            if not self.improved_in_sweep:
                self.step *= self.shrink
            self.improved_in_sweep = False

    def stop(self):
        if self.sweep >= self.n_sweeps:
            return "sweeps"
        if self.step < self.min_step:
            return "min_step"
        return None

    def step_info(self):
        return {"sweep": self.sweep, "param": self.pos, "rel_step": self.step}


# ------------------------------------------------------------ scipy local
class ScipyLocal(Optimizer):
    """scipy.optimize.minimize driven through ask/tell by running it in a
    generator-like fashion: each objective call becomes one ask/tell."""

    def __init__(self, z0, *, method: str = "nelder-mead", max_evals: int = 2000):
        import threading
        import queue

        self.name = method
        self._q_ask, self._q_tell = queue.Queue(maxsize=1), queue.Queue(maxsize=1)
        self._done = None
        z0 = np.clip(np.asarray(z0, float), 0, 1)

        def f(z):
            self._q_ask.put(np.clip(np.asarray(z, float), 0, 1))
            out = self._q_tell.get()
            if out is None:
                raise _Abort()
            return out

        def run():
            from scipy.optimize import minimize

            try:
                opts = {"maxfev": max_evals} if method.lower() in ("nelder-mead", "powell") else {"maxfun": max_evals}
                if method.lower() == "nelder-mead":
                    opts.update({"xatol": 1e-5, "fatol": 1e-8, "adaptive": len(z0) > 10})
                res = minimize(f, z0, method=method, bounds=[(0, 1)] * len(z0), options=opts)
                self._done = f"scipy:{res.message}"
            except _Abort:
                self._done = "aborted"
            except Exception as exc:  # noqa: BLE001
                self._done = f"scipy error: {exc}"
            self._q_ask.put(None)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        self._pending = None

    def ask(self):
        z = self._q_ask.get()
        if z is None:
            self._pending = None
            return []
        self._pending = z
        return [z]

    def tell(self, zs, losses):
        if zs:
            self._q_tell.put(float(losses[0]))

    def stop(self):
        return self._done

    def __getstate__(self):
        raise TypeError("scipy local optimizers cannot be checkpointed; resume restarts from the best point")


class _Abort(Exception):
    pass


OPTIMIZERS = ("cmaes", "mcff", "soppe", "nelder-mead", "powell")
