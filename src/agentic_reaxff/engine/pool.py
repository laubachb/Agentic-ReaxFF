"""Parallel evaluation of many candidate force fields over many structures.

Workers are separate processes started with the `spawn` method (never
`fork`: each worker owns a serial LAMMPS instance and an MPI singleton,
which must not be inherited). Each worker loads the training structures
once, at start-up; a task is then just (candidate ffield text, structure
names), and the reply is the per-structure results the loss needs.

Candidate ffields are written to worker-private node-local scratch
(/dev/shm when available), never to the shared filesystem: a fit writes
one ffield per evaluation, which on Lustre would be millions of metadata
operations.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Optional

import numpy as np

from ..io.structures import Structure
from .evaluator import EvalResult, EvalSettings, Evaluator

# --- worker side -------------------------------------------------------------
_W = {}


def _scratch_dir() -> str:
    for base in (os.environ.get("AGENTIC_REAXFF_SCRATCH"), "/dev/shm", os.environ.get("TMPDIR"), tempfile.gettempdir()):
        if base and os.path.isdir(base) and os.access(base, os.W_OK):
            return tempfile.mkdtemp(prefix="reaxff-agent-", dir=base)
    return tempfile.mkdtemp(prefix="reaxff-agent-")


def _init_worker(structures: dict, settings: dict, needs: dict, masses: dict, env: dict, owner=None):
    os.environ.update(env or {})
    os.environ["OMP_NUM_THREADS"] = "1"
    old = _W.get("evaluator")
    _W.clear()
    if old is not None:
        old.settings = EvalSettings.from_dict(settings)
    _W.update(structures=structures, needs=needs, masses=masses,
              evaluator=old or Evaluator(EvalSettings.from_dict(settings)), dir=_scratch_dir(), owner=owner)
    _W["ffield"] = os.path.join(_W["dir"], "ffield")
    import atexit
    import shutil

    atexit.register(lambda: shutil.rmtree(_W.get("dir", ""), ignore_errors=True))


def _strip(r: EvalResult, need: dict) -> EvalResult:
    if not need.get("geometry"):
        r.positions = None
        r.cell = None
    if not need.get("forces"):
        r.forces = None
    return r


def _run_task(task):
    """task = (key, ffield_text, names). Returns (key, {name: EvalResult}, {name: seconds})."""
    key, text, names = task
    path = _W["ffield"]
    with open(path, "w") as f:
        f.write(text)
    if _W.get("last_text_checked") != hash(text):
        try:
            from ..io import ffield as F

            F.parse_string(text)
            _W["last_text_checked"] = hash(text)
        except Exception as exc:  # noqa: BLE001 - LAMMPS would segfault on it
            return key, {n: EvalResult(name=n, ok=False, error=f"invalid ffield: {exc}") for n in names}, {}
    out, times = {}, {}
    ev = _W["evaluator"]
    for name in names:
        s = _W["structures"][name]
        need = _W["needs"].get(name, {})
        t0 = time.perf_counter()
        r = ev.evaluate(path, s, masses=_W["masses"], want_forces=need.get("forces", False), check_ffield=False)
        times[name] = time.perf_counter() - t0
        out[name] = _strip(r, need)
    return key, out, times


# --- parent side -------------------------------------------------------------
class EvalPool:
    """Evaluate candidates (ffield texts) on a fixed set of structures.

    n_workers = 1 evaluates in this process (no subprocesses), which is what
    tests and quick checks want."""

    def __init__(self, structures: list, settings: EvalSettings, needs: dict, masses: dict,
                 n_workers: Optional[int] = None):
        self.structures = {s.name: s for s in structures}
        self.names = [s.name for s in structures]
        self.settings = settings
        self.needs = needs
        self.masses = masses
        self.n_workers = max(1, int(n_workers or default_workers()))
        # cost model, refined from measured times (seconds per structure)
        self.cost = {s.name: _guess_cost(s) for s in structures}
        self._pool = None
        import uuid

        self._token = uuid.uuid4().hex  # not id(self): CPython reuses ids of freed pools
        self.n_structure_evals = 0
        self.eval_seconds = 0.0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _ensure(self):
        if self.n_workers == 1:
            if _W.get("owner") != self._token:
                _init_worker(self.structures, self.settings.to_dict(), self.needs, self.masses, {}, owner=self._token)
            return
        if self._pool is None:
            # ProcessPoolExecutor, not multiprocessing.Pool: when a worker dies
            # (segfault in a native library, OOM kill, a script without a
            # __main__ guard) it raises BrokenProcessPool instead of silently
            # respawning workers and waiting forever for the lost tasks.
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            env = {k: v for k, v in os.environ.items() if k.startswith("AGENTIC_REAXFF")}
            self._pool = ProcessPoolExecutor(max_workers=self.n_workers, mp_context=mp.get_context("spawn"),
                                             initializer=_init_worker,
                                             initargs=(self.structures, self.settings.to_dict(), self.needs, self.masses,
                                                       env))

    def close(self):
        if _W.get("owner") == getattr(self, "_token", None):
            _W.pop("owner", None)
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def _chunks(self, names: list, n_chunks: int) -> list:
        """Greedy longest-processing-time split into n_chunks balanced lists."""
        n_chunks = max(1, min(n_chunks, len(names)))
        order = sorted(names, key=lambda n: -self.cost[n])
        bins = [[] for _ in range(n_chunks)]
        loads = [0.0] * n_chunks
        for n in order:
            k = int(np.argmin(loads))
            bins[k].append(n)
            loads[k] += self.cost[n]
        return [b for b in bins if b]

    def evaluate(self, texts: list, names: Optional[list] = None) -> list:
        """Results for each candidate text: list of {name: EvalResult}."""
        self._ensure()
        names = list(names or self.names)
        n_cand = len(texts)
        # enough tasks to keep every worker busy, a few per worker for balance
        per_cand = max(1, int(np.ceil(2.0 * self.n_workers / max(n_cand, 1)))) if self.n_workers > 1 else 1
        chunks = self._chunks(names, per_cand)
        tasks = [((i, c), texts[i], chunk) for i in range(n_cand) for c, chunk in enumerate(chunks)]
        results = [dict() for _ in range(n_cand)]
        t0 = time.perf_counter()
        if self.n_workers == 1:
            it = map(_run_task, tasks)
        else:
            from concurrent.futures import as_completed
            from concurrent.futures.process import BrokenProcessPool

            futures = [self._pool.submit(_run_task, t) for t in tasks]

            def _results():
                try:
                    for fut in as_completed(futures):
                        yield fut.result()
                except BrokenProcessPool as exc:
                    self._pool = None
                    raise RuntimeError("a LAMMPS worker process died (crash or out of memory); if you drive "
                                       "agentic_reaxff from your own script, it needs an `if __name__ == '__main__':` "
                                       "guard because workers are started with the spawn method") from exc
            it = _results()
        for (i, _c), out, times in it:
            results[i].update(out)
            for n, dt in times.items():
                self.cost[n] = 0.7 * self.cost[n] + 0.3 * dt
                self.eval_seconds += dt
            self.n_structure_evals += len(out)
        self.wall_seconds = time.perf_counter() - t0
        return results


def _guess_cost(s: Structure) -> float:
    base = 1e-3 + 2e-5 * s.natoms ** 1.3
    return base * (30.0 if s.mode == "relax" else 200.0 if s.mode == "relax_cell" else 1.0)


def default_workers() -> int:
    """Worker processes to use: AGENTIC_REAXFF_WORKERS if set; inside a Slurm
    job every CPU of the allocation; elsewhere (a shared login node) at most
    AGENTIC_REAXFF_LOGIN_WORKERS (default 8) — heavy work belongs in a job
    (`--machine`)."""
    v = os.environ.get("AGENTIC_REAXFF_WORKERS")
    if v and v.isdigit():
        return max(1, int(v))
    try:
        n = max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        n = os.cpu_count() or 1
    if not os.environ.get("SLURM_JOB_ID"):
        cap = os.environ.get("AGENTIC_REAXFF_LOGIN_WORKERS", "8")
        n = min(n, int(cap) if cap.isdigit() else 8)
    return n
