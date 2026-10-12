"""Optimize ReaxFF parameters against a training set.

The protocol follows what the optimizer benchmarks found, not one long run:

  1. `restarts` independent runs, each for `short_evals` evaluations. Run 0
     starts from the given force field; the others from random points in
     the central half of every range (Shchygol's "random guess": the best
     of their 150 runs always came from one [Sh19]).
  2. Only the `keep` best runs continue to `max_evals` ("run many short
     CMA-ES runs and continue only the best" [Sh19]; GloMPO automates the
     same idea [GW22]).
  3. The result is the run with the lowest *validation* loss when a
     validation set is given (near-degenerate minima are common, and the
     held-out set is how to choose among them [Sh19, P21]), else training.
  4. Optional local polish of the winner (SOPPE or Nelder-Mead).

Every run keeps its best ffield; together they are the committee whose
disagreement `refine-select` uses to pick new training data ("add data
where near-optimal solutions disagree" [Sh19]).

Outputs: fit.json (all runs, the choice and why), best.ffield,
best_report.json (errors per kind/group, train and validation), and
runs/run_XX/{history.jsonl, best.ffield, best_val.ffield, checkpoint.pkl}.
Interrupted fits (time limit, a 1 h queue) resume with `--continue`;
`--machine` submits the whole thing as a chain of Slurm jobs that do so.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from ..engine.objective import Objective
from ..io import atomic
from ..io import params as P
from ..io import trainset as T
from ..optim.base import Budget, Runner
from ..optim.optimizers import CMAES, MCFF, SOPPE, ScipyLocal, default_popsize
from . import _remote

NAME = "fit"
SUMMARY = "Fit active ReaxFF parameters (CMA-ES default; multi-start with pruning, validation-based choice, resumable, Slurm-ready)."
SUPPORTS_DRY_RUN = True
SCHEMA = {
    "type": "object",
    "required": ["trainset", "params"],
    "properties": {
        "trainset": {"type": "string"},
        "validset": {"type": ["string", "null"], "description": "Held-out set: early stopping and choice among runs."},
        "params": {"type": "string", "description": "params.yaml from params-select."},
        "optimizer": {"type": "string", "enum": ["cmaes", "mcff", "soppe", "nelder-mead", "powell"], "default": "cmaes"},
        "restarts": {"type": "integer", "default": 4, "description": "Independent runs (Shchygol: 10 for a final fit)."},
        "keep": {"type": "integer", "default": 2, "description": "Runs continued after the short phase."},
        "short_evals": {"type": "integer", "default": 1500, "description": "Budget of every run before pruning."},
        "max_evals": {"type": "integer", "default": 6000, "description": "Budget of the runs that continue."},
        "starts": {"type": "string", "enum": ["mixed", "ffield", "random"], "default": "mixed",
                   "description": "mixed: run 0 from the ffield, others random in the central half of the box."},
        "sigma0": {"type": "number", "default": 0.25, "description": "CMA-ES initial step, fraction of each range [Sh19]."},
        "popsize": {"type": ["integer", "null"], "description": "CMA-ES population (default 4 + 3 ln n)."},
        "mcff_iter": {"type": "integer", "default": 3000},
        "soppe_step": {"type": "number", "default": 0.05},
        "polish": {"type": ["string", "null"], "enum": [None, "soppe", "nelder-mead", "powell"], "description": "Local polish of the winner."},
        "polish_evals": {"type": "integer", "default": 800},
        "patience_evals": {"type": ["integer", "null"], "description": "Early stopping: evaluations without validation improvement."},
        "stall_evals": {"type": ["integer", "null"], "description": "Stop a run after this many evaluations without improvement."},
        "seed": {"type": "integer", "default": 0},
        "workers": {"type": ["integer", "null"]},
        "time_limit_min": {"type": ["number", "null"], "description": "Stop cleanly (resumable) after this long; default: the Slurm job's remaining time minus 5 min."},
        "continue_run": {"type": "boolean", "default": False, "description": "Resume an interrupted fit in --output-dir."},
        **_remote.MACHINE_SCHEMA,
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--trainset", default=None)
    parser.add_argument("--validset", default=None)
    parser.add_argument("--params", default=None)
    parser.add_argument("--optimizer", default="cmaes", choices=["cmaes", "mcff", "soppe", "nelder-mead", "powell"])
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--keep", type=int, default=2)
    parser.add_argument("--short-evals", dest="short_evals", type=int, default=1500)
    parser.add_argument("--max-evals", dest="max_evals", type=int, default=6000)
    parser.add_argument("--starts", default="mixed", choices=["mixed", "ffield", "random"])
    parser.add_argument("--sigma0", type=float, default=0.25)
    parser.add_argument("--popsize", type=int, default=None)
    parser.add_argument("--mcff-iter", dest="mcff_iter", type=int, default=3000)
    parser.add_argument("--soppe-step", dest="soppe_step", type=float, default=0.05)
    parser.add_argument("--polish", default=None, choices=[None, "soppe", "nelder-mead", "powell"])
    parser.add_argument("--polish-evals", dest="polish_evals", type=int, default=800)
    parser.add_argument("--patience-evals", dest="patience_evals", type=int, default=None)
    parser.add_argument("--stall-evals", dest="stall_evals", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--time-limit-min", dest="time_limit_min", type=float, default=None)
    parser.add_argument("--continue", dest="continue_run", action="store_true", default=False)
    _remote.add_machine_arguments(parser)


def _slurm_time_left_s():
    jid = os.environ.get("SLURM_JOB_ID")
    if not jid:
        return None
    try:
        out = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%L"], capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    if not out or "INVALID" in out or "UNLIMITED" in out:
        return None
    d = 0
    if "-" in out:
        d, out = out.split("-", 1)
        d = int(d)
    parts = [int(x) for x in out.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return d * 86400 + h * 3600 + m * 60 + s


def _optimizer(args, spec, z0, seed, budget_evals):
    n = len(z0)
    if args.optimizer == "cmaes":
        return CMAES(z0, sigma0=args.sigma0, popsize=args.popsize, seed=seed)
    if args.optimizer == "mcff":
        from ..engine.pool import default_workers

        return MCFF(z0, n_iter=args.mcff_iter, seed=seed, replicas=max(1, min(8, (args.workers or default_workers()) // 4)))
    if args.optimizer == "soppe":
        return SOPPE(z0, spec.to_x, step=args.soppe_step, n_sweeps=max(1, budget_evals // max(1, 3 * n)))
    return ScipyLocal(z0, method=args.optimizer, max_evals=budget_evals)


def _start_z(args, spec, k):
    z_ff = np.clip(spec.to_z(spec.x0), 0, 1)
    if args.starts == "ffield" or (args.starts == "mixed" and k == 0):
        return z_ff
    rng = np.random.default_rng(10_000 + args.seed + k)
    return rng.uniform(0.25, 0.75, size=len(z_ff))


def _load_runner(obj, run_dir, budget, log):
    ck = Runner.load_checkpoint(run_dir)
    if not ck:
        return None
    if ck["ids"] != obj.spec.ids:
        raise ValueError(f"{run_dir}: checkpoint was made for different parameters; use a new --output-dir or --force")
    opt = ck["optimizer"]
    st = ck["state"]
    if opt is None:  # non-resumable optimizer: the caller restarts it from the best point, keeping the run state
        return st
    return Runner(obj, opt, run_dir, budget=budget, state=st, log=log)


def run(args) -> dict:
    if getattr(args, "machine", None):
        return _remote.submit(args, NAME, resumable=True)
    if getattr(args, "dry_run", False):
        raise ValueError("--dry-run applies with --machine (it previews the job script)")
    out = Path(args.output_dir or ".")
    out.mkdir(parents=True, exist_ok=True)
    runs_dir = out / "runs"
    runs_dir.mkdir(exist_ok=True)
    state_path = out / "fit_state.json"
    fp = fingerprint(args)
    if not args.continue_run and state_path.exists():
        # a fresh fit in a directory that has one: start clean (the manifest already made the caller say --force)
        reset(out)
        runs_dir.mkdir()
    fstate = atomic.read_json(state_path) or {"phase": "short", "runs": {}}
    if fstate.get("fingerprint") and fstate["fingerprint"] != fp:
        raise ValueError(f"{out} holds a fit made with different inputs (training/validation set, parameters or fit "
                         "settings changed): rerun without --continue to start over there, or use a new --output-dir")
    fstate["fingerprint"] = fp
    prev = atomic.read_json(out / "fit.json")
    if args.continue_run and fstate.get("finished") and prev and prev.get("status") == "done":
        prev["note"] = "already finished; returned the recorded result (rerun without --continue to refit)"
        return prev

    ts = T.load(args.trainset)
    vs = T.load(args.validset) if args.validset else None
    spec = P.load(args.params)
    t_start = time.time()
    left = _slurm_time_left_s()
    limit = args.time_limit_min * 60 if args.time_limit_min else (left - 300 if left else None)

    def remaining():
        return None if limit is None else max(0.0, limit - (time.time() - t_start))

    log_lines = []

    def log(msg):
        log_lines.append(msg)
        print(msg, flush=True)

    with Objective(ts, spec, validation=vs, n_workers=args.workers) as obj:
        z0 = np.clip(spec.to_z(spec.x0), 0, 1)
        start_eval = obj.evaluate_x([spec.x0])[0]
        start_val = obj.evaluate_x([spec.x0], validation=True)[0].loss if vs else None
        log(f"start loss {start_eval.loss:.6g} (failed structures: {start_eval.n_failed}); "
            f"{len(spec.ids)} parameters, popsize {args.popsize or default_popsize(len(spec.ids))}")
        if start_eval.n_failed:
            log(f"WARNING: the starting force field fails on {start_eval.n_failed} structure(s): {start_eval.errors}")

        def do_run(k, budget_evals):
            rd = runs_dir / f"run_{k:02d}"
            budget = Budget(max_evals=budget_evals, time_limit_s=remaining(), patience_evals=args.patience_evals,
                            stall_evals=args.stall_evals)
            loaded = _load_runner(obj, rd, budget, log) if rd.exists() else None
            runner = loaded if isinstance(loaded, Runner) else None
            if runner is None:
                zs = _start_z(args, spec, k)
                if rd.exists() and (rd / "best.json").exists():
                    bp = json.loads((rd / "best.json").read_text())["params"]
                    zs = np.clip(spec.to_z(np.array([bp[p] for p in spec.ids])), 0, 1)
                runner = Runner(obj, _optimizer(args, spec, zs, args.seed + k, budget_evals), rd, budget=budget, log=log,
                                state=loaded if loaded is not None and not isinstance(loaded, Runner) else None)
            st = runner.run()
            rec = {"run": k, "status": st.status, "n_evals": st.n_evals, "loss": st.best_loss,
                   "val_loss": st.best_val_loss if vs else None, "dir": str(rd)}
            fstate["runs"][str(k)] = rec
            atomic.write_json(state_path, fstate, indent=1)
            return st

        interrupted = False
        # phase 1: short runs
        if fstate["phase"] == "short":
            for k in range(args.restarts):
                rec = fstate["runs"].get(str(k))
                if rec and rec["status"] != "time_limit":
                    continue
                st = do_run(k, args.short_evals)
                if st.status == "time_limit":
                    interrupted = True
                    break
            if not interrupted:
                ranked = sorted(fstate["runs"].values(), key=lambda r: (r["val_loss"] if vs else r["loss"]))
                fstate["kept"] = [r["run"] for r in ranked[: max(1, args.keep)]]
                fstate["phase"] = "long" if args.max_evals > args.short_evals else "polish"
                atomic.write_json(state_path, fstate, indent=1)
        # phase 2: continue the best runs
        if not interrupted and fstate["phase"] == "long":
            for k in fstate["kept"]:
                if fstate["runs"][str(k)].get("long_done"):
                    continue
                st = do_run(k, args.max_evals)
                fstate["runs"][str(k)]["long_done"] = st.status != "time_limit"
                if st.status == "time_limit":
                    interrupted = True
                    break
            if not interrupted:
                fstate["phase"] = "polish"
                atomic.write_json(state_path, fstate, indent=1)

        # choose
        runs = list(fstate["runs"].values())
        key = (lambda r: r["val_loss"]) if vs else (lambda r: r["loss"])
        best = min(runs, key=key) if runs else None
        best_dir = Path(best["dir"]) if best else None
        which = "best_val" if (vs and best_dir and (best_dir / "best_val.json").exists()) else "best"
        result = {"status": "time_limit" if interrupted else "done", "n_runs": len(runs),
                  "n_evals_total": int(sum(r["n_evals"] for r in runs)), "start_loss": start_eval.loss,
                  "start_val_loss": start_val, "runs": sorted(runs, key=key), "chosen_by": "validation" if vs else "training"}
        if best_dir is not None:
            bj = json.loads((best_dir / f"{which}.json").read_text())
            xbest = np.array([bj["params"][p] for p in spec.ids])
            if not interrupted and fstate["phase"] == "polish" and args.polish and not fstate.get("polished"):
                rd = out / "polish"
                if rd.exists():
                    shutil.rmtree(rd)  # a polish cut short earlier restarts from the unpolished best
                zb = np.clip(spec.to_z(xbest), 0, 1)
                opt = SOPPE(zb, spec.to_x, step=args.soppe_step / 2, n_sweeps=3) if args.polish == "soppe" else \
                    ScipyLocal(zb, method=args.polish, max_evals=args.polish_evals)
                st = Runner(obj, opt, rd, budget=Budget(max_evals=args.polish_evals, time_limit_s=remaining()), log=log).run()
                if st.best_x is None or st.status == "time_limit":
                    interrupted = True
                    result["status"] = "time_limit"
                else:
                    cand = st.best_x
                    if vs:
                        vb = obj.evaluate_x([xbest], validation=True)[0].loss
                        vc = obj.evaluate_x([cand], validation=True)[0].loss
                        accept = vc <= vb
                    else:
                        accept = st.best_loss < bj["loss"]
                    fstate["polished"] = True
                    fstate["polish"] = {"status": st.status, "loss": st.best_loss, "accepted": bool(accept),
                                        "x": [float(v) for v in cand] if accept else None}
                    atomic.write_json(state_path, fstate, indent=1)
            if fstate.get("polish"):
                if fstate["polish"].get("x"):
                    xbest = np.array(fstate["polish"]["x"])
                result["polish"] = {k: v for k, v in fstate["polish"].items() if k != "x"}
            obj.write_ffield(xbest, out / "best.ffield")
            rep = {"train": obj.report(xbest)}
            if vs:
                rep["validation"] = obj.report(xbest, validation=True)
            rep["start"] = {"train": obj.report(spec.x0)}
            frac = rep["start"]["train"].get("top10_loss_fraction", 0.0)
            if frac > 0.5 and len(ts.items) > 40:
                result["warning_dominant_items"] = (
                    f"10 of {len(ts.items)} items carry {frac:.0%} of the starting loss "
                    f"({[t['id'] for t in rep['start']['train']['top_items'][:5]]}...): the fit is mostly about them. "
                    "Check their sigmas/weights (trainset-build --high-energy-scale, --group-sigma) [Sh19]")
            (out / "best_report.json").write_text(json.dumps(rep, indent=1, default=str))
            result["best"] = {"run": best["run"], "loss": rep["train"]["loss"],
                              "val_loss": rep.get("validation", {}).get("loss"), "ffield": str(out / "best.ffield"),
                              "report": str(out / "best_report.json"),
                              "kinds": {k: {m: v.get(m) for m in ("rmse", "mae", "units")} for k, v in rep["train"]["kinds"].items()}}
            moved = []
            for pid, x0, xb, lo, hi in zip(spec.ids, spec.x0, xbest, spec.lo, spec.hi):
                if min(xb - lo, hi - xb) < 0.01 * (hi - lo):
                    moved.append(pid)
            if moved:
                result["at_bounds"] = moved[:30]
                result["at_bounds_note"] = ("parameters within 1% of a bound: the bound, not the data, set them. Widen it "
                                            "only if the value stays physical; Larsson et al. found ranges matter more than "
                                            "the optimizer [L13]")
            result["committee"] = sorted(str(Path(r["dir"]) / "best.ffield") for r in runs if (Path(r["dir"]) / "best.ffield").exists())
        if interrupted:
            result["next"] = "time limit reached: resubmit / rerun with --continue (all runs resume from their checkpoints)"
    if not interrupted:
        fstate["finished"] = True
        atomic.write_json(state_path, fstate, indent=1)
    atomic.write_json(out / "fit.json", result, indent=1)
    return result


def fingerprint(args) -> str:
    """Hash of everything that defines the fit (input file contents and the
    settings that change the result); a resume must match it."""
    import hashlib

    h = hashlib.sha256()
    for f in (args.trainset, args.validset, args.params):
        if f:
            h.update(Path(f).read_bytes())
            ts_struct = None
            if f.endswith((".yaml", ".yml")):
                try:
                    import yaml

                    d = yaml.safe_load(Path(f).read_text()) or {}
                    sp = d.get("structures") or d.get("ffield")
                    ts_struct = (Path(f).parent / sp) if sp and not Path(sp).is_absolute() else (Path(sp) if sp else None)
                except Exception:  # noqa: BLE001
                    ts_struct = None
            if ts_struct and ts_struct.is_file():
                st = ts_struct.stat()
                h.update(f"{ts_struct}:{st.st_size}:{st.st_mtime_ns}".encode())
    keys = ("optimizer", "restarts", "keep", "short_evals", "max_evals", "starts", "sigma0", "popsize", "mcff_iter",
            "soppe_step", "polish", "polish_evals", "patience_evals", "stall_evals", "seed")
    h.update(json.dumps({k: getattr(args, k, None) for k in keys}, sort_keys=True).encode())
    return h.hexdigest()[:16]


def reset(out: Path) -> None:
    """Remove a previous fit's state from `out` (fresh start)."""
    for name in ("runs", "polish"):
        if (out / name).exists():
            shutil.rmtree(out / name)
    for name in ("fit_state.json", "fit.json", "best.ffield", "best_report.json"):
        (out / name).unlink(missing_ok=True)
