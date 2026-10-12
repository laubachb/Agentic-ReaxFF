"""Which parameters does the training set actually constrain?

Fitting parameters the data cannot see adds noise and overfitting, not
accuracy; Freitas Gustavo, Hellström and Verstraelen showed that optimizing
a small, sensitive subset converges faster to comparable loss with less
overfitting [FH23], and ParAMS' disulfide refit first scanned all 701
parameters one at a time and kept the 35 most sensitive [P21].

Methods:
  oat    one-at-a-time: each parameter moved ±delta (fraction of its
         range) from the start, everything else fixed: 2n+1 evaluations.
         Local, cheap, and it flags "fragile" parameters whose small
         changes break structures (NaN, QEq failure).
  hsic   global: `samples` uniform random points in the box, sensitivity of
         log(loss) to each parameter by the Hilbert-Schmidt independence
         criterion (the ParAMS/[FH23] measure), normalized to sum 1.

Output `sensitivity.json` (ranking with per-group loss changes for oat),
consumed by `params-select --from-sensitivity --top N`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..engine.objective import Objective
from ..io import params as P
from ..io import trainset as T
from . import _remote

NAME = "sensitivity"
SUMMARY = "Rank active parameters by how much the training-set loss depends on them (one-at-a-time or HSIC)."
SUPPORTS_DRY_RUN = True
SCHEMA = {
    "type": "object",
    "required": ["trainset", "params"],
    "properties": {
        "trainset": {"type": "string"},
        "params": {"type": "string", "description": "params.yaml from params-select (usually a broad preset)."},
        "method": {"type": "string", "enum": ["oat", "hsic"], "default": "oat"},
        "delta": {"type": "number", "default": 0.1, "description": "oat: step as a fraction of each parameter's range."},
        "samples": {"type": "integer", "default": 500, "description": "hsic: random samples."},
        "seed": {"type": "integer", "default": 0},
        "threshold": {"type": "number", "default": 0.01, "description": "Influential if sensitivity >= threshold x the largest."},
        "workers": {"type": ["integer", "null"]},
        **_remote.MACHINE_SCHEMA,
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--trainset", default=None)
    parser.add_argument("--params", default=None)
    parser.add_argument("--method", default="oat", choices=["oat", "hsic"])
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=None)
    _remote.add_machine_arguments(parser)


def _hsic(x, y):
    n = len(x)

    def gram(v):
        d = (v[:, None] - v[None, :]) ** 2
        med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
        return np.exp(-d / med)

    K, L = gram(np.asarray(x, float)), gram(np.asarray(y, float))
    H = np.eye(n) - 1.0 / n
    return float(np.trace(K @ H @ L @ H) / (n - 1) ** 2)


def run(args) -> dict:
    if getattr(args, "machine", None):
        return _remote.submit(args, NAME)
    if getattr(args, "dry_run", False):
        raise ValueError("--dry-run applies with --machine (it previews the job script)")
    ts = T.load(args.trainset)
    spec = P.load(args.params)
    out_dir = Path(args.output_dir or ".")
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(spec.params)
    z0 = np.clip(spec.to_z(spec.x0), 0, 1)
    with Objective(ts, spec, n_workers=args.workers) as obj:
        if args.method == "oat":
            Z = [z0]
            for i in range(n):
                for sgn in (-1, 1):
                    z = z0.copy()
                    z[i] = np.clip(z[i] + sgn * args.delta, 0, 1)
                    Z.append(z)
            evs = obj.evaluate_z(Z, details=True)
            base = evs[0]
            base_groups = T.summarize_errors(ts, base.preds)["groups"]
            ranking = []
            for i, pid in enumerate(spec.ids):
                em, ep = evs[1 + 2 * i], evs[2 + 2 * i]
                dl = [e.loss - base.loss for e in (em, ep)]
                s = float(max(abs(d) for d in dl)) / max(base.loss, 1e-12)
                gm = T.summarize_errors(ts, em.preds)["groups"]
                gp = T.summarize_errors(ts, ep.preds)["groups"]
                top_groups = sorted(((g, max(abs(gm[g]["loss"] - b["loss"]), abs(gp[g]["loss"] - b["loss"])))
                                     for g, b in base_groups.items()), key=lambda t: -t[1])[:3]
                ranking.append({"id": pid, "sensitivity": s, "dloss_minus": dl[0], "dloss_plus": dl[1],
                                "fragile": bool(em.n_failed or ep.n_failed) and not base.n_failed,
                                "groups": [g for g, v in top_groups if v > 0]})
            result = {"method": "oat", "delta": args.delta, "base_loss": base.loss, "n_evals": len(Z)}
        else:
            rng = np.random.default_rng(args.seed)
            Z = rng.uniform(0, 1, size=(args.samples, n))
            evs = obj.evaluate_z(list(Z))
            y = np.log10(np.array([max(e.loss, 1e-12) for e in evs]))
            ok = np.array([e.n_failed == 0 and e.violation == 0 for e in evs])
            if ok.sum() < 20:
                raise ValueError(f"only {ok.sum()} of {args.samples} random samples evaluated cleanly: the bounds are too "
                                 "wide for a global screen; use --method oat, or narrower bounds")
            h = np.array([_hsic(Z[ok, i], y[ok]) for i in range(n)])
            h = h / h.sum() if h.sum() > 0 else h
            ranking = [{"id": pid, "sensitivity": float(h[i])} for i, pid in enumerate(spec.ids)]
            result = {"method": "hsic", "samples": args.samples, "n_clean": int(ok.sum()),
                      "frac_failed": float(1 - ok.mean()), "n_evals": len(Z)}
    ranking.sort(key=lambda r: -r["sensitivity"])
    smax = ranking[0]["sensitivity"] if ranking else 0.0
    for r in ranking:
        r["influential"] = bool(smax > 0 and r["sensitivity"] >= args.threshold * smax)
    result.update({"n_params": n, "ranking": ranking, "n_influential": sum(r["influential"] for r in ranking),
                   "insensitive": [r["id"] for r in ranking if not r["influential"]][:50]})
    path = out_dir / "sensitivity.json"
    path.write_text(json.dumps(result, indent=1))
    return {"sensitivity": str(path), "method": result["method"], "n_params": n, "n_influential": result["n_influential"],
            "top": [(r["id"], round(r["sensitivity"], 5)) for r in ranking[:15]],
            "fragile": [r["id"] for r in ranking if r.get("fragile")][:20],
            "next": f"params-select --ffield <ffield> --from-sensitivity {path} --top N"}
