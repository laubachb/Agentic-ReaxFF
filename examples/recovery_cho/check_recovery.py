"""Compare a recovery fit with the published parameters it should find."""

import json
import sys
from pathlib import Path

import numpy as np

from agentic_reaxff import config
from agentic_reaxff.io import ffield as F
from agentic_reaxff.io import params as P

out = Path(sys.argv[1])
fit = json.loads((out / "03_fit/fit/fit.json").read_text())
spec = P.load(out / "03_fit/params/params.yaml")
truth = np.array(F.read(config.FFIELD_LIBRARY_DIR / "ffield.reax.cho").get_many(spec.ids))
best = np.array(F.read(out / "03_fit/fit/best.ffield").get_many(spec.ids))
rel = (best - truth) / np.where(truth == 0, 1, np.abs(truth))
print(f"runs: {[round(r['loss'], 4) for r in fit['runs']]}  (start loss from random points; truth has loss 0)")
print(f"best loss {fit['best']['loss']:.4g}, validation {fit['best']['val_loss']}")
for pid, r in sorted(zip(spec.ids, rel), key=lambda t: -abs(t[1])):
    print(f"  {pid:28s} rel. error {r:+.4f}")
print("recovered" if np.max(np.abs(rel)) < 0.02 else
      "NOT fully recovered: insensitive parameters or a degenerate minimum (compare their sensitivity)")
