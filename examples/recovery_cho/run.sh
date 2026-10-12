#!/usr/bin/env bash
# Parameter-recovery benchmark: can the fitting setup find a KNOWN answer?
#
# Labels come from a published force field (Chenoweth 2008 C/H/O), every fit
# run starts from a random point inside the bounds (--starts random), and the
# test is whether the fit returns to the published parameters and loss ~ 0.
# No DFT needed; it exercises data generation, the training-set builder,
# parameter selection, sensitivity, the optimizer protocol and evaluation.
# A fit that cannot recover a known force field from its own data will not
# recover a good one from DFT.
#
#   source ../../env.sh && ./run.sh [--machine dane]   (fit locally, or on one Slurm node)
set -euo pipefail
MACHINE="${2:-}"
[ "${1:-}" = "--machine" ] || MACHINE=""
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HERE/run}"
FF="$(python -c 'from agentic_reaxff import config; print(config.FFIELD_LIBRARY_DIR)')/ffield.reax.cho"
R=reaxff-agent

$R study --init "$OUT" --name recovery-cho --goal "recover Chenoweth 2008 C/H/O bond parameters from their own labels" --elements C,H,O >/dev/null
cd "$OUT"
$R data-generate --json-in "$HERE/gen.json" --output-dir 01_data/gen | python -c "import json,sys; d=json.load(sys.stdin); print('frames', d['n_frames'], d['by_kind'])"
$R label --structures 01_data/gen/structures.extxyz --backend ffield --ffield "$FF" --output-dir 01_data/ref | python -c "import json,sys; d=json.load(sys.stdin); print('labeled', d['n_labeled'], 'failed', d['n_failed'])"
$R trainset-build --structures 01_data/ref/labeled.extxyz --validation-fraction 0.2 --output-dir 02_trainset | python -c "import json,sys; d=json.load(sys.stdin); print('items', d['counts'], d['hint'])"
$R params-select --ffield "$FF" --preset sigma_bond --trainset 02_trainset/trainset.yaml --output-dir 03_fit/params | python -c "import json,sys; d=json.load(sys.stdin); print('active', d['n_active'], d['warnings'])"
FIT=(fit --trainset 02_trainset/trainset.yaml --validset 02_trainset/validset.yaml --params 03_fit/params/params.yaml
     --starts random --restarts 4 --keep 2 --short-evals 1500 --max-evals 4000 --output-dir 03_fit/fit)
if [ -n "$MACHINE" ]; then
  $R "${FIT[@]}" --machine "$MACHINE" --walltime-hours 1 --dry-run | python -c "import json,sys; d=json.load(sys.stdin); print('job script:', d['job_files'][0])"
  echo "Review the script, then submit:  (cd $OUT && $R ${FIT[*]} --machine $MACHINE --walltime-hours 1)"
  echo "Afterwards: python $HERE/check_recovery.py $OUT"
else
  $R "${FIT[@]}" > /dev/null
  python "$HERE/check_recovery.py" "$OUT"
fi
