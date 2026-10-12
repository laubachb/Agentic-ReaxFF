# Getting started

## 1. Install

```bash
./install.sh --machine dane --account <bank>   # or: ./install.sh  (generic Linux)
source env.sh
reaxff-agent doctor --machine dane
```

`doctor` must say `ok` for `lammps`, `reaxff_numerics` and `worker_pool`.
If `mpi_lib` fails on an OpenMPI system: `pip install mpich` into the venv
and rerun `reaxff-agent setup --component mpi`.

## 2. A study

```bash
reaxff-agent study --init /scratch/me/cho-study --name cho --goal "..." --elements C,H,O
```

Work inside it; every stage you run with an `--output-dir` there is
journaled. Record decisions with `study --note`.

## 3. Pick a starting force field

```bash
reaxff-agent ffield-library --output-dir 00_lib            # index deps/ffields (or --dirs yours)
reaxff-agent ffield-info --ffield <candidate> --elements C,H,O
# new element:  ffield-merge --base B --add Si=donor --analog Si=C --output-dir 00_start
```

## 4. Data, labels, training set

```bash
reaxff-agent data-generate --json-in gen.json --output-dir 01_data/gen
reaxff-agent label --structures 01_data/gen/structures.extxyz --backend xtb --output-dir 01_data/xtb   # shakedown
reaxff-agent label --structures 01_data/gen/structures.extxyz --backend qe --machine dane \
    --qe '{"ecutwfc": 40, "ecutrho": 200}' --lanes 8 --output-dir 01_data/qe --dry-run         # then submit
reaxff-agent label --collect 01_data/qe
reaxff-agent trainset-build --structures 01_data/qe/labeled.extxyz --validation-fraction 0.15 --output-dir 02_trainset
```

## 5. Fit in stages

```bash
reaxff-agent params-select --ffield start.ffield --preset bond --trainset 02_trainset/trainset.yaml --output-dir 03_fit/bond
reaxff-agent sensitivity --trainset 02_trainset/trainset.yaml --params 03_fit/bond/params.yaml --output-dir 03_fit/sens
reaxff-agent params-select --ffield start.ffield --from-sensitivity 03_fit/sens/sensitivity.json --top 20 \
    --trainset 02_trainset/trainset.yaml --output-dir 03_fit/bond_top
reaxff-agent fit --trainset 02_trainset/trainset.yaml --validset 02_trainset/validset.yaml \
    --params 03_fit/bond_top/params.yaml --restarts 6 --machine dane --chunks 2 --output-dir 03_fit/fit_bond --dry-run
```

Then the next preset, starting from `03_fit/fit_bond/best.ffield`.

## 6. Validate, refine, deploy

```bash
reaxff-agent evaluate --ffields 03_fit/fit_bond/best.ffield --ffields start.ffield --labels fit,start \
    --trainset ... --validset ... --pair-check --plots --output-dir 04_validate/eval
reaxff-agent data-generate --boxes '[{"name": "mix", "molecules": {"CH4": 10, "O2": 20}, "density": 0.3}]' --output-dir 04_validate/boxes
reaxff-agent md-check --ffields 03_fit/fit_bond/best.ffield --ffields start.ffield --systems 04_validate/boxes/boxes.extxyz \
    --temperatures 300,2500 --output-dir 04_validate/md
reaxff-agent refine-select --committee ... --trajectories ... --trainset ... --output-dir 05_refine/r01
reaxff-agent deploy --ffield 03_fit/fit_bond/best.ffield --fit 03_fit/fit_bond --evaluate 04_validate/eval \
    --md-check 04_validate/md --output-dir 06_deploy
```
