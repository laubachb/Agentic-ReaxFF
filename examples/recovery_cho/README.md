# Parameter-recovery benchmark (C/H/O, no DFT)

Labels are generated with the published Chenoweth 2008 C/H/O force field,
the sigma-bond parameters are set free within ±20% bounds, and every fit
run starts from a random point inside the bounds. A working setup returns
to the published values (loss ~ 0). Parameters that are not recovered are
the ones the data does not constrain (check `sensitivity`) or belong to
degenerate minima: both are the classic ReaxFF difficulties (Larsson 2013;
Shchygol 2019).

```bash
source ../../env.sh
./run.sh                    # local (minutes on a workstation)
./run.sh --machine dane     # prepares the Slurm fit (dry run) and prints the submit command
```
