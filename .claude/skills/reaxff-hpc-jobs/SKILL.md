---
name: reaxff-hpc-jobs
description: Anything that touches Slurm, Quantum ESPRESSO, LAMMPS MPI or Lustre in reaxff-agent - machine profiles, dry runs and approval, chunked jobs on 1 h debug queues, job-status verdicts, file-count quotas, MPI library issues. Use before submitting or diagnosing any job.
---

# HPC jobs

- Profiles: `src/agentic_reaxff/machines/profiles/*.yaml` (dane, stampede3,
  generic) or your own YAML (`setup --init-profile my.yaml` detects Slurm
  settings). Account from `REAXFF_ACCOUNT` (written to env.sh by
  `./install.sh --account`).
- Submitting stages: `label --backend qe`, and `fit`, `sensitivity`,
  `md-check` with `--machine`. Each renders ONE sbatch script; always run
  with `--dry-run` first and show it; the approval hook asks the user.
- Every job requests a full node (`--ntasks-per-node` from the profile).
  On Dane a bare `-N 1` gets 1 CPU and 2.3 GB.
- Job directories must be on a shared filesystem (never /tmp): the
  submission refuses node-local paths.
- Short queues: `--chunks N --walltime-hours 1` chains N jobs with
  afterany; fit resumes from checkpoints, QE frames already done are
  skipped and interrupted pw.x runs restart (max_seconds).
- Status: `reaxff-agent job-status --work-dir DIR` (QUEUED / RUNNING /
  SUCCEEDED / FAILED + fix / COMPLETED_WITHOUT_RESULTS). Results of a
  stage job are in `DIR/job_result.json` (the stage's own JSON). Delegate
  waiting to `reaxff-job-monitor`; never sleep-poll in the main thread.
- QE labeling: one frame directory per frame (pw.in, pw.out, pw.err); the
  `out/` wavefunction scratch is deleted when a frame finishes. Thousands
  of frames are fine; tens of thousands are not on a file-count quota:
  label in batches and tar finished label directories to archive storage.
- Lustre (LLNL /p/lustre2): the binding limit is the number of files
  (~1.05M); `doctor --machine` reports usage.
- MPI: the LAMMPS wheel needs an MPICH-ABI `libmpi.so.12` (mvapich2,
  cray-mpich, Intel MPI or pip `mpich`); `setup --component mpi` finds and
  verifies one. Evaluations run one serial LAMMPS per worker process
  (spawned, never forked); MPI ranks are not used for fitting.
- LAMMPS for big MD: `setup --component lammps --machine M` builds from
  source (shared lib + python module + lmp, OPENMP; `--lammps-kokkos`).
