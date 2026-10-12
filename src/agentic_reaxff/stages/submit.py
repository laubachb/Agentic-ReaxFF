"""Generic Slurm submit/status/cancel/dry-run, on top of hpc/slurm.py (which
wraps sbatch through hpc/slurm.py). Every other
HPC-submitting stage calls into hpc.slurm directly rather than shelling out
to this subcommand, but it's also exposed standalone for ad hoc use --
resubmitting, checking on a job, or previewing an sbatch script with
--dry-run before spending an allocation.
"""

from __future__ import annotations

import subprocess

from .. import hpc, machines

NAME = "submit"
SUMMARY = "Submit/status/cancel a Slurm job, or --dry-run to preview the sbatch script."
SUPPORTS_DRY_RUN = True
SCHEMA = {
    "type": "object",
    "properties": {
        "machine": {"type": "string"},
        "job_name": {"type": "string"},
        "commands": {"type": "array", "items": {"type": "string"}},
        "work_dir": {"type": "string"},
        "nodes": {"type": "integer", "default": 1},
        "ntasks_per_node": {"type": ["integer", "null"]},
        "walltime_hours": {"type": "number", "default": 1.0},
        "queue": {"type": "string", "default": "debug"},
        "status_of": {"type": ["string", "null"], "description": "Job ID to check status of instead of submitting."},
        "cancel": {"type": ["string", "null"], "description": "Job ID to cancel instead of submitting."},
        "job_file_prefix": {"type": "string", "default": "run", "description": "Job file name prefix (<prefix>.cmd, <prefix>.2.cmd, ...); use a new one for a second chain in the same directory."},
        "repeat": {"type": "integer", "default": 1, "description": "Submit this many copies chained with afterany dependencies (restartable work in chunks on a short queue)."},
        "dependency": {"type": ["string", "null"], "description": "Slurm --dependency for the job (with --repeat: for the first link), e.g. afterany:12345."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--machine", default=None)
    parser.add_argument("--job-name", dest="job_name", default="reaxff-agent-job")
    parser.add_argument("--command", dest="commands", action="append", default=None, help="A job body command line (repeatable, in order).")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--ntasks-per-node", dest="ntasks_per_node", type=int, default=None)
    parser.add_argument("--walltime-hours", dest="walltime_hours", type=float, default=1.0)
    parser.add_argument("--queue", default="debug")
    parser.add_argument("--status-of", dest="status_of", default=None, help="Job ID to check status of.")
    parser.add_argument("--cancel", default=None, help="Job ID to cancel.")
    parser.add_argument("--repeat", type=int, default=1, help="Chain this many copies (afterany).")
    parser.add_argument("--job-file-prefix", dest="job_file_prefix", default="run")
    parser.add_argument("--dependency", default=None)


def run(args) -> dict:
    if getattr(args, "status_of", None):
        proc = subprocess.run(["squeue", "-j", args.status_of], capture_output=True, text=True)
        return {"job_id": args.status_of, "in_queue": args.status_of in proc.stdout, "squeue_output": proc.stdout}

    if getattr(args, "cancel", None):
        proc = subprocess.run(["scancel", args.cancel], capture_output=True, text=True)
        return {"job_id": args.cancel, "cancelled": proc.returncode == 0, "stderr": proc.stderr}

    if not args.machine:
        raise ValueError("submit requires --machine unless --status-of/--cancel is given")
    if not args.commands:
        raise ValueError("submit requires at least one --command (or 'commands' in --json-in)")
    work_dir = getattr(args, "work_dir", None) or getattr(args, "output_dir", None)
    if not work_dir:
        raise ValueError("submit requires --output-dir (or work_dir in --json-in) as the job's work_dir")

    profile = machines.load_profile(args.machine)
    repeat = int(getattr(args, "repeat", 1) or 1)
    if repeat > 1:
        handles = hpc.submit_chain(profile, n_jobs=repeat, job_name=args.job_name, commands=args.commands, work_dir=work_dir,
                                   first_dependency=getattr(args, "dependency", None),
                                   job_file_prefix=getattr(args, "job_file_prefix", "run") or "run",
                                   nodes=args.nodes, ntasks_per_node=args.ntasks_per_node, walltime_hours=args.walltime_hours,
                                   queue=args.queue, dry_run=bool(getattr(args, "dry_run", False)))
        return {"job_ids": [h.job_id for h in handles], "job_id": handles[0].job_id, "job_files": [str(h.job_file) for h in handles],
                "dry_run": handles[0].dry_run, "machine": handles[0].machine, "queue": handles[0].queue, "work_dir": str(handles[0].work_dir)}
    handle = hpc.submit_job(
        profile,
        job_name=args.job_name,
        commands=args.commands,
        work_dir=work_dir,
        nodes=args.nodes,
        ntasks_per_node=args.ntasks_per_node,
        walltime_hours=args.walltime_hours,
        queue=args.queue,
        dry_run=bool(getattr(args, "dry_run", False)),
        dependency=getattr(args, "dependency", None),
        job_file=f"{getattr(args, 'job_file_prefix', 'run') or 'run'}.cmd",
    )

    return {
        "job_id": handle.job_id,
        "job_name": handle.job_name,
        "work_dir": str(handle.work_dir),
        "job_file": str(handle.job_file),
        "dry_run": handle.dry_run,
        "machine": handle.machine,
        "queue": handle.queue,
    }
