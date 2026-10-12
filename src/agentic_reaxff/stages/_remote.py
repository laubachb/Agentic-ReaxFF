"""Run a stage as a Slurm job: the same CLI invocation, minus `--machine`.

A stage called with `--machine` writes its own inputs (without the machine
keys) to `<output-dir>/job_input.json` and submits

    reaxff-agent <stage> --json-in job_input.json --output-dir <dir> --force [--continue]

so what runs on the compute node is exactly what would run locally, with
the full node as its worker pool. `--chunks N` chains N jobs (afterany);
resumable stages (`fit`) pick up from their checkpoint in each link, which
is how a long fit runs on 1 h debug queues.
"""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from .. import hpc, machines

MACHINE_KEYS = ("machine", "queue", "nodes", "ntasks_per_node", "walltime_hours", "chunks", "dependency")
_NON_INPUT = ("json_in", "json_out", "describe", "stage", "force", "dry_run", "output_dir")


def submit(args, stage: str, *, resumable: bool = False, extra_input: dict = None) -> dict:
    out_dir = Path(args.output_dir).resolve() if getattr(args, "output_dir", None) else None
    if out_dir is None:
        raise ValueError(f"{stage} --machine needs --output-dir (a shared-filesystem directory for the job)")
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = machines.load_profile(args.machine)
    inputs = {k: v for k, v in vars(args).items()
              if not k.startswith("_") and k not in MACHINE_KEYS and k not in _NON_INPUT and v is not None}
    for k, v in list(inputs.items()):  # absolute paths: the job runs from another directory
        if isinstance(v, str) and Path(v).exists():
            inputs[k] = str(Path(v).resolve())
        elif isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            inputs[k] = [str(Path(x).resolve()) if Path(x).exists() else x for x in v]
    if resumable and not getattr(args, "continue_run", False) and not getattr(args, "dry_run", False):
        # every chunk runs with --continue; the FIRST submission of a new fit must not resume an old one
        from . import fit as _fit

        _fit.reset(out_dir)
    inputs.pop("continue_run", None)
    inputs.update(extra_input or {})
    (out_dir / "job_input.json").write_text(json.dumps(inputs, indent=1, default=str))
    cli = f"{shlex.quote(sys.executable)} -m agentic_reaxff.cli"
    cmd = f"{cli} {stage} --json-in job_input.json --output-dir {shlex.quote(str(out_dir))} --force" + \
          (" --continue" if resumable else "") + " --json-out job_result.json"
    ntasks = getattr(args, "ntasks_per_node", None) or profile.default_ntasks_per_node
    body = ["cd " + shlex.quote(str(out_dir)), f"export AGENTIC_REAXFF_WORKERS={ntasks}", "export OMP_NUM_THREADS=1", cmd]
    handles = hpc.submit_chain(profile, n_jobs=int(getattr(args, "chunks", 1) or 1), job_name=stage, commands=body,
                               work_dir=out_dir, nodes=1, ntasks_per_node=ntasks,
                               walltime_hours=float(getattr(args, "walltime_hours", 1.0) or 1.0),
                               queue=getattr(args, "queue", "debug") or "debug", dry_run=bool(getattr(args, "dry_run", False)),
                               first_dependency=getattr(args, "dependency", None), expect=["job_result.json"])
    return {"submitted": not handles[0].dry_run, "dry_run": handles[0].dry_run, "job_ids": [h.job_id for h in handles],
            "job_files": [str(h.job_file) for h in handles], "work_dir": str(out_dir), "machine": profile.name,
            "queue": handles[0].queue, "ntasks_per_node": ntasks,
            "next": f"reaxff-agent job-status --work-dir {out_dir}; results in {out_dir}/job_result.json"}


def add_machine_arguments(parser, *, walltime: float = 1.0) -> None:
    parser.add_argument("--machine", default=None, help="Submit as a Slurm job on this machine profile.")
    parser.add_argument("--queue", default="debug")
    parser.add_argument("--ntasks-per-node", dest="ntasks_per_node", type=int, default=None)
    parser.add_argument("--walltime-hours", dest="walltime_hours", type=float, default=walltime)
    parser.add_argument("--chunks", type=int, default=1, help="Chain this many jobs (afterany), each resuming the last.")
    parser.add_argument("--dependency", default=None)


MACHINE_SCHEMA = {
    "machine": {"type": ["string", "null"], "description": "Run as a Slurm job (one full node) on this machine profile."},
    "queue": {"type": "string", "default": "debug"},
    "ntasks_per_node": {"type": ["integer", "null"]},
    "walltime_hours": {"type": "number", "default": 1.0},
    "chunks": {"type": "integer", "default": 1, "description": "Chained jobs (afterany), each resuming the previous one."},
    "dependency": {"type": ["string", "null"]},
}
