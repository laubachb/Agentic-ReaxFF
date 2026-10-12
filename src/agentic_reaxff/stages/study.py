"""A study: one directory holding every phase of one ReaxFF force field's
development, marked by `study.json`.

`study.json` records the goal and a registry of the artifacts each phase
produced (`trainset`, `params_spec`, `fit`, `ffield`, `md_check`, ...), so
later phases, the usage accounting and the final report find them without
guessing paths. Standard layout, which `--init` creates:

    <study>/study.json  STUDY.md  journal.jsonl
    01_data/       reference structures and their DFT labels
    02_trainset/   trainset.yaml / validset.yaml (+ the packed extxyz)
    03_fit/        params.yaml, sensitivity, one directory per fit
    04_validate/   evaluate / md-check / scans of candidate ffields
    05_refine/     iterative refinement rounds (MD -> select -> label -> refit)
    06_deploy/     the delivered ffield + its card
    usage/

Stages that write inside a study (any `--output-dir` below a `study.json`)
add their login-node CPU time to `usage/local.jsonl` automatically (see
cli.py); Slurm jobs are found by `usage` through sacct's working directory.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from ..io import atomic
from ..io import fs

NAME = "study"
SUMMARY = "Create a study directory, register phase artifacts, or show a study's status."
USES_OUTPUT_DIR = False
LAYOUT = ("01_data", "02_trainset", "03_fit", "04_validate", "05_refine", "06_deploy", "usage")
KNOWN_ARTIFACTS = {
    "start_ffield": "the force field the fit starts from (published or merged)",
    "structures": "01_data: labeled structures (packed extxyz)",
    "trainset": "02_trainset: trainset.yaml",
    "validset": "02_trainset: validset.yaml (held out for early stopping / model choice)",
    "params_spec": "03_fit: params.yaml (active parameters, bounds, constraints)",
    "sensitivity": "03_fit: sensitivity.json",
    "fit": "03_fit: a fit output directory (fit.json)",
    "ffield": "the chosen force field",
    "evaluate": "04_validate: evaluate output directory (evaluate.json)",
    "md_check": "04_validate: md-check output directory (md_check.json)",
    "refine": "05_refine: refinement rounds (refine_status.json)",
    "usage": "usage_report.json from usage",
    "deploy": "06_deploy: delivered force field package",
}
SCHEMA = {
    "type": "object",
    "properties": {
        "init": {"type": ["string", "null"], "description": "Create a study at this directory (on a shared filesystem for HPC work)."},
        "study": {"type": ["string", "null"], "description": "An existing study directory (or any path inside one)."},
        "name": {"type": ["string", "null"]},
        "goal": {"type": ["string", "null"], "description": "The user's request, verbatim."},
        "elements": {"type": ["array", "null"], "items": {"type": "string"}},
        "register": {"type": ["array", "null"], "items": {"type": "string"},
                     "description": "key=path entries; keys: " + ", ".join(KNOWN_ARTIFACTS) + "."},
        "extra_roots": {"type": ["array", "null"], "items": {"type": "string"},
                        "description": "Other directories whose Slurm jobs belong to this study (for usage accounting)."},
        "status": {"type": "boolean", "default": False,
                   "description": "Add a one-screen dashboard (phases, jobs, what waits on the user, CPU-hours) and write STATUS.md."},
        "note": {"type": ["string", "null"], "description": "Add a note to the study journal: a decision and its reason, a finding, an issue, a plan or a result -- what the automatic stage entries cannot say."},
        "note_kind": {"type": "string", "enum": ["decision", "finding", "issue", "plan", "result", "note"], "default": "note"},
        "note_phase": {"type": ["string", "null"], "description": "data, fit, active learning, md, benchmark, deploy, ... (free text)."},
        "note_refs": {"type": ["array", "null"], "items": {"type": "string"}, "description": "Paths or job ids the note is about."},
        "note_at": {"type": ["string", "null"], "description": "ISO timestamp for a note recorded after the fact (backfilling a study's history); default now."},
        "journal_backfill": {"type": "boolean", "default": False, "description": "For a study begun before the journal existed: add (backfilled) entries for every stage output directory found on disk, from its manifest."},
        "journal": {"type": "boolean", "default": False, "description": "Render the journal (every stage run in the study + notes) to JOURNAL.md and return its path and counts."},
    },
}


def add_arguments(parser) -> None:
    parser.add_argument("--init", default=None)
    parser.add_argument("--study", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--goal", default=None)
    parser.add_argument("--elements", type=lambda s: [x for x in s.split(",") if x], default=None)
    parser.add_argument("--register", action="append", default=None, help="key=path (repeatable)")
    parser.add_argument("--extra-root", dest="extra_roots", action="append", default=None)
    parser.add_argument("--status", action="store_true", default=False,
                        help="One-screen dashboard: phases, jobs, what waits on the user, CPU-hours (also written to STATUS.md).")
    parser.add_argument("--note", default=None, help="Add a note (decision, finding, issue, plan, result) to the study journal.")
    parser.add_argument("--note-kind", dest="note_kind", default="note", choices=["decision", "finding", "issue", "plan", "result", "note"])
    parser.add_argument("--note-phase", dest="note_phase", default=None)
    parser.add_argument("--note-ref", dest="note_refs", action="append", default=None)
    parser.add_argument("--note-at", dest="note_at", default=None)
    parser.add_argument("--journal", action="store_true", default=False, help="Render JOURNAL.md from journal.jsonl.")
    parser.add_argument("--journal-backfill", dest="journal_backfill", action="store_true", default=False)


def find_study(path) -> Path | None:
    """The nearest ancestor (or self) containing study.json."""
    p = Path(path).resolve()
    for d in [p] + list(p.parents):
        if (d / "study.json").is_file():
            return d
    return None


def load(root: Path) -> dict:
    return json.loads((root / "study.json").read_text())


def save(root: Path, data: dict) -> None:
    atomic.write_json((root / "study.json"), data, indent=1)


def artifact(root: Path, key: str):
    """Registered path for `key`, else the standard-layout default if present."""
    data = load(root)
    reg = data.get("artifacts", {})
    if key in reg:
        return reg[key]
    defaults = {
        "structures": root / "01_data" / "structures.extxyz",
        "trainset": root / "02_trainset" / "trainset.yaml",
        "validset": root / "02_trainset" / "validset.yaml",
        "params_spec": root / "03_fit" / "params.yaml",
        "sensitivity": root / "03_fit" / "sensitivity" / "sensitivity.json",
        "usage": root / "usage" / "usage_report.json",
        "deploy": root / "06_deploy",
    }
    d = defaults.get(key)
    return str(d) if d is not None and d.exists() else None


def _status(root: Path) -> dict:
    data = load(root)
    arts = {k: artifact(root, k) for k in KNOWN_ARTIFACTS}
    return {"study": str(root), "name": data.get("name"), "goal": data.get("goal"), "elements": data.get("elements"),
            "created_at": data.get("created_at"), "artifacts": arts,
            "missing": [k for k, v in arts.items() if not v],
            "extra_roots": data.get("extra_roots", [])}


PHASES = (
    ("Starting force field", ("start_ffield",), "ffield-info / ffield-library / ffield-merge; reaxff-literature (branches)"),
    ("Reference data", ("structures",), "reaxff-trainset-designer: data-generate -> label (--machine for DFT)"),
    ("Training set", ("trainset", "validset"), "trainset-build (and --split for the validation set)"),
    ("Parameters", ("params_spec",), "params-select (preset per fitting stage), sensitivity"),
    ("Fit", ("fit", "ffield"), "reaxff-fitter: fit (several seeds), then register the chosen ffield"),
    ("Validation", ("evaluate", "md_check"), "reaxff-validator: evaluate (vs start ffield), md-check"),
    ("Refinement", ("refine",), "refine rounds: md-check -> select -> label -> trainset-build --merge -> fit"),
    ("Deploy", ("deploy",), "deploy --study"),
)
_PRUNE = {"__pycache__", ".git", "deps", "runs"}


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return None


def _artifact_json(path, name):
    """A registered artifact may be the JSON file or its directory (possibly with a run/ level)."""
    if not path:
        return None
    p = Path(path)
    if p.is_file():
        return _read(p)
    for cand in (p / name, p / "run" / name):
        if cand.is_file():
            return _read(cand)
    return None


def _headline(root: Path, arts: dict) -> dict:
    """One short fact per phase, read from the artifacts that exist."""
    out = {}
    fit = _artifact_json(arts.get("fit"), "fit.json")
    if fit:
        best = fit.get("best") or {}
        out["Fit"] = (f"loss {best.get('loss'):.4g} from {fit.get('n_runs', 1)} run(s), "
                      f"{fit.get('n_evals_total')} evaluations") if isinstance(best.get("loss"), (int, float)) else "fit finished"
    ev = _artifact_json(arts.get("evaluate"), "evaluate.json")
    if ev and ev.get("ffields"):
        f0 = ev["ffields"][0]
        kinds = (f0.get("train") or {}).get("kinds") or {}
        e = kinds.get("energy", {}).get("rmse")
        out["Validation"] = f"energy RMSE {e:.2f} kcal/mol (train)" if isinstance(e, float) else "evaluated"
    mc = _artifact_json(arts.get("md_check"), "md_check.json")
    if mc:
        out["Validation"] = (out.get("Validation", "") + f"; MD: {mc.get('verdict')}").strip("; ")
    rf = _artifact_json(arts.get("refine"), "refine_status.json")
    if rf:
        out["Refinement"] = f"{rf.get('verdict')} after {rf.get('n_rounds')} round(s)"
    return out


def _jobs(roots) -> tuple:
    """(submitted jobs, prepared-but-unsubmitted job directories) under the study, with one squeue call."""
    import getpass
    import os
    import subprocess

    submitted, prepared = [], []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _PRUNE and not d.startswith("frame_")]
            if dirpath[len(str(root)):].count(os.sep) >= 6:
                dirnames[:] = []
            if "job.json" in filenames:
                rec = _read(Path(dirpath) / "job.json") or {}
                if rec.get("job_id"):
                    submitted.append({"work_dir": dirpath, **{k: rec.get(k) for k in ("job_id", "job_name", "submitted_at", "expect")}})
            elif "run.cmd" in filenames and "stdoutmsg" not in filenames:
                # a rendered job script with no job record and no Slurm output: a dry run nobody has submitted
                prepared.append(dirpath)
    queue = {}
    try:
        q = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i|%T|%M"], capture_output=True, text=True, timeout=20)
        for ln in q.stdout.splitlines():
            t = ln.split("|")
            if len(t) == 3:
                queue[t[0]] = (t[1], t[2])
    except (OSError, subprocess.TimeoutExpired):
        queue = None
    for j in submitted:
        expect = j.pop("expect") or []
        if queue is not None and str(j["job_id"]) in queue:
            state, elapsed = queue[str(j["job_id"])]
            j.update({"state": "QUEUED" if state == "PENDING" else "RUNNING", "elapsed": elapsed})
        elif expect and all(Path(e).is_file() for e in expect):
            j["state"] = "FINISHED"
        elif expect:
            j["state"] = "NO_RESULTS"       # left the queue without its result files: job-status says why
        else:
            j["state"] = "FINISHED?" if queue is not None else "UNKNOWN"
    submitted.sort(key=lambda j: -(j.get("submitted_at") or 0))
    return submitted, sorted(prepared)


def dashboard(root: Path) -> dict:
    from .. import journal

    data = load(root)
    journal_recent = journal.read(root)[-12:]
    arts = {k: artifact(root, k) for k in KNOWN_ARTIFACTS}
    facts = _headline(root, arts)
    phases = []
    for name, keys, how in PHASES:
        have = [k for k in keys if arts.get(k)]
        state = "done" if len(have) == len(keys) else "partial" if have else "not started"
        phases.append({"phase": name, "state": state, "have": have, "missing": [k for k in keys if k not in have],
                       "headline": facts.get(name), "how": how})
    report = root / "REPORT.md"
    narrative_left = report.is_file() and "<!-- NARRATIVE" in report.read_text()
    phases.append({"phase": "Report", "state": "partial" if narrative_left else "done" if report.is_file() else "not started",
                   "have": ["REPORT.md"] if report.is_file() else [], "missing": [] if report.is_file() else ["REPORT.md"],
                   "headline": "narrative placeholders remain" if narrative_left else None,
                   "how": "study-report --study, then the reaxff-report-writer agent"})
    roots = [root] + [Path(p) for p in data.get("extra_roots", []) if Path(p).is_dir()]
    submitted, prepared = _jobs(roots)
    active = [j for j in submitted if j["state"] in ("QUEUED", "RUNNING")]
    problems = [j for j in submitted if j["state"] == "NO_RESULTS"]
    waiting = [f"approve or discard the prepared job in {p}" for p in prepared]
    writers = journal.recent_writers(root, minutes=60)
    if len(writers) > 1:
        waiting.insert(0, "several login hosts ran stages in this study in the last hour (" +
                       ", ".join(f"{h}: {n}" for h, n in sorted(writers.items())) +
                       "): is more than one agent or person driving it? Settle who owns the study before submitting more")
    waiting += [f"job {j['job_id']} ({j['job_name']}) left the queue without results: reaxff-agent job-status --work-dir {j['work_dir']}"
                for j in problems]
    usage = _read(arts.get("usage")) if arts.get("usage") else None
    local_s = 0.0
    ledger = root / "usage" / "local.jsonl"
    if ledger.is_file():
        for ln in ledger.read_text().splitlines():
            try:
                local_s += float(json.loads(ln).get("cpu_s", 0.0))
            except ValueError:
                pass
    cpu = {"login_node_cpu_hours": round(local_s / 3600.0, 3)}
    if usage:
        cpu.update({"slurm_charged_cpu_hours": usage.get("total_cpu_hours"), "slurm_used_cpu_hours": usage.get("total_used_cpu_hours")})
    else:
        cpu["note"] = "Slurm CPU-hours not collected yet: reaxff-agent usage --study <dir>"
    nxt = next((p for p in phases if p["state"] != "done"), None)
    return {"journal_recent": journal_recent, "phases": phases, "jobs_active": active, "jobs_recent": submitted[:5], "n_jobs_submitted": len(submitted),
            "waiting_on_user": waiting, "cpu_hours": cpu,
            "next": f"{nxt['phase']} ({nxt['state']}): {nxt['how']}" if nxt else "all phases done"}


def render_dashboard(status: dict, dash: dict) -> str:
    mark = {"done": "[x]", "partial": "[~]", "not started": "[ ]"}
    L = [f"# {status.get('name')}: status", "", f"Goal: {status.get('goal') or '(none recorded)'}", "", "## Phases", ""]
    for p in dash["phases"]:
        tail = f": {p['headline']}" if p.get("headline") else ""
        miss = f" (missing: {', '.join(p['missing'])})" if p["state"] == "partial" and p["missing"] else ""
        L.append(f"- {mark[p['state']]} {p['phase']}{tail}{miss}")
    L += ["", f"Next: {dash['next']}", "", "## Jobs", ""]
    if dash["jobs_active"]:
        L += [f"- {j['state']} {j['job_id']} {j['job_name']} ({j.get('elapsed')}) in {j['work_dir']}" for j in dash["jobs_active"]]
    else:
        L.append(f"- none queued or running ({dash['n_jobs_submitted']} submitted so far)")
    L += ["", "## Waiting on you", ""] + ([f"- {w}" for w in dash["waiting_on_user"]] or ["- nothing"])
    c = dash["cpu_hours"]
    L += ["", "## CPU-hours", ""]
    if "slurm_charged_cpu_hours" in c:
        L.append(f"- Slurm: {c['slurm_charged_cpu_hours']} charged, {c['slurm_used_cpu_hours']} used")
    else:
        L.append(f"- {c['note']}")
    L.append(f"- login node: {c['login_node_cpu_hours']}")
    recent = dash.get("journal_recent") or []
    if recent:
        L += ["", "## Recent journal (newest last; full record: study --journal)", ""]
        for e in recent:
            when = e.get("at", "")[:16].replace("T", " ")
            if e.get("type") == "note":
                text = e.get("text", "")
                L.append(f"- {when} {e.get('kind')}: {text[:220]}{'...' if len(text) > 220 else ''}")
            else:
                who = f" from {e['host']}" if e.get("host") else ""
                jobs = f" jobs {e.get('chunk_job_ids') or e.get('job_ids') or e.get('job_id')}" if (e.get("job_id") or e.get("job_ids")) else ""
                L.append(f"- {when} {e.get('stage')} {e.get('status')} -> {e.get('output_dir')}{jobs}{who}")
    return "\n".join(L) + "\n"


def run(args) -> dict:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    if getattr(args, "init", None):
        root = Path(args.init).resolve()
        fs.ensure_dir(root)
        if (root / "study.json").is_file():
            raise ValueError(f"{root} is already a study; use --study {root}")
        for d in LAYOUT:
            (root / d).mkdir(exist_ok=True)
        data = {"name": getattr(args, "name", None) or root.name, "goal": getattr(args, "goal", None),
                "elements": getattr(args, "elements", None), "created_at": now, "artifacts": {},
                "extra_roots": [str(Path(p).resolve()) for p in (getattr(args, "extra_roots", None) or [])],
                "log": [{"at": now, "event": "created"}]}
        save(root, data)
        if not (root / "STUDY.md").exists():
            (root / "STUDY.md").write_text(f"# {data['name']}\n\n**Goal:** {data['goal'] or '(fill in)'}\n\n"
                                           "## Decisions\n\n## Phase status\n")
    else:
        where = getattr(args, "study", None)
        if not where:
            raise ValueError("give --init DIR or --study DIR")
        root = find_study(where)
        if root is None:
            raise ValueError(f"no study.json at or above {where}")
        data = load(root)
        changed = False
        for k in ("name", "goal", "elements"):
            if getattr(args, k, None):
                data[k] = getattr(args, k)
                changed = True
        for p in getattr(args, "extra_roots", None) or []:
            rp = str(Path(p).resolve())
            if rp not in data.setdefault("extra_roots", []):
                data["extra_roots"].append(rp)
                changed = True
        for entry in getattr(args, "register", None) or []:
            key, _, path = entry.partition("=")
            if key not in KNOWN_ARTIFACTS:
                raise ValueError(f"unknown artifact key {key!r}; known: {sorted(KNOWN_ARTIFACTS)}")
            if not Path(path).exists():
                raise FileNotFoundError(f"{key}: {path} does not exist")
            path = str(Path(path).resolve())
            if key == "md_runs":
                runs = data.setdefault("artifacts", {}).setdefault("md_runs", [])
                if path not in runs:
                    runs.append(path)
            else:
                data.setdefault("artifacts", {})[key] = path
            data.setdefault("log", []).append({"at": now, "event": f"registered {key}={path}"})
            changed = True
        if changed:
            save(root, data)
    from .. import journal

    if getattr(args, "init", None):
        journal.add_note(root, f"Study created. Goal: {data.get('goal') or '(none given)'}; elements: {data.get('elements')}",
                         kind="plan")
    if getattr(args, "note", None):
        note = journal.add_note(root, args.note, kind=getattr(args, "note_kind", "note") or "note",
                                phase=getattr(args, "note_phase", None), refs=getattr(args, "note_refs", None),
                                at=getattr(args, "note_at", None))
    backfilled = journal.backfill_from_manifests(root) if getattr(args, "journal_backfill", False) else None
    status = _status(root)
    if backfilled is not None:
        status["journal_backfilled"] = backfilled
    if getattr(args, "note", None):
        status["note_added"] = note
    if getattr(args, "journal", False):
        entries = journal.read(root)
        status["journal_md"] = str(journal.render(root, title=data.get("name")))
        status["journal_counts"] = {"stage_runs": sum(e.get("type") == "stage" for e in entries),
                                    "notes": sum(e.get("type") == "note" for e in entries),
                                    "failed": sum(e.get("status") == "failed" for e in entries)}
    if getattr(args, "status", False):
        dash = dashboard(root)
        text = render_dashboard(status, dash)
        atomic.write_text(root / "STATUS.md", text)
        status.update({"dashboard": dash, "dashboard_text": text, "status_md": str(root / "STATUS.md")})
    return status
