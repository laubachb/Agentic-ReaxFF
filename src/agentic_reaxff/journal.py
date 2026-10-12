"""The study journal: an append-only record of everything done in a study.

Every `reaxff-agent` stage that writes inside a study directory (an
`--output-dir`, `--study` or `--collect` path below a `study.json`) appends
one entry to `<study>/journal.jsonl`: when, which stage, the inputs that
differ from the stage's defaults, a compact summary of the outputs (scalars,
job ids, errors), the status (done, failed, dry run, short-circuited), wall
time, host and the toolkit version (git commit, `+dirty` when the working
tree has uncommitted changes). Entries are written by the CLI wrapper, so no
stage can forget to log itself.

What commands cannot capture -- why a choice was made, what a result means,
what went wrong -- goes in as notes: `reaxff-agent study --study S --note
"..." --note-kind decision|finding|issue|plan|result`. `study --journal`
renders the whole history as `JOURNAL.md`, grouped by day and tagged by
phase; the report writer and case-study pages are written from it.

Runs inside Slurm jobs are not journaled one by one (a chunk of 112 MD runs
would bury the record); the submission is, with its job ids, and so is
whatever later collects or analyzes the job's output. The journal must never
break a stage: every failure here is swallowed.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
from pathlib import Path

JOURNAL = "journal.jsonl"
NOTE_KINDS = ("decision", "finding", "issue", "plan", "result", "note")
PHASES = {"01_data": "data", "02_trainset": "trainset", "03_fit": "fit", "04_validate": "validate",
          "05_refine": "refine", "06_deploy": "deploy", "usage": "accounting"}
_MAX_STR = 300
_MAX_LIST = 12

_version_cache = None


def tool_version() -> str:
    """git short commit of the toolkit checkout, '+dirty' with uncommitted changes."""
    global _version_cache
    if _version_cache is None:
        root = Path(__file__).resolve().parents[2]
        try:
            sha = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                 timeout=20).stdout.strip()
            dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                   text=True, timeout=20).stdout.strip()
            _version_cache = (sha or "unknown") + ("+dirty" if dirty else "")
        except Exception:  # noqa: BLE001
            _version_cache = "unknown"
    return _version_cache


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _compact(value, depth: int = 0):
    """JSON-safe, size-bounded copy: long strings cut, long lists summarized,
    nesting limited. Enough to see what was run and what came out."""
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    if isinstance(value, str):
        return value if len(value) <= _MAX_STR else value[:_MAX_STR] + f"... (+{len(value) - _MAX_STR} chars)"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        if depth >= 2:
            return f"[{len(value)} items]"
        items = [_compact(v, depth + 1) for v in value[:_MAX_LIST]]
        return items + ([f"... (+{len(value) - _MAX_LIST} more)"] if len(value) > _MAX_LIST else [])
    if isinstance(value, dict):
        if depth >= 2:
            return f"{{{len(value)} keys}}"
        return {str(k): _compact(v, depth + 1) for k, v in list(value.items())[:40]}
    return str(value)[:_MAX_STR]


def summarize_outputs(result: dict) -> dict:
    """Scalars, short lists and job/error fields of a stage result; big
    nested blocks (per-frame tables, reports) are reduced to their size."""
    if not isinstance(result, dict):
        return {}
    keep = {}
    for k, v in result.items():
        if k in ("stage_log",):
            continue
        if isinstance(v, (dict, list)) and len(json.dumps(v, default=str)) > 2000:
            keep[k] = f"<{type(v).__name__}, {len(v)} entries>"
        else:
            keep[k] = _compact(v)
    return keep


def phase_of(root: Path, path) -> str | None:
    try:
        rel = Path(path).resolve().relative_to(Path(root).resolve())
    except (ValueError, OSError, TypeError):
        return None
    return PHASES.get(rel.parts[0]) if rel.parts else None


def _rel(root: Path, path):
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except (ValueError, OSError, TypeError):
        return str(path) if path else None


def append(root: Path, entry: dict) -> dict:
    root = Path(root)
    entry = {"at": _now(), **entry}
    with open(root / JOURNAL, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    return entry


def read(root: Path) -> list:
    path = Path(root) / JOURNAL
    if not path.is_file():
        return []
    out = []
    for ln in path.read_text().splitlines():
        try:
            out.append(json.loads(ln))
        except ValueError:
            continue  # a line cut by a crash; skip it
    return out


def record_stage(root: Path, *, stage: str, inputs: dict, defaults: dict, status: str, result=None, error=None,
                 wall_s=None, output_dir=None) -> None:
    """One journal entry for a stage invocation. Inputs equal to the stage's
    defaults are left out (the stage's --describe documents them)."""
    try:
        if os.environ.get("SLURM_JOB_ID"):
            return
        changed = {k: _compact(v) for k, v in inputs.items() if v is not None and v != defaults.get(k, object())
                   and not (v is False and k not in defaults)}
        entry = {"type": "stage", "stage": stage, "status": status, "phase": phase_of(root, output_dir),
                 "output_dir": _rel(root, output_dir), "inputs": changed, "wall_s": round(wall_s, 2) if wall_s else None,
                 "host": os.uname().nodename, "tool_version": tool_version()}
        if result is not None:
            entry["outputs"] = summarize_outputs(result)
            for k in ("job_id", "job_ids", "chunk_job_ids"):
                if result.get(k):
                    entry[k] = result[k]
        if error:
            entry["error"] = _compact(str(error))
        append(root, entry)
    except Exception:  # noqa: BLE001 - the journal must never break a stage
        pass


def add_note(root: Path, text: str, kind: str = "note", phase: str = None, refs=None, at: str = None) -> dict:
    if kind not in NOTE_KINDS:
        raise ValueError(f"note kind must be one of {NOTE_KINDS}, not {kind!r}")
    entry = {"type": "note", "kind": kind, "phase": phase, "text": text, "refs": list(refs or []), "tool_version": tool_version()}
    if at:
        entry["at"] = at
        with open(Path(root) / JOURNAL, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry
    return append(root, entry)


def _fmt_inputs(inputs: dict) -> str:
    parts = []
    for k, v in (inputs or {}).items():
        sv = json.dumps(v, default=str) if not isinstance(v, str) else v
        parts.append(f"`{k}`={sv if len(sv) <= 80 else sv[:77] + '...'}")
    return ", ".join(parts)


_HEADLINE_KEYS = ("n_frames", "n_converged", "n_submitted", "n_already_done", "n_harvested", "status", "step", "target_step",
                  "relative_force_error", "reduced_force_rmse", "rmse_force_kcal_mol_ang", "recommended", "n_replicas",
                  "lanes", "mpi_ranks", "natoms_range", "warnings")


def _fmt_outputs(entry: dict) -> str:
    out = entry.get("outputs") or {}
    bits = [f"{k}={json.dumps(out[k], default=str)[:120]}" for k in _HEADLINE_KEYS if k in out and out[k] not in (None, [], {})]
    for k in ("job_id", "job_ids", "chunk_job_ids"):
        if entry.get(k):
            bits.append(f"{k}={entry[k]}")
    return "; ".join(bits)


def render(root: Path, *, title: str = None) -> Path:
    """JOURNAL.md: the study's history, oldest first, grouped by day."""
    root = Path(root)
    entries = sorted(read(root), key=lambda e: e.get("at", ""))
    lines = [f"# Journal: {title or root.name}", "",
             "Every stage run in this study (written automatically by `reaxff-agent`) and every decision, finding and",
             "issue noted along the way (`reaxff-agent study --note`). Generated from `journal.jsonl` by",
             "`reaxff-agent study --journal`; do not edit by hand.", ""]
    counts = {}
    for e in entries:
        key = e.get("kind") if e.get("type") == "note" else e.get("status")
        counts[key] = counts.get(key, 0) + 1
    lines += ["Totals: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())), ""]
    day = None
    for e in entries:
        d = e.get("at", "")[:10]
        if d != day:
            day = d
            if lines[-1] != "":
                lines.append("")
            lines += [f"## {d}", ""]
        t = e.get("at", "")[11:16]
        ph = f" [{e['phase']}]" if e.get("phase") else ""
        if e.get("type") == "note":
            text = e.get("text", "").replace("\n", "\n  ")
            refs = f" (see {', '.join(f'`{r}`' for r in e.get('refs') or [])})" if e.get("refs") else ""
            lines += [f"- **{t} {e.get('kind', 'note')}**{ph}: {text}{refs}"]
        else:
            status = e.get("status")
            flag = {"failed": " **FAILED**", "dry_run": " (dry run)", "short_circuit": " (unchanged inputs: reused)"}.get(status, "")
            where = f" → `{e['output_dir']}`" if e.get("output_dir") else ""
            lines += [f"- {t} `{e.get('stage')}`{flag}{ph}{where}"]
            if e.get("inputs"):
                lines += [f"  - inputs: {_fmt_inputs(e['inputs'])}"]
            o = _fmt_outputs(e)
            if o:
                lines += [f"  - outputs: {o}"]
            if e.get("error"):
                lines += [f"  - error: {e['error']}"]
    path = root / "JOURNAL.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def backfill_from_manifests(root: Path) -> int:
    """Stage entries for a study begun before the journal existed, rebuilt
    from each output directory's `.manifest-<stage>.json` (the latest run
    there: status, outputs, finish time; inputs are not kept by manifests).
    Directories below another manifest directory are runs inside a job
    (e.g. one trajectory of a submitted batch) and are skipped. Entries are
    marked `backfilled`; directories already in the journal are skipped, so
    running it twice adds nothing."""
    root = Path(root).resolve()
    known = {e.get("output_dir") for e in read(root) if e.get("type") == "stage"}
    found = sorted(root.rglob(".manifest-*.json"))
    dirs = {m.parent for m in found}
    n = 0
    for m in found:
        if any(a in dirs for a in m.parent.parents if a != m.parent and root in (a, *a.parents)):
            continue
        rel = _rel(root, m.parent)
        if rel in known:
            continue
        try:
            d = json.loads(m.read_text())
        except (OSError, ValueError):
            continue
        try:
            at = datetime.datetime.fromtimestamp(float(d.get("finished_at")), datetime.timezone.utc).isoformat(timespec="seconds")
        except (TypeError, ValueError):
            at = None
        outputs = d.get("outputs") or {}
        entry = {"at": at or _now(), "type": "stage", "stage": d.get("stage"), "status": d.get("status"), "backfilled": True,
                 "phase": phase_of(root, m.parent), "output_dir": rel, "inputs": {}, "outputs": summarize_outputs(outputs),
                 "tool_version": "unknown (backfilled)"}
        for k in ("job_id", "job_ids", "chunk_job_ids"):
            if isinstance(outputs, dict) and outputs.get(k):
                entry[k] = outputs[k]
        with open(root / JOURNAL, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        n += 1
    return n


def recent_writers(root: Path, minutes: float = 60.0) -> dict:
    """Login hosts that ran stages in this study within the last `minutes`
    (host -> number of stage runs). Two hosts at once usually means two agents
    or people are driving the same study, which ends in duplicate jobs or
    overwritten results."""
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=minutes)
    out = {}
    for e in read(root):
        if e.get("type") != "stage" or not e.get("host") or e.get("backfilled"):
            continue
        try:
            at = datetime.datetime.fromisoformat(e["at"])
        except (KeyError, ValueError):
            continue
        if at >= cutoff:
            out[e["host"]] = out.get(e["host"], 0) + 1
    return out
