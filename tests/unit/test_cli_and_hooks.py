import json
import subprocess
import sys
from pathlib import Path

import pytest

from agentic_reaxff import cli

ROOT = Path(__file__).resolve().parents[2]


def _run(argv, capsys):
    rc = cli.main(argv)
    return rc, json.loads(capsys.readouterr().out)


def test_every_stage_describes(capsys):
    for name in [m.replace("_cmd", "").replace("_", "-") for m in cli.STAGE_MODULE_NAMES]:
        rc, out = _run([name, "--describe"], capsys)
        assert rc == 0 and out["stage"] == name and out["summary"]


def test_json_in_refuses_unknown_keys_and_types(tmp_path, capsys):
    p = tmp_path / "in.json"
    p.write_text(json.dumps({"ffeld": "x"}))
    rc, out = _run(["ffield-info", "--json-in", str(p)], capsys)
    assert rc == 1 and "unknown key" in out["error"]
    p.write_text(json.dumps({"ffield": "x", "diff_tol": "big"}))
    rc, out = _run(["ffield-info", "--json-in", str(p)], capsys)
    assert rc == 1 and "must be" in out["error"]


def test_errors_are_one_json_object(capsys, tmp_path):
    rc, out = _run(["ffield-info", "--ffield", str(tmp_path / "missing")], capsys)
    assert rc == 1 and "error" in out


def test_manifest_short_circuits_and_refuses_changed_inputs(tmp_path, capsys):
    from conftest import make_ffield

    f = tmp_path / "ffield"
    make_ffield().write(f)
    args = ["params-select", "--ffield", str(f), "--preset", "bond", "--output-dir", str(tmp_path / "p")]
    rc, out1 = _run(args, capsys)
    assert rc == 0, out1
    rc, out2 = _run(args, capsys)
    assert rc == 0 and out2["n_active"] == out1["n_active"]
    rc, out3 = _run(args[:-2] + ["--scale", "0.3", "--output-dir", str(tmp_path / "p")], capsys)
    assert rc == 1


def test_dry_run_rejected_by_local_stages(capsys):
    rc, out = _run(["ffield-info", "--ffield", "x", "--dry-run"], capsys)
    assert rc == 1 and "dry-run" in out["error"]


def test_approval_gate():
    sys.path.insert(0, str(ROOT / ".claude" / "hooks"))
    import approval_gate as g

    assert g.reason_to_ask("reaxff-agent fit --trainset t --params p --machine dane")
    assert g.reason_to_ask("reaxff-agent fit --machine dane --dry-run") is None
    assert g.reason_to_ask("python3 -m agentic_reaxff.cli label --structures s")          # qe is the default backend
    assert g.reason_to_ask("reaxff-agent label --backend xtb --structures s") is None
    assert g.reason_to_ask("reaxff-agent label --collect d") is None
    assert g.reason_to_ask("cd x && sbatch run.cmd")
    assert g.reason_to_ask("reaxff-agent evaluate --ffields a") is None
