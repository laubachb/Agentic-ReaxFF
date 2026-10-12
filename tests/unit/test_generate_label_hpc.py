import numpy as np
import pytest
from ase.build import molecule

from agentic_reaxff.generate import builders as B
from agentic_reaxff.io.structures import from_atoms
from agentic_reaxff.labeling import qe as QE


def _mol(name):
    a = molecule(name)
    a.info["name"] = f"{name}_eq"
    return from_atoms(a)


def test_bond_scan_sets_exact_distance_and_restraint():
    scans = B.bond_scans(_mol("C2H6"), factors=[0.9, 1.0, 1.5])
    assert scans
    for s in scans:
        i, j = (int(x) for x in s.info["scan_atoms"].split(","))
        d = np.linalg.norm(s.positions[i] - s.positions[j])
        assert d == pytest.approx(s.info["scan_value"], abs=1e-6)
        assert s.mode == "relax" and s.restraints[0][:3] == ["bond", i, j]
    kinds = {tuple(sorted((s.symbols[int(s.info['scan_atoms'].split(',')[0])], s.symbols[int(s.info['scan_atoms'].split(',')[1])])))
             for s in scans}
    assert kinds == {("C", "C"), ("C", "H")}       # one per element pair


def test_angle_and_torsion_scans():
    a = B.angle_scans(_mol("H2O"), deltas=[-10, 0, 10])
    assert len(a) == 3 and a[0].restraints[0][0] == "angle"
    t = B.torsion_scans(_mol("C2H6"), step=60)
    assert len(t) == 6 and t[0].restraints[0][0] == "dihedral"


def test_dimers_and_pack_box():
    d = B.dimers(["C", "H"])
    assert {s.group for s in d} == {"dimer_CC", "dimer_CH", "dimer_HH"}
    box = B.pack_box({"H2O": 8}, density=0.5, seed=1)
    D = []
    from ase import Atoms

    at = Atoms(box.symbols, positions=box.positions, cell=box.cell, pbc=True)
    dist = at.get_all_distances(mic=True)
    np.fill_diagonal(dist, 10)
    assert dist.min() > 0.8 and box.natoms == 24


def test_qe_input_constraints_spin_and_kpoints(tmp_path):
    scans = B.bond_scans(_mol("C2H6"), factors=[2.0])
    cc = next(s for s in scans if {s.symbols[int(x)] for x in s.info["scan_atoms"].split(",")} == {"C"})
    st = QE.QESettings(pseudo_dir=str(tmp_path), pseudopotentials={"C": "c.UPF", "H": "h.UPF"})
    text = QE.render_pw_in(cc, st, {"C": 4.0, "H": 1.0}, {"C": 12.011, "H": 1.008})
    assert "calculation = 'relax'" in text and "CONSTRAINTS" in text and "distance" in text
    assert "nspin = 2" in text and "C1" in text and "C2" in text     # broken symmetry on a stretched C-C
    assert "ion_dynamics = 'damp'" in text                            # pw.x: no constraints with BFGS
    assert "K_POINTS gamma" in text
    m = _mol("CH4")
    m.mode = "relax"
    eq = QE.render_pw_in(m, st, {"C": 4.0, "H": 1.0}, {"C": 12.0, "H": 1.0})
    assert "nspin" not in eq and "calculation = 'relax'" in eq
    crystal = B.eos(["C:diamond:3.567"], volume_ratios=[1.0])[0]
    t2 = QE.render_pw_in(crystal, st, {"C": 4.0, "H": 1.0}, {"C": 12.0})
    assert "K_POINTS automatic" in t2 and "calculation = 'scf'" in t2


def test_qe_job_command_and_lowdin_parse():
    cmd = QE.frame_command("srun", 28, 112, lowdin=True)
    assert "srun --exact -n 28 -N 1" in cmd and "projwfc.x" in cmd and "rm -rf out" in cmd
    text = """
     Lowdin Charges:

     Atom #   1: total charge =   4.2000, s =  1.2
     Atom #   2: total charge =   0.9000, s =  0.9
     Atom #   3: total charge =   0.9000, s =  0.9

     Spilling Parameter:   0.01
"""
    q = QE.parse_lowdin(text, ["C", "H", "H"], {"C": 4.0, "H": 1.0})
    assert q.sum() == pytest.approx(0.0, abs=1e-9) and q[0] < 0


def test_sbatch_render_always_requests_full_node(tmp_path, monkeypatch):
    monkeypatch.setenv("REAXFF_ACCOUNT", "testbank")
    from agentic_reaxff import hpc, machines

    prof = machines.load_profile("dane")
    h = hpc.submit_job(prof, job_name="t", commands=["echo hi"], work_dir=tmp_path, dry_run=True)
    script = h.job_file.read_text()
    assert "--ntasks-per-node 112" in script and "-A testbank" in script and "-p pdebug" in script
