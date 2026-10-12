"""Unit conventions.

Everything inside the toolkit (training-set reference values, predictions,
errors, sigmas) is in LAMMPS `real` units, the units ReaxFF parameters are
defined in:

    energy  kcal/mol        force   kcal/mol/Å      distance  Å
    angle   degree          charge  e               stress    GPa (converted from atm)

Structure files (`structures.extxyz`) keep ASE's conventions (eV, eV/Å,
eV/Å^3) so any ASE-aware tool reads them correctly; conversion happens when
a training set is built and when predictions are compared.
"""

EV_TO_KCALMOL = 23.060547830619026  # 1 eV = 96.485332123 kJ/mol / 4.184
KCALMOL_TO_EV = 1.0 / EV_TO_KCALMOL
HARTREE_TO_EV = 27.211386245988
RY_TO_EV = HARTREE_TO_EV / 2.0
BOHR_TO_ANG = 0.529177210903
ATM_TO_GPA = 1.01325e-4
EV_PER_A3_TO_GPA = 160.21766208
