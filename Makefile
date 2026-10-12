# Convenience targets (all of them work from a fresh clone).
MACHINE ?= generic

.PHONY: install test doctor clean-pycache

install:          ## create .venv, install, set up components, run doctor
	./install.sh --machine $(MACHINE)

test:             ## unit tests (LAMMPS-dependent ones skip if not installed)
	.venv/bin/python -m pytest -q

doctor:
	.venv/bin/reaxff-agent doctor --machine $(MACHINE)

clean-pycache:
	find . -name __pycache__ -prune -exec rm -rf {} +
