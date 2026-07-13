# Mutation Master Plan — Task 5b: Tests for `orchestrator/membrane.py`

**Date:** 2026-07-12
**Source:** `mutation-plans/Process-plan-master.md` Task 5 (second half). Independent of
the mutation work. Completes Task 5.

## What was done

Same local+Modal split as Task 5a (ligands).

### 1. `tests/test_membrane.py` — local, fully mocked (20 tests)
- **`_resolve_insane`** — 3-tier lookup via patched `os.path.isfile` + `shutil.which`:
  config path is a file (PATH not consulted), `insane.py` on PATH, compiled `insane`
  fallback, and nothing-found → `None`.
- **`_lipid_name`** — parametrized: `None`/`""` → `POPC`, lowercase known → mapped upper,
  unknown → uppercased passthrough.
- **`embed_in_membrane_gromacs`** — patched `_resolve_insane` + `subprocess.run`, real
  `tmp_path`: insane-not-found → `RuntimeError`; success returns `(gro, top)` and puts
  `POPC:1` in `-l`; `span` present → `-center` in command; no `span` → no `-center`;
  non-zero exit → `RuntimeError` (with output); exit-0-but-missing-outputs →
  `RuntimeError`.
- **`embed_in_membrane_openmm`** — fake `openmm` / `openmm.app` / `openmmforcefields`
  injected via `sys.modules`: openmm-missing → `RuntimeError`; openmmforcefields-missing
  → `RuntimeError`; `addMembrane` raising → wrapped `RuntimeError` naming the lipid
  (`'POPC'`); success → returns the same modeller and passes normalised `lipidType`.

### 2. `modal_app.py::test_membrane_modal` — real binaries on Modal
CPU `@app.function` (no GPU), run with `modal run modal_app.py::test_membrane_modal`.
Mirrors the real `simulation.py` OpenMM path exactly:
`PDBFixer → ForceField("charmm36.xml","charmm36/water.xml","charmm36/lipids.xml") →
Modeller.addHydrogens(pH=7) → embed_in_membrane_openmm` (real `Modeller.addMembrane`) on
a small soluble peptide, asserting the atom count grows after membrane+water are added.
Each stage returns diagnostics (`*_ok` / `*_error`) rather than hard-crashing.

## Modal image coverage note
The Modal image has `openmm` **and** `openmmforcefields` (both via micromamba), so
`test_membrane_modal` genuinely exercises the CHARMM36m `addMembrane` builder. It does
**not** have insane.py (Tieleman-lab download, like gnina/acpype), so
`embed_in_membrane_gromacs` has **mocked-only** coverage — flagged rather than implied.
Real coverage there would require adding insane.py to the image.

## Verification
`python -m pytest tests/test_membrane.py -q` in the `ProPredict` conda env: **20 passed**.
Full suite (`test_membrane + test_ligands + test_agent + test_mutation_scan +
test_boltz`): **80 passed, 3 skipped** (skips: 2 Boltz GPU + 1 ProteinMPNN integration).
`modal_app.py` parses clean.

## Status
This completes **Task 5** and, with it, all six tasks of the resequenced mutation master
plan (Tasks 0–5). See the other `Process/mutation-task-*.md` write-ups for the rest.
