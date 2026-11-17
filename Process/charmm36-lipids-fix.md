# CHARMM36 membrane force field: remove nonexistent `charmm36/lipids.xml`

**Date:** 2026-11-17
**Status:** Complete. Found while verifying step 2 of Process/pyproject-adoption.md.

## Problem

Every CHARMM36 membrane force field was loaded as
`ForceField("charmm36.xml", "charmm36/water.xml", "charmm36/lipids.xml")`. **OpenMM ships no
`charmm36/lipids.xml`**: the lipid parameters (POPC, POPE, DPPC, DMPC, …) are defined in
`charmm36.xml` itself. Checked against OpenMM 8.6.1 in the worker image. Its `app/data` holds
`charmm36.xml`, `charmm36_2024.xml` and `charmm36/{water,spce,tip3p-pme-b,tip3p-pme-f,tip4pew,
tip4p2005,tip5p,tip5pew}.xml`, and the lipid templates are in `charmm36.xml`. Modal's OpenMM 8.5.2
fails the same way.

So `ForceField(...)` **always raised** `ValueError: Could not locate file "charmm36/lipids.xml"`:

| Site | Effect |
|---|---|
| `orchestrator/simulation.py` (OpenMM path, `membrane_context` set) | caught by `except Exception` → **silently fell back to AMBER14** (`amber14-all.xml` includes Lipid17, so `addMembrane` still worked). Every OpenMM membrane run so far was AMBER14, not CHARMM36m; the only trace is a `CHARMM36m XML not found` warning in the logs. |
| `modal_app.py::test_membrane_modal` | failed at step 2, through an early `return` with no `print`, so `modal run` showed nothing. Process/mutation-task-5b-membrane-tests.md records only a parse check, so this test had never passed. |
| `orchestrator/membrane.py` docstring | showed the same broken call. |

## Change

- **`orchestrator/membrane.py`**: new constant `CHARMM36_MEMBRANE_FORCEFIELDS =
  ("charmm36.xml", "charmm36/water.xml")`. The docstring now uses it.
- **`orchestrator/simulation.py`**: `ForceField(*CHARMM36_MEMBRANE_FORCEFIELDS)`, so the real code
  and the regression test read one list. The AMBER14 fallback is **kept** (see below).
- **`modal_app.py::test_membrane_modal`**: same two files (it mirrors the pipeline, and the Modal
  image only mounts `orchestrator/`, so it keeps a literal), plus `print(results)` on the early-return path.
- **`tests/test_membrane.py`**: `test_charmm36_membrane_forcefields_load_with_lipid_templates`.
  It loads the real constant with real OpenMM and asserts POPC/POPE/DPPC/DMPC templates exist.
  `importorskip("openmm.app")`, so it skips on the host (no OpenMM) and runs in the worker image.

### Decisions

- **Kept `charmm36.xml` rather than switching to `charmm36_2024.xml`.** Picking a force-field
  version is a scientific choice, and this is a bug fix: it makes the code do what it always said
  it did.
- **AMBER14 fallback left as-is**, even though it is what hid this bug for so long. Turning it into
  a hard failure, or recording which force field ran in the result, is a behaviour change worth
  doing separately.
- **Results impact:** no membrane result produced so far used CHARMM36m. Any of them that matter
  should be re-run.

## Verification

- **Worker image, real OpenMM 8.6.1** (`docker run … propredict-celery_worker`, tree mounted read-only):
  - `tests/test_membrane.py` → 21 passed, 0 skipped (the new test ran for real).
  - Real membrane setup on `myprotein.pdb` (33 residues): old file list →
    `ValueError Could not locate file "charmm36/lipids.xml"`; new list loads;
    `addHydrogens` → `embed_in_membrane_openmm(POPC)` → 567 → 30,542 atoms, 110 lipids
    (OpenMM names POPC residues `POP`), 5,068 waters, Na⁺/Cl⁻; `ff.createSystem` builds the full
    CHARMM36 system.
- **Modal, `test_membrane_modal`** (served from cache, no image rebuild): `charmm36_ff_ok: True`,
  `addmembrane_ok: True`, 56 → 31,438 atoms. Two earlier attempts were cut off by a local network
  drop (client heartbeat lost, then DNS failure), not by the code.
- Host: `ruff check` / `ruff format --check` clean; `tests/test_membrane.py` + `tests/test_boltz.py`
  44 passed, 3 skipped (the new test skips without OpenMM).

## Sources

- OpenMM bundled force fields:
  http://docs.openmm.org/latest/userguide/application/02_running_sims.html#force-fields
- `Modeller.addMembrane`:
  http://docs.openmm.org/latest/api-python/generated/openmm.app.modeller.Modeller.html
