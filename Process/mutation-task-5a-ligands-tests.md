# Mutation Master Plan — Task 5a: Tests for `orchestrator/ligands.py`

**Date:** 2026-07-12
**Source:** `mutation-plans/Process-plan-master.md` Task 5 (split: ligands first,
membrane next). Independent of the mutation work.

## What was done

Two complementary layers, per the "real binaries on Modal, mocked locally" decision:

### 1. `tests/test_ligands.py` — local, fully mocked (27 tests)
None of RDKit / Vina / meeko / GNINA / ACPYPE / OpenFF are installed in the local dev
env, so every test is mocked and runs anywhere:
- **`smiles_to_3d`** — fake `rdkit` + `rdkit.Chem` injected via `sys.modules`:
  RDKit-missing → `RuntimeError` (real ImportError), invalid SMILES
  (`MolFromSmiles`→None) → `ValueError`, ETKDG embed fail (`EmbedMolecule`→-1) →
  `RuntimeError`, and success → returns `<out_dir>/<name>.sdf`.
- **`_ca_centroid` / `_all_ca_coords`** — pure-Python PDB parsing against a synthetic
  PDB (CA subset centroid, HETATM "CA" calcium ignored, non-matching residues → all-CA
  fallback, missing file → `[]` / `(0,0,0)`).
- **`dock_gnina`** — mocked `shutil.which` + `subprocess.run`: binary-missing, binding-
  site command (`--center_x/--size_x`, no `--autobox_ligand`), blind command
  (`--autobox_ligand`, no `--center_x`), non-zero exit, and exit-0-but-no-`docked.sdf`.
- **`dock_vina`** — import guards only (vina-missing, rdkit-missing). The real happy
  path is covered on Modal (see below), not mocked — wiring up fake Vina+meeko+PDBQT
  file I/O would be brittle and low-value.
- **`parameterize_ligand_acpype`** — bin-missing, success (collects `itp`/`gro`/`mol2`
  from `<name>.acpype/`), non-zero exit, exit-0-but-outputs-missing (empty dict).
- **`parameterize_ligand_openff`** — toolkit-missing → `RuntimeError`.
- **`prepare_ligands`** — the full fallback chain by patching the module-level step
  functions: no-SMILES skip, conformer-fail skip, GNINA-success (acpype),
  GNINA-fail→Vina, GNINA+Vina-fail→undocked conformer, `use_openff=True`, and
  parameterization-fail → entry still returned with `parameterizer="none"`.

### 2. `modal_app.py::test_ligands_modal` — real binaries on Modal
A CPU `@app.function` (no GPU), mirroring the existing `test_boltz_gpu`. Run with
`modal run modal_app.py::test_ligands_modal`. It runs the actual
**RDKit ETKDG → Vina blind dock → OpenFF SMIRNOFF** pipeline on ethanol against a small
embedded real receptor, then runs `prepare_ligands(..., use_openff=True)` end-to-end.
Each step is wrapped to return diagnostics (`*_ok` / `*_error`) rather than hard-crash,
and returns a summary dict like `test_boltz_gpu`.

## Modal image coverage note (important)
The Modal image (`modal_app.py`) installs RDKit, Vina, meeko, OpenFF, GROMACS — but
**NOT gnina and NOT acpype** (gnina is a release-binary download; acpype isn't in
`requirements.txt` or the conda spec). Consequences:
- `test_ligands_modal` genuinely exercises the **GNINA-absent → Vina fallback** and the
  `use_openff=True` branch — a real test of the fallback logic, not a contrivance.
- `dock_gnina` and `parameterize_ligand_acpype` have **no real-binary coverage anywhere**
  — mocked-only. Real coverage would require adding the gnina release binary and
  `pip install acpype` (+ AmberTools) to the image. Flagged here rather than implied.

## Verification
`python -m pytest tests/test_ligands.py -q` in the `ProPredict` conda env: **27 passed**.
Full relevant suite (`test_ligands + test_agent + test_mutation_scan + test_boltz`):
**60 passed, 3 skipped** (skips: 2 Boltz GPU + 1 ProteinMPNN integration). `modal_app.py`
parses clean. The Modal function is invoked separately via `modal run` (not part of the
local pytest run).

## Next
Task 5b — `tests/test_membrane.py` (mocked) for `orchestrator/membrane.py`
(`_resolve_insane`, `_lipid_name`, `embed_in_membrane_gromacs`,
`embed_in_membrane_openmm`), same local+Modal split if a real membrane-build path is
worth a Modal smoke test.
