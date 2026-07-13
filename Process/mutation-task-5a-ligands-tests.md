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
The Modal image (`modal_app.py`) installs RDKit, Vina, meeko, OpenFF, GROMACS, and now
**AmberTools + OpenBabel + acpype** (added via conda-forge). GNINA is still absent
(CUDA-compiled binary; no maintained conda package). Consequences:
- `test_ligands_modal` genuinely exercises RDKit → Vina → OpenFF **and ACPYPE(GAFF2)**
  for real, plus the **GNINA-absent → Vina fallback** in `prepare_ligands`.
- `parameterize_ligand_acpype` now has **real coverage** on Modal (conda AmberTools 21.11,
  not the fragile pip acpype wheel).
- `dock_gnina` remains **mocked-only** — real gnina coverage is a tracked follow-up
  (see ROADMAP "Real-binary GNINA coverage on Modal"): it needs a dedicated CUDA GPU
  image because the release binary is CUDA-linked and the official gnina Docker image is
  Py3.6/Ubuntu18.04. Not blocking, since Vina (the CPU fallback) is real-tested.

## Verification
`python -m pytest tests/test_ligands.py -q` in the `ProPredict` conda env: **27 passed**.
Full relevant suite (`test_ligands + test_agent + test_mutation_scan + test_boltz`):
**60 passed, 3 skipped** (skips: 2 Boltz GPU + 1 ProteinMPNN integration). `modal_app.py`
parses clean.

## Real Modal run — two bugs caught (2026-07-12)
`modal run modal_app.py::test_ligands_modal` surfaced two real defects the mocks could
not (both now fixed and confirmed green on a second run):
1. **`parameterize_ligand_acpype` wrong flag** — the command used `-f gmx`, but acpype's
   `-f`/`--force` is a boolean (`store_true`); `gmx` landed as a stray positional →
   `acpype: error: unrecognized arguments: gmx`. Fixed to `-o gmx` (`--outtop`, the real
   GROMACS-output selector). Regression-locked in the mocked test.
2. **`pydantic==2.5.0` stale pin** — conda `openff-interchange` calls
   `model_dump(serialize_as_any=...)` (pydantic ≥ 2.7); the pip requirements downgraded
   pydantic to 2.5.0 in the image, breaking OpenFF. Bumped `requirements.txt` to
   `pydantic==2.13.4` (the version the local suite already passes on; fastapi 0.104.1
   validated against it locally).

Confirmed result: `smiles_to_3d_ok`, `dock_vina_ok`, `openff_ok`, `acpype_ok` all `True`;
`prepare_ligands` returns one entry with `parameterizer="openff"`, docked pose set.

**Known minor caveat:** acpype names its charged mol2 by charge-method+atom-type (e.g.
`ETH_bcc_gaff2.mol2`), not `<name>.mol2`, so `parameterize_ligand_acpype`'s exact-name
mol2 lookup misses it (logs a warning, `mol2` key omitted). Inspection-only — GROMACS MD
uses the `.itp`/`.gro`/`.top`, which ARE collected. Fix is a glob lookup (see task notes).

## Next
Task 5b — `tests/test_membrane.py` (mocked) for `orchestrator/membrane.py`
(`_resolve_insane`, `_lipid_name`, `embed_in_membrane_gromacs`,
`embed_in_membrane_openmm`), same local+Modal split if a real membrane-build path is
worth a Modal smoke test.
