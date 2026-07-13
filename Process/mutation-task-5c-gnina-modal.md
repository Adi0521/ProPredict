# Test-infra follow-up: real-binary GNINA coverage on Modal

**Date:** 2026-07-13
**Source:** ROADMAP "Real-binary GNINA coverage on Modal (test-infra follow-up)" —
the one gap left after Task 5a (`tests/test_ligands.py`), where `dock_gnina` was the
only ligand function with mocked-only coverage.

## What was done

Added `modal_app.py::test_gnina_modal`, a real-binary smoke test for
`orchestrator/ligands.py::dock_gnina` on a real GPU, plus the dedicated image it needs.

### `gnina_image` (dedicated, separate from the main `image`)
GNINA can't join the main Modal image the way ACPYPE/Vina/OpenFF did:
- No maintained conda package.
- The release binary is CUDA-compiled and dynamically linked.
- The official `gnina/gnina` Docker image is Py3.6 / Ubuntu 18.04 — conflicts with the
  project's Py3.11 stack.

So the prebuilt **v1.3 release binary** is layered onto
`nvidia/cuda:12.2.2-runtime-ubuntu22.04` (`add_python="3.11"`) with `openbabel`,
`libopenbabel-dev`, `libgomp1`, plus `rdkit==2024.3.5` and `numpy<2`. The image is kept
**separate** from the main `image` so the ~200 MB CUDA gnina layer doesn't ride along on
every other Modal function. The build runs `gnina --version || true` so a missing
runtime lib fails loudly at build time rather than opaquely at dock time.

Import safety: `orchestrator/__init__.py` is empty, so
`from orchestrator.ligands import smiles_to_3d, dock_gnina` in the test does **not** pull
in torch/boltz/openff — the minimal image is sufficient.

### `test_gnina_modal` (CPU-imports, T4 GPU)
Mirrors `test_ligands_modal`'s diagnostic style (`*_ok` / `*_error` per step, never a
hard crash). On the same minimal 3-residue receptor (ALA-GLY-SER, real CA coords) with
ethanol as the ligand:
1. `smiles_to_3d` — RDKit ETKDG conformer.
2. **Blind docking** — `dock_gnina(..., binding_site=None)` → the `--autobox_ligand`
   branch, into its own out_dir so `docked.sdf` doesn't collide.
3. **Binding-site docking** — `dock_gnina(..., binding_site=[1,2,3])` → the
   `--center_x/--size_x` branch (CA centroid of residues 1-3), into a separate out_dir.

Both branches are exercised in one run.

## Verification
`modal run modal_app.py::test_gnina_modal` — **green on the first run**, no lib
iteration needed:
```
{'gnina_on_path': True, 'smiles_to_3d_ok': True,
 'dock_gnina_blind_ok': True, 'dock_gnina_site_ok': True}
```
The v1.3 binary loaded cleanly on `cuda:12.2.2-runtime` with only `openbabel` +
`libgomp1` — no `libtiff`/`libxml2`/`BABEL_DATADIR` fixups required.

## Coverage state after this
- `dock_gnina` — **real binary** (Modal, both branches) + mocked (`tests/test_ligands.py`).
- The mocked `tests/test_ligands.py` stays the CI path (no GPU needed); `test_gnina_modal`
  is the on-demand real check, matching the local-mocked / Modal-real split used for the
  rest of the ligand and membrane pipelines.
- ROADMAP follow-up marked DONE.

## Notes / future
- `gnina_image` pins the binary to a hardcoded release URL (`v1.3`). If gnina cuts a new
  release, bump `GNINA_RELEASE`; the build's `--version` step will flag an incompatible
  binary early.
- gnina is still **not** in the production `image` — production `prepare_ligands` uses the
  GNINA-absent → Vina fallback (also real-tested in `test_ligands_modal`). Wiring gnina
  into the production path would be a separate decision, not just test infra.
