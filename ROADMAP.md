# ProPredict Roadmap

## Architecture Summary

**End goal:** Simulate proteins in real environmental conditions (pH, membranes, ligands) using an
adaptive agent loop, with support for multi-model ensemble prediction.

**Request flow:** Client → FastAPI (`api/main.py`) → Celery task or Modal function → orchestrator pipeline → response stored in Postgres.

---

## Completed Stages

### ~~Stage A — Fix Existing Stubs~~ DONE
- Clash detection via BioPython `NeighborSearch` (CA-CA < 3.8 Å)
- Multi-seed ensemble: `ENSEMBLE_NUM_SEEDS` seeds with best-pLDDT selection
- GROMACS trigger uses `context.get("membrane")` / `context.get("ligands")`
- Redis GET/SET caching with `CACHE_TTL`

### ~~Stage B — pH-Aware Protonation~~ DONE
- PropKa3 per-residue pKa computation
- pH-dependent protonation states for HIS, ASP, GLU passed to `pdb2gmx`

### ~~Stage C — Full MD Protocol~~ DONE
- GROMACS: PropKa → pdb2gmx → editconf → solvate → genion → EM → NVT → NPT → Production → RMSD/Rg
- OpenMM: PDBFixer → addHydrogens(pH) → solvate → EM → NVT → NPT → Production → RMSD/Rg
- `OPENMM_ENABLED` takes priority over `GROMACS_ENABLED`

### ~~Stage D — True Agentic Loop~~ DONE
- Claude API tool-use loop with `analyze_structure`, `run_rosetta_relax`, `run_simulation`, `run_boltz_prediction`, `accept_structure`, `escalate_structure`
- Falls back to threshold logic when `AGENT_ENABLED=False` or anthropic not installed

### ~~Stage E — Multi-Model Ensemble~~ DONE
- ESMFold (local + remote), Boltz-2 (CLI subprocess), RoseTTAFold2/OpenFold stubs
- Inter-model structural alignment via BioPython Superimposer
- Per-residue CA RMSD disagreement scoring with high-disagreement region detection
- Iterative refinement loop with plateau detection

### ~~Stage F — Membrane & Ligand Environments~~ DONE
- Membrane embedding: insane.py (GROMACS) and Modeller.addMembrane (OpenMM/CHARMM36m)
- Ligand pipeline: RDKit ETKDG → GNINA/Vina docking → ACPYPE GAFF2 or OpenFF SMIRNOFF
- Both GROMACS and OpenMM MD pipelines accept membrane and ligand contexts

---

## Remaining Work

### Boltz-2 MSA Support
- `BOLTZ_USE_MSA=True` queries the ColabFold MSA server for higher accuracy
- Currently defaulted to `False` (single-sequence mode) for offline/fast operation
- Benchmark with MSA enabled to measure accuracy improvement on CASP15 targets

### Chai-1 Integration (Specialist Backend)
- Chai-1 for experimental constraints and antibody-protein interactions
- Conditional routing: only fires when `context.constraints` is populated
- Non-commercial license — gate behind `CHAI1_ENABLED` flag

### Mutation Workflow
- `apply_mutation` agent tool: mutate sequence at a position, re-predict, compare
- Wire into `context.mutations` field (schema already supports it)

### Simulation Validation
- Validate MD results: check for NaN energy, RMSD > 2.0 nm blowup, Rg divergence
- Auto-escalate when validation fails

### Progress Reporting
- Celery `update_state()` calls at each pipeline stage (folding → post-processing → simulation → finalizing)
- Update `/predict/{run_id}/status` to read real progress from `task.info`

### Real-binary GNINA coverage on Modal (test-infra follow-up)
- `orchestrator/ligands.py::dock_gnina` currently has **mocked-only** coverage
  (`tests/test_ligands.py`). ACPYPE, Vina, RDKit, OpenFF are all real-tested in
  `modal_app.py::test_ligands_modal`, but GNINA is not in the Modal image.
- GNINA has no maintained conda package; the release binary is CUDA-compiled and
  dynamically linked, and the official `gnina/gnina` Docker image is Ubuntu 18.04 /
  Python 3.6–3.7 (conflicts with the project's 3.11). So real coverage needs a
  dedicated CUDA GPU Modal image (gnina binary + CUDA runtime + OpenBabel layered onto
  a Py3.11 env) and a `test_gnina_modal` function calling `smiles_to_3d` → `dock_gnina`.
- Lower priority because Vina (the CPU fallback that `prepare_ligands` uses when GNINA
  is absent) already has real end-to-end coverage.

---

## Tools Summary

| Purpose | Tool | Install |
|---------|------|---------|
| Clash detection | BioPython | `pip install biopython` |
| pH / protonation | PropKa3 | `pip install propka` |
| Python-native MD | OpenMM | `conda install -c conda-forge openmm` |
| Trajectory analysis | MDAnalysis | `pip install mdanalysis` |
| Agent loop | Anthropic SDK | `pip install anthropic` |
| Ligand prep | RDKit + ACPYPE | `conda install -c conda-forge rdkit && pip install acpype` |
| Ligand docking | GNINA / Vina | download from gnina releases / `pip install vina` |
| Membrane builder | insane.py | download from Tieleman lab |
| Structure prediction | Boltz-2 | `pip install git+https://github.com/jwohlwend/boltz` |
| Multi-model (stubs) | RoseTTAFold2, OpenFold | see respective GitHub repos |
