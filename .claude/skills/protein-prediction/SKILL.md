---
name: protein-prediction
description: Patterns for working with structure prediction backends (ESMFold, Boltz-2, new backends)
---

# Protein Prediction Patterns

When working with structure prediction backends:

- ESMFold returns pLDDT in B-factor column as 0-1 float. Always multiply by 100.
- Boltz-2 uses YAML input format, not FASTA. See `backends/boltz.py` for the template.
- Boltz-2 outputs CIF, which must be converted to PDB. Use `_cif_to_pdb()` in boltz.py.
- All backends must return a `StructurePrediction` (see `models/schemas.py`).
- New backends: add to `orchestrator/backends/`, add a feature flag in `config.py` + `.env.example`, and wire into `_run_prediction_core()` in `tasks.py`.