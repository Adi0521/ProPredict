---
name: simulation
description: MD simulation patterns for GROMACS, OpenMM, membrane, and ligand workflows
---

# MD Simulation Patterns

- GROMACS pipeline: PropKa → pdb2gmx → editconf → solvate → genion → EM → NVT → NPT → Production → analysis
- OpenMM pipeline: PDBFixer → addHydrogens(pH) → solvate → EM → NVT → NPT → Production → analysis
- OpenMM takes priority over GROMACS when both are enabled (`OPENMM_ENABLED` checked first)
- Membrane support: insane.py for GROMACS, Modeller.addMembrane for OpenMM
- Ligand support: RDKit ETKDG conformer → GNINA/Vina docking → ACPYPE GAFF2 or OpenFF SMIRNOFF parameterization
- All simulation functions are in `orchestrator/simulation.py` (1000+ lines) — read the function you need, don't load the whole file
- Energy values should be checked for NaN — see ROADMAP.md "Simulation Validation"