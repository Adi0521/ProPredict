# ProPredict Roadmap

## Architecture Review Summary

The infrastructure (FastAPI + Celery + Redis + Postgres + ESMFold + PyRosetta + GROMACS) is sound.
The mental model — predict → assess → refine → simulate — is correct for an agentic system.
The gap is the simulation layer does the minimum viable operation rather than what the end goals require.

**End goal:** Simulate proteins in real environmental conditions (pH, membranes, ligands) using an
adaptive agent loop, with support for multi-model ensemble prediction.

---

## What's Working

- **Local ESMFold inference** via HuggingFace Transformers (`facebook/esmfold_v1`); MPS/CUDA/CPU auto-detected; remote API kept as opt-in fallback (`ESMFOLD_LOCAL=False`)
- ESMFold integration: retry logic, B-factor pLDDT extraction, caching key
- Celery task orchestration with webhook callbacks
- PostgreSQL job persistence and result storage
- Progressive refinement decision (accept/refine/escalate)
- PyRosetta FastRelax (guarded by ROSETTA_ENABLED flag)
- GROMACS EM subprocess pipeline (guarded by GROMACS_ENABLED flag)
- Schema design: `Context` already models pH, temperature, ions, membrane, ligands, mutations

---

## Known Stubs / Bugs to Fix First

| Location | Issue | Fix |
|----------|-------|-----|
| `orchestrator/tasks.py:304` | `num_clashes` hardcoded to `0` | BioPython `NeighborSearch` on CA atoms < 3.8 Å |
| `orchestrator/tasks.py:326-406` | `ENSEMBLE_NUM_SEEDS` configured but task always uses seed=0 | Loop N predictions, average pLDDT |
| `orchestrator/tasks.py:~270` | GROMACS trigger checks string "membrane"/"ligands" in raw JSON | Check `context.get("membrane") is not None` |
| `api/main.py` | Cache key generated but never stored or retrieved | Wire to Redis GET/SET |

---

## Stage A — Fix Existing Stubs
**Files:** `orchestrator/tasks.py`

1. Clash detection via BioPython `NeighborSearch` (CA-CA < 3.8 Å)
2. Ensemble: loop `ENSEMBLE_NUM_SEEDS`, collect predictions, average pLDDT, pick best seed
3. Fix GROMACS trigger to use actual context fields, not string matching
4. Wire cache key to Redis for result reuse

**Install:** `pip install biopython`

---

## Stage B — Wire pH to Protonation State Assignment
**Files:** `orchestrator/tasks.py` (before `pdb2gmx` call)

`Context.pH` is accepted in the API but never used — pH 5 and pH 7 currently produce identical simulations.

Steps:
1. Add `run_propka(pdb_string) -> Dict[str, float]` — returns per-residue pKa values
2. Before `pdb2gmx`: determine protonation states at `context["pH"]`
3. Pass histidine protonation flags to `pdb2gmx -his`
4. Extend to aspartate/glutamate protonation via `pdb2gmx` titratable residue options

**Install:** `pip install propka`

---

## ~~Stage C — Full MD Protocol~~ DONE

**Delivered:**
- `_make_nvt_mdp()`, `_make_npt_mdp()`, `_make_production_mdp()` — parameterised by temperature (K)
- `run_gromacs_md(pdb_string, pH, temperature_c, production_ns)` — full pipeline:
  PropKa → pdb2gmx → editconf → solvate → genion → EM → NVT (100 ps) → NPT (100 ps) → Production → RMSD/Rg analysis
- `run_openmm_simulation(pdb_string, pH, temperature_c, production_ns)` — Python-native:
  addHydrogens(pH) → solvate (TIP3P + 0.15 M NaCl) → EM → NVT (50 ps) → NPT (50 ps) → Production → RMSD/Rg
- `_compute_openmm_trajectory_metrics()` — CA RMSD + Rg from numpy position arrays
- `_analyze_gromacs_trajectory()` — calls `gmx rms` + `gmx gyrate`, parses XVG
- Main task updated: `OPENMM_ENABLED` takes priority over `GROMACS_ENABLED`; both receive `pH` and `temperature_c` from context; results stored in `post_proc.simulation_metrics`
- `PostProcessingResult.simulation_metrics: Optional[Dict]` added to schema

**New config vars:** `OPENMM_ENABLED=False`, `MD_PRODUCTION_NS=0.1`
**Install:** `conda install -c conda-forge openmm` → set `OPENMM_ENABLED=True`

---

## ~~Stage D — True Agentic Loop~~ DONE
**Files:** `orchestrator/tasks.py`, `models/schemas.py`, `config.py`, `.env.example`

Current "agent" is an if/else on two thresholds. Replace with a Claude API tool-use loop:

```python
import anthropic

client = anthropic.Anthropic()

tools = [
    {"name": "run_rosetta_relax", ...},
    {"name": "run_openmm_simulation", ...},
    {"name": "run_propka", ...},
    {"name": "apply_mutation", ...},
    {"name": "accept_structure", ...},
]

response = client.messages.create(
    model="claude-opus-4-6",
    tools=tools,
    messages=[{
        "role": "user",
        "content": (
            f"Per-residue pLDDT: {plddt_scores}\n"
            f"Mean pLDDT: {mean_plddt}\n"
            f"Context: pH={context['pH']}, temp={context['temperature_c']}C, "
            f"membrane={context.get('membrane')}, ligands={context.get('ligands')}\n"
            "Decide the best refinement strategy and call the appropriate tools."
        )
    }]
)
```

The agent can:
- Identify which residue ranges need attention from per-residue pLDDT
- Select the appropriate protocol based on environmental context
- Propose stabilizing mutations (uses `mutations` context field)
- Iterate: re-predict after mutation, re-simulate after relaxation

**Install:** `pip install anthropic`
**Config:** Add `ANTHROPIC_API_KEY` to `.env` and `config.py`

---

## ~~Stage E — Multi-Model Ensemble~~ DONE
**Files:** `orchestrator/tasks.py`

For "in conjunction with different models":
1. Add `call_rosettafold2_api()` as second prediction backend
2. Add `call_openfold_local()` for GPU environments
3. Collect predictions from N models, compute structural consensus
4. Flag residues where models disagree as "low-confidence" — prioritize for MD
5. Use RMSD between model predictions as a refinement trigger

**Tools:** RoseTTAFold2 (lighter, runs on CPU), OpenFold (AlphaFold2-compatible, GPU-ideal)

---

## Stage F — Membrane & Ligand Environments
**Files:** `orchestrator/tasks.py`, new `orchestrator/membrane.py`, `orchestrator/ligands.py`

`membrane` and `ligands` context fields are accepted but never processed:

**Membrane:**
- Use `insane.py` or CHARMM-GUI API to embed protein in lipid bilayer
- Support `MembraneContext.type` (e.g., "POPC", "POPE") and `span` (TM residue range)
- Add CHARMM36m lipid force field to GROMACS/OpenMM setup

**Ligands:**
- Use GNINA or AutoDock-GPU for docking (`LigandContext.smiles`, `binding_site`)
- Generate GAFF2 parameters via ACPYPE or OpenFF
- Add ligand topology to GROMACS/OpenMM before MD

**Install:** `pip install rdkit acpype`

---

## Implementation Priority

1. **Stage A** — stubs are undermining existing logic (clash score = 0 means nothing)
2. **Stage B** — PropKa3 is one function + one GROMACS flag; highest impact for lowest effort
3. **Stage C (OpenMM)** — replaces fragile subprocess chain with Python-native simulation
4. **Stage D** — Claude API agent loop; makes the system genuinely adaptive
5. **Stage E** — multi-model ensemble for confidence
6. **Stage F** — membrane/ligand environments (most complex, requires most new infra)

---

## Tools Summary

| Purpose | Tool | Install |
|---------|------|---------|
| Clash detection | BioPython | `pip install biopython` |
| pH / protonation | PropKa3 | `pip install propka` |
| Python-native MD | OpenMM | `conda install -c conda-forge openmm` |
| Trajectory analysis | MDAnalysis | `pip install mdanalysis` |
| True agent loop | Anthropic SDK | `pip install anthropic` |
| Ligand prep | RDKit + ACPYPE | `conda install -c rdkit rdkit && pip install acpype` |
| Membrane builder | insane.py (standalone script) | download from Tieleman lab |
| Multi-model prediction | RoseTTAFold2 | see RoseTTAFold2 GitHub |
