# ProPredict

Agentic protein structure prediction service. Accepts an amino acid sequence + environmental context (pH, ligands, membrane, ions), predicts the 3D structure using one or more backends, optionally refines and simulates, and returns a scored PDB.

## Prediction Backends

| Backend | Accuracy | Hardware | Install |
|---------|----------|----------|---------|
| **ESMFold** (local) | Good | CPU / MPS / CUDA | Included in `requirements.txt` |
| **ESMFold** (remote) | Good | None (API call) | Set `ESMFOLD_LOCAL=False` |
| **Boltz-2** | AlphaFold3-class | GPU (A10G+) | `pip install git+https://github.com/jwohlwend/boltz` |
| RoseTTAFold2 | Stub | GPU | Not yet implemented |
| OpenFold | Stub | GPU | Not yet implemented |

When multiple backends are enabled, the pipeline runs all of them with multi-seed sampling and picks the best prediction by pLDDT. Inter-model structural disagreement is computed via BioPython Superimposer.

## Pipeline

```
Sequence + Context
    |
    v
1. Structure prediction (ESMFold, Boltz-2)
2. Multi-model ensemble + disagreement scoring
3. Iterative refinement (Boltz-2 re-seeds + Rosetta relax)
4. MD simulation (OpenMM or GROMACS, with membrane/ligand support)
5. Agentic refinement via Claude API (optional)
6. Post-processing: scoring, clash detection, accept/refine/escalate
    |
    v
Scored PDB + metrics
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env as needed — defaults work for local ESMFold inference

# Start all services (Postgres, Redis, API, Celery worker, Flower)
docker compose up
```

On first run, the Celery worker downloads the `facebook/esmfold_v1` weights (~2 GB) and caches them.

### Submit a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "MKTAYIAKQRQISFVKSHFSRQDILDLWQYVQG",
    "context": {"pH": 7.4, "temperature_c": 25},
    "priority": "fast"
  }'
```

### Poll for results

```bash
curl http://localhost:8000/predict/<run_id>
curl http://localhost:8000/predict/<run_id>/status
curl http://localhost:8000/predict/<run_id>/pdb          # download PDB file
curl http://localhost:8000/predict/<run_id>/simulation-pdb  # post-solvation system
```

## Architecture

```
Client --> FastAPI (api/main.py)
               |
       +-------+-------+
       |               |
   Celery task    Modal function
   (local dev)    (GPU cloud)
       |               |
       +-------+-------+
               |
    _run_prediction_core()
               |
    orchestrator pipeline
               |
    Postgres (results) + Redis (cache)
```

**Two execution modes:**
- **Local/Docker** (`MODAL_ENABLED=False`): FastAPI dispatches to Celery. Redis is both broker and result cache.
- **Modal** (`MODAL_ENABLED=True`): FastAPI dispatches to `modal_app.py::run_prediction` on a GPU (A10G).

**Orchestrator modules:**

| Module | Responsibility |
|--------|---------------|
| `orchestrator/tasks.py` | Celery app, caching, webhook, main pipeline, refinement loop |
| `orchestrator/backends/esmfold.py` | ESMFold local + remote, pLDDT parsing |
| `orchestrator/backends/boltz.py` | Boltz-2 CLI wrapper, CIF-to-PDB conversion |
| `orchestrator/backends/stubs.py` | RoseTTAFold2 / OpenFold placeholders |
| `orchestrator/ensemble.py` | Multi-model alignment + disagreement scoring |
| `orchestrator/simulation.py` | Rosetta FastRelax, GROMACS EM/MD, OpenMM, protonation (PropKa) |
| `orchestrator/scoring.py` | Clash detection, post-processing decision logic |
| `orchestrator/agent.py` | Claude tool-use refinement loop + tool handlers |
| `orchestrator/ligands.py` | RDKit conformer gen, GNINA/Vina docking, ACPYPE/OpenFF params |
| `orchestrator/membrane.py` | insane.py (GROMACS) and OpenMM CHARMM36m membrane embedding |

## Environmental Context

The `context` field supports:

```json
{
  "pH": 7.4,
  "temperature_c": 25,
  "ions": {"Na+": 150, "Cl-": 150},
  "membrane": {"type": "POPC", "span": [20, 45]},
  "ligands": [{"name": "ATP", "smiles": "...", "binding_site": [45, 46]}],
  "mutations": [{"pos": 12, "from": "A", "to": "V"}]
}
```

- **pH** drives PropKa3 protonation state assignment (HIS/ASP/GLU) before MD
- **Membrane** triggers insane.py (GROMACS) or CHARMM36m addMembrane (OpenMM)
- **Ligands** with SMILES trigger RDKit conformer generation, GNINA docking, and force-field parameterization
- **Temperature** sets thermostat for NVT/NPT equilibration and production MD

## Optional Tools

All optional tools are gated by feature flags in `.env`. Enable after installing:

| Tool | Flag | Install |
|------|------|---------|
| PyRosetta | `ROSETTA_ENABLED=True` | `conda install -c rosettacommons pyrosetta` |
| GROMACS | `GROMACS_ENABLED=True` | `brew install gromacs` (Mac) / `apt-get install gromacs` |
| OpenMM | `OPENMM_ENABLED=True` | `conda install -c conda-forge openmm` |
| Boltz-2 | `BOLTZ_ENABLED=True` | `pip install git+https://github.com/jwohlwend/boltz` (GPU required) |
| GNINA | `GNINA_BIN=gnina` | Download from [gnina releases](https://github.com/gnina/gnina/releases) |
| Claude agent | `AGENT_ENABLED=True` | `pip install anthropic` + set `ANTHROPIC_API_KEY` |

## Modal (GPU Cloud)

For GPU-accelerated predictions (Boltz-2, ESMFold on CUDA):

```bash
# GPU smoke test
modal run modal_app.py::test_boltz_gpu --sequence MKTAYIAK

# Benchmark against CASP15 targets
modal run benchmark_modal.py

# Deploy the full API + worker
modal deploy modal_app.py
```

Configure Modal secrets:
```bash
modal secret create propredict-secrets \
  BOLTZ_ENABLED=True \
  BOLTZ_DIFFUSION_SAMPLES=1 \
  BOLTZ_SAMPLING_STEPS=200 \
  ESMFOLD_LOCAL=True \
  MODAL_ENABLED=True \
  GROMACS_ENABLED=True \
  OPENMM_ENABLED=True
```

## Testing

```bash
# Unit tests (no GPU, no Postgres, no Redis needed)
pytest tests/test_boltz.py -k "not integration"
pytest tests/test_esmfold_local.py -k "not integration"

# API tests (need Postgres running)
pytest tests/test_api.py

# Integration tests (need GPU + tools installed)
pytest tests/test_boltz.py -k "integration"
```

## Agent Decision Logic

When `AGENT_ENABLED=True`, a Claude tool-use loop assesses the predicted structure and decides:

- **Accept**: mean pLDDT >= 75, <= 2 clashes, no special context
- **Refine**: run Rosetta relax, re-predict with Boltz-2, or run MD simulation
- **Escalate**: flag for human review when quality is too low or backends unavailable

The agent has access to tools: `analyze_structure`, `run_rosetta_relax`, `run_simulation`, `run_boltz_prediction`, `accept_structure`, `escalate_structure`.

When `AGENT_ENABLED=False`, a threshold-based policy makes the same decisions automatically.
