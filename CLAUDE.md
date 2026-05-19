# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

ProPredict is an agentic protein structure prediction service. It accepts an amino acid sequence + environmental context (pH, ligands, membrane, ions), predicts the 3D structure using one or more backends, optionally refines and simulates, and returns a scored PDB.

## Commands

```bash
# Run all services locally (Postgres, Redis, API, Celery worker, Flower)
docker compose up

# Run API server standalone (requires Postgres + Redis running)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Run Celery worker standalone
celery -A orchestrator.tasks worker --loglevel=info

# Run tests (unit tests don't need running services; integration tests need Postgres/Redis)
pytest tests/test_boltz.py          # Boltz-2 unit tests (all mocked, no GPU needed)
pytest tests/test_api.py            # API tests (need Postgres or will fail on DB connect)
pytest tests/test_boltz.py -k "not integration"  # skip GPU-dependent tests

# Run on Modal (GPU cloud — production path for Boltz-2)
modal run modal_app.py::run_prediction     # single prediction
modal run modal_app.py::test_boltz_gpu     # GPU smoke test
modal run benchmark_modal.py               # benchmark suite against CASP15 targets

# Deploy to Modal
modal deploy modal_app.py
```

## Architecture

**Request flow:** Client → FastAPI (`api/main.py`) → Celery task or Modal function → orchestrator pipeline → response stored in Postgres.

**Two execution modes:**
1. **Local/Docker** (`MODAL_ENABLED=False`): FastAPI dispatches to Celery (`orchestrator/tasks.py`), which runs the pipeline on the local worker. Redis is both Celery broker and result cache.
2. **Modal** (`MODAL_ENABLED=True`): FastAPI dispatches to `modal_app.py::run_prediction` on a GPU (A10G). The Modal image bundles GROMACS, OpenMM, RDKit, Boltz-2, and all conda deps.

**Orchestrator modules:**
- `orchestrator/tasks.py` — Celery app, caching, webhook, main pipeline (`_run_prediction_core`), refinement loop
- `orchestrator/backends/esmfold.py` — ESMFold local + remote, pLDDT parsing
- `orchestrator/backends/boltz.py` — Boltz-2 CLI wrapper, CIF-to-PDB conversion
- `orchestrator/backends/stubs.py` — RoseTTAFold2/OpenFold placeholders
- `orchestrator/ensemble.py` — multi-model alignment + inter-model disagreement scoring
- `orchestrator/simulation.py` — Rosetta FastRelax, GROMACS EM/MD, OpenMM, protonation (PropKa)
- `orchestrator/scoring.py` — clash detection, post-processing decision logic
- `orchestrator/agent.py` — Claude tool-use refinement loop + tool handlers

**Pipeline stages (orchestrated by `tasks.py`):**
1. Structure prediction (`backends/esmfold.py`, `backends/boltz.py`)
2. Multi-model ensemble + disagreement scoring (`ensemble.py`)
3. Iterative refinement: Boltz-2 re-seeds + Rosetta relax (`simulation.py`)
4. MD simulation: GROMACS or OpenMM with membrane/ligand support (`simulation.py`)
5. Agentic refinement via Claude API (`agent.py`, optional, AGENT_ENABLED)
6. Post-processing: scoring, clash detection, decision (`scoring.py`)

**Feature flags in `config.py`** gate every optional tool. Each tool (PyRosetta, GROMACS, OpenMM, Boltz-2, GNINA, Claude agent) must be installed separately and enabled via `.env`.

## Key Conventions

- All config via environment variables loaded in `config.py` (single source of truth). Copy `.env.example` → `.env`.
- Pydantic v2 schemas in `models/schemas.py` — use `model_dump()` not `.dict()`.
- The Celery app is defined in `orchestrator/tasks.py` (not a separate celery.py).
- `_run_prediction_core(request_data: dict)` is the shared pipeline entry point used by both Celery and Modal.
- ESMFold model is lazy-loaded once per worker process (heavy ~2GB download on first run).
- Boltz-2 runs via CLI subprocess (`boltz predict`) with YAML input, not Python API.
- PDB files are stored as strings in task results and Postgres `result_json`, not on disk.
- pLDDT is normalized to 0-100 scale everywhere (ESMFold B-factor is 0-1, multiplied by 100 on parse).
- Docker images use `mambaorg/micromamba` base for ARM64-compatible conda packages (openmm, rdkit, etc.).

## Testing Notes

- `tests/test_boltz.py` mocks subprocess and filesystem — runs without GPU or Boltz-2 installed.
- `tests/test_api.py` uses FastAPI TestClient but needs a Postgres connection (init_db runs on import). To run without Postgres, the DB init needs mocking.
- Integration tests (marked by `test_*_integration` naming) require real tools installed and are intended for Modal/GPU environments.
