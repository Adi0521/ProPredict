# CLAUDE.md

ProPredict: agentic protein structure prediction service. Sequence + environmental context (pH, ligands, membrane, ions) → 3D structure via composable backends (ESMFold, Boltz-2) → optional refinement/MD → scored PDB.

## Commands

```bash
# Full stack (Postgres, Redis, API, Celery, Flower)
docker compose up

# API standalone (needs Postgres + Redis running)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Celery worker standalone
celery -A orchestrator.tasks worker --loglevel=info

# Install (editable) — deps live in pyproject.toml; see extras for ml/modal/gpu/dev
pip install -e ".[all]"

# Lint / format (config in pyproject.toml [tool.ruff])
ruff check .            # add --fix for safe auto-fixes
ruff format .

# Tests — unit tests are fully mocked, no services needed
pytest tests/test_boltz.py                        # Boltz-2 (mocked, no GPU)
pytest tests/test_boltz.py -k "not integration"  # skip GPU tests
docker compose up -d postgres redis              # services test_api.py needs
pytest tests/test_api.py                          # Postgres + Redis (/predict enqueues to Celery)
pytest tests/test_esmfold_local.py                # integration test skips unless weights cached

# Modal (GPU cloud)
modal run modal_app.py::run_prediction      # single prediction
modal run benchmark_modal.py                # CASP15 benchmark suite
modal deploy modal_app.py                   # deploy
```

## Architecture

Request flow: Client → FastAPI (`api/main.py`) → Celery task or Modal function → orchestrator pipeline → Postgres.

Two execution modes controlled by `MODAL_ENABLED`:
- **Local/Docker**: FastAPI → Celery (`orchestrator/tasks.py`) → local worker. Redis = broker + cache.
- **Modal**: FastAPI → `modal_app.py::run_prediction` on GPU (A10G).

Pipeline stages (orchestrated by `_run_prediction_core` in `tasks.py`):
1. Structure prediction → `backends/esmfold.py`, `backends/boltz.py`
2. Multi-model ensemble + disagreement → `ensemble.py`
3. Iterative refinement: re-seeds + Rosetta relax → `simulation.py`
4. MD simulation (GROMACS/OpenMM) with membrane/ligand → `simulation.py`, `membrane.py`, `ligands.py`
5. Agentic refinement via Claude API → `agent.py` (when `AGENT_ENABLED=True`)
6. Scoring, clash detection, accept/refine/escalate → `scoring.py`

See @ROADMAP.md for completed stages and remaining work.

## Critical Conventions

- **All config via env vars** loaded in `config.py` (single source of truth). Copy `.env.example` → `.env`.
- **Feature flags gate every optional tool** — each must be installed separately AND enabled in `.env`. Never assume a tool is available; always check the flag first (e.g. `if BOLTZ_ENABLED`).
- **Pydantic v2** — use `model_dump()` never `.dict()`. Schemas in `models/schemas.py`.
- **Celery app** is defined in `orchestrator/tasks.py`, not a separate celery.py.
- **`_run_prediction_core(request_data: dict, progress_cb=None)`** is the shared entry point for both Celery and Modal — keep the dict-in invariant. `progress_cb(percent, stage)` is optional; each wrapper relays it to its own transport (Celery `update_state`, Modal `modal.Dict`). Stage names/percents live in `orchestrator/progress.py`.
- **PDB files are strings** stored in task results and Postgres `result_json`, never on disk.
- **pLDDT is always 0-100** — ESMFold B-factor is 0-1, multiply by 100 on parse.
- **Boltz-2 runs via CLI subprocess** (`boltz predict` with YAML input), not Python API.
- **ESMFold model lazy-loads** once per worker (~8 GB `facebook/esmfold_v1` download on first run, into the HF cache). Don't import at module level.
- Docker images use `mambaorg/micromamba` base for ARM64 conda compat. Currently optimized for Apple M3.
- **Dependencies are declared twice** — `pyproject.toml` and `requirements*.txt` (Docker + Modal still install from the latter). Change both together; Boltz's exact-commit pin must match in `pyproject.toml`, `requirements-gpu.txt`, and `modal_app.py`.

## Anti-Patterns (things that have caused bugs)

- NEVER commit `.env` or API keys. The `.env.example` has `ANTHROPIC_API_KEY=<your-...>` as a placeholder.
- Don't remove `.env` from `.dockerignore` — both Dockerfiles `COPY . .`, which would bake the API key into the image. Compose bind-mounts `.:/app`, so the image never needs it.
- Don't add new config by hardcoding values — always go through `config.py` with `os.getenv()` + add to `.env.example`.
- Don't call `import pyrosetta` / `import openmm` at module top level — they're optional deps. Guard with try/except or the feature flag.
- `test_api.py` calls `init_db()` in a session-scoped autouse fixture and **deletes every row in `jobs`** — never point it at a database with data you care about. New test files that trigger `init_db` (the fixture, or `with TestClient(app)` lifespan) need Postgres unless you mock `models.database.init_db`.
- The webhook SSRF validator in `schemas.py` resolves DNS — don't remove it or weaken it.

## Testing

When modifying orchestrator code, always run `pytest tests/test_boltz.py` to verify nothing broke — it's the fastest feedback loop (fully mocked). For API changes, `test_api.py` needs Postgres + Redis (`docker compose up -d postgres redis`) or mocking `models.database.init_db`. Run `ruff check .` and `ruff format --check .` before committing.

## Workflow Rules

- **Step by step** — do NOT batch multiple sub-tasks in one pass. Complete one, discuss the next.
- **Discuss before acting** — always propose the plan for the next step and get approval before writing code or making changes.
- **Document in `Process/`** — each completed step gets a short write-up in the `Process/` folder describing what was done, why, and any decisions made. Cite sources (docs, issues, etc.) when applicable.