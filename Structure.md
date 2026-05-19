# Structure Review Prompt

Review the ProPredict codebase file by file. For each file, assess:

1. **Current role** — what does this file do in the pipeline?
2. **Structural issues** — dead code, misplaced responsibilities, tight coupling, unclear boundaries
3. **Suggested changes** — concrete refactors (split, merge, move, rename) with rationale

## Codebase Overview

ProPredict is an agentic protein structure prediction service. The pipeline:
- FastAPI accepts sequence + context → dispatches to Celery (local) or Modal (GPU cloud)
- Orchestrator runs: ESMFold / Boltz-2 (multi-seed) → iterative refinement loop → Rosetta relax → GROMACS/OpenMM MD → scoring → accept/refine/escalate
- Results stored in Postgres, cached in Redis

## Files to Review (in order)

1. `config.py` — all env vars and feature flags
2. `models/schemas.py` — Pydantic v2 request/response schemas
3. `models/database.py` — SQLAlchemy Job model + session management
4. `orchestrator/tasks.py` — Celery app, all prediction backends, refinement loop, MD, agent loop, post-processing (~2200 lines)
5. `api/main.py` — FastAPI endpoints, Modal/Celery dispatch
6. `modal_app.py` — Modal image definition + GPU worker function
7. `benchmark_modal.py` — CASP15 benchmark runner
8. `orchestrator/ligands.py` — ligand docking helpers
9. `orchestrator/membrane.py` — membrane embedding helpers

## What I'm Looking For

- Is `orchestrator/tasks.py` too monolithic? What should be split out?
- Are the abstraction boundaries between prediction backends (ESMFold, Boltz-2, RoseTTAFold2, OpenFold) clean enough for adding new backends (e.g. Chai-1)?
- Does the iterative refinement loop belong in tasks.py or should it be its own module?
- Is the agent refinement (Claude API calls) properly separated from the threshold-based pipeline?
- Are there functions that should move to `orchestrator/ligands.py` or `orchestrator/membrane.py` but currently live in `tasks.py`?
- Any suggestions for making the mutation workflow (coming next) easier to integrate?

## Output Format

For each file, respond with:

```
### <filename>
**Assessment:** <1-2 sentence summary>
**Issues:**
- ...
**Recommended changes:**
- ...
```

Then provide a final **Proposed Architecture** section with a suggested file/module layout and migration priority (what to refactor first vs. what can wait).
