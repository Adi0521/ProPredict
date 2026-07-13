# Task 6 (post-master-plan): wire `scan_mutations` agent tool + container reachability

**Date:** 2026-07-13
**Source:** The follow-on telegraphed by master-plan Task 3 ("a validated structural
scorer exists but is not yet a tool") + open question #2 ("where does `PROTEINMPNN_PATH`
live?"). All numbered master-plan tasks (0–5) plus the gnina Modal follow-up were already
complete; this exposes the ProteinMPNN scorer to the agent and makes it reachable in the
container images.

## What was done

### 1. `scan_mutations` agent tool — `orchestrator/agent.py`
- Imports `score_candidate_mutations` from `orchestrator.mutation_scan` and the config
  values `PROTEINMPNN_PATH` / `PROTEINMPNN_MODEL_NAME`.
- New entry in `_AGENT_TOOLS` (placed before `apply_mutation` so scan→apply reads in
  order). Optional `positions` (list[int], restrict the scan) and `top_k` (default 10).
  Description is explicit that the score is **structural compatibility only — not a
  proxy for function/stability/fitness**, and read-only.
- New branch in `_execute_agent_tool()`:
  - `PROTEINMPNN_PATH` unset → `{"error": "PROTEINMPNN_PATH not configured …"}` (graceful
    degradation; the agent is told to fall back).
  - Validates `positions`/`top_k`, calls `score_candidate_mutations(current_pdb,
    sequence, positions, top_k, proteinmpnn_dir, model_name)`, wraps any scorer exception
    as `{"error": "mutation scan failed: …"}`.
  - Returns `{status, note, candidates}`.
- `_AGENT_SYSTEM` prompt: replaced the old "not yet available as a tool" paragraph with a
  "two mutation tools work together" paragraph describing the scan→apply flow and the
  read-only/not-a-fitness-proxy caveat.

**Decision preserved:** `apply_mutation` still does NOT call the scorer internally, and
the scorer stays a distinct read-only tool — same separation the master plan mandated.
The agent chooses whether to chain scan → apply.

### 2. Container reachability (open question #2)
The scorer needs the ProteinMPNN clone present *inside* the container; both images already
carry torch, so the subprocess scorer runs in-place.
- **Modal (`modal_app.py` main image) — primary:** `git clone --depth 1` into
  `/opt/ProteinMPNN` + `.env({"PROTEINMPNN_PATH": "/opt/ProteinMPNN"})`. No `.env` is
  copied into the Modal image, so this image env var is authoritative — clean, no override
  problem. This is where real agent runs happen.
- **`Dockerfile.celery`:** added `git` to the apt list, clone into `/opt/ProteinMPNN`,
  `ENV PROTEINMPNN_PATH=/opt/ProteinMPNN`.
- **`docker-compose.yml`:** added `PROTEINMPNN_PATH=/opt/ProteinMPNN` to
  `celery_worker.environment`, with an inline caveat comment.

**The one caveat (docker-compose only):** `config.py` calls `load_dotenv(override=True)`
and `celery_worker` bind-mounts `.:/app`, so a `PROTEINMPNN_PATH` set in the **host
`.env`** overrides the baked-in `/opt/ProteinMPNN` inside the container. To use the baked
clone, set `PROTEINMPNN_PATH=/opt/ProteinMPNN` in the Docker `.env` (or leave it unset).
If it points at a host-only path, the tool simply reports "unavailable" — no crash.
`.env.example` now documents this.

### 3. Tests — `tests/test_agent.py` (4 new, fully mocked)
Patch `orchestrator.agent.score_candidate_mutations` + `orchestrator.agent.PROTEINMPNN_PATH`:
- unconfigured path → error, scorer never called
- happy path → candidates returned verbatim + scorer called with exact args (pdb,
  sequence, positions, top_k, dir, model)
- non-integer `positions` → validation error before the scorer runs
- scorer raises → wrapped `"mutation scan failed: …"`, no crash

## Verification
`python -m pytest tests/test_agent.py -q` → **18 passed** (14 prior + 4 new).
Full mocked suite (`test_agent + test_mutation_scan + test_boltz + test_ligands +
test_membrane`) → **84 passed, 3 skipped** (skips: 2 Boltz GPU + 1 ProteinMPNN
integration). `modal_app.py` + `agent.py` parse clean. Modal image change not run
(costs GPU compute) — the clone pattern mirrors the verified gnina image work.

## Not done / follow-ups
- Modal image not rebuilt/run here — will happen on the next `modal run`. The `git clone`
  layer sits before `add_local_dir`, so heavy conda/torch layers stay cached.
- Open question #1 (empirical `model_name` choice, `v_48_020` vs higher-noise `v_48_030`
  given we always re-fold) is still open — a benchmark, not a blocker.
- Combinatorial/multi-site mutation search remains Phase 2, out of scope.
