# ProPredict — Model Backend Migration Plan

## Context

ProPredict currently depends on Meta's ESMFold public API (`api.esmatlas.com`), which has had reliability issues and could be shut down at any time. This plan migrates the project to local model inference with a three-tier backend strategy:

1. **ESMFold Local** (HuggingFace Transformers) — CPU-friendly fallback, drop-in replacement
2. **Boltz-2** — primary GPU backend, AlphaFold3-class accuracy, MIT license
3. **Chai-1** — specialist backend for experimental constraints and antibody targets

Each phase is self-contained. Complete Phase 1 before starting Phase 2, etc.

**Repo structure reference:**
- `config.py` — feature flags and env vars
- `orchestrator/tasks.py` — all prediction backends and main Celery task
- `models/schemas.py` — Pydantic request/response models
- `api/main.py` — FastAPI endpoints
- `requirements.txt` — dependencies
- `docker-compose.yml` / `Dockerfile.celery` — deployment
- `tests/test_api.py` — tests

---

## Phase 1: Local ESMFold via HuggingFace Transformers

**Goal:** Eliminate the external API dependency. Everything downstream (pLDDT parsing, caching, post-processing, agent loop) stays untouched because the output format is identical.

### 1.1 Add config flags for local ESMFold

- [ ] In `config.py`, add:
  ```python
  ESMFOLD_LOCAL = os.getenv("ESMFOLD_LOCAL", "True") == "True"
  ESMFOLD_MODEL_NAME = os.getenv("ESMFOLD_MODEL_NAME", "facebook/esmfold_v1")
  ```
- [ ] In `.env.example`, add:
  ```
  # ESMFold: use local model (True) or remote API (False)
  ESMFOLD_LOCAL=True
  ESMFOLD_MODEL_NAME=facebook/esmfold_v1
  ```
- [ ] Default `ESMFOLD_LOCAL=True` so new installs never depend on the API.

### 1.2 Implement `call_esmfold_local()` in `orchestrator/tasks.py`

- [ ] Add lazy-loaded global for the model and tokenizer (heavy — only load once per worker):
  ```python
  _esmfold_model = None
  _esmfold_tokenizer = None

  def _get_esmfold_local():
      global _esmfold_model, _esmfold_tokenizer
      if _esmfold_model is None:
          from transformers import EsmForProteinFolding, AutoTokenizer
          import torch
          _esmfold_tokenizer = AutoTokenizer.from_pretrained(ESMFOLD_MODEL_NAME)
          _esmfold_model = EsmForProteinFolding.from_pretrained(ESMFOLD_MODEL_NAME)
          _esmfold_model.eval()
          # Use MPS on Apple Silicon, CUDA if available, else CPU
          if torch.cuda.is_available():
              _esmfold_model = _esmfold_model.cuda()
          elif torch.backends.mps.is_available():
              _esmfold_model = _esmfold_model.to("mps")
      return _esmfold_model, _esmfold_tokenizer
  ```
- [ ] Implement `call_esmfold_local(sequence, seed=0) -> StructurePrediction`:
  - Tokenize the sequence
  - Run inference with `torch.no_grad()`
  - Convert output to PDB string via `model.output_to_pdb(output)[0]`
  - Parse pLDDT from PDB using existing `_parse_plddt_from_pdb()`
  - Return a `StructurePrediction` with `model_name="esmfold_local"`
- [ ] Note: ESMFold is deterministic — the `seed` parameter doesn't change the output. For ensemble seeds, the existing `ENSEMBLE_NUM_SEEDS` loop will call this N times but get the same result. This is a known limitation of ESMFold (not a bug in ProPredict). Document this in a comment.

### 1.3 Update `call_esmfold_api()` to dispatch

- [ ] Rename the current `call_esmfold_api()` to `_call_esmfold_remote()` (keep as fallback).
- [ ] Create a new `call_esmfold_api()` that dispatches:
  ```python
  def call_esmfold_api(sequence: str, seed: int = 0) -> StructurePrediction:
      if ESMFOLD_LOCAL:
          return call_esmfold_local(sequence, seed)
      return _call_esmfold_remote(sequence, seed)
  ```
- [ ] This keeps the function signature unchanged so nothing else in `tasks.py` needs editing.

### 1.4 Update dependencies

- [ ] In `requirements.txt`, add to the core section (these pip-install cleanly):
  ```
  transformers>=4.36.0
  torch>=2.1.0
  ```
- [ ] Note: `torch` is large (~2 GB). For Docker images that already have it (e.g., the Celery worker with OpenMM), this adds no size. For the API-only container, it's not needed — ESMFold runs in the Celery worker, not the API server.

### 1.5 Update Docker

- [ ] In `Dockerfile.celery`, add `torch` and `transformers` to the pip install if not already present.
- [ ] In `docker-compose.yml`, add `ESMFOLD_LOCAL=True` to the celery_worker environment.
- [ ] **Remove `platform: linux/arm64`** from ALL services in `docker-compose.yml`. Let Docker auto-detect. This unblocks x86 users and CI systems.

### 1.6 Test

- [ ] Add `tests/test_esmfold_local.py`:
  - Test that `call_esmfold_local()` returns a `StructurePrediction` with valid pLDDT scores for a short sequence (e.g., `"MKTAYIAK"`)
  - Test that `_parse_plddt_from_pdb()` correctly parses the output PDB
  - Test that the dispatch in `call_esmfold_api()` routes to local when `ESMFOLD_LOCAL=True`
- [ ] Run end-to-end: submit a prediction via the API and confirm it completes without any network call to `api.esmatlas.com`.

### 1.7 Update docs

- [ ] Update `ROADMAP.md` to reflect that Stages A–F are implemented (the current file is misleading).
- [ ] Update `README.md` to remove the statement "Code currently optimized for Apple Mac M3" and document the local ESMFold setup.
- [ ] Update `QUICKSTART.md` if it references the ESMFold API.

---

## Phase 2: Boltz Integration (Primary GPU Backend)

**Goal:** Add Boltz-2 as the highest-accuracy prediction backend, behind a feature flag. When enabled with GPU access, it becomes the primary model. ESMFold becomes the fast/CPU fallback.

### 2.1 Add config flags

- [ ] In `config.py`, add:
  ```python
  BOLTZ_ENABLED = os.getenv("BOLTZ_ENABLED", "False") == "True"
  BOLTZ_SAMPLES = int(os.getenv("BOLTZ_SAMPLES", 1))
  BOLTZ_STEPS = int(os.getenv("BOLTZ_STEPS", 200))
  BOLTZ_USE_MSA = os.getenv("BOLTZ_USE_MSA", "True") == "True"
  ```
- [ ] In `.env.example`, add corresponding entries with comments.

### 2.2 Implement `call_boltz()` in `orchestrator/tasks.py`

- [ ] Implement `call_boltz(sequence: str, seed: int = 0) -> StructurePrediction`:
  - Write a temp YAML input file in Boltz's format:
    ```yaml
    version: 1
    sequences:
      - protein:
          id: A
          sequence: <SEQUENCE>
          msa: empty  # or server if BOLTZ_USE_MSA
    ```
  - Run `boltz predict <yaml_path> --out_dir <tmpdir> --samples <N>` via subprocess
  - Parse the output CIF file from `<tmpdir>/boltz_results_<name>/predictions/`
  - Extract pLDDT confidence scores from the CIF B-factor column
  - Convert CIF to PDB string (use BioPython `MMCIFIO` → `PDBIO`, or a simpler line-by-line parser)
  - Return a `StructurePrediction` with `model_name="boltz"`
- [ ] Handle Boltz's `--use_msa_server` flag (requires network) vs `msa: empty` (fully local, lower accuracy). Tie this to the `BOLTZ_USE_MSA` config.
- [ ] Handle the `seed` parameter by setting `--seed` in the Boltz CLI.

### 2.3 Wire Boltz into the main task

- [ ] In `predict_protein_structure()`, add Boltz alongside the existing ESMFold + RoseTTAFold + OpenFold block (around line 1716):
  ```python
  if BOLTZ_ENABLED:
      try:
          boltz_pred = call_boltz(sequence, seed=0)
          predictions.append(boltz_pred)
          logger.info(f"[boltz] mean pLDDT={boltz_pred.mean_plddt:.2f}")
      except (RuntimeError, FileNotFoundError) as e:
          logger.warning(f"Boltz skipped: {e}")
  ```
- [ ] The existing `best_prediction = max(predictions, key=lambda p: p.mean_plddt)` will automatically pick Boltz if it produces a higher-confidence result.

### 2.4 Replace the RoseTTAFold2 and OpenFold stubs

- [ ] Since Boltz is more accurate and easier to install than either of those, consider removing or deprecating the `call_rosettafold2()` and `call_openfold()` stubs. If you want to keep them for future use, that's fine, but Boltz makes them lower priority.
- [ ] Update the `ROSETTAFOLD_ENABLED` and `OPENFOLD_ENABLED` docs to note that Boltz is the recommended multi-model backend.

### 2.5 Optional: Boltz-2 affinity prediction

- [ ] Boltz-2 can predict binding affinities natively. If `context.ligands` is populated, you could add affinity estimation to the Boltz YAML:
  ```yaml
  version: 1
  sequences:
    - protein:
        id: A
        sequence: <SEQUENCE>
    - ligand:
        id: B
        smiles: <SMILES>
  ```
  This could eventually supplement or replace the GNINA docking pipeline for ligand affinity. **Mark this as a future enhancement** — don't block Phase 2 on it.

### 2.6 Update dependencies

- [ ] Add to `requirements.txt` (in an optional/GPU section):
  ```
  # Boltz (GPU recommended): pip install boltz
  # boltz  # uncomment when BOLTZ_ENABLED=True
  ```
- [ ] Or better: create a `requirements-gpu.txt` that extends the base requirements with `boltz` and other GPU-only deps.

### 2.7 Agent loop awareness

- [ ] Update `_AGENT_SYSTEM` prompt in `tasks.py` to tell the agent about Boltz:
  ```
  Available prediction backends: ESMFold (fast, CPU), Boltz (accurate, GPU).
  If Boltz is enabled and produced a prediction, prefer its structure for refinement.
  ```
- [ ] Update the `_AGENT_TOOLS` `run_simulation` description to note that Boltz structures may already include ligand poses.

### 2.8 Test

- [ ] Add `tests/test_boltz.py`:
  - Mock test: verify YAML generation for a given sequence
  - Mock test: verify CIF-to-PDB conversion and pLDDT extraction
  - Integration test (requires GPU): run a short sequence through `call_boltz()` and verify output
- [ ] Test the multi-model ensemble: enable both ESMFold local and Boltz, verify that `_align_and_compare_structures()` fires and produces disagreement metrics.

---

## Phase 3: Chai-1 Integration (Specialist Backend)

**Goal:** Add Chai-1 for cases involving experimental constraints or antibody-protein interactions. Only fires when explicitly needed.

### 3.1 Add config flags

- [ ] In `config.py`, add:
  ```python
  CHAI1_ENABLED = os.getenv("CHAI1_ENABLED", "False") == "True"
  ```
- [ ] In `.env.example`, add:
  ```
  # Chai-1: specialist backend for experimental constraints / antibody targets
  # Non-commercial license — do NOT enable for commercial deployments
  CHAI1_ENABLED=False
  ```

### 3.2 Implement `call_chai1()` in `orchestrator/tasks.py`

- [ ] Implement `call_chai1(sequence: str, seed: int = 0, constraints: dict = None) -> StructurePrediction`:
  - Chai-1 uses a Python API, not a CLI. The integration pattern is:
    ```python
    import chai_lab
    from chai_lab.chai1 import run_inference
    ```
  - Write a FASTA input with the sequence
  - If `constraints` is provided (from `context.constraints`), format as Chai-1 constraint features
  - Parse the output CIF, extract pLDDT, convert to PDB
  - Return a `StructurePrediction` with `model_name="chai1"`
- [ ] Chai-1 accepts optional MSAs but also works in single-sequence mode (outperforms ESMFold in that mode).

### 3.3 Wire Chai-1 into the main task with conditional routing

- [ ] Chai-1 should NOT run on every request. Add it with a condition:
  ```python
  # Chai-1: only when constraints are provided or agent requests it
  has_constraints = bool(context.get("constraints"))
  if CHAI1_ENABLED and has_constraints:
      try:
          chai_pred = call_chai1(sequence, seed=0, constraints=context.get("constraints"))
          predictions.append(chai_pred)
          logger.info(f"[chai1] mean pLDDT={chai_pred.mean_plddt:.2f}")
      except (RuntimeError, ImportError) as e:
          logger.warning(f"Chai-1 skipped: {e}")
  ```

### 3.4 Add Chai-1 as an agent tool

- [ ] Add a new tool to `_AGENT_TOOLS`:
  ```python
  {
      "name": "run_chai1_prediction",
      "description": (
          "Re-predict the structure using Chai-1, which handles experimental "
          "constraints (crosslinks, epitope data) and is stronger for antibody-protein "
          "interfaces. Use when constraints are present or inter-model disagreement is "
          "high on interface regions. Requires CHAI1_ENABLED=True."
      ),
      "input_schema": {
          "type": "object",
          "properties": {
              "use_constraints": {
                  "type": "boolean",
                  "description": "Whether to pass context.constraints to Chai-1",
              }
          },
          "required": [],
      },
  }
  ```
- [ ] Implement the corresponding handler in `_execute_agent_tool()`.

### 3.5 Update `StructurePrediction` schema

- [ ] In `models/schemas.py`, verify that `model_name` field accepts the new values. Currently it defaults to `"esmfold"` but is a free-form string, so `"esmfold_local"`, `"boltz"`, and `"chai1"` will all work without schema changes.

### 3.6 Test

- [ ] Add `tests/test_chai1.py`:
  - Mock test: verify constraint formatting
  - Integration test (requires GPU + Chai-1 installed): run a short sequence
- [ ] Test the routing logic: confirm Chai-1 only fires when `context.constraints` is populated.

---

## Phase 4: Cleanup and Hardening

**Goal:** Fix the issues surfaced during the codebase review that aren't directly about model backends but affect reliability.

### 4.1 Fix `requirements.txt`

- [ ] Split into:
  - `requirements.txt` — core deps that pip-install cleanly (fastapi, celery, redis, pydantic, biopython, propka, anthropic, transformers, torch, requests, etc.)
  - `requirements-gpu.txt` — includes `requirements.txt` plus GPU deps (`boltz`, etc.)
  - `requirements-conda.txt` or `environment.yml` — for conda-only deps (openmm, pdbfixer, rdkit, openff-toolkit, pyrosetta)
- [ ] Remove deps from `requirements.txt` that don't pip-install: `openmm`, `rdkit`, `vina`, `meeko`, `openff-interchange`, `openmmforcefields`. These currently cause `pip install -r requirements.txt` to fail on a fresh environment.

### 4.2 Fix Docker platform lock

- [ ] Remove `platform: linux/arm64` from ALL services in `docker-compose.yml`.
- [ ] If ARM-specific builds are needed, use a separate `docker-compose.arm64.yml` override.

### 4.3 Clean up repo

- [ ] Add to `.gitignore`:
  ```
  __pycache__/
  *.pyc
  *.pyo
  *.pdb
  *.pka
  .env
  ```
- [ ] Remove committed `__pycache__/` directories:
  ```bash
  git rm -r --cached __pycache__ api/__pycache__ models/__pycache__ orchestrator/__pycache__
  ```
- [ ] Remove test artifacts from repo root: `myprotein.pdb`, `result.pdb`, `input.pka` (move to `tests/fixtures/` or `.gitignore` them).

### 4.4 Add unit tests for core functions

- [ ] Create `tests/test_orchestrator.py` with tests for:
  - `_parse_plddt_from_pdb()` — pass a known PDB string, verify scores
  - `_count_clashes()` — pass a PDB with known clashes, verify count
  - `generate_cache_key()` — verify determinism and that different contexts produce different keys
  - `compute_post_processing()` — verify accept/refine/escalate thresholds
  - `_determine_protonation_states()` — verify HIS/ASP/GLU states at different pH values

### 4.5 Add simulation validation

- [ ] In `orchestrator/tasks.py`, add `_validate_simulation_metrics(sim_result: dict) -> bool`:
  - Check for NaN potential energy
  - Check for RMSD > 2.0 nm (likely blowup)
  - Check for Rg divergence (> 3x initial value)
  - Return False if any check fails
- [ ] Call this after `run_openmm_simulation()` / `run_gromacs_md()` returns
- [ ] If validation fails, set `post_proc.decision = "escalate"` with a reason

### 4.6 Add progress reporting

- [ ] In `predict_protein_structure()`, add progress updates at each stage:
  ```python
  self.update_state(state='PROGRESS', meta={'progress_percent': 10, 'stage': 'folding'})
  # ... after ESMFold
  self.update_state(state='PROGRESS', meta={'progress_percent': 40, 'stage': 'post_processing'})
  # ... after Rosetta
  self.update_state(state='PROGRESS', meta={'progress_percent': 60, 'stage': 'simulation'})
  # ... after MD
  self.update_state(state='PROGRESS', meta={'progress_percent': 90, 'stage': 'finalizing'})
  ```
- [ ] Update `get_job_status()` in `api/main.py` to read progress from `task.info` instead of hardcoding 0/50/100.

### 4.7 Add the `apply_mutation` agent tool

- [ ] Implement the mutation tool that the agent system prompt references but doesn't exist:
  ```python
  {
      "name": "apply_mutation",
      "description": "Mutate the sequence at a given position and re-predict structure.",
      "input_schema": {
          "type": "object",
          "properties": {
              "position": {"type": "integer"},
              "from_aa": {"type": "string"},
              "to_aa": {"type": "string"},
          },
          "required": ["position", "to_aa"],
      },
  }
  ```
- [ ] Handler: modify the sequence string, re-run `call_esmfold_api()` (or Boltz if enabled), update `state["current_pdb"]` and `state["plddt_scores"]`.

---

## Verification Checklist

After all phases, confirm:

- [ ] `docker compose up` works on both ARM64 and x86
- [ ] A prediction completes with `ESMFOLD_LOCAL=True` and no network access
- [ ] A prediction completes with `BOLTZ_ENABLED=True` on a GPU machine
- [ ] Multi-model ensemble fires when 2+ backends are enabled (check logs for "Inter-model comparison")
- [ ] The agent loop (AGENT_ENABLED=True) can invoke Boltz and Chai-1 tools
- [ ] `pip install -r requirements.txt` succeeds on a fresh Python 3.11 venv
- [ ] All tests pass: `pytest tests/`
- [ ] No `__pycache__` or `.pyc` files in the repo
- [ ] `ROADMAP.md` and `README.md` reflect the actual state of the codebase