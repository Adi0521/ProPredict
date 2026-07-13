# Mutation Master Plan — Task 3: `apply_mutation` agent tool

**Date:** 2026-07-12
**Source:** `mutation-plans/Process-plan-master.md` Task 3 (from hardening doc)
**Depends on:** Task 0 (`AGENT_MAX_MUTATIONS` config). Standalone from Task 1/2 — does
**not** call `score_candidate_mutations`; only the system prompt mentions the scorer.

## What was done

Added an `apply_mutation` tool to the Claude agent loop so the agent can mutate the
sequence at a position and re-predict the structure, gated by a per-session cap.

### `orchestrator/agent.py`
- **Imports:** added `AGENT_MAX_MUTATIONS` (config) and `call_esmfold_api`
  (`orchestrator.backends.esmfold`).
- **Module constant:** `_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")` — duplicated from
  `PredictionRequest.validate_sequence` rather than imported, to avoid an
  `agent -> schemas` validator coupling.
- **`_AGENT_TOOLS`:** new `apply_mutation` entry. Tool args are `position` / `from_aa`
  / `to_aa` (not the request schema's `pos`/`from`/`to` — `from` is a Python keyword,
  and these are Claude's call args, not the request body).
- **`_execute_agent_tool()` branch:**
  - Cap check first: `len(state["mutations_applied"]) >= AGENT_MAX_MUTATIONS` → error.
  - Validates `position` in range, `to_aa` a standard AA, and (if given) `from_aa`
    matches the actual residue — each returns a JSON error and leaves state untouched.
  - Re-predicts via `call_boltz` if `BOLTZ_ENABLED` else `call_esmfold_api`, mirroring
    `orchestrator/tasks.py` — a Boltz session won't silently downgrade to ESMFold.
  - **Rollback on failure:** the backend runs *before* any `state` mutation, so a
    prediction exception returns an error with `state["sequence"]`/`current_pdb`
    unchanged.
  - On success: updates `sequence`, `current_pdb`, `plddt_scores`, `mean_plddt`,
    `num_clashes` (recomputed on the new PDB), and appends `"{from}{pos}{to}"` to
    `mutations_applied`. Surfaces `affinity_kcal_mol` when the backend returns one.
- **`_AGENT_SYSTEM`:** added guidance to use `apply_mutation` only when
  `context.mutations` requests it or analysis justifies it, plus an honest note that
  `orchestrator/mutation_scan.py` (validated against ProteinGym) exists but is not yet
  a callable tool. (Deviation from plan text: added "validated against ProteinGym"
  since Task 2's gate has now actually passed.)
- **`run_agent_refinement()`:** initializes `state["mutations_applied"] = []` and, before
  returning, sets `post_proc.mutations_applied = state["mutations_applied"] or None`.

### `models/schemas.py`
- `PostProcessingResult` gains `mutations_applied: Optional[List[str]] = None`.

## Scope boundaries (deliberately not done here)
- **Stale `num_clashes` in the final result (`agent.py`, the `num_clashes=` arg of the
  returned `PostProcessingResult`) is untouched** — that is Task 4. The new branch only
  keeps `state["num_clashes"]` correct for its own returned JSON.
- No `scan_mutations` tool and no internal call to `score_candidate_mutations` — the
  scorer stays standalone per the earlier decision.

## Tests — `tests/test_agent.py` (new, fully mocked)
Patches `call_boltz` / `call_esmfold_api` / `count_clashes` / `BOLTZ_ENABLED` /
`AGENT_MAX_MUTATIONS` in the `orchestrator.agent` namespace and calls
`_execute_agent_tool` directly. Cases: happy path (state updated, backend gets mutated
seq), lowercase `to_aa` normalization, `from_aa` mismatch, invalid `to_aa`, out-of-range
positions (0/-1/99, parametrized), missing `position`, backend-failure rollback,
Boltz-vs-ESMFold routing + affinity surfacing, and cap enforcement (neither backend
called at limit).

## Verification
`python -m pytest tests/test_agent.py tests/test_boltz.py -q` in the `ProPredict` conda
env: **22 passed, 2 skipped** (skips are `test_boltz` GPU integration tests). No
regression.

Note: bare `pytest` on a multi-file invocation mis-set `sys.path` and failed to import
`orchestrator` at collection; `python -m pytest` (prepends cwd) is the reliable form.

## Next
Task 4 — fix the stale `num_clashes` reported after refinement (both edits already
documented in the master plan: recompute in the relax/simulation branches, and change
the `PostProcessingResult(num_clashes=...)` arg to read `state["num_clashes"]`).
