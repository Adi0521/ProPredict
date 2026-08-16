# Phase 2 — Task P2-3: Two-stage funnel + real re-fold validation

**Date:** 2026-08-03
**Source:** `mutation-plans/Process-plan-phase2.md` § "P2-3 — Two-stage funnel + real re-fold
validation"
**Scope:** the tier-3 re-fold funnel + `MutationCandidate` re-fold fields. No agent tool / CLI
/ config-callable wiring yet (that's P2-4).

## What was built
Appended to `orchestrator/mutation_search.py`:
- `refold_validate(wild_type, candidates, context, max_refolds, refold_fn, seed)` — re-folds
  the top `max_refolds` cheap-oracle candidates through the real backend, attaches structural
  metrics, and re-ranks. Returns `(reordered_candidates, refolds_used)`.
- `search_and_validate(...)` — the composer: cheap `adalead_search`, then (if
  `max_refolds > 0`) the funnel. `max_refolds == 0` → no-op, cheap result unchanged.
- `_default_refold_fn` — Boltz when `BOLTZ_ENABLED` else ESMFold, backends **lazy-imported**
  inside the function (never pulls torch/boltz at module import — verified in a test).
- `_refold_one` — helper attaching the metrics to a candidate via `model_copy`.

`MutationCandidate` gained five optional fields (all `None` until re-folded): `refold_plddt`,
`refold_num_clashes`, `refold_score`, `refold_affinity`, `refold_affinity_probability`.

## Design decisions

### Ranking key = structural score, NOT affinity (changed from the P2-3 plan)
The original plan said "rank by affinity when ligands present." Reading the repo first (as
instructed) surfaced `research_plan/rowA-boltz-affinity-invariance.md`: an **open,
GPU-blocked** investigation into whether Boltz-2's affinity head is *blind to point mutations*
(slope-0 hypothesis). Until that resolves, ranking mutants by affinity would be unsound. So:
- **Ranking key = `refold_score = mean_plddt - 5*num_clashes`** — the SAME formula
  `scoring.compute_post_processing` uses for accept/refine/escalate, so the funnel and the
  main pipeline agree on "structurally good." Higher = better.
- **Affinity is recorded as metadata only.** Both `affinity_pred_value` (log10 IC50 µM, lower
  = tighter — the repo's affinity-always-`None` bug was fixed in commit `3206d7e`, and the
  units corrected from the old bogus "kcal/mol") and `affinity_probability_binary` (binder-vs-
  decoy, a separate head) are stored on the candidate for later Δ-affinity analysis, never
  used to order. A test (`..._affinity_recorded_not_ranked`) pins this so a future change
  can't quietly promote affinity to the ranking key.

### `score` / `oracle` stay the CHEAP result
A re-folded candidate keeps its cheap-oracle `score` and `oracle` ("score_only"); all re-fold
info lives in the `refold_*` fields. This preserves the cheap-vs-refold comparison for the
P2-5 ablation, and the top-level `MutationSearchResult.oracle` still names the *search*
oracle (a non-zero `refolds_used` is what signals validation happened).

### Dependency injection for the backend
`refold_fn` is injected (default `_default_refold_fn`), exactly like `oracle` in P2-2, so the
whole funnel is unit-testable with a stub — no ESMFold/Boltz/GPU. This also keeps the
module import-light.

### Failure handling
A re-fold that raises is logged and skipped; that candidate stays in the un-refolded
remainder rather than sinking the funnel — mirrors `agent.apply_mutation`'s "re-prediction
failed, mutation not applied." `refolds_used` counts only successes.

### Merge semantics
Re-folded candidates (sorted by `refold_score` desc) float to the front; un-refolded
candidates keep their cheap-oracle order below.

## Verification
- `pytest tests/test_mutation_search.py` — **40 passed** (8 new funnel tests): budget cap
  (exactly `max_refolds` re-folds), structural re-ranking (a lower cheap-ranked but
  better-folding candidate floats to #1), clash penalty (`plddt - 5*clashes` via a patched
  `count_clashes`), affinity-recorded-not-ranked, failure-skip, budget > candidate count,
  zero-budget no-op, and an end-to-end `search_and_validate`.
- Import sanity: loading `mutation_search` imports **neither torch nor boltz** (lazy backend
  import holds); new schema fields default to `None`.
- `pytest tests/test_mutation_scan.py tests/test_boltz.py` — 35 passed, 3 skipped (schema
  change didn't break existing consumers).

## Notes for whoever wires P2-4
- `search_and_validate` takes params, not config — P2-4 supplies `MUTATION_SEARCH_ROUNDS /
  _CANDIDATES_PER_ROUND / _MAX_SITES / _MAX_REFOLDS` and gates on `MUTATION_SEARCH_ENABLED`.
- The cheap search oracle for real use is `score_only_oracle` bound to a WT `pdb_string` +
  `PROTEINMPNN_PATH`; wrap it as `lambda seqs: score_only_oracle(pdb, seqs, ...)`.
- `context` flows straight to `call_boltz` (ligands/membrane + affinity); ESMFold ignores it.

## Next
P2-4 — `search_mutations` agent tool (read-only cheap search by default; optional budgeted
re-fold validation) wired like `scan_mutations` in `orchestrator/agent.py`, plus a
`python -m orchestrator.mutation_search` CLI. Then P2-5 — the honest ablation write-up.
