# Phase 2 — Task P2-4: `search_mutations` agent tool + CLI entrypoint

**Date:** 2026-08-16
**Source:** `mutation-plans/Process-plan-phase2.md` § "P2-4 — Agent tool + CLI entrypoint"
**Scope:** expose the combinatorial search to the agent + offline CLI, and wire the
`MUTATION_SEARCH_*` config (first consumers). No auto-adopt (deferred — see below); no
ablation write-up (that's P2-5).

## What was built

### `search_mutations` agent tool (`orchestrator/agent.py`)
Registered in `_AGENT_TOOLS` after `apply_mutation`, dispatched in `_execute_agent_tool`.
**Read-only** (matches `scan_mutations`): it returns a ranked shortlist of MULTI-SITE
candidates for the agent to feed into `apply_mutation`; it does **not** modify
`state["sequence"]` / `state["current_pdb"]`.

- Gated on `MUTATION_SEARCH_ENABLED` + `PROTEINMPNN_PATH` (else a clear "disabled" /
  "unavailable" error, no work done).
- Binds the tier-2 `score_only_oracle` to the CURRENT structure and calls
  `search_and_validate`. `validate=false` (default) → cheap search only; `validate=true` →
  tier-3 re-fold ranking with a budgeted number of re-folds.
- **Config values are CEILINGS** (`_clamped` helper): the agent may request smaller
  `rounds` / `candidates_per_round` / `max_sites` / `max_refolds`, but a larger request is
  clamped down to the operator-configured budget — guards against runaway subprocesses /
  re-folds. `max_refolds` is forced to 0 unless `validate=true`.
- **Two distinct seeds:** AdaLead's RNG `seed=0` (search reproducibility) is separate from
  `PROTEINMPNN_SEED` (must stay non-zero — used inside the oracle's decoding order). No
  collision.
- Return payload = `result.model_dump()` + a `note` repeating the structural-compatibility
  caveat and "affinity is metadata, not ranked."
- System-prompt guidance updated: the "two mutation tools" section is now three, describing
  `search_mutations` as the multi-site/epistasis tool and flagging its cost.

### CLI (`python -m orchestrator.mutation_search`)
`_cli()` mirrors `mutation_scan.py`'s argparse entrypoint: `--pdb`, `--sequence`,
`--proteinmpnn-dir`, search knobs (`--rounds`, `--candidates-per-round`, `--max-sites`,
`--top-k`, `--seed`), and `--validate` / `--max-refolds` for the tier-3 funnel. Prints a
header line + one ranked line per candidate (`A3D+A6K  score=… [refold_score=… pLDDT=…
clashes=… affinity=…]`).

## Deferred: auto-adopt (future work, needs more calculation + proven results)
`search_mutations` is intentionally read-only for now. A future **auto-adopt** mode would
write the best re-folded candidate straight into agent `state` when it beats the current
structure's `refold_score`, saving the current read-only+`apply_mutation` **double-fold**
(the funnel re-folds the winner to rank it, then `apply_mutation` re-folds the same mutant
again to adopt it). It is deferred deliberately because it requires:
1. a fair current-vs-candidate comparison on the same `refold_score` metric;
2. counting adopted mutations against `AGENT_MAX_MUTATIONS` (it changes the pipeline
   trajectory, unlike a read-only shortlist);
3. enough confidence in the re-fold ranking to let a tool change state autonomously — which
   in turn depends on the open affinity/re-fold trust question in
   `research_plan/rowA-boltz-affinity-invariance.md`.
This rationale is captured inline at the dispatch site so the next person sees why it's a
shortlist, not an adopter.

## Verification
- `pytest tests/test_agent.py` — 24 passed, incl. 5 new `search_mutations` tests: disabled
  flag, missing `PROTEINMPNN_PATH`, happy-path defaults (called with the WT sequence,
  `max_refolds=0`, AdaLead `seed=0`; state untouched — read-only), `validate=true` passes
  the re-fold budget + `context`, ceiling-clamping (999→config max on every knob), and
  wrapped-failure.
- `pytest tests/test_mutation_search.py` — 41 passed, incl. a CLI test driving `_cli()` with
  a stubbed oracle + temp PDB (no ProteinMPNN).
- Full mocked suite (`test_boltz`, `test_mutation_scan`, `test_mutation_search`,
  `test_agent`) — **100 passed, 3 skipped**.
- **Real-binary CLI smoke test**: `python -m orchestrator.mutation_search` against the actual
  ProteinMPNN clone on a 68-residue monomer (`6MRR`, 3 rounds) produced ranked single- AND
  multi-site candidates with real `score_only` scores (`refolds_used=0`, no `--validate`).
- Import sanity: `agent` → `mutation_search` → `scoring` chain imports cleanly (no circular
  import), and loading the module still pulls neither torch nor boltz.

## Next
P2-5 — the honest ablation write-up: does tier-2 AdaLead beat naive top-N single-site
stacking on a case with known epistasis? Stated as a methodology/ablation check, NOT a new
ProteinGym gate (multi-mutant experimental ground truth is scarce). The synthetic
hidden-synergy result from P2-2 is the core evidence; P2-5 documents it as the deliverable.
