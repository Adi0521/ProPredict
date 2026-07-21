# Phase 2 — Task P2-0: Config + schema plumbing

**Date:** 2026-07-21
**Source:** `mutation-plans/Process-plan-phase2.md` § "P2-0 — Config + schema plumbing"
**Scope:** config + schema only; no search/oracle logic yet (that is P2-1/P2-2).
**Decisions confirmed before starting** (from the plan's "Decisions to confirm"):
tier-2 `score_only` search + tier-3 re-fold funnel; hand-roll AdaLead (no `flexs`); build
the **thin slice P2-0→P2-2 first** and re-evaluate the expensive funnel after seeing the
search work.

## What was done

### P2-0a — Phase 2 config knobs (`config.py`)
Added a Phase 2 block after the ProteinMPNN block, all via `os.getenv` per the
"all config through `config.py`" convention:

| Var | Default | Meaning |
|---|---|---|
| `MUTATION_SEARCH_ENABLED` | `False` | master gate for all mutation-search code |
| `MUTATION_SEARCH_ROUNDS` | `10` | AdaLead rounds |
| `MUTATION_SEARCH_CANDIDATES_PER_ROUND` | `20` | AdaLead λ — candidates proposed/evaluated per round |
| `MUTATION_SEARCH_MAX_SITES` | `3` | cap on simultaneous mutations k |
| `MUTATION_SEARCH_MAX_REFOLDS` | `5` | tier-3 re-fold validation budget — the only *expensive* knob |

Defaults are conservative starting points; every knob is `.env`-overridable. Nothing
imports these yet — they are consumed in P2-1 (oracle) / P2-2 (AdaLead) / P2-3 (funnel).

**Naming decision.** Identifiers use a descriptive `MUTATION_SEARCH_*` prefix (matching the
planned `orchestrator/mutation_search.py` module), **not** a plan-label like `PHASE2_*` —
plan phases get renumbered across plans, so the code names shouldn't depend on them. This
`Process/` doc keeps the "P2-0" label because it's a record of the plan, not code.

### P2-0b — `.env.example` mirror
Mirrored the five vars with the same defaults and a header comment noting the tier-2 loop
is cheap (one ProteinMPNN subprocess/round) so real spend = `PHASE2_MAX_REFOLDS`.

### P2-0c — `MutationSearchResult` schema (`models/schemas.py`)
Added two Pydantic-v2 models before `PredictionResponse`:

- `MutationCandidate` — one multi-site mutant: `mutations` (`["A12V", "G45S"]`, 1-indexed
  `<wt><pos><mut>`), full `sequence`, `score`, and `oracle` (`additive`/`score_only`/`refold`).
- `MutationSearchResult` — `wild_type_sequence`, ranked `candidates`, primary `oracle`,
  `rounds`, `total_evaluated`, and `refolds_used` (defaults `0`; stays 0 until P2-3 wires
  the tier-3 funnel).

**Decision — split `MutationCandidate` into its own model** rather than parallel lists on
the result. Keeps each ranked entry self-describing (its sequence, mutation list, and which
oracle scored it travel together), which matters once tier-2 and tier-3 scores coexist.

## Verification
- `python -c "import config; ..."` — all five knobs load with correct defaults.
- `MutationSearchResult(...).model_dump()` round-trips (uses `model_dump`, not `.dict()`).
- `pytest tests/test_boltz.py` — 11 passed, 2 skipped (fast mocked suite, nothing broke).

## Next
P2-1 — `orchestrator/mutation_search.py`: `additive_oracle` (pure) + batched
`score_only_oracle` wrapping `--score_only --path_to_fasta`, unit-tested at the subprocess
boundary like `tests/test_mutation_scan.py`.
