# Phase 2 Task Plan — Combinatorial / Multi-Site Mutation Search

**Status:** Proposed, not started. Supersedes the "Phase 2 sketch" in
`Process-plan-mutation-scan (1).md` (that was a landscape survey with open questions;
this resolves them into an executable task sequence).

**Prereqs (all satisfied):** single-site ProteinMPNN scorer exists (Task 1) and is
ProteinGym-validated (Task 2) + checkpoint-benchmarked (Task 7); scorer is now
reproducible after the determinism fix (seeded + decoding-order-averaged, 2026-07-16);
`apply_mutation` exists as a ready re-fold oracle (Task 3); `scan_mutations` exposes the
single-site scorer to the agent (Task 6).

**Execution rule:** per `CLAUDE.md`, do these one at a time with sign-off between each
and a `Process/` write-up per task — same cadence as Tasks 0–7. Everything gated behind
a `PHASE2_ENABLED` flag; no new heavy dependencies.

---

## The three open questions (from the sketch), answered

### Q1 — Oracle
Three tiers of fitness signal for a candidate multi-mutant:

| Tier | What | Cost | Captures epistasis? |
|---|---|---|---|
| **Additive** | sum of single-site log-odds from one WT-structure `scan_mutations` call | ~free | **No** |
| **Context-aware (`score_only`)** | thread the mutant sequence onto the WT backbone, run ProteinMPNN `--score_only --path_to_fasta` → `global_score` (mean NLL) | ~1 subprocess/round (batched) | **Yes** — autoregressive sequence context |
| **Re-fold** | re-fold the mutant (ESMFold/Boltz), score by pLDDT/clashes, or Boltz affinity when ligands present | minutes/candidate | Yes (+ real geometry) |

**Critical design point.** AdaLead over the **additive** oracle is vacuous — the optimum
of a sum of independent per-site terms is just "pick the best substitution at each site,"
no search required. A real combinatorial search only earns its keep against a
**non-additive** oracle. Therefore:
- **Search-loop oracle = tier-2 `score_only`** (epistasis-aware, no re-fold, and
  *batchable*: one ProteinMPNN model-load scores an entire round's candidate fasta).
- **Tier-1 additive** = fast seeding / pre-filter only.
- **Tier-3 re-fold** = final validation of the top handful, budget-capped.

This is a **cheap → expensive funnel**, which also answers Q2.

**Verified capability (2026-07-20):** `protein_mpnn_run.py` supports
`--score_only 1 --path_to_fasta <file>`; it loops over all sequences in the fasta in a
single model load (`protein_mpnn_run.py:239–279`) and emits a per-sequence `global_score`.
So tier-2 is real and batchable — the plan does not assume a capability ProteinMPNN lacks.

### Q2 — Budget
Config knobs, all under `PHASE2_ENABLED`:
- `PHASE2_ROUNDS` — AdaLead rounds
- `PHASE2_CANDIDATES_PER_ROUND` — AdaLead λ (candidates proposed/evaluated per round)
- `PHASE2_MAX_SITES` — cap on simultaneous mutations k
- `PHASE2_MAX_REFOLDS` — the only *expensive* budget: tier-3 validation count (default small, e.g. 5)

The tier-2 loop is cheap (one subprocess/round), so real spend = the fixed re-fold budget.

### Q3 — `flexs` package vs hand-roll
**Hand-roll AdaLead.** `flexs` last released ~2020–21, pulls heavy deps (TensorFlow), and
would be the only unmaintained dependency in the tree — against the project's
"no fragile optional deps" ethos. AdaLead is ~100–150 lines of pure numpy and we control
the oracle interface. Clear-cut.

---

## Task breakdown

### P2-0 — Config + schema plumbing
- Add `PHASE2_ENABLED`, `PHASE2_ROUNDS`, `PHASE2_CANDIDATES_PER_ROUND`, `PHASE2_MAX_SITES`,
  `PHASE2_MAX_REFOLDS` to `config.py` + `.env.example` (via `os.getenv`, per conventions).
- Add a `MutationSearchResult` schema to `models/schemas.py` (ranked sequences, per-candidate
  mutation lists, oracle used, scores).
- No search logic yet.

### P2-1 — Oracle module (`orchestrator/mutation_search.py`)
- `additive_oracle(log_p, muts)` — pure function, reuses the benchmark's proven additive
  math (`sum of log_p[to] - log_p[from]`).
- `score_only_oracle(pdb_string, sequences, proteinmpnn_dir, ...)` — wraps
  `--score_only --path_to_fasta`, **batched** (one subprocess for a list of sequences),
  returns one fitness per sequence (negated `global_score` so higher = better).
- Fully unit-testable: additive against a synthetic `log_p`; `score_only` mocked at the
  subprocess boundary, mirroring `tests/test_mutation_scan.py`.

### P2-2 — Hand-rolled AdaLead
- Population + greedy-around-best + recombination; `PHASE2_MAX_SITES` k-cap; per-round
  budget; deterministic given seed.
- Operates against the oracle interface from P2-1 (default: tier-2 `score_only`).
- Tests on a **synthetic non-additive landscape with a planted epistatic pair**: assert it
  finds the known optimum, respects the k-cap and per-round budget. (This is where the
  search is proven to beat naive top-N single-site stacking.)

### P2-3 — Two-stage funnel + real re-fold validation
- Take AdaLead's top-M candidates, re-fold each via the existing `apply_mutation` /
  prediction path, rank by pLDDT/clashes (or Boltz affinity if ligands present), capped at
  `PHASE2_MAX_REFOLDS`.
- Connects the search to the real pipeline.

### P2-4 — Agent tool + CLI entrypoint
- `search_mutations` agent tool (read-only cheap search by default; optional budgeted
  re-fold validation), wired the same way as `scan_mutations` in `orchestrator/agent.py`.
- `python -m orchestrator.mutation_search` CLI for offline use.

### P2-5 — Validation write-up
- Honest ablation: does tier-2 AdaLead beat naive top-N single-site stacking on a case with
  known epistasis? Document in `Process/`.
- Caveat stated up front: multi-mutant experimental ground truth is scarce in ProteinGym,
  so this is a methodology/ablation check, **not** a new ProteinGym gate like Task 2.

---

## Scope / cost
Cheap to run (loop ≈ `PHASE2_ROUNDS` subprocesses + a few re-folds), self-contained, no new
dependencies, all behind `PHASE2_ENABLED`. ~5 small tasks — done one at a time with sign-off.

## Decisions to confirm before P2-0
1. **Oracle strategy** — tier-2 `score_only` search + tier-3 re-fold funnel (recommended)
   vs additive-only (advised against — makes the search vacuous).
2. **Hand-roll AdaLead** (recommended) vs `flexs`.
3. **Scope now** — full P2-0→P2-5, or a thin slice first (P2-0→P2-2: config + oracle +
   AdaLead with the cheap oracle, no pipeline re-fold wiring yet) and decide the expensive
   funnel after seeing it work.

---

## Still out of scope (future / Phase 3)
- **Bayesian optimization** — the natural Phase 3 if AdaLead's sample efficiency becomes the
  bottleneck once paying for real re-folds per evaluation.
- **CMA-ES / CbAS / DbAS** — surveyed in the original sketch; not planned.
