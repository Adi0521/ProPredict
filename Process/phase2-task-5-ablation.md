# Phase 2 — Task P2-5: Validation write-up (AdaLead vs naive single-site stacking)

**Date:** 2026-08-16
**Source:** `mutation-plans/Process-plan-phase2.md` § "P2-5 — Validation write-up"
**Deliverable:** an honest ablation — does the tier-2 AdaLead-lite search beat naive top-N
single-site stacking on a case with known epistasis, and *when*?
**Reproduce:** `python -m benchmarks.ablation_mutation_search`

## Caveat stated up front (per the plan)
This is a **methodology / ablation check on a synthetic landscape, NOT a ProteinGym-style
gate** like Task 2. Multi-mutant experimental ground truth is scarce (the Row A HIV-PR work
found only 28 single-mutant isolates in 2171, and multi-mutant labels are worse), so there is
no real-data label to score against here. The claim under test is narrow and qualitative: *a
combinatorial search fuses epistatic combinations that a single-site ranker structurally
cannot* — and the honest follow-up: *at what epistasis strength does that advantage turn on?*

## Setup
Hidden-synergy landscape (8-residue WT, same family as the P2-2 unit test): two additive
decoy singles (`C1`, `C8`, +1.0 each — the best singles), a decent-but-not-top epistatic pair
(`D3`, `K6`, +0.6 each) that earns an extra **synergy** bonus only when both are present, and
−0.5 for anything else. The naive baseline is the `scan_mutations`-style strategy: score every
single substitution, greedily stack the best distinct-position ones up to the k-cap (k=3),
evaluate the combination. It is deterministic, ranks the two decoys first, and never fuses
`D3+K6`. AdaLead: 30 seeds, rounds=25, λ=30, k=3.

## Results (synergy swept; 30 seeds/point)

| synergy | naive | AdaLead mean | AdaLead max | found pair | beats naive |
|--------:|------:|-------------:|------------:|-----------:|------------:|
| 0.0 | 2.60 | 2.36 | 2.60 | 10% | 0% |
| 0.5 | 2.60 | 2.48 | 2.70 | 87% | 77% |
| 1.0 | 2.60 | 2.97 | 3.20 | 87% | 83% |
| 2.0 | 2.60 | 3.90 | 4.20 | 90% | 90% |
| 3.0 | 2.60 | 4.84 | 5.20 | 90% | 90% |
| 5.0 | 2.60 | 6.64 | 7.20 | 90% | 90% |

## What this honestly shows

1. **When epistasis is real and strong (synergy ≳ 2), AdaLead wins reliably.** It finds the
   `D3+K6` combination in ~90% of seeds and beats the best naive single-site stack ~90% of the
   time, by a margin that grows with the synergy — a combination the single-site ranker
   *cannot represent by construction*, regardless of budget. This validates the core P2-2
   claim quantitatively.

2. **When epistasis is absent, the search does not help — and can slightly hurt.** At
   synergy 0 the landscape is purely additive; naive stacking (2.60) ties or beats AdaLead's
   mean (2.36), and AdaLead never strictly wins. Its stochasticity is a small *liability*
   here. So `search_mutations` is not a drop-in upgrade over `scan_mutations`; it is the right
   tool only when combinatorial/epistatic effects are plausible.

3. **There is a crossover, not a cliff.** At weak synergy (0.5) AdaLead already finds the pair
   often (87%) and its *best* run beats naive (2.70 > 2.60), but its *mean* (2.48) still trails
   naive because the ~13% of seeds that miss drag the average below the deterministic baseline.
   The advantage becomes robust (mean and worst-case) only once the synergy clears the gap
   between the pair and the best additive stack.

4. **The advantage is bought with compute.** AdaLead tier-2 costs `1 + rounds` = 26 oracle
   calls (ProteinMPNN `score_only` subprocesses) versus **one** single-site conditional-probs
   pass for the naive `scan_mutations` stack — ~26× more. The search does not find "better
   singles"; it pays ~26× to reach multi-site combinations the single pass structurally omits.
   This is exactly why `search_mutations` is gated, budget-capped, and documented as expensive.

## The honest open question (analogous to Row A)
This establishes the *method* is sound where epistasis exists and strong. It does **not**
establish that real single-chain backbones, scored in ProteinMPNN `score_only` space, exhibit
synergy large enough to clear the crossover — that is an empirical question with no
answer here. Settling it needs either scarce multi-mutant experimental data or a Modal
experiment measuring `score_only` epistasis on real proteins (the mutation-search counterpart
to `research_plan/rowA-boltz-affinity-invariance.md` for affinity). Until then the tier-3
re-fold funnel (P2-3) is the backstop: whatever the cheap search proposes is validated against
real predicted structure before anything is trusted.

## Artifacts
- `benchmarks/ablation_mutation_search.py` — the reproducible harness (self-contained, no
  ProteinMPNN; uses the in-memory landscape and the real `adalead_search`).
- `tests/test_mutation_search.py::test_adalead_beats_naive_stacking_over_fixed_seeds` — the CI
  guard (fixed seeds, majority margin) that keeps the synergy=3 result from regressing.

## Phase 2 thin slice — status
P2-0 → P2-5 complete: config/schema (P2-0), two-tier oracle (P2-1), AdaLead-lite search
(P2-2), re-fold funnel (P2-3), agent tool + CLI (P2-4), this ablation (P2-5). All behind
`MUTATION_SEARCH_ENABLED`, no new dependencies. Deferred/future: auto-adopt mode (P2-4 note),
and a real-data epistasis-magnitude study (above). Phase 3 (Bayesian optimization / sample
efficiency) remains out of scope per the plan.
