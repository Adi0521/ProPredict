# Phase 2 — Task P2-2: Hand-rolled AdaLead-lite search

**Date:** 2026-07-21
**Source:** `mutation-plans/Process-plan-phase2.md` § "P2-2 — Hand-rolled AdaLead"
**Scope:** the combinatorial search (`adalead_search`) + its tests, against an in-memory
oracle. No ProteinMPNN and no pipeline re-fold wiring (that's P2-3).

## What was built
`adalead_search(wild_type, oracle, rounds, candidates_per_round, max_sites, seed, ...)`
appended to `orchestrator/mutation_search.py`. Oracle-agnostic (`List[str] -> List[float]`,
higher = better, batched), so it is fully testable without ProteinMPNN. Returns a
`MutationSearchResult` (P2-0 schema): candidates ranked best-first (WT excluded),
`total_evaluated`, `rounds`, `refolds_used=0`.

Loop: evaluate the seed set (default `[wild_type]`) → each round, select an elite parent
pool, breed up to λ new candidates by **uniform crossover** (recombination) and **capped
random mutation**, dedup against everything measured, score the batch in **one oracle call**,
and fold results back into a cumulative `measured` dict. Total oracle calls = `1 + rounds`.

## Two deliberate deviations from textbook AdaLead (both load-bearing, both documented in code)

### (1) No greedy rollout — the honest consequence of being model-free
Textbook AdaLead grows each candidate one residue at a time, keeping a step only if it
improves, scored against a cheap **surrogate model** (`model.get_fitness` in FLEXS). That is
exactly what makes a query-per-mutation-step affordable. This search has **no surrogate
layer**: ProteinMPNN is simultaneously the scorer and the thing being optimized, so a rollout
would necessarily hit the *expensive* oracle once per step. Dropping it is therefore the
honest consequence of being model-free, not merely a subprocess-saving trick. The real cost
is **sample efficiency**: we lose intra-round exploitation and lean harder on λ, rounds, and
crossover to assemble multi-site combos. Between-round elitism (the elite band re-seeds each
round's parent pool) recovers the "greedy-around-best" behavior across rounds. What remains
is essentially an **elitist GA with uniform crossover and capped random mutation** — hence
"AdaLead-lite / AdaLead-inspired," so a future reader doesn't come looking for the rollout.

### (2) Range-normalized elite band — not FLEXS's multiplicative band
Verified against FLEXS `adalead.py`:
```python
top_inds = measured["true_score"] >= top_fitness * (1 - np.sign(top_fitness) * self.threshold)   # default threshold=0.05
```
This is **multiplicative toward max**, and it **degenerates when max fitness ≈ 0**: the band
collapses to `[0, 0]` and excludes every below-max sequence, so an epistatic pair whose
components each score below the current best could never enter the recombination pool to be
fused. We instead use a band normalized by the observed fitness **range**:
`fitness >= max - kappa*(max - min)` — sign-agnostic, keeps below-best-but-decent singles
eligible. Because the geometry differs, **`kappa` here is not comparable to FLEXS's 0.05**:
`kappa=0.5` means "top half of the observed fitness range," a deliberately wide default,
described on its own terms.

## Determinism contract
Every random draw goes through `np.random.default_rng(seed)` **and samples only from ordered
structures** — the insertion-ordered `measured` dict and lists derived from it, never by
iterating a Python `set` (whose order is subject to per-process hash randomization, which
would make "same seed → same result" flake in a fresh interpreter). The one `set` in the loop
(`seen_this_round`) is used only for membership tests, never for sampling. Verified: same
`(inputs, seed)` → identical candidates.

## Edge cases handled
- **Singleton pool** (round 1 = `{WT}`): `_recombine(WT, WT) == WT`; parent sampling is with
  replacement, so a one-element pool is safe.
- **Dedup starvation**: generation is capped at `25*λ` attempts, so a small k-cap / small
  elite band emits "up to λ" and moves on instead of spinning.
- **k-cap**: enforced after crossover+mutation by reverting *random excess* mutations back to
  WT (not to a parent) — self-contained and unbiased.

## Validation — the epistasis ablation (headline)
Planted **hidden-synergy** landscape (`_landscape` in the test): decoys `pos1→C`/`pos8→C`
(+1.0, the best singles, no synergy) vs. an epistatic pair `pos3→D`/`pos6→K` (+0.6 each,
**+3.0 together**). The pair (+4.2) beats any additive stack of the best singles (naive k=3
stack = +2.6), but its components rank *below* the decoys, so **naive top-N single-site
stacking never fuses them**. Over the fixed seed set 0–7, the search finds the fused pair in
**7/8** seeds and reaches the global optimum (+5.2); the test asserts a safe majority (≥6/8)
so a future change that makes it knife-edge trips. Deterministic properties get hard
assertions: k-cap respected, exactly `1 + rounds` oracle calls each ≤ λ, reproducibility,
WT excluded from candidates.

### Honest finding: deceptive needle ≠ hidden synergy
The first landscape attempt made the pair's components individually *worse* than the decoys
(a deceptive needle). The rollout-free search found it in **0/20** seeds — correctly, and
instructively: once positive decoys enter `measured`, the range-normalized band's cutoff
rises and re-excludes the deleterious components, so they never become parents to be fused.
That is precisely the sample-efficiency cost of dropping the rollout, made visible. A *fair,
winnable* epistasis test is hidden synergy (components decent-but-not-top), which is what was
committed. The additive sanity test hit a related subtlety — two *specific* point optima are
harder than the synergy case (no gradient pulls them together), so the committed sanity
landscape is **dense** (any mutation at a few positions rewarded), isolating "does it stack?"
from "can it discover a needle?" (solved 10/10).

## Verification
- `pytest tests/test_mutation_search.py` — **32 passed** (7 pure/oracle-mock from P2-1 + the
  new AdaLead tests).
- `pytest tests/test_mutation_scan.py tests/test_boltz.py` — 22 passed, 3 skipped (nothing
  broke).
- Robustness spot-check (scratch, not committed): 30-seed sweep confirmed `rounds=25, λ=30,
  kappa=0.5` is the robust operating point (~90% pair-discovery); wider bands did not help.

## Next
P2-3 — two-stage funnel: take the top-M candidates and re-fold each via the existing
`apply_mutation` / prediction path, ranked by pLDDT/clashes (or Boltz affinity with ligands),
capped at `MUTATION_SEARCH_MAX_REFOLDS`. This connects the search to the real pipeline and is
where `refolds_used` becomes non-zero. (Thin-slice checkpoint: per the P2 plan we re-evaluate
whether to build the expensive funnel now, having seen the search work.)
