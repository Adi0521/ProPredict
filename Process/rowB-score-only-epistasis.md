# Row B — Does ProteinMPNN `score_only` capture epistasis on real proteins? (sub-question A)

**Date:** 2026-10-02 (study committed `588ca3a`, 2026-08-17; results artifact replaced in `bbef345`)
**Companion:** `research_plan/rowB-score-only-epistasis.md` — the study design and narrative.
This document is the engineering record plus an **independent re-verification** of the numbers
against the committed artifacts.
**Deliverable:** settle whether the tier-2 search oracle has epistasis for AdaLead to exploit.
**Answer: yes — and it replicates.** Non-additivity is ~**0.39–0.41** of the single-mutation
effect scale across two near-independent draws.
**Reproduce:** `python -m benchmarks.benchmark_score_only_epistasis --n-doubles 300 --noise-check`

## Why this had to be answered

`Process/phase2-task-5-ablation.md` showed AdaLead beats naive single-site stacking *when
epistasis exists and is strong* — on a **synthetic** landscape. It left an explicit open
question: do real backbones, scored in `score_only` space, actually exhibit epistasis? If
`score_only` were effectively additive, the P2-1/P2-2 warning fires and **AdaLead over it is
near-vacuous on real proteins** — the search would be paying ~26× the oracle calls to find
combinations that, by construction, could not exist.

This is the mutation-search counterpart to Row A. Unlike Row A it needs **no GPU**:
`score_only` runs on CPU, so the whole study ran locally on the M3.

Epistasis, in fitness convention (`f = -global_score`, higher = better):
```
eps(m1, m2) = f(WT+m1+m2) - f(WT+m1) - f(WT+m2) + f(WT)
```
Zero = additive. The quantity that matters is not `|eps|` itself but **`|eps|` relative to the
single-mutation effect scale** — epistasis has to be large *compared to main effects* before a
combinatorial search can exploit it.

## What was built

`benchmarks/benchmark_score_only_epistasis.py` — four steps, each with a decision worth recording:

1. **Variant selection** (`load_variants`). Reads the cached ProteinGym parquet for
   `AMFR_HUMAN_Tsuboyama_2023_4G3O` and keeps only doubles whose **both** constituent singles
   are also measured — without both singles the epistasis formula cannot be evaluated at all.
   Yields 820 singles and 2103 usable doubles.
2. **Backbone preparation** (`trim_pdb_to_dms`). Takes the real RCSB structure, extracts the
   model-0 first chain, asserts it is a **single gap-free fragment**, locates the DMS target
   sequence inside it, slices exactly that span and **renumbers 1..L** so `score_only` threads
   each variant onto the correct positions. It raises rather than guessing if the slice isn't
   found — the right call, since a silent off-by-one here would corrupt every number
   downstream while still producing plausible output.
3. **One batched scoring call.** WT + every involved single + all sampled doubles go through a
   single `score_only_oracle` invocation. This is what makes the study cheap (~82 s on CPU at
   8 decoding orders) and, more importantly, guarantees every sequence in the epistasis
   expression is scored under an identical decoding-order draw — the same protocol-matching
   discipline the QM/MM oracle enforces for WT/mutant.
4. **Noise check** (`--noise-check`). Scores the same doubles at two ProteinMPNN decoding
   seeds (37, 38) and reports signal vs noise, since `eps` is a difference of four scores and
   could plausibly be entirely decoding-order artifact.

Why this system: 47-residue domain (small, fast), **X-ray** not predicted, single gap-free
chain, dense singles+doubles coverage, and **stability** as the observable — the matched
quantity for a fold-compatibility model like `score_only`.

## Result — verified, and it replicated by accident

The committed artifact was regenerated at a different sample size during today's commit, which
means the repo now holds **two near-independent draws** from the same 2103 doubles. They
overlap in only **24 of 150 pairs (16%)** — `rng.choice(n, size=150)` and `size=300` from
`default_rng(0)` are different draws, not nested subsets. Both were recomputed from the
committed JSONL files:

| metric | 300-pair draw (`588ca3a`) | 150-pair draw (`HEAD`) | replicates? |
|---|--:|--:|:--|
| median \|eps\| | 0.0247 | 0.0209 | yes |
| median single \|Δ\| | 0.0603 | 0.0544 | yes |
| **median \|eps\| / \|Δ\|** | **0.409** | **0.385** | **yes** |
| eps std | 0.0285 | 0.0296 | yes |
| \|eps\| median, contacting (CA ≤ 8 Å) | 0.0221 (n=198) | 0.0219 (n=100) | yes |
| \|eps\| median, distal (CA > 8 Å) | 0.0286 (n=102) | 0.0188 (n=50) | **no — inverts** |

**The headline conclusion is robust.** `score_only` epistasis is ~40% of the single-effect
scale, reproduced across two largely non-overlapping samples. **The search is not vacuous on
real proteins** — the additive-oracle concern does not materialize, and this is the internal
green light for the combinatorial search.

**One secondary claim does not survive.** `research_plan/rowB` states distal pairs show
*slightly larger* median `|eps|` (0.029) than contacting pairs (0.022), concluding
`score_only` epistasis is a sequence-context rather than contact phenomenon. The 300-pair
numbers do reproduce exactly — but the independent 150-pair draw **inverts the direction**
(contacting 0.0219 > distal 0.0188). The difference in both runs is small relative to the
`eps` spread (~0.029) and the distal bin is the smaller one (n=102, then n=50), so this was
never a well-powered comparison. The defensible statement is **"no resolvable contact
dependence in either direction,"** not "distal is larger." The broader inference — that this
is sequence-context, not physical contacts — may still be right for the reason given (it is
how ProteinMPNN decodes), but **this data does not establish it.**

## Signal vs noise (from `research_plan/rowB`, not re-verified here)

| K (decoding orders) | eps std (signal) | noise std | corr(seed37, seed38) | SNR |
|--:|--:|--:|--:|--:|
| 8 | 0.028 | 0.016 | 0.71 | 1.7 |
| 32 | 0.025 | 0.008 | 0.89 | 3.0 |

Signal magnitude is stable while noise falls ~1/√K, so `eps` is genuine coupling rather than
decoding-order artifact. **Concrete recommendation:** when `MUTATION_SEARCH_ENABLED` is used,
raise `PROTEINMPNN_NUM_DECODING_ORDERS` to ≥32 so the search ranks on signal, not noise.

Two caveats on this table that the plan doc does not state: `noise_check` runs on
`doubles[:60]`, so **n=60**, not 300; and the K=32 row required a second invocation at
`--decoding-orders 32`. The independent corroboration is in
`Process/determinism-gate.md`, which measured residual cross-seed spread at N=8 by a different
route (1.15e-02 mean per-residue std) and reached the same ≥32 recommendation.

## Honest scope

- **One domain, one protein, one assay.** Stability on a 47-residue X-ray domain.
- **`|eps|` is in `score_only` fitness units** and is *not* comparable to the synthetic
  ablation's arbitrary-unit synergy crossover. This study says epistasis **exists** at ~40% of
  main-effect scale; it does **not** say that clears the crossover where AdaLead starts winning.
  Those are different claims in different units and should not be conflated.
- **Sub-question A only.** It shows the oracle *has* epistasis, not that it matches
  *experimental* epistasis. Sub-question B is the next step.

## Defect found

**The committed results artifact no longer matches the documented run.** `research_plan/rowB`
reports its reproduce line as `--n-doubles 300` and its results table is the 300-pair run, but
`bbef345` replaced `benchmarks/score_only_epistasis.jsonl` with a **150-pair** run
(the harness default is `--n-doubles 150`, and it writes with mode `"w"`, so any default-flag
re-run silently overwrites the documented artifact). The 300-pair data is recoverable at
`git show 588ca3a:benchmarks/score_only_epistasis.jsonl`.

This is the second instance of the same failure mode in one commit — `bbef345` also replaced
`benchmarks/proteinmpnn_checkpoint_results.json`'s three-seed result with a single-seed run
(see `Process/determinism-gate.md`). Both are results artifacts overwritten by a lower-effort
re-run at default flags. Worth a guard: have these harnesses write to a filename that encodes
the run parameters, or refuse to overwrite an existing file without `--force`.

In this instance the accident was informative — it produced the independent replicate above.
That is luck, not a reason to keep the behavior.

## Follow-ups

- **Restore or supersede the 300-pair artifact**, and correct the contact-dependence claim in
  `research_plan/rowB` to "no resolvable contact dependence."
- **Parameter-stamp or write-protect results artifacts** (both this and the ProteinMPNN
  checkpoint benchmark).
- **Sub-question B** — correlate `eps_model` against experimental epistasis
  `eps_exp = DMS(m12) − DMS(m1) − DMS(m2) + DMS(WT)` from the same parquet: a stability arm
  (Tsuboyama ΔG doubles, this domain + 2 more) and one functional DMS arm. Readout is
  Spearman/Pearson and slope. Run at ≥32 decoding orders per the SNR finding. A near-zero
  slope is still a publishable negative and a reason to lean on the tier-3 re-fold funnel.
- Re-run the contact split with a larger distal bin if the sequence-context claim is wanted.

Until sub-question B lands, the tier-3 re-fold funnel
(`Process/phase2-task-3-refold-funnel.md`) remains the backstop: whatever the cheap search
proposes is validated against real predicted structure before it is trusted.

## Artifacts

- `benchmarks/benchmark_score_only_epistasis.py` — harness (parquet extract → PDB trim →
  batched `score_only` → epistasis stats + `--noise-check`).
- `benchmarks/epistasis_structures/4G3O.raw.pdb` — cached RCSB structure (also the default
  input for `benchmarks/check_determinism.py`).
- `benchmarks/score_only_epistasis.jsonl` — per-pair `eps_model`, single deltas, CA distance.
  **Currently the 150-pair run;** 300-pair run at `588ca3a`.
