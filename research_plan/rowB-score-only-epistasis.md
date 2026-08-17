# Row B — Does ProteinMPNN `score_only` capture epistasis on real proteins?

**Status: Step 1 (sub-question A) complete and run locally on CPU. Result: YES, `score_only`
is meaningfully non-additive on a real protein — modest but real signal, sharpened by more
decoding orders. Sub-question B (correlation vs experimental epistasis) is the next step.**

This is the mutation-search analogue of Row A (`rowA-boltz-affinity-invariance.md`). Row A
asks whether Boltz-2's *affinity* head responds to mutations; Row B asks whether the tier-2
search oracle — ProteinMPNN `score_only` — has *epistasis* for the combinatorial search
(`orchestrator/mutation_search.py`) to exploit. Unlike Row A, this needs **no GPU**:
`score_only` runs on CPU, so the whole study ran locally on the M3.

## Why this matters (the P2-5 open question)
The Phase 2 ablation (`Process/phase2-task-5-ablation.md`) proved AdaLead beats naive
single-site stacking *when epistasis exists and is strong*, on a synthetic landscape. The
honest open question it left: **do real backbones, scored in `score_only` space, actually
exhibit epistasis** — or is `score_only` effectively additive, which would make the search
near-vacuous on real proteins (the P2-1/P2-2 "AdaLead over an additive oracle is vacuous"
warning)? This settles the *internal* half of that question.

## Two sub-questions
- **(A) Internal — does `score_only` show non-additivity at all?** Pure model probe, no
  experimental labels. **This step.**
- **(B) External — does that non-additivity track experimental epistasis?** Needs real
  multi-mutant fitness. Next step (below).

Epistasis for a double mutant (fitness convention, higher = better; `f = -global_score`):
```
eps(m1, m2) = f(WT+m1+m2) - f(WT+m1) - f(WT+m2) + f(WT)
```
Zero = additive. `|eps|` large *relative to the single-mutation effect scale* = real coupling
the search can fuse that naive stacking cannot.

## Method (Step 1)
- **System:** `AMFR_HUMAN_Tsuboyama_2023_4G3O` — a 47-residue domain from the Tsuboyama-2023
  megascale set in the cached ProteinGym parquet. Chosen because it is small, **X-ray**, a
  single gap-free chain, and has dense singles+doubles coverage (820 singles, 2103 doubles
  whose *both* constituent singles are also present). Stability is the matched observable for
  a fold-compatibility model like `score_only`.
- **Backbone:** the real PDB (`4G3O`), trimmed to the exact DMS residue slice and renumbered
  1..L so `score_only` threads each variant onto the correct positions. Verified the trimmed
  backbone sequence == the DMS wild-type sequence.
- **Scoring:** one batched `score_only` call over WT + all involved singles + 300 sampled
  doubles. ~82 s on CPU at 8 decoding orders.
- **Reproduce:** `python -m benchmarks.benchmark_score_only_epistasis --n-doubles 300 --noise-check`
  (harness + cached structure + results committed under `benchmarks/`).

## Results

**`score_only` is clearly non-additive** (300 real doubles, 8 decoding orders):

| metric | value |
|---|---|
| median `|eps|` | 0.025 (fitness units) |
| median single-effect `|delta|` | 0.060 |
| **median `|eps|` / `|delta|`** | **0.41** — epistasis ≈ 40% of the main-effect scale |
| pairs with `|eps|` > ½·median `|delta|` | 116 / 300 |

**The signal is real, not decoding-order noise.** Scoring the same doubles at two independent
ProteinMPNN decoding seeds and comparing `eps` (noise ~ 1/√K, K = decoding orders):

| K | eps std (signal) | noise std | corr(seed37, seed38) | SNR |
|--:|--:|--:|--:|--:|
| 8  | 0.028 | 0.016 | 0.71 | 1.7 |
| 32 | 0.025 | 0.008 | 0.89 | 3.0 |

The signal magnitude is stable across K while noise falls as ~1/√K — so `eps` is genuine
coupling, and quadrupling decoding orders (8→32) roughly halves the noise and lifts per-pair
SNR from ~1.7 to ~3.0.

**No strong spatial-contact dependence.** Distal pairs (CA–CA > 8 Å) show *slightly larger*
median `|eps|` (0.029) than contacting pairs (0.022). `score_only` epistasis is an
autoregressive **sequence-context** phenomenon, not primarily physical contacts — consistent
with how ProteinMPNN decodes.

## Conclusions
1. **The search is not vacuous on real proteins.** `score_only` has real epistatic structure
   (~40% of the single-effect scale) for AdaLead to fuse — the P2-1/P2-2 additive-oracle
   concern does **not** materialize here. This is the internal green light for the
   combinatorial search.
2. **Per-pair epistasis is modest and noisy at the default 8 decoding orders** (SNR ~1.7).
   Concrete recommendation: when `MUTATION_SEARCH_ENABLED` is used, raise
   `PROTEINMPNN_NUM_DECODING_ORDERS` (≥32 gives SNR ~3) so the search ranks on signal, not
   noise — at a compute cost linear in K. The cheap-search savings vs re-folding leave ample
   room for this.
3. **Caveats (honest scope):** one small domain, one protein; `|eps|` in `score_only` fitness
   units is not directly comparable to the synthetic ablation's arbitrary-unit crossover; and
   this is sub-question (A) only — it shows the oracle *has* epistasis, not yet that it
   matches *experimental* epistasis.

## Next — Step 2 (sub-question B)
Correlate model epistasis `eps_model` against **experimental** epistasis
`eps_exp = DMS(m12) - DMS(m1) - DMS(m2) + DMS(WT)` from the same parquet:
- **Stability arm:** Tsuboyama ΔG doubles (this domain + 2 more) — the matched observable.
- **Function arm** (per the study scoping): one functional DMS assay, expecting weaker/noisier
  correlation since function depends on more than fold stability (a documented confound).
- Readout: Spearman/Pearson of `eps_model` vs `eps_exp` and the slope. Slope > 0 and
  significant ⇒ `score_only` epistasis points the right way; near-zero ⇒ the internal
  non-additivity is real but experimentally uninformative (still a publishable negative,
  and a reason to lean on the tier-3 re-fold funnel).
- Run at ≥32 decoding orders given the SNR finding above.

Until Step 2 lands, the tier-3 re-fold funnel (`Process/phase2-task-3-refold-funnel.md`)
remains the backstop: whatever the cheap search proposes is validated against real predicted
structure before it is trusted.

## Artifacts
- `benchmarks/benchmark_score_only_epistasis.py` — harness (extract → trim PDB → batched
  `score_only` → epistasis stats + `--noise-check`).
- `benchmarks/epistasis_structures/4G3O.raw.pdb` — cached RCSB structure.
- `benchmarks/score_only_epistasis.jsonl` — per-pair `eps_model`, single deltas, CA distance
  (the committed 300-pair run).
