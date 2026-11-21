# Row B — Does ProteinMPNN `score_only` capture epistasis on real proteins?

**Status: Step 1 (sub-question A) and Step 2's stability arm (sub-question B) both complete,
run locally on CPU.**
**(A) YES** — `score_only` is meaningfully non-additive on a real protein (~40% of the
single-mutation effect scale).
**(B) YES, weakly, and only coarsely** — that non-additivity tracks *experimental* stability
epistasis in the same direction on all three domains tested (pooled ρ = +0.18 [+0.03, +0.32]),
but the signal sits almost entirely in **which position pairs couple**, not in which
substitutions couple there (within-pair median ρ = +0.07 over 24 pairs, sign test p = 0.54),
and scoring the double outright gives no reliable held-out win over summing the two singles.
Full results: `Process/rowB-step2-stability-arm.md`. The functional arm is still open.

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
  (harness + cached structure + results committed under `benchmarks/`). The harness
  parameter-stamps its output — `score_only_epistasis_n300_k8.jsonl` — and refuses to
  overwrite an existing artifact without `--force`.

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

Two caveats on that table: `--noise-check` scores `doubles[:60]`, so it is **n=60**, not 300;
and the K=32 row is a **second invocation** at `--decoding-orders 32`, not part of the default
run.

**No resolvable contact dependence, in either direction.** In the 300-pair draw distal pairs
(CA–CA > 8 Å) show slightly larger median `|eps|` (0.029) than contacting pairs (0.022) — but
an independent 150-pair draw (`score_only_epistasis_n150_k8.jsonl`, only 24/150 pairs shared)
**inverts** it (contacting 0.0219 > distal 0.0188). Both gaps are small against the `eps`
spread (~0.029) and the distal bin is the smaller one (n=102, then n=50), so the comparison was
never well-powered. The defensible statement is that this data resolves **no** contact
dependence. That `score_only` epistasis is a sequence-context rather than physical-contact
phenomenon remains plausible from how ProteinMPNN decodes autoregressively, but **this data
does not establish it**; a larger distal bin would be needed.

## Conclusions
1. **The search is not vacuous on real proteins.** `score_only` has real epistatic structure
   (~40% of the single-effect scale) for AdaLead to fuse — the P2-1/P2-2 additive-oracle
   concern does **not** materialize here. This is the internal green light for the
   combinatorial search.
2. **Per-pair epistasis is modest and noisy at the default 8 decoding orders** (SNR ~1.7).
   Concrete recommendation: when `MUTATION_SEARCH_ENABLED` is used, raise
   `PROTEINMPNN_NUM_DECODING_ORDERS` (then ≥32 for SNR ~3; **superseded** — under protocol
   v2, K=8 suffices for the score the search ranks, see the K-sweep below) so it ranks on signal, not
   noise — at a compute cost linear in K. The cheap-search savings vs re-folding leave ample
   room for this.
3. **Caveats (honest scope):** one small domain, one protein; `|eps|` in `score_only` fitness
   units is not directly comparable to the synthetic ablation's arbitrary-unit crossover; and
   this is sub-question (A) only — it shows the oracle *has* epistasis, not yet that it
   matches *experimental* epistasis.

## Step 2 — sub-question B, stability arm (DONE)

Model epistasis vs **experimental** epistasis
`eps_exp = DMS(m12) - DMS(m1) - DMS(m2) + DMS(WT)`, over three X-ray Tsuboyama-2023 domains
(AMFR/4G3O 47 aa, HECD1/3DKM 72 aa, UBE4B/3L1X 69 aa), 300 doubles each at K=32 decoding
orders. Engineering record, method decisions and limitations:
**`Process/rowB-step2-stability-arm.md`**.

> **Superseded numerics — read with §"Re-run under scoring protocol v2" below.** The run in
> this table used scoring protocol `score_only/v1`, whose scores are not invariant to batch
> composition. The same panel was re-scored under `score_only/v2`; the v1 table is kept because
> the committed artifacts and the earlier write-up cite it, but the v2 row is the current
> result.

| domain | Spearman (residue-pair block 95% CI) | detrended ρ | within-pair median ρ | reliability (K=32) | held-out r: additive → full |
|---|---|--:|--:|--:|---|
| 4G3O | +0.113 [−0.118, +0.358] | +0.316 | +0.025 (4/7) | 0.89 | −0.329 → −0.448 |
| 3DKM | +0.251 [−0.006, +0.459] | +0.214 | −0.108 (3/8) | 0.40 | +0.193 → +0.266 |
| 3L1X | +0.230 [−0.071, +0.498] | +0.171 | +0.180 (7/9) | 0.51 | +0.093 → +0.134 |

Pooled (secondary, within-domain centred): **ρ = +0.181 [+0.031, +0.319]**, n = 900.

Four findings worth carrying forward:

1. **Direction is consistent, magnitude is small, per-domain CIs include zero.** Only the
   pooled estimate excludes zero, over three proteins — weak evidence, not a generality claim.
2. **The signal is positional.** Cross-seed reliability decomposes into ~0.85–1.00 *between*
   position pairs and only 0.21–0.79 *within* a pair. The oracle knows which positions
   interact; it does not resolve which substitutions interact there. On 4G3O, where within-pair
   reliability is high (0.79), the near-zero within-pair correlation is a genuine miss rather
   than a noise ceiling.
3. **No usable predictive win over the additive baseline.** Held-out gains of +0.07 and +0.04
   on two domains, a loss on the third, and paired block-bootstrap deltas that straddle zero.
4. **Experimental ΔΔG epistasis here is mostly *global***: the additive-null fit
   `DMS(AB) ~ a + b·(DMS(A)+DMS(B))` gives b ≈ 0.39–0.54 on every candidate domain. Detrending
   that curvature out *strengthens* the association on 4G3O (+0.11 → +0.32), so the model
   lines up better with the pair-specific part of experiment than with raw `eps_exp`.

**The WT caveat.** These assays have no wild-type row, and because b ≪ 1 the additive-null
intercept cannot recover it either, so `DMS(WT) = 0` is a stated convention. All primary
readouts are within-domain and provably invariant to the offset; `eps_exp` signs and magnitudes
are not interpretable, and uncentered values are never pooled across domains.

**Correction to the Step 1 recommendation.** The ≥32 decoding-orders advice above was
calibrated on 4G3O — the domain with by far the largest epistasis magnitude. Measured on all
300 analysed pairs, K=32 under v1 gives per-double reliability of 0.89 on 4G3O but only
**0.40 / 0.51** on 3DKM / 3L1X: noise was roughly constant in absolute units while epistasis
magnitude varies ~3× across domains, so no fixed K delivered a fixed SNR. Under v2 (below)
most of that noise turns out to have been a batching artifact rather than an irreducible
decoding-order variance, and K=8 outperforms v1's K=32 on every domain. K still does not buy a
fixed reliability for every protein, so it stays a per-protein question — but the operating
point moved by roughly 4× in cost.

## Re-run under scoring protocol v2

The "not invariant to batch composition" note above was diagnosed, traced to ProteinMPNN's
per-process torch seeding (a sequence's decoding-order draws depend on its index in the batch),
and fixed by scoring every sequence against a **shared** set of K decoding orders drawn once
from a dedicated generator. Full record: **`Process/mutation-scoring-consistency.md`**.

The 0.28–0.92 cross-invocation correlation quoted above is **withdrawn**: it was not
reproducible from any artifact, described the four-term epistasis contrast rather than raw
scores, and conflated a seed change with an invocation change. Under v2, a re-invocation at the
same seed is bit-identical; a re-invocation at a different seed reproduces `eps_model` at
ρ ≈ 0.99.

Same 300 doubles per domain, same singles, same structures, same replicate subsets — verified
element-by-element against the v1 artifacts, not assumed.

| domain | Spearman (block 95% CI) | detrended ρ | within-pair median ρ | reliability (K=32) | between / within pair | held-out r: additive → full |
|---|---|--:|--:|--:|--:|---|
| 4G3O | +0.103 [−0.102, +0.313] | +0.307 | +0.145 (6/7) | 0.99 | 1.00 / 0.99 | −0.338 → −0.449 |
| 3DKM | +0.390 [+0.082, +0.614] | +0.246 | +0.036 (5/8) | 0.99 | 0.99 / 0.99 | +0.235 → +0.272 |
| 3L1X | +0.286 [−0.113, +0.574] | +0.286 | +0.257 (9/9) | 0.99 | 1.00 / 0.99 | +0.108 → +0.147 |

Pooled (secondary, within-domain centred): **ρ = +0.208 [+0.034, +0.372]**, n = 900.

What changed, and what did not:

1. **Repeatability, decisively.** Per-double reliability 0.89 / 0.40 / 0.51 → **0.99** on all
   three. Noise SD fell 4–14× while signal SD was preserved (0.0229→0.0224, 0.0070→0.0079,
   0.0112→0.0101), so the estimand is unchanged. Finding 2 above needs restating: the old
   between/within split (≈0.85–1.00 vs 0.21–0.79) was **mostly a measurement artifact**. Under
   v2 within-pair reliability is 0.99 everywhere.
2. **Within-pair substitution ranking is now consistently positive.** Pooled over the same 24
   position pairs: median ρ +0.069 with 14/24 positive (sign-test p = 0.54) → median **+0.196**
   with **20/24 positive (p = 0.002)**. Still weak in magnitude, over non-independent pairs
   from three proteins — "visible and weak", not "usable".
3. **The association is a little stronger and still not reliable prediction.** Two of three
   domains move up, 3DKM's CI now excludes zero, two per-domain CIs still straddle zero. This
   is the attenuation ceiling lifting, not new biology.
4. **Finding 3 is unchanged: still no predictive win over the additive baseline.** Held-out
   Δ Pearson −0.120 / +0.073 / +0.041 → −0.111 / +0.037 / +0.039 — a clear loss on 4G3O
   (where both predictors are negatively correlated with experiment) and two small gains.
   Cleaner numerics did not make the non-additive term a better predictor, which is what one
   expects if the limitation is the biological model rather than the arithmetic.
5. **Finding 4 (global epistasis) and the WT caveat are untouched.** Detrending still helps,
   and `DMS(WT) = 0` is still an unverified convention with every primary readout invariant to
   it by construction.

### How many decoding orders (I-23, resolved)

Swept K ∈ {2, 4, 8, 16} on all three panels under v2, two draws of the shared orders per point,
K=32 reused from the committed artifacts. Harness `benchmarks/sweep_decoding_orders.py`;
artifact `benchmarks/decoding_order_sweep_n300_k2-4-8-16_v2.{json,md}`; discussion in
`Process/mutation-scoring-consistency.md` §7.

| K | single-delta reliability (4G3O / 3DKM / 3L1X) | eps reliability (4G3O / 3DKM / 3L1X) |
|--:|---|---|
| 2 | 0.961 / 0.935 / 0.951 | 0.930 / 0.827 / 0.892 |
| 4 | 0.980 / 0.972 / 0.934 | 0.952 / 0.927 / 0.896 |
| 8 | **0.990 / 0.982 / 0.980** | 0.979 / 0.958 / 0.963 |
| 16 | 0.993 / 0.996 / 0.993 | 0.987 / 0.982 / 0.990 |
| 32 | — | 0.992 / 0.992 / 0.994 |

The two curves separate by ~4× in K. `single_delta` = `f(mut) − f(WT)`, the quantity the search
ranks; `eps` = the four-term contrast, which every earlier K figure measured and which the
pipeline never computes. So **the ≥32 advice was provisioned for the wrong quantity**, and K=8
is sufficient for the right one on all three domains. Noise falls as K^−0.44…−0.68 against an
ideal of −0.50 — plain Monte-Carlo averaging — so unmeasured K is predictable without
re-measuring. The association with experiment is flat in K on 3DKM/3L1X and climbs only on
4G3O, so past K≈8 additional orders buy numerical self-consistency rather than predictive
power. K<8 is not supported here: 3L1X at K=4 scores below its own K=2 (0.934 vs 0.951), a
non-monotonicity produced by the two-draw design, which would need ≥3 draws per point to
resolve.

**What this means for the search.** Unchanged in substance. The cheap oracle is now internally
consistent enough to *compare* candidates — which it was not before — but it is still only
trustworthy for *proposing* combinations at coupled positions, not for establishing that a
proposal is better protein. The tier-3 re-fold funnel
(`Process/phase2-task-3-refold-funnel.md`) stays load-bearing: whatever the cheap search
proposes is screened against real predicted structure, and even that is a plausibility screen,
not experimental validation.

## Still open — Step 2's function arm
One functional DMS assay, expecting weaker/noisier correlation since function depends on more
than fold stability (a documented confound). It needs domain-slice threading: those assays'
`target_seq` are full-length and will not match a PDB chain gap-free, unlike the Tsuboyama
domains.

## Artifacts
- `benchmarks/benchmark_score_only_epistasis.py` — harness, both arms
  (`--arm internal` = sub-question A, `--arm stability` = sub-question B).
- `benchmarks/epistasis_stats.py` — the statistics (associations, block bootstraps,
  reliability + decomposition, grouped held-out additive-vs-full), pure and unit-tested in
  `tests/test_epistasis_stats.py`.
- `benchmarks/score_only_epistasis_stability_{4G3O,3DKM,3L1X}_n300_k32.json` + `.png`, and
  `..._summary_n300_k32.{json,md}` — Step 2 per-domain artifacts, plots and comparison table
  (scoring protocol `score_only/v1`).
- `benchmarks/score_only_epistasis_stability_{4G3O,3DKM,3L1X}_n300_k32_v2.json` + `.png`, and
  `..._summary_n300_k32_v2.json` — the same panel re-scored under `score_only/v2`. The protocol
  is part of both the config hash and the filename, so a v2 run can neither reuse nor overwrite
  a v1 result.
- `orchestrator/mpnn_score_only.py` — protocol v2; `benchmarks/diagnose_score_only_batching.py`
  + `..._4G3O_k8_d30.json` — the batch-sensitivity diagnostic that motivated it.
- `benchmarks/epistasis_structures/4G3O.raw.pdb` — cached RCSB structure.
- `benchmarks/score_only_epistasis_n300_k8.jsonl` — per-pair `eps_model`, single deltas, CA
  distance for the 300-pair run reported above.
- `benchmarks/score_only_epistasis_n150_k8.jsonl` — a near-independent 150-pair draw
  (24/150 pairs shared with the 300 draw) that replicates the headline ratio (0.385 vs 0.409)
  and is what the contact-dependence correction above rests on.
