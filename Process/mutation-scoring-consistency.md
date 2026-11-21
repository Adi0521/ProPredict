# Mutation scoring consistency — diagnosing and fixing `score_only` batch sensitivity

**Date:** 2026-11-18
**Companions:** `Process/rowB-step2-stability-arm.md` (the stability arm this follows up),
`research_plan/rowB-score-only-epistasis.md` (study design).

## What this was

The stability arm left a loose end: it recorded, in passing, that `score_only` scores are "not
invariant to batch composition" at a fixed seed. If true that undermines every comparison the
mutation pipeline makes, because the combinatorial search accumulates scores across one
ProteinMPNN invocation per round and then ranks them against each other. This step diagnosed
that claim, fixed what it found, and re-ran the stability arm to see whether the fix changes
the study's conclusions.

**Answers, up front.**

1. **Cause:** ProteinMPNN seeds torch once per process and draws decoding orders sequentially
   through the fasta, so a sequence's score depends on its *index in the batch*. Confirmed from
   source, then measured.
2. **Guarantees now:** a `score_only` score is a function of
   `(backbone, sequence, model, K, seed)` alone. Reordering, chunking, inserting unrelated
   sequences and one-call-per-sequence all return bit-identical numbers.
3. **Finalist rankings:** more consistent, and now reported with uncertainty instead of as a
   point estimate from the draw that won.
4. **Biological predictive performance:** marginally better, not meaningfully. Pooled
   model-vs-experiment epistasis ρ +0.181 → +0.208; within-position-pair substitution ranking
   went from a coin flip (14/24 pairs positive, p = 0.54) to consistently positive (20/24,
   p = 0.002) but still weak (median ρ ≈ 0.20). That is noise removal raising a ceiling, not
   the model knowing more biology. See §6.
5. **Has the combinatorial search earned an advantage over the additive baseline?** No.

## Audit finding: the original claim was not reproducible

The write-up's figure — "the same 60 doubles, same seed 37, same K=32, scored in two different
invocations, correlate at only 0.28 (3DKM) / 0.55 (3L1X) / 0.92 (4G3O)" — is not reproducible
from any committed artifact, and should not have been stated as a measurement.

What the artifacts actually store is `replicate.eps_seed_a` vs `replicate.eps_seed_b` with
`mpnn_seed_a = 37, mpnn_seed_b = 38`: Pearson r = 0.931 (4G3O) / 0.063 (3DKM) / 0.379 (3L1X).
Those are cross-**seed** reliability numbers, they match the reliability table rather than the
batch-composition sentence, and they are ordered differently from the 0.28/0.55/0.92 triple.
The quoted numbers appear to come from a transient exploratory run that was never persisted.

Two things follow. The quantity was an **epistasis contrast** (a four-score difference), not a
raw score, so the figure could not have been quoted as raw-score instability either. And the
mechanism, though real, had never actually been measured — which is what made step 1 below a
genuine experiment rather than a confirmation.

## 1. The mechanism, traced through the pinned clone (8907e66)

`protein_mpnn_run.py:29` calls `torch.manual_seed(seed)` **once per process**. The `score_only`
branch then loops `for fc in range(1 + loop_c)` over the fasta sequences (`:244`) and, inside
each, `for j in range(NUM_BATCHES)` draws `randn_1 = torch.randn(chain_M.shape)` (`:256`) from
that single global stream. `randn_1` is the decoding-order draw. With `--batch_size 1`, fasta
sequence *k* consumes stream draws `[K·k … K·k+K−1]`, where `k = 0` is the native-PDB pass.

Two properties that must not be conflated:

| property | holds? |
|---|---|
| exact reproducibility under identical inputs | **yes** — identical fasta content *and order* replays bit-for-bit |
| invariance to batch composition | **no** — a sequence moved to another index gets different orders |

Ruled out as causes: **backbone noise** (`--backbone_noise` defaults to `0.00`, so
`augment_eps = 0` and no coordinate jitter is applied) and **batch-dim nondeterminism**
(`--batch_size 1`). Decoding order is the only stochastic input.

One latent hazard found while reading: `S[:, :input_seq_length] = S_input` mutates `S` in place
and never restores it, so a shorter sequence following a longer one would inherit the previous
sequence's tail. The oracle enforces equal lengths, which is why it has not bitten; v2 clones
`S` per sequence regardless.

## 2. The diagnostic

`benchmarks/diagnose_score_only_batching.py`. Panel: 4G3O, 47 aa, 1 WT + 41 singles + 30
doubles = 72 sequences, K=8, seed 37, model/backbone/K/seed fixed across every condition. The
panel is built doubles-first so every double's two constituents are present and the epistasis
contrast is always computable. Artifact:
`benchmarks/score_only_batching_diagnostic_4G3O_k8_d30.json`.

| condition | exact? | ‖Δscore‖ mean / max | ρ score | ‖Δeps‖ mean | ρ eps | top-5 overlap | cost |
|---|---|---|--:|--:|--:|---|--:|
| repeat | **yes** | 0.0000 / 0.0000 | 1.000 | 0.0000 | 1.000 | 5/5 | 15 s |
| reorder | no | 0.0114 / 0.0332 | 0.978 | 0.0283 | **0.273** | 4/5 | 16 s |
| insert unrelated | no | 0.0109 / 0.0322 | 0.968 | 0.0205 | **0.427** | 4/5 | 15 s |
| chunked (3 invocations) | no | 0.0065 / 0.0389 | 0.984 | 0.0116 | 0.768 | 5/5 | 19 s |
| individual (72 invocations) | no | 0.0206 / 0.0538 | 0.983 | 0.0232 | 0.603 | 4/5 | 82 s |

Reference score SD 0.0652; reference eps SD **0.0233**. Stream-offset probe — one fixed double
scored at fasta index 1–6 with nothing else varying: range 0.0210, SD 0.0083.

**Raw scores are mildly perturbed; the epistasis contrast is destroyed.** Perturbations are
10–32% of the score SD and leave rank correlation at 0.97–0.98, but top-5 overlap still falls
to 4/5 — so even raw-score selection is unsafe at the top, which is exactly where a search
operates. On epistasis it is categorical: ‖Δeps‖ (0.012–0.028) is at or above the entire eps SD
(0.0233), and ρ_eps collapses to 0.27–0.77. Four scores, four independent stream offsets, and a
signal three times smaller than the score spread. This is why correlation alone is insufficient
as a report: ρ_score = 0.978 and ρ_eps = 0.273 describe the same perturbation.

### Shared draws: the measurement that decided the fix

Scoring so that every sequence sees the *same* K decoding orders, then re-drawing via a second
seed (the `individual` condition realises shared draws exactly — every sequence lands at index
1 — and is used here as the stand-in):

| protocol | eps noise SD | eps observed SD | implied signal SD | cross-seed eps reliability |
|---|--:|--:|--:|--:|
| batched (v1) | 0.0159 | 0.0233 | 0.0170 | **0.533** |
| shared draws | **0.0024** | 0.0156 | 0.0154 | **0.977** |

Noise SD falls 6.6×. The **implied signal SD** — √(observed² − noise²) — agrees across
protocols (0.0170 vs 0.0154), which is the evidence that shared draws remove noise rather than
manufacture signal: the estimand is preserved and only the error structure changes. Each score
remains an unbiased mean over K orders drawn from the same distribution; what changes is that
the order-dependent component becomes *common* to the four terms of the contrast and cancels in
the difference. Classic common random numbers.

**Shared draws and per-sequence seeding are not interchangeable.** Deriving a stable per-sequence
seed (sha256 of sequence+structure, never Python's process-randomized `hash()`) would also give
batch invariance, but would leave the four errors independent — it buys the 0.533 column, not
the 0.977 one. That is why the fix shares draws rather than seeding per sequence.

## 3. The fix — scoring protocol `score_only/v2`

`orchestrator/mpnn_score_only.py`. One model load; backbone featurized once; K decoding-order
tensors drawn once from a **private** `torch.Generator` and reused for every sequence in the
call. The private generator matters: it makes the draws a function of `seed` alone rather than
of how much unrelated randomness ran earlier in the process.

Selected via `score_only_oracle(..., protocol=...)`, defaulting to v2, with
`PROTEINMPNN_SCORE_PROTOCOL` in `config.py` / `.env.example`, a `--score-protocol` flag on both
CLIs, and the agent's `search_mutations` call site passing it through. `seed = 0` stays rejected
for parity with the CLI. v1 remains callable — the old artifacts cannot be reproduced without
it, and `cmd_internal` (sub-question A) must be run with `--score-protocol score_only/v1` to
reproduce its committed numbers.

Cost: v2 is comparable to v1 batched and far cheaper than per-sequence CLI invocations
(82 s → ~15 s for the 72-sequence panel), because the model loads once instead of 72 times.

### Verification (`tests/test_mpnn_score_only.py`, real weights on CPU)

- **Invariance is exact.** Reordered, chunked, one-call-per-sequence and decoy-spliced batches
  return bit-identical floats, asserted with `==` rather than `approx` — v2's draws are a
  function of the seed, so there is no float wobble to absorb. The epistasis contrast has its
  own test, since that is the quantity that actually broke.
- **Fidelity to upstream is exact.** Fed the decoding orders the CLI would have used,
  `_score_with_orders` reproduces v1's `global_score` to **8e-8** (float32 precision). This
  required replicating upstream's *call order*, not just its seed: constructing `ProteinMPNN`
  initialises weights from the global torch RNG and so offsets the stream before any decoding
  order is drawn. This is the guard on reaching into upstream internals — if the pinned clone's
  model construction, featurization or `_scores` reduction moves, it fails with a message
  naming the expected commit.
- **Negative controls** so invariance cannot pass vacuously: different seeds must give
  different scores, and distinct sequences must give distinct scores.
- Mocked tests cannot establish any of this. The four tests in `tests/test_mutation_search.py`
  that patch `subprocess.run` necessarily test v1 and now pin it explicitly.

**v2 does not equal v1 numerically on the same inputs**, and is not meant to: v1 additionally
burns K draws on its native-PDB pass, so even a one-sequence batch gets different orders
(measured max |v1 − v2| = 0.030 at K=4, the scale of order noise). A claim to the contrary in an
early draft of the module docstring was wrong and was corrected.

## 4. Finalist evaluation

`evaluate_finalists` in `orchestrator/mutation_search.py`, exposed as `--finalists` on the
search CLI. Deliberately **not** wired into the agent tool: that would be the agent redesign
this task excludes.

The audit it follows from: the only place the pipeline compared scores across stochastic
conditions was `adalead_search`'s `measured` dict, which accumulates across one invocation per
round and is read by both `_elite_parents` and the final ranking. Under v1, a candidate's draw
was decided by its generation order; under v2 that is fixed. There is no score cache anywhere
in the mutation path, so nothing persisted incomparable numbers.

What remains after v2 is not comparability but **selection bias**: a candidate reaches the top
of `measured` partly because its draw was favourable, so its search score is biased upward. The
stage therefore re-scores a bounded shortlist under **independent** draws — `eval_seeds` must be
non-zero, distinct, and must not include the search seed, all enforced — and reports mean, SD,
SEM, per-draw scores, per-draw ranks, mean pairwise Spearman across draws, rank spread, and the
stable top-3. Raw per-draw scores are kept so an alternative analysis needs no rescoring. One
draw reports `sd = None` rather than 0, which would imply certainty. The additive proposer runs
through the same AdaLead machinery so the two candidate sets differ in their oracle rather than
in how they were generated, merging is round-robin so a long list from one proposer cannot crowd
out the other, and scoring cost is tracked separately from the search's own.

This fixes numerical consistency and the selection of noisy winners. It does **not** correct
model bias: where ProteinMPNN is systematically wrong about a backbone — on 4G3O both the
additive and the full predictor were *negatively* correlated with measured stability —
re-scoring reproduces that error more precisely.

### Smoke run (4G3O, `--rounds 3 --candidates-per-round 6 --max-sites 2`, K=4, eval seeds 101/102/103)

Two observations worth more than the mechanics:

- **Every finalist scored below wild type** (Δ vs WT −0.004 to −0.040). On a native 1.6 Å X-ray
  backbone the model prefers the wild-type sequence to everything the search proposed, which is
  the expected and honest result.
- **The additive baseline and the `score_only` search returned the same five candidates.** The
  non-additive oracle only re-ranked a shortlist both proposers had already found. Caveat: both
  searches shared the AdaLead seed, so they explored the same sequence neighbourhood by
  construction, and this was a tiny budget. Tracked as **I-25** with a proper proposal-overlap
  measurement as the next step.

Ranking consistency across the three independent draws: mean pairwise ρ = 0.933, max rank
spread 1.

## 5. Pipeline and agent behaviour preserved

- `search_mutations` stays **read-only**; no auto-adoption of its top candidate was added.
- Affinity remains metadata, never a ranking signal.
- Tool descriptions and agent prose audited. Four unsupported claims removed:
  1. `scan_mutations` called the scorer "(validated against ProteinGym)" without saying at
     what — narrowed to single-substitution ranking against single-mutant DMS, with an explicit
     note that this says nothing about ranking combinations.
  2. `search_mutations` opened with "Search for **beneficial** MULTI-SITE mutation
     combinations" — now `EXPERIMENTAL`, with the stability-arm result stated and the additive
     view named as the baseline.
  3. The system prompt's "then apply_mutation on the most promising candidate **to confirm it
     improves the structure**" — a screen described as a confirmation. Rewritten.
  4. The `search_mutations` result note now says the non-additive signal has not been shown to
     beat an additive baseline.
- The agent prompt now carries the reasoning chain explicitly: a high score identifies a
  candidate hypothesis; repeat scoring assesses numerical consistency; re-folding screens
  predicted structural plausibility; experimental benefit remains unvalidated. Compatibility
  score, uncertainty, pLDDT and clash count stay separately interpretable.
- The additive scorer is **not** described as generally validated by this study. Its single-site
  ProteinGym check predates this work and covers single substitutions only.

## 6. The rerun

The whole stability arm was re-run under v2 on the **same sampled variants**: same 300 doubles
per domain, same constituent singles, same replicate subset, same structures. Verified rather
than assumed — doubles, singles, `eps_exp` and the raw-PDB SHA-256 are identical element by
element across all three domains, and the replicate subset lists match. The v1 artifacts were
not touched; v2 results carry a `_v2` filename tag and a different `config_id`.

```
benchmarks/score_only_epistasis_stability_{4G3O,3DKM,3L1X}_n300_k32_v2.json  (+ .png)
benchmarks/score_only_epistasis_stability_summary_n300_k32_v2.json
```

### 6.1 Batch sensitivity and repeatability — the fix works, and it was the whole story

Cross-seed reliability of `eps_model` over all 300 analysed pairs (draw A = seed 37, draw B = a
separate invocation at seed 38):

| domain | v1 reliability [95% CI] | v2 reliability [95% CI] | noise SD v1 → v2 | signal SD v1 → v2 |
|---|---|---|--:|--:|
| AMFR 4G3O | 0.885 [0.848, 0.913] | **0.992** [0.989, 0.994] | 0.0083 → 0.0020 | 0.0229 → 0.0224 |
| HECD1 3DKM | 0.401 [0.305, 0.485] | **0.992** [0.990, 0.994] | 0.0085 → 0.0007 | 0.0070 → 0.0079 |
| UBE4B 3L1X | 0.514 [0.424, 0.595] | **0.994** [0.993, 0.995] | 0.0109 → 0.0008 | 0.0112 → 0.0101 |

Noise SD falls 4–14×; **signal SD is preserved** (0.0229→0.0224, 0.0070→0.0079, 0.0112→0.0101).
That is the point: v2 removes variance without moving the quantity being measured. The
decomposition shows where the old noise lived:

| domain | between position pairs (v1 → v2) | within a position pair (v1 → v2) |
|---|--:|--:|
| AMFR 4G3O | 0.997 → 0.996 | 0.791 → **0.987** |
| HECD1 3DKM | 0.854 → 0.994 | 0.206 → **0.990** |
| UBE4B 3L1X | 0.942 → 0.997 | 0.301 → **0.990** |

Between-pair reliability was already high under v1 and barely moves. Essentially all of the
old instability was **within** a position pair — exactly the substitution-level resolution the
search needs. On 3DKM the within-pair signal under v1 (reliability 0.21) was close to pure
batching artifact.

This also retires the "scores correlate 0.28–0.92 across invocations" line. That figure was not
reproducible from any artifact, described the four-term epistasis contrast rather than raw
scores, and conflated a seed change with an invocation change. Under v2 a re-invocation at the
same seed is bit-identical, and a re-invocation at a *different* seed reproduces `eps_model` at
ρ ≈ 0.99.

### 6.2 Model-versus-experimental epistasis association — slightly better, still weak

| domain | v1 ρ [residue-pair block 95% CI] | v2 ρ [block 95% CI] | v1 Pearson | v2 Pearson |
|---|---|---|--:|--:|
| AMFR 4G3O | +0.113 [−0.118, +0.358] | +0.103 [−0.102, +0.313] | +0.184 | +0.178 |
| HECD1 3DKM | +0.251 [−0.006, +0.459] | +0.390 [+0.082, +0.614] | +0.296 | +0.450 |
| UBE4B 3L1X | +0.230 [−0.071, +0.498] | +0.286 [−0.113, +0.574] | +0.187 | +0.263 |

Pooled (secondary, within-domain centred, n = 900, 32 blocks): **+0.181 [+0.031, +0.319] →
+0.208 [+0.034, +0.372]**. Detrended (global-epistasis residual): +0.316/+0.214/+0.171 →
+0.307/+0.246/+0.286.

Two of three domains move up, one is flat, and 3DKM's CI now excludes zero where it previously
did not. This is the direction attenuation theory predicts — less measurement noise, less
attenuation — but it is **not** a change of conclusion. Three proteins, two per-domain CIs
still straddling zero, a pooled estimate that remains an order of magnitude short of useful
prediction. The association was never established as reliable and still is not. Removing
measurement noise raised the *ceiling*; it did not demonstrate that the model knows more
biology than it did yesterday.

### 6.3 Within-position-pair substitution ranking — the one qualitative change

This is the readout the stability arm called a genuine miss: given a position pair that does
couple, does the model rank *which substitutions* couple there?

| | v1 | v2 |
|---|---|---|
| AMFR 4G3O (7 pairs) | median ρ +0.025, 4/7 positive | median ρ +0.145, 6/7 positive |
| HECD1 3DKM (8 pairs) | median ρ −0.108, 3/8 positive | median ρ +0.036, 5/8 positive |
| UBE4B 3L1X (9 pairs) | median ρ +0.180, 7/9 positive | median ρ +0.257, 9/9 positive |
| pooled, same 24 pairs | median **+0.069**, 14/24 positive, sign-test **p = 0.54** | median **+0.196**, 20/24 positive, sign-test **p = 0.002** |

Under v1 the within-pair association was indistinguishable from a coin flip. Under v2 the sign
test is clearly positive. Read carefully: the effect size is still small (median ρ ≈ 0.20), the
sign test is a consistency-of-direction test over 24 pairs from three proteins and says nothing
about magnitude, and the 24 pairs are not independent of each other (they share single
mutations). What changed is that a signal previously buried under its own measurement noise is
now visible. "Visible and weak" is the honest description.

### 6.4 Additive versus full prediction — unchanged, and still no win

Grouped held-out Pearson (group-disjoint by residue pair, linear calibration fitted on train
groups only):

| domain | v1 additive → full (Δ) | v2 additive → full (Δ) |
|---|---|---|
| AMFR 4G3O | −0.329 → −0.448 (**−0.120**) | −0.338 → −0.449 (**−0.111**) |
| HECD1 3DKM | +0.193 → +0.266 (+0.073) | +0.235 → +0.272 (+0.037) |
| UBE4B 3L1X | +0.093 → +0.134 (+0.041) | +0.108 → +0.147 (+0.039) |

The pattern is identical to v1: a clear loss on one domain, two small gains on the others, and
on 4G3O both predictors are *negatively* correlated with experiment. The in-sample paired
block-bootstrap Δ Pearson CIs (a different, in-sample quantity from the held-out point
estimates above, and reported as such) are +0.031/+0.259 on 4G3O, −0.015/+0.059 on 3DKM and
−0.019/+0.059 on 3L1X — two of three straddle zero, and the third disagrees in sign with its
own held-out estimate.

So the fix did **not** rescue combinatorial scoring. Cleaner numbers did not turn the
non-additive term into a better predictor of double-mutant effects than adding the two singles.
That is the expected outcome if the limitation is the biological model rather than the
numerics, and it is consistent with the note in §4: rescoring removes selection bias, not model
bias.

### 6.5 Runtime and scoring cost

| domain | sequences scored (main) | v1 main | v2 main | v1 full-sample replicate | v2 full-sample replicate |
|---|--:|--:|--:|--:|--:|
| AMFR 4G3O | 392 | 306 s | 380 s | 291 s | 329 s |
| HECD1 3DKM | 531 | 684 s | 915 s | 610 s | (unusable) |
| UBE4B 3L1X | 510 | 637 s | 671 s | 555 s | 1365 s |

v2 costs roughly 1.05–1.35× v1 per sequence at the same K. The reason is structural: v1 scores
a batch in one subprocess with the model loaded once and the featurisation done once, and it
burns K draws on an initial native-PDB pass; v2 runs in-process, reuses a cached model, and
threads every sequence through the *same* K orders. The extra time is the per-sequence loop,
not extra model work.

Two of these numbers are wall-clock-contaminated and should not be used: the 3DKM v2
full-sample replicate recorded 14,133 s because the host slept mid-run, and the 3DKM v2 n=60
replicate (1,228 s) ran under CPU contention. They are listed here only so the artifact's
`runtime_seconds` field is not read as clean measurement. The clean comparisons above are
4G3O and 3L1X.

The relevant cost conclusion is not the ratio but what it buys: v2 at **K=8** reached eps
reliability 0.977 in the diagnostic, where v1 at **K=32** managed 0.40–0.89. Per unit of
reliability, v2 is several times cheaper.

### 6.6 What the rerun changes, and what it does not

**Numerical improvement, firmly:** repeatability 0.40–0.89 → 0.99 across three domains, with
the signal variance preserved and the gain concentrated exactly where the pipeline needs it.

**Biological prediction, barely:** the association moved up a little, one CI crossed zero, and
within-pair ranking became consistently positive instead of a coin flip. Every one of those is
a *ceiling* effect from removing measurement noise, measured on the same three proteins, with
the same unverified WT convention and the same dependence-aware block bootstraps. None of it
establishes reliable substitution-level epistasis prediction, and none of it is evidence that a
higher compatibility score means a better protein.

**Combinatorial search, not at all:** it has still not beaten the additive baseline on held-out
double-mutant effects, and the smoke run in §4 returned the same five candidates from both
proposers. That remains open as I-25.

## 7. How many decoding orders? (the K-sweep)

`PROTEINMPNN_NUM_DECODING_ORDERS >= 32` was calibrated against v1's noise on one domain. With
v2 in place that number needed re-deriving, so the same three panels were swept over
K ∈ {2, 4, 8, 16}, two independent draws of the shared orders per point (seeds 37/38), with
K=32 reused free from the committed artifacts. Harness:
`benchmarks/sweep_decoding_orders.py`; artifact
`benchmarks/decoding_order_sweep_n300_k2-4-8-16_v2.{json,md}`.

**Two curves, because only one of them is the config value's question.** Every K figure on
record until now measured the four-term epistasis contrast
`f(AB) − f(A) − f(B) + f(WT)`. The pipeline does not compute that quantity — it ranks
`f(mut) − f(WT)`. The contrast sums four perturbations and is correspondingly noisier, so the
old numbers were a conservative proxy for the setting they were being used to choose. Both come
free from the same scored panel, so the sweep reports both.

| K | single-delta reliability (4G3O / 3DKM / 3L1X) | eps reliability (4G3O / 3DKM / 3L1X) |
|--:|---|---|
| 2 | 0.961 / 0.935 / 0.951 | 0.930 / 0.827 / 0.892 |
| 4 | 0.980 / 0.972 / **0.934** | 0.952 / 0.927 / 0.896 |
| 8 | **0.990 / 0.982 / 0.980** | 0.979 / 0.958 / 0.963 |
| 16 | 0.993 / 0.996 / 0.993 | 0.987 / 0.982 / 0.990 |
| 32 | — (not stored) | 0.992 / 0.992 / 0.994 |

The curves separate by roughly **4× in K**, as the four-term structure predicts: single-delta
reaches ≥0.98 on all three domains by K=8, where the contrast needs K≈32 for the same. K=32 was
provisioned for a quantity the pipeline never evaluates.

**It is plain Monte-Carlo averaging.** Fitting `noise_sd ~ K^a` over the six curves gives
a ∈ [−0.68, −0.44] against an ideal of −0.50, so reliability at an unmeasured K is predictable
rather than needing measurement. Extrapolated: single-delta wants K ≈ 2 for 0.95 and K ≈ 6–12
for 0.99; eps wants K ≈ 4–5 for 0.95 and K ≈ 20–26 for 0.99. K=8 sits just past the
single-delta 0.99 crossing on two of three domains.

**Going below 8 is not supported.** 3L1X at K=4 gives single-delta 0.934 — *lower* than its own
K=2 value of 0.951, with barely-overlapping CIs. That non-monotonicity is a limitation of the
design rather than a finding about K: a two-draw reliability estimate conditions on one
particular pair of order-draws, and the bootstrap CI resamples variants while holding those
draws fixed, so the interval understates the true uncertainty. Three or more draws per point
would resolve it and were not run. The honest reading is that K=4 is within noise of K=8 on two
domains and visibly worse on one.

**Raising K buys repeatability, not biology.** The association with experiment is nearly flat in
K on 3DKM (+0.351 → +0.390 over K = 2 → 32) and 3L1X (+0.273 → +0.286) — saturated by K=2. Only
4G3O climbs materially (+0.013 → +0.103), and that is the domain with the weakest association in
absolute terms. Past K ≈ 8 the spend buys decimal places of numerical self-consistency.

**Recommendation: leave the default at 8** — a 4× reduction against the old ≥32 advice, now
measured on all three domains instead of extrapolated from one. The default was not changed as
part of this work; that is a config commitment, left to the maintainer.

**The sweep's cost column is unusable.** `s/seq/order` reads 0.41 s on 3DKM's K=4 point against
≈0.04 s everywhere else — host contention, the same cause as the wall-clock contamination noted
in §6.5. The reliability figures are computed from scores and are unaffected. No cost curve
should be read off that column.

## 8. Limitations

- The diagnostic is **one domain** (4G3O, 47 aa, 30 doubles, K=8). The mechanism is
  domain-independent — it is a property of the RNG plumbing — but the *magnitudes* are not.
- Shared draws were measured via the `individual` condition, which realises them exactly but is
  not the shipped implementation. The shipped implementation is separately verified for
  invariance and for fidelity to upstream arithmetic.
- Cross-seed reliability under shared draws measures stability against re-drawing the shared
  orders. It is not a measure of experimental reliability, and **decoding reliability says
  nothing about whether the model is right**.
- High repeatability between position pairs does not establish that those pairs are
  biologically correct — the stability arm's central caveat, unchanged by anything here.
- The experimental WT reference was never verified; `DMS(WT) = 0` remains an explicit
  convention, so absolute experimental epistasis and its sign stay uninterpretable. Every
  primary readout is invariant to that offset by construction.
- v2 depends on upstream internals at the pinned commit rather than on the CLI contract. The
  fidelity test is the mitigation, not a guarantee.

## Follow-ups

- **I-23** — **done** (§7). Swept K ∈ {2,4,8,16} under v2 on all three domains, K=32 reused.
  The knee is at K=8 for the quantity the search ranks; the default of 8 stands and K=32 is
  retired. Residual gap: two draws per point, so the per-point reliability CI understates
  draw-level uncertainty (3L1X is non-monotonic between K=2 and K=4).

- **I-25** — decide whether `search_mutations` earns its keep over the additive baseline.
- **I-26** — nothing in the pipeline measures experimental benefit; closing it needs an
  independent oracle, not a better compatibility score.
- **I-24** — assay selection for position-pair diversity.
- **I-18** — the function arm, still untouched and explicitly out of scope here.

## Artifacts and provenance

- `benchmarks/diagnose_score_only_batching.py` + `..._4G3O_k8_d30.json` — the diagnostic, with
  config, panel membership, reference scores, reference epistasis, per-condition metrics,
  stream-offset probe, shared-draw comparison and per-condition scoring cost.
- `orchestrator/mpnn_score_only.py` — protocol v2.
- `benchmarks/sweep_decoding_orders.py` + `decoding_order_sweep_n300_k2-4-8-16_v2.{json,md}` —
  the K-sweep of §7. Reads the committed v2 panels read-only; writes no stability-arm artifact.
- `tests/test_mpnn_score_only.py` (17 tests), `tests/test_finalist_evaluation.py` (19 tests).
- Model: ProteinMPNN `v_48_020`, clone pinned at `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`,
  CPU (M3), `augment_eps = 0`.
- Dataset: ProteinGym `DMS_substitutions.parquet` via `benchmarks/_proteingym.py`; structures
  from RCSB, cached under `benchmarks/epistasis_structures/`.
- Protocol identity is part of both the artifact config hash and the artifact filename, so a v2
  run can neither reuse nor overwrite a v1 result. v1 artifacts keep the names cited in
  `research_plan/` and the earlier write-up; v2 results carry a `_v2` tag.
