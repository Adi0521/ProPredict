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
4. **Biological predictive performance:** see "The rerun" below.
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

_(filled in below once the three-domain v2 run completes)_

## 7. Limitations

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

- **I-23** — re-tune `PROTEINMPNN_NUM_DECODING_ORDERS` from the v2 rerun; K≥32 was calibrated
  against v1's noise and is likely over-provisioned.
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
- `tests/test_mpnn_score_only.py` (17 tests), `tests/test_finalist_evaluation.py` (19 tests).
- Model: ProteinMPNN `v_48_020`, clone pinned at `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`,
  CPU (M3), `augment_eps = 0`.
- Dataset: ProteinGym `DMS_substitutions.parquet` via `benchmarks/_proteingym.py`; structures
  from RCSB, cached under `benchmarks/epistasis_structures/`.
- Protocol identity is part of both the artifact config hash and the artifact filename, so a v2
  run can neither reuse nor overwrite a v1 result. v1 artifacts keep the names cited in
  `research_plan/` and the earlier write-up; v2 results carry a `_v2` tag.
