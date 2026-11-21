# Row C — How far do the `score_only` results generalize?

**Status:** design / pre-registration. Nothing has been run. Written 2026-11-21.
**Companions:** `research_plan/rowB-score-only-epistasis.md` (the study this extends),
`Process/mutation-scoring-consistency.md` (protocol v2 and the K-sweep).

Two studies, designed together because they share one estimator and one flaw to fix, but kept
separate because they answer different questions and need different statistics.

---

## The flaw both studies exist to fix: the two-draw design

Every reliability figure in Row B — the stability arm's 0.89 / 0.40 / 0.51, protocol v2's
0.99s, the K-sweep's whole table — comes from **exactly two scorings** of a panel: draw A at
decoding seed 37, draw B at seed 38. Reliability is then

```
noise_var = Var(a − b)/2 ;  observed_var = (Var(a) + Var(b))/2 ;  reliability = 1 − noise/observed
```

There are two independent sources of randomness, and the design samples only one of them:

| level | what varies | samples |
|---|---|--:|
| variant | which mutants are in the panel | ~300 ✓ |
| draw | which K decoding orders were drawn | **2** ✗ |

`es.reliability`'s bootstrap resamples variants (`idx = rng.integers(0, n, size=n)`, applied to
`a` and `b` together) while **conditioning on those two particular order-sets**. So the interval
answers "with these same two draws, what if the variants differed?" — not "what if different
orders had been drawn?" Two draws can be jointly lucky or unlucky in a way shared across every
variant at once, and no amount of resampling variants can see it.

It demonstrably bit: 3L1X single-delta reliability was **0.951 [0.935, 0.962] at K=2** and
**0.934 [0.909, 0.953] at K=4**. Reliability cannot fall as K rises in expectation (noise
variance falls as ~1/K, which the sweep's own log-log fits confirm), so the reversal is
draw-level luck — and the intervals were nearly disjoint, confidently asserting a difference
that is an artifact.

**Fix: D = 5 draws and a one-way random-effects estimator.** For variant *i* and draw *d*,
`x_id = μ_i + e_id`; within-variant variance across draws is noise, between-variant variance is
signal, and their ratio is an intraclass correlation. ICC(1,1) is the reliability of a *single*
draw, so it is directly comparable to every two-draw number already on record. The CI is
bootstrapped over **both** variants and draws.

Large effects on record are safe — a 0.40 → 0.99 jump is far too big to be draw luck. Fine
distinctions are not: "K=4 vs K=8" and "is 0.990 better than 0.982" are exactly what the old
design cannot support.

---

## Study 1 — Does the K=8 operating point generalize? (unlabelled)

### Question
Does `PROTEINMPNN_NUM_DECODING_ORDERS = 8` hold across proteins beyond three 47–72 aa domains,
and does any cheap backbone property predict the K a given protein needs?

### Why this needs no experimental data
Reliability is test-retest of the model against itself. DMS labels are needed only for the
*association* claim. Dropping them removes every data dependency — no doubles requirement, no
assay family, no ProteinGym — so the backbone set can be chosen for diversity instead of for
what happens to have labels. This is the single biggest reason to run Study 1 first.

### Design
- **Backbones: N = 16**, stratified by length, pre-registered as an explicit list before any
  scoring: 8 small (≤ 80 aa, drawn from the 45 Tsuboyama PDBs, which already thread), 5 medium
  (~150–250 aa), 3 large (~350–450 aa). Eligibility: X-ray, resolution ≤ 2.5 Å, single
  gap-free chain. The medium/large entries need a curated list; it goes in this file before the
  run, not after.
- **Variants: M = 100** single mutants per backbone, sampled uniformly over (position,
  non-WT amino acid) at a fixed seed. Plus WT.
- **Draws: D = 5** (seeds 37, 38, 39, 40, 41 — distinct and nonzero).
- **K ∈ {2, 4, 8, 16}**, protocol `score_only/v2`.
- **Estimand:** ICC(1,1) of `f(mut) − f(WT)`.

### Analysis, fixed in advance
- Primary: ICC per (backbone, K), with a two-level bootstrap CI over variants and draws.
- **Report the conditional (variants-only) interval alongside the full one.** The gap between
  them is a direct measurement of how much the old two-draw design understated uncertainty —
  worth having as a number rather than an argument.
- Pre-registered functional form: `noise_sd ∝ K^-0.5`. The K-sweep measured exponents in
  [−0.68, −0.44] over six curves. A clear departure is informative, so it is declared now.
- Secondary (**exploratory, declared as such**): does ICC at K=8 track length, mean contact
  number, or fraction buried?

### Decision rule, fixed in advance
Retain K=8 as the default iff **ICC(K=8) ≥ 0.95 on ≥ 90% of backbones with a full-bootstrap
lower bound ≥ 0.90**. Otherwise raise it. Separately report the minimum qualifying K per
backbone, so the answer is a distribution rather than one number.

### Cost
Scales with sequence length as well as K, so the size-stratified set is not uniform:
roughly 0.6 ks per small backbone, 1.7 ks per medium, 3.5 ks per large → **≈ 6–8 h** as a
background run. Trim M or drop K=2 if that proves optimistic; the measured throughput figures
behind this estimate are contaminated by host contention (see
`Process/mutation-scoring-consistency.md` §6.5, §7) and should be treated as a lower bound.

---

## Study 2 — Does the epistasis association generalize across domains? (Row B Step 3)

### Question
Row B found a weak positive model-vs-experiment epistasis association, pooled ρ = +0.208
[+0.034, +0.372] over **three** domains. Does it hold across many more, and — the question
that actually matters — **what should be expected on a new protein?**

### Scope, stated honestly up front
This expands along **protein** variety, not **assay** variety. All 45 candidate domains are
Tsuboyama 2023: one lab, one measurement technology, 37–72 aa, stability ΔΔG. The parquet
carries no PDB mapping (`DMS_score, DMS_score_bin, mutated_sequence, target_seq, mutant,
DMS_id`); the Tsuboyama IDs simply embed theirs, which is why they are free. The 16
non-Tsuboyama assays with ≥ 300 doubles span 12 families and 93–770 aa, but need curated
UniProt→PDB mappings and gap-free domain-slice threading that mostly will not exist at those
lengths. **Assay-family generalization remains open after this study** and is tracked as a
follow-up, not claimed by it.

### Design
- **Domains: every Tsuboyama assay passing a pre-registered rule**, with the eligibility table
  dumped before any association is computed (the existing harness already does this):
  - ≥ 300 usable doubles (both constituent singles present, parseable, in range, WT letter
    matching `target_seq`);
  - X-ray, resolution ≤ 2.5 Å, single gap-free chain containing `target_seq`;
  - length ≤ 80.
  Expected yield 12–20 of 45. **If more qualify than the compute budget allows, select among
  qualifiers at random under a recorded seed — not first-come.** Row B's three domains were
  chosen by a quota that stopped at 3, so the eligibility *ordering* picked them; that is a
  selection artifact and it is removed here.
- 300 doubles per domain, existing fixed-seed sampling.
- **K = 8** (justified by the sweep), **D = 5 draws**, protocol v2.
- WT convention unchanged: `DMS(WT) = 0` is a stated convention, never verified; every primary
  readout is invariant to the offset by construction, and uncentered values are not pooled.

### Analysis, fixed in advance — this is what is actually new
- Per domain: Spearman ρ of `eps_exp` on `eps_model` with the residue-pair block bootstrap, as
  now.
- **Primary estimand: a random-effects meta-analysis across domains** on Fisher-z(ρ) —
  pooled mean, between-domain heterogeneity τ² and I², and a **95% prediction interval for a
  new domain**. The prediction interval *is* the generalization claim; Row B's pooled CI is a
  statement about the three domains measured, not about the next protein, and with three
  domains τ² was not estimable at all.
- Per-domain results are **descriptive**. With 12–20 domains, reading the best one as the
  headline is exactly the forking path this pre-registration exists to close.
- Attenuation: D = 5 gives a proper per-domain ICC, so raw and disattenuated ρ can both be
  reported — with the standing caveat that disattenuation corrects for **model decoding noise
  only**, never experimental noise, and is not a Spearman correction.
- **Within-position-pair ranking: cluster by domain.** Row B's pooled sign test (20/24 pairs
  positive, p = 0.002) treated 24 position pairs as independent when pairs within a domain
  share single mutations. That p-value is anti-conservative and is **superseded** by a
  domain-clustered test here.
- Additive vs full: per-domain held-out Δ, meta-analysed the same way.

### Decision rule, fixed in advance
The association is called **generalizable** only if the meta-analytic mean excludes zero *and*
the prediction interval's lower bound is positive. A positive mean with a prediction interval
spanning zero means "it works on average over these domains and cannot be relied on for the
next one" — a likely outcome, and an acceptable one.

### Cost
≈ 500 sequences × K=8 × D=5 per domain ≈ 0.8 ks plus bootstrap → **≈ 3.5–6 h** for 16 domains.

---

## Shared engineering (none of it written yet)

1. `epistasis_stats.icc_reliability(x)` — one-way random-effects ICC over an
   `(n_variants, D)` matrix with a two-level bootstrap CI. **Additive**: `es.reliability` stays
   for the committed two-draw artifacts, which must keep reproducing their published numbers.
2. `epistasis_stats.meta_analyze(...)` — random-effects pooling on Fisher-z, returning mean,
   τ², I², CI and prediction interval.
3. `benchmark_score_only_epistasis.py` — D-draw support (`--draws`, `--seeds`), remove the
   3-domain quota, raise the domain cap.
4. `benchmarks/benchmark_score_reliability.py` — new, unlabelled, for Study 1.

Artifacts stay parameter-stamped and overwrite-guarded; D and the seed list enter the config
hash, so a D=5 run can neither reuse nor overwrite a D=2 one.

## What neither study can establish

Reliability is the model agreeing with itself. **A backbone reaching ICC 0.99 buys no
experimental validity.** Study 2's prediction interval describes association with measured
ΔΔG epistasis on small single-domain proteins under an assumed WT reference — not that a
higher-scoring sequence is a better protein. Nothing here measures experimental benefit; that
needs an independent oracle (ISSUES I-26) and is out of scope.

## Follow-ups this study deliberately defers
- Assay-family generalization (Axis B): the 16 non-Tsuboyama assays, gated on domain-slice
  threading. Also unblocks Row B's function arm (I-18).
- Whether `search_mutations` earns its keep over the additive baseline (I-25).
