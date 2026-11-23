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
- **Backbones: N = 16**, stratified by length, listed explicitly below. Every entry was
  screened from the actual PDB file (method, resolution, model count, chain count, gap-free
  fragment count, nonstandard residues) — nothing is asserted from memory. Eligibility:
  X-ray, resolution ≤ 2.7 Å, single gap-free chain of standard residues.
**The 16 backbones (pre-registered).** Lengths are the *structure chain* length, which is what
Study 1 scores; for the Tsuboyama entries this differs from the DMS slice length (4G3O is a
53-residue chain carrying a 47-residue DMS slice).

| stratum | PDB | protein | L | res (Å) | fold |
|---|---|---|--:|--:|---|
| small | 4G3O | AMFR domain | 53 | 1.60 | — |
| small | 1LP1 | — | 55 | 2.30 | — |
| small | 1TUD | — | 60 | 1.77 | — |
| small | 1I2T | — | 61 | 1.04 | — |
| small | 1R69 | — | 63 | 2.00 | — |
| small | 1TG0 | — | 66 | 0.97 | — |
| small | 1F0M | — | 71 | 2.20 | — |
| small | 3DKM | HECD1 domain | 79 | 1.60 | — |
| medium | 1STN | staph nuclease | 136 | 1.70 | α/β |
| medium | 1MBO | myoglobin | 153 | 1.60 | α |
| medium | 1PPN | papain | 212 | 1.60 | α+β |
| medium | 2PTN | trypsin | 223 | 1.55 | β |
| medium | 1BTL | TEM-1 β-lactamase | 263 | 1.80 | α/β |
| large | 5CPA | carboxypeptidase A | 307 | 1.54 | α/β |
| large | 1AJ8 | citrate synthase | 371 | 1.90 | α |
| large | 3GRS | glutathione reductase | 461 | 1.54 | α/β |

The small stratum is drawn from the 11 Tsuboyama X-ray domains at ≤ 2.5 Å, and includes 4G3O
and 3DKM for continuity with Row B. Screened alternates, equally clean, if a substitution is
wanted: 5JRT (64, 1.53), 1ORC (64, 1.54), 1YU5 (67, 1.40); 7RSA 124, 3CHY 128, 1RX2 159,
1L63 162, 4AKE 214, 2CGA 245, 2CBA 258; 1EZM 298, 1OMP 370, 1QLP 372.

**Rejected by the screen**, recorded so the selection is auditable: 1EMA (GFP — 6 fragments,
longest 64 of 225 aa, chain is gapped), 1STP (2.60 Å) and 1HTI (2.80 Å) over the cap at the
time of screening, 8TLN (2 fragments, 316 of 318, chain `E`), 1A8I (4 fragments, longest 357
of 813), 1GPB (823 aa, past the size target). Losing 1EMA cost the set its β-barrel; 2PTN and
1BTL carry the β and α/β load instead.

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

### Cost — measured, not extrapolated
Throughput was measured directly on five of the chosen backbones with the model cache warm and
the fixed cost amortised over 41 sequences, giving ≈ `0.0004·L + 0.004` s per sequence per
decoding order:

| L | 66 | 136 | 263 | 370 | 461 |
|---|--:|--:|--:|--:|--:|
| s/seq/order | 0.030 | 0.057 | 0.118 | 0.151 | 0.187 |

At M=100, K-sum 30 and D=5 that is ≈ 7 min per small backbone, 14–30 min per medium, 34–47 min
per large: **≈ 4.7 h total** (58 min + 105 min + 119 min). Single-threaded with no contention,
so treat it as a floor — earlier wall-clock figures in this project were inflated by host
contention and by a sleeping laptop (`Process/mutation-scoring-consistency.md` §6.5, §7).

A first attempt at this measurement was discarded: at 5 sequences and K=2, model load and
featurisation dominated so completely that the 66-residue backbone appeared *slower* than the
263-residue one. Per-sequence throughput must be measured with the fixed cost amortised.

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
lengths. Within Tsuboyama, the limit is structural method rather than labels: 45 assays clear
the doubles bar, only 12 have a crystal structure (see below). **Assay-family generalization remains open after this study** and is tracked as a
follow-up, not claimed by it.

### Backbone provenance in this corpus — and why the strata exist

All 45 candidates were screened from file. The methods are **not** uniform, which was not
anticipated when this study was drafted:

| method | count | what it is |
|---|--:|---|
| X-ray diffraction | **12** | one crystal-averaged structure, carries a resolution |
| Solution NMR | **32** | an *ensemble* (9–50 models) consistent with NMR restraints; no resolution number |
| Powder diffraction | **1** | 2O2W — microcrystalline powder, far lower information content |

So **X-ray availability, not double-mutant count, is the binding constraint**: 45 assays have
≥ 300 doubles but only 12 have a crystal structure. The original "expected yield 12–20 of 45"
was wrong about the mechanism and right about the number only by luck.

Resolution is the finest detail distinguishable, in Å, lower being better: ~1 Å resolves atoms,
1.5–2.0 Å places side chains confidently, 2.5–3.0 Å gives a reliable backbone with
increasingly modelled side chains. **The cap is set at ≤ 2.7 Å**, deliberately not the tidier
2.5 Å: 3L1X is 2.60 Å and is one of Row B's three published domains, and excluding real data to
keep a round number would break comparability for no physical reason. Resolution is recorded
per domain so it can be checked afterwards as a covariate. Note that resolution error is a
*fixed* property of a backbone — it adds no variance across draws, so it cannot affect a
reliability estimate; it can only bias an association systematically.

NMR entries are included as a **separate, pre-declared stratum, never pooled with X-ray.**
ProteinMPNN needs only backbone coordinates and an NMR model 0 is a gap-free single chain, so
they score without special handling; Row B's own code already records X-ray-only as "a SCOPE
RESTRICTION… not a claim that NMR ensembles are unsuitable". Two caveats carried explicitly:
model 0 is conventional but arbitrary, and the ensemble spread is information this design
discards. Keeping the strata separate turns backbone provenance into its own generalization
axis rather than a confound.

2O2W (powder diffraction) is **excluded** from both strata; one assay, and the method is not
comparable to either.

**A latent fragility in Row B worth recording.** Its candidate list was
`4G3O, 3DKM, 3L1X, 2BTT, 1PV0, 1V1C`, and the last three are all NMR — they were logged as
"not evaluated — quota already filled" and so never screened. Had any of the first three
failed, that study would have run on fewer than three domains. This does not affect its result.

### Design
- **Domains: every Tsuboyama assay passing a pre-registered rule**, with the eligibility table
  dumped before any association is computed (the existing harness already does this):
  - ≥ 300 usable doubles (both constituent singles present, parseable, in range, WT letter
    matching `target_seq`);
  - single gap-free chain containing `target_seq`, all standard residues;
  - length ≤ 80;
  - **primary stratum:** X-ray, resolution ≤ 2.7 Å → **12 domains**
    (4G3O 1.60, 1LP1 2.30, 1I2T 1.04, 5JRT 1.53, 1TUD 1.77, 1R69 2.00, 1ORC 1.54, 1TG0 0.97,
    1YU5 1.40, 1F0M 2.20, 3L1X 2.60, 3DKM 1.60);
  - **secondary stratum:** solution NMR, model 0 → up to **32 domains**.
  Expected total ≈ 44. **If more qualify than the compute budget allows, select among
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
  new domain**. Run **within each stratum separately**; the X-ray result is the headline and
  the NMR result is reported beside it. A between-strata comparison is declared here as a
  secondary question (does backbone provenance change the association?) and is not pooled into
  the primary estimand. The prediction interval *is* the generalization claim; Row B's pooled CI is a
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
≈ 500 sequences × K=8 × D=5 per domain. At the measured 0.030 s/seq/order for ~60-residue
chains that is ≈ 0.6 ks per domain: **≈ 2 h for the 12 X-ray domains**, ≈ 7 h more if the full
32-domain NMR stratum is run. Both are background runs; the NMR stratum can be split.

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
