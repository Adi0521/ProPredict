# Row B — Step 2 (sub-question B), stability arm: does `score_only` epistasis track *experiment*?

**Date:** 2026-11-17
**Companion:** `research_plan/rowB-score-only-epistasis.md` (study design + narrative).
**Closes:** ISSUES I-18 (stability arm). The functional arm remains out of scope — see Follow-ups.
**Deliverable:** settle whether ProteinMPNN `score_only`'s non-additivity (established in
sub-question A) *corresponds to measured stability interactions*, and whether scoring the
double mutant outright beats summing the two single-mutant scores.

**Reproduce (CPU, ~45 min + ~30 min for the full-sample replicate):**
```bash
python -m benchmarks.benchmark_score_only_epistasis --arm stability \
    --n-doubles 300 --decoding-orders 32 --replicate-subset 60 --plots
python -m benchmarks.benchmark_score_only_epistasis --arm stability \
    --n-doubles 300 --decoding-orders 32 --reanalyze --replicate-full --plots --force
```

---

## Answer, up front

**Yes, weakly, and only at the level of *which position pairs* couple — not which substitutions.**

1. Model epistasis correlates **positively with experimental epistasis on all three domains**
   (Spearman +0.11, +0.25, +0.23), but every per-domain residue-pair block CI includes zero.
   Only the pooled, within-domain-centred estimate excludes zero: **ρ = +0.18 [+0.03, +0.32]**,
   over three proteins and 32 position pairs — consistent direction, small magnitude, weak
   evidence.
2. The correlation lives **between position pairs**, not within them. Across substitution
   identities at a *fixed* pair of positions, median ρ = +0.07 over 24 pairs, 14/24 positive
   (sign test p = 0.54). The model has an opinion about which positions couple; it has
   essentially no resolution on which amino-acid combinations couple there.
3. **Non-additive scoring beats the additive baseline on 2 of 3 domains, marginally**
   (held-out Pearson +0.193 → +0.266 on 3DKM, +0.093 → +0.134 on 3L1X), but the paired
   block-bootstrap deltas straddle zero, and on 4G3O the full model is **worse** (−0.329 →
   −0.448) in a regime where both models are anti-correlated with experiment. This is not a
   usable predictive win.
4. **The decoding noise is far larger than sub-question A's single-domain measurement
   suggested**, and it is domain-dependent: full-sample cross-seed reliability at K=32 is
   0.89 / 0.40 / 0.51. The ≥32 decoding-orders recommendation (ISSUES I-11) was calibrated on
   the one domain where it happens to suffice.

---

## What was built

`--arm {internal,stability}` on the existing harness (`benchmarks/benchmark_score_only_epistasis.py`),
sharing `load_assay` / `trim_pdb_to_dms` / `_eps_for_sample` with sub-question A, which keeps
its previous behaviour byte-for-byte (regression-tested, see Tests). Supporting modules:

- **`benchmarks/epistasis_stats.py`** — all statistics as pure functions: epistasis
  arithmetic, associations, individual-double and residue-pair **block** bootstraps, test-retest
  reliability with a bootstrap CI, between/within-group reliability decomposition,
  group-disjoint folds, and the additive-vs-full comparison with grouped held-out calibration.
  Pure so the arithmetic every conclusion rests on is testable without weights or data.
- **`benchmarks/_proteingym.py`** — the ProteinGym parquet download, lifted out of
  `benchmark_proteinmpnn_checkpoints.py` so both harnesses share one implementation. Before
  this, the epistasis harness hardcoded the cache path and would have crashed on a fresh
  checkout.
- Config-hashed artifact **reuse**, `--reanalyze` (recompute analysis from stored scores, no
  rescoring) and `--replicate-full` (one extra scoring pass, no repeat of the main pass).

---

## Decisions recorded before any correlation was computed

### Domain selection — executed, not asserted

Candidates were evaluated in the fixed order of the study design, each check actually run
(PDB header method, exact DMS-sequence match, gap-free trim, usable-double count):

| PDB | method | resolution | L | usable doubles | verdict |
|---|---|--:|--:|--:|---|
| 4G3O (AMFR_HUMAN) | X-ray | 1.60 Å | 47 | 2103 | **selected** (retained from sub-question A) |
| 3DKM (HECD1_HUMAN) | X-ray | 1.60 Å | 72 | 4288 | **selected** |
| 3L1X (UBE4B_HUMAN) | X-ray | 2.60 Å | 69 | 2370 | **selected** |
| 2BTT, 1PV0, 1V1C | solution NMR | — | 61/44/65 | 2253/1936/1982 | not evaluated — quota filled first |

All three selections pass gap-free trimming and exact sequence match. The quota filled before
the NMR candidates were reached, so no NMR structure was rejected on quality grounds; **X-ray
only is a scope restriction** (uniform backbone provenance with sub-question A), not a claim
that NMR ensembles are unsuitable for `score_only`.

### Experimental score semantics — and an offset that cannot be verified

Source column `DMS_score`; ProteinGym reference metadata gives the Tsuboyama-2023 source as
`ddG_ML_float`, direction +1 (higher = more stable). Verified against the data rather than
assumed:

- **No wild-type row exists** in any of these assays, so `DMS(WT)` cannot be read off.
- The values are consistent with a **WT-relative** quantity (mostly negative, continuous,
  range ≈ [−4.6, +2.3]), not an absolute ΔG.
- **No censoring and no missing data**: every value distinct, exactly one point at each range
  end, zero NaNs, zero duplicates.
- The additive-null fit `DMS(AB) ~ a + b·(DMS(A)+DMS(B))` gives **b = 0.39–0.54** across all
  six candidate domains. Because b ≪ 1, its intercept does **not** identify the WT reference —
  that estimate is only valid at b = 1. So the fit is reported and deliberately **not** used
  to back out WT.

**Consequence, carried through the whole analysis:** `DMS(WT) = 0` is an explicit convention,
not a verified fact. A wrong reference `c` shifts every `eps_exp` in a domain by the same
constant, so within-domain Spearman, Pearson and OLS slope *with an intercept* are invariant
(unit-tested), while **eps_exp signs and absolute magnitudes are not interpretable** and
uncentered values are **never pooled** across domains (the pooled row uses within-domain
centering and domain-specific intercepts).

That b ≈ 0.4 is itself a finding: experimental ΔΔG epistasis here is dominated by **global,
nonspecific** epistasis (diminishing returns as variants approach the unfolded limit), not by
pair-specific coupling — which motivates the detrended secondary readout below.

### Exclusions

Seven rules, all counted per domain (`provenance.exclusions` in each artifact): unparseable
mutant, position out of range, mutant WT letter disagreeing with `target_seq`, sequence-length
mismatch, missing score, duplicate mutant, and doubles missing one of their constituent
singles. On these three assays only the last fired: **49 / 54 / 134** doubles dropped. Nothing
else was excluded.

### The dependence structure is much worse than "doubles share singles"

The 300 sampled doubles per domain span only **7 (4G3O), 16 (3DKM), 9 (3L1X) distinct position
pairs** — these assays are many substitution identities at a handful of positions. The
effective number of independent units for any cross-pair claim is therefore single-digit to
low-double-digit, not 300. This drove three design choices: the individual-double bootstrap is
labelled **conditional/descriptive**, the primary CIs come from a **residue-pair block
bootstrap** (which still leaves residual dependence through shared singles across blocks), and
a **within-position-pair** analysis was added that carries no cross-pair dependence at all.

---

## Observed results

Full table (also at `benchmarks/score_only_epistasis_stability_summary_n300_k32.md`):

| domain | PDB | n doubles | position pairs | Spearman (block 95% CI) | Pearson | slope | within-pair median ρ (pos/tested) | reliability (n=300) | reliability between / within pair | held-out r: additive → full |
|---|---|--:|--:|---|--:|--:|---|--:|--:|---|
| AMFR_HUMAN | 4G3O | 300 | 7 | +0.113 [−0.118, +0.358] | +0.184 | +6.02 | +0.025 (4/7) | 0.89 | 1.00 / 0.79 | −0.329 → −0.448 |
| HECD1_HUMAN | 3DKM | 300 | 16 | +0.251 [−0.006, +0.459] | +0.296 | +22.99 | −0.108 (3/8) | 0.40 | 0.85 / 0.21 | +0.193 → +0.266 |
| UBE4B_HUMAN | 3L1X | 300 | 9 | +0.230 [−0.071, +0.498] | +0.187 | +11.05 | +0.180 (7/9) | 0.51 | 0.94 / 0.30 | +0.093 → +0.134 |

Pooled (secondary, within-domain centred): **ρ = +0.181 [+0.031, +0.319]**, r = +0.198, n = 900.

The OLS slopes (6–23) are large because model fitness units (|eps| ~ 0.01–0.03) and ddG units
(~1 kcal/mol) differ by roughly two orders of magnitude. **Slope = 1 is not the target**; only
its sign and significance are meaningful here.

### Detrended for global epistasis (secondary, in-sample fit)

Replacing `eps_exp` with the residual of `DMS(AB)` on a quadratic in `DMS(A)+DMS(B)`:

| domain | ρ (detrended) | block 95% CI |
|---|--:|---|
| 4G3O | **+0.316** | [+0.032, +0.536] |
| 3DKM | +0.214 | [−0.005, +0.365] |
| 3L1X | +0.171 | [+0.011, +0.331] |

Removing the nonspecific trend **strengthens** the association on 4G3O (+0.11 → +0.32) and
pushes two of three block CIs off zero. Read as: the model's non-additivity lines up better
with the *pair-specific* part of experimental epistasis than with raw `eps_exp`, which is
mostly global curvature. The detrending is fitted in-sample on the analysed pairs, so this is
descriptive, not a held-out claim.

### Where the signal is — and where it is not

Within a fixed position pair, across substitution identities (groups with n ≥ 20, 24 pairs
total across the three domains): median ρ = **+0.069**, 14/24 positive, sign test **p = 0.54**,
Wilcoxon p = 0.24. Individual pairs range from +0.67 (4G3O 28–41) to −0.50 (3DKM 26–28).

Cross-seed reliability decomposition (full sample, n=300) explains part of this and sharpens
the rest:

| domain | reliability, between position pairs | reliability, within pair |
|---|--:|--:|
| 4G3O | 1.00 | 0.79 |
| 3DKM | 0.85 | 0.21 |
| 3L1X | 0.94 | 0.30 |

The model is **highly self-consistent about which position pairs couple** and much noisier
about which substitutions couple at a given pair. On 3DKM/3L1X, within-pair reliability of
0.21–0.30 (attenuation ≈ 0.46–0.55) means weak within-pair tracking is partly a noise ceiling.
**On 4G3O it is not**: within-pair reliability is 0.79 (attenuation ≈ 0.89), so the near-zero
within-pair correlation there (+0.025) is a genuine failure to resolve substitution-level
coupling, not noise.

### Decoding noise, and a correction to the sub-question A recommendation

Reliability was first estimated on a 60-pair subset, then on all 300 pairs:

| domain | reliability (n=60 subset) | reliability (n=300, full) |
|---|--:|--:|
| 4G3O | 0.930 | 0.885 |
| 3DKM | **0.062** (CI [−0.17, +0.29], flagged unstable) | **0.401** (CI [+0.31, +0.49]) |
| 3L1X | 0.378 | 0.514 |

The subset estimate on 3DKM was badly unrepresentative and was flagged as unstable by its own
CI rather than reported as a usable attenuation factor — which is why the full-sample pass was
run before any interpretation. Taking the full-sample numbers: at **K = 32 decoding orders**,
per-double model epistasis is ~89% reliable on 4G3O but only **40–51% reliable** on the two
larger domains. √reliability bounds Pearson attenuation **from model decoding noise alone**
(independent equal-variance additive errors); it is not a Spearman correction and not a
total-noise ceiling — experimental noise attenuates further and is not estimated here.

A related measurement, made in passing but worth recording: **`score_only` scores are not
invariant to batch composition** at a fixed decoding seed. The same 60 doubles, same seed 37,
same K=32, scored in two different invocations, correlate at only 0.28 (3DKM) / 0.55 (3L1X) /
0.92 (4G3O). A sequence's decoding orders depend on the invocation, so the only honest
replicate is a separate invocation — which is what both reliability estimates use.

### Context: the model does track single mutants

| domain | ρ(model single effect, DMS single) | ρ(additive sum, measured double effect) |
|---|--:|--:|
| 4G3O | +0.364 | −0.132 |
| 3DKM | +0.548 | +0.317 |
| 3L1X | +0.589 | +0.219 |

`score_only` ranks single-mutant stability respectably, and degrades sharply when predicting
double-mutant effects — consistent with the strong global epistasis (b ≈ 0.4) plus accumulated
noise, and the reason the additive-vs-full comparison on 4G3O happens in an anti-correlated
regime where "improvement" is not a meaningful notion.

---

## Interpretation (separate from the observations above)

- **The tier-2 search oracle's epistasis is real but coarse.** Sub-question A showed
  `score_only` is non-additive; this step shows that non-additivity is **not arbitrary** — it
  points, weakly and consistently, in the same direction as measured stability coupling, and
  does so mostly by identifying *which positions* interact.
- **It is not accurate enough to rank specific double mutants.** Within-pair resolution is
  indistinguishable from zero, and the additive-baseline comparison yields no reliable
  held-out win. For the mutation search this means the cheap oracle can be trusted to *propose*
  combinations involving coupled positions, but not to *rank* the substitution choices at
  those positions.
- **The tier-3 re-fold funnel stays load-bearing**
  (`Process/phase2-task-3-refold-funnel.md`). This result argues *against* removing it and
  *for* keeping the cheap search in a proposal role.
- **ISSUES I-11 (`PROTEINMPNN_NUM_DECODING_ORDERS` ≥ 32) needs qualifying.** K = 32 was
  calibrated on 4G3O, the domain with by far the largest epistasis magnitude; on 3DKM and 3L1X
  it leaves per-double reliability at 0.40–0.51. Noise is roughly constant in absolute units
  while epistasis magnitude varies ~3× across domains, so a fixed K cannot deliver a fixed
  SNR. Either raise K further for search use, or average the oracle over several invocations.

---

## Limitations

- **Three proteins, one assay type, one lab.** All three are small Tsuboyama-2023 domains
  measured by the same method. The pooled row is a secondary, within-domain-centred summary;
  it is not evidence of broad generalization.
- **The experimental WT reference is unverified** (see above). Signs and magnitudes of
  `eps_exp` carry an unknown per-domain offset; only within-domain, intercept-bearing readouts
  are used.
- **Few independent units.** 300 doubles sit on 7–16 position pairs. Block CIs reflect this;
  they still leave residual dependence through singles shared across blocks.
- **Reliability estimates are themselves noisy**, and the subset-vs-full discrepancy on 3DKM
  (0.06 vs 0.40) is a concrete demonstration. Only the full-sample numbers are interpreted.
- **The detrending is in-sample** and descriptive.
- **No experimental-noise estimate.** ProteinGym's parquet carries no replicate information
  here, so the total attenuation ceiling is unknown; only the model-side component is measured.
- **Functional arm not run.** Whether this transfers to function-based assays is untested.

---

## Tests

`tests/test_epistasis_stats.py` — 28 tests, no weights/GPU/parquet required:
epistasis arithmetic; **WT-offset invariance** (ρ/r/slope unchanged, intercept absorbs the
shift); reliability against a hand-computed formula, plus its handling of identical, pure-noise
(flagged unstable via CI), anti-correlated (invalid) and tiny-sample inputs; block bootstrap
accepting list groups, counting distinct pairs, and being provably wider than the individual
bootstrap when pairs cluster; group-disjoint folds; `additive_vs_full` detecting a real
epistatic gain and correctly reporting *no* gain when the model's extra term is noise;
`load_assay`'s seven exclusion rules on a synthetic parquet; semantics recording the unknown
WT offset; PDB header parsing; domain selection order, quota and shortfall reporting; and two
**internal-arm regression tests** (same epistasis formula and output format, refuses to
overwrite). `pytest tests/test_boltz.py tests/test_mutation_search.py tests/test_mutation_scan.py`
→ 76 passed, 3 skipped. `ruff check .` / `ruff format --check .` clean.

---

## Follow-ups

- **Functional arm** (the other half of sub-question B): a function-based DMS, with
  domain-slice threading since those assays' `target_seq` are full-length and will not match a
  PDB chain gap-free. Expect weaker correlation — function depends on more than fold stability.
- **Qualify ISSUES I-11** with the domain-dependence above, and decide between a larger K and
  multi-invocation averaging for search use.
- **More position pairs, not more doubles.** The binding constraint on this study is the number
  of distinct position pairs (7–16), not the number of variants. An assay selection that
  maximises pair diversity would do more for statistical power than scoring more doubles.
- **Test the proposal-not-ranking reading directly:** does restricting AdaLead to pairs the
  oracle marks as coupled, with substitution choice delegated to tier 3, beat the current
  arrangement?

---

## Artifacts

- `benchmarks/score_only_epistasis_stability_{4G3O,3DKM,3L1X}_n300_k32.json` — per domain: raw
  WT/single/double model scores, experimental values, derived effects and epistasis, variant
  IDs and positions, CA distances, exclusions, structure identity (method, resolution,
  sequence + raw-PDB sha256), dataset provenance, ProteinMPNN checkpoint and git revision, all
  seeds, decoding orders, config hash, runtimes, both replicates, and the full analysis.
- `benchmarks/score_only_epistasis_stability_*_n300_k32.png` — per-domain panels.
- `benchmarks/score_only_epistasis_stability_summary_n300_k32.{json,md}` — comparison table,
  pooled secondary analysis, and the domain-eligibility record.
- `benchmarks/epistasis_stats.py`, `benchmarks/_proteingym.py`, `tests/test_epistasis_stats.py`.
- `benchmarks/epistasis_structures/{3DKM,3L1X}.raw.pdb` — cached RCSB structures. These are
  **untracked**: `.gitignore` excludes `*.pdb`, and `4G3O.raw.pdb` is in the repo only because
  it was force-added. They re-download on demand, so nothing breaks without them; force-add
  them for parity if offline reproducibility matters
  (`git add -f benchmarks/epistasis_structures/{3DKM,3L1X}.raw.pdb`).
