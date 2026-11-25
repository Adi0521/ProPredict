# Row C Study 1 — does the K=8 decoding-order operating point generalize?

**Date:** 2026-11-23
**Companions:** `research_plan/rowC-reliability-generalization.md` (the pre-registration, including
amendment A1), `Process/mutation-scoring-consistency.md` §7 (the three-domain K-sweep this
extends), `Process/rowB-step2-stability-arm.md`.

## What this was

The K-sweep set `PROTEINMPNN_NUM_DECODING_ORDERS = 8` from three domains of 47–79 residues,
with two decoding draws per point. Two things were left open. The sweep could not say whether
K=8 holds on proteins unlike those three, and its two-draw design could not support fine
distinctions between K values at all (ISSUES I-23(b)) — 3L1X scored *lower* at K=4 than at K=2,
which is impossible in expectation.

This study answers both on 16 X-ray backbones spanning 53–461 residues and four fold classes,
with D = 5 draws and a one-way random-effects estimator. It uses **no experimental data**:
test-retest reliability compares the model against itself, so the backbone set could be chosen
for structural diversity rather than for what happens to carry DMS labels.

**Answers, up front.**

1. **Does K=8 generalize?** Yes. **16/16 backbones pass** the pre-registered rule (ICC(1,1) ≥
   0.95 with F-interval lower bound ≥ 0.90), from 53 to 461 residues. ICC at K=8 ranges
   0.9851–0.9970.
2. **Is K=8 *necessary*?** No. **K=2 clears the identical bar on 15 of 16 backbones**; only
   1PPN needs K=4. The rule was written to test sufficiency and that is all it tested. This
   resolves I-23(b) toward K being reducible by 4×, but see §3 — reliability is not the
   constraint that should decide it.
3. **Does noise follow the pre-registered K^−0.5?** Yes, tightly: 16/16 log-log slopes,
   mean **−0.503**, range [−0.558, −0.438]. The K-sweep's looser [−0.68, −0.44] was itself
   two-draw noise. Noise at unmeasured K is now predictable analytically.
4. **Did D = 5 change anything, and how?** Yes, but not by the mechanism the design predicted.
   The old two-draw intervals are **wider**, not narrower; their failure is *miscalibration* —
   they miss the D=5 estimate in 7/64 points against a nominal 5%, and the old reading of this
   very data still produces one impossible reversal. See §5, which corrects amendment A1's
   framing.
5. **Does any cheap backbone property predict the K a protein needs?** Not establishable here.
   The descriptor correlations are real but rest on 16 points and one influential backbone;
   declared exploratory in advance and reported as a hypothesis (§7).

**What none of this establishes.** Reliability is the model agreeing with itself. A backbone
reaching ICC 0.99 buys no experimental validity, and nothing here measures whether a
higher-scoring sequence is a better protein.

---

## 1. Design as run

| | |
|---|---|
| Backbones | 16, pre-registered, X-ray ≤ 2.7 Å, single gap-free chain of standard residues |
| Variants | 100 single mutants per backbone, uniform over the L × 19 (position, non-WT aa) grid, seed 20261121 |
| Draws | D = 5 (decoding seeds 37, 38, 39, 40, 41) |
| K | 2, 4, 8, 16 |
| Protocol | `score_only/v2`, ProteinMPNN `v_48_020` |
| Estimand | ICC(1,1) of `f(mut) − f(WT)` over the (100, 5) matrix |
| Scored | 64 (backbone, K) points; per backbone 101 sequences × 5 draws × (2+4+8+16) orders |

Eligibility is enforced in the harness rather than trusted: `Backbone.__init__` re-screens the
PDB file and refuses an ineligible code. Nothing in the backbone list is asserted from memory —
`benchmarks/screen_backbones.py` is the provenance, and it rejected four candidates that looked
fine (1EMA is six fragments, longest 64 of 225).

The quantity analysed is `f(mut) − f(WT)`, the per-sequence score the mutation search ranks by.
It is **not** the four-term epistasis contrast that every K figure before the sweep measured and
that the pipeline never computes.

## 2. Result — K=8 holds everywhere tested

| backbone | L | res (Å) | ICC(1,1) @ K=8 | F 95% CI | signal SD | noise SD | K^x slope | min qualifying K |
|---|--:|--:|--:|---|--:|--:|--:|--:|
| 4G3O | 53 | 1.60 | 0.9949 | [0.993, 0.996] | 0.0429 | 0.0031 | −0.455 | 2 |
| 1LP1 | 55 | 2.30 | 0.9970 | [0.996, 0.998] | 0.0412 | 0.0022 | −0.519 | 2 |
| 1TUD | 60 | 1.77 | 0.9927 | [0.990, 0.995] | 0.0338 | 0.0029 | −0.486 | 2 |
| 1I2T | 61 | 1.04 | 0.9944 | [0.993, 0.996] | 0.0359 | 0.0027 | −0.493 | 2 |
| 1R69 | 63 | 2.00 | 0.9936 | [0.991, 0.995] | 0.0322 | 0.0026 | −0.526 | 2 |
| 1TG0 | 66 | 0.97 | 0.9951 | [0.993, 0.996] | 0.0358 | 0.0025 | −0.519 | 2 |
| 1F0M | 71 | 2.20 | 0.9949 | [0.993, 0.996] | 0.0334 | 0.0024 | −0.558 | 2 |
| 3DKM | 79 | 1.60 | 0.9951 | [0.993, 0.996] | 0.0299 | 0.0021 | −0.492 | 2 |
| 1STN | 136 | 1.70 | 0.9935 | [0.991, 0.995] | 0.0193 | 0.0016 | −0.531 | 2 |
| 1MBO | 153 | 1.60 | 0.9909 | [0.988, 0.993] | 0.0127 | 0.0012 | −0.486 | 2 |
| 1PPN | 212 | 1.60 | 0.9851 | [0.980, 0.989] | 0.0096 | 0.0012 | −0.523 | 4 |
| 2PTN | 223 | 1.55 | 0.9887 | [0.985, 0.992] | 0.0096 | 0.0010 | −0.465 | 2 |
| 1BTL | 263 | 1.80 | 0.9920 | [0.989, 0.994] | 0.0082 | 0.0007 | −0.535 | 2 |
| 5CPA | 307 | 1.54 | 0.9912 | [0.988, 0.994] | 0.0074 | 0.0007 | −0.504 | 2 |
| 1AJ8 | 371 | 1.90 | 0.9923 | [0.990, 0.994] | 0.0060 | 0.0005 | −0.515 | 2 |
| 3GRS | 461 | 1.54 | 0.9930 | [0.991, 0.995] | 0.0048 | 0.0004 | −0.438 | 2 |

Full 64-row table with every interval:
`benchmarks/score_reliability_n100_k2-4-8-16_d5_v2.md`.

The decision rule was fixed before the run and it passes unambiguously: 16/16, no backbone
anywhere near the 0.90 lower-bound floor. The lowest ICC in the whole study, 1PPN at K=2
(0.9482, F lower bound 0.931), is the single point that fails, and it passes at K=4.

**The ≥32 advice is now dead on evidence rather than on argument.** It was calibrated on the
epistasis contrast, which is ~4× noisier and which the pipeline never computes; the quantity
that is actually ranked reaches ≥0.985 at K=8 on every backbone from a 53-residue domain to a
461-residue flavoenzyme.

## 3. K=8 is sufficient, not necessary — and why that is not a licence to lower it

Minimum qualifying K is **2 on 15 of 16 backbones**. On reliability grounds alone the default
could drop 4× and the oracle cost with it.

That conclusion should not be acted on from this study, for a reason this project keeps running
into: reliability is self-consistency, and the question the pipeline cares about is predictive
accuracy. The K-sweep found association with *experimental* epistasis flat in K on 3DKM and
3L1X but still climbing on 4G3O — so there is at least one measured case where extra orders
bought something reliability cannot see. "K=2 agrees with itself" is not "K=2 predicts as
well," and treating the first as evidence for the second is precisely the substitution of a
numerical improvement for a biological one that this work is supposed to avoid.

What the evidence supports now: **K=8 is comfortably sufficient and K<8 is no longer
unsupported**, which is a change from I-23(b). What would settle it: association-versus-K on
more than three domains, which is Study 2's territory, not Study 1's.

## 4. Noise scaling — the pre-registered form, confirmed

| | slope of log(noise SD) on log(K) |
|---|---|
| Pre-registered expectation | −0.5 (plain Monte-Carlo averaging) |
| Measured, 16 backbones | mean **−0.503**, range [−0.558, −0.438] |
| K-sweep, 3 domains × 2 curves, two draws | [−0.68, −0.44] |

Every backbone independently reproduces the half-power law, and the spread collapsed relative
to the K-sweep. That spread was draw noise, not protein-to-protein variation in how averaging
works. Two consequences: noise at an unmeasured K can be projected as
`noise_sd(K) = noise_sd(8)·(K/8)^−0.5`, and a future departure from −0.5 would be a real signal
that something other than order-averaging is in play.

Signal SD is flat in K on every backbone (e.g. 4G3O 0.0430/0.0431/0.0429/0.0430 at K =
2/4/8/16). Raising K removes noise without touching the signal, which is what the estimator
assumes and worth having confirmed rather than asserted.

## 5. What D = 5 bought — a correction to amendment A1's framing

The pre-registration said the two-draw design *understates* uncertainty. Measured on this data,
that is wrong as stated:

| over all 64 points | mean interval width |
|---|--:|
| D=5 ANOVA F interval | 0.0082 |
| Old two-draw variant bootstrap (draws 1–2 of the same panel) | 0.0115 |

The old intervals are **1.41× wider**. The failure is not narrowness but **miscalibration**:

- The old 95% interval **fails to cover the D=5 ICC in 7 of 64 points (11%)**, against a
  nominal 5%. An interval that is wider *and* misses more often is not conservative.
- The old reading of this very data still produces an **impossible reversal**: 1F0M at 0.9956
  (K=4) → 0.9951 (K=8). Reliability cannot fall as K rises in expectation. This is the same
  symptom that exposed the flaw on 3L1X, now reproduced on a different protein.
- The **D=5 series is monotone in K at all 64 points**. Zero reversals.

So A1's conclusion stands — the two-draw design cannot support fine K distinctions, and D ≥ 3
fixes it — but the mechanism recorded in the pre-registration needs amending: the two-draw
interval is wide and wrongly centred, not narrow.

The declared draw-sensitivity diagnostic passes cleanly. The leave-one-draw-out jackknife
spread exceeds the F interval's width in **0 of 64 points**, so the F interval's normality
assumption is not doing work the data fails to support anywhere in this study. The variant-row
bootstrap interval agrees with the F interval throughout, which is why only one of the two is
quoted above.

## 6. Signal range collapses with length — the one operational warning

Nobody asked this and it is the most consequential thing in the study.

| | L = 53 (4G3O) | L = 461 (3GRS) | ratio |
|---|--:|--:|--:|
| signal SD of `f(mut) − f(WT)` | 0.0429 | 0.0048 | 8.9× |
| noise SD at K=8 | 0.0031 | 0.0004 | 7.8× |
| ICC at K=8 | 0.9949 | 0.9930 | — |

`global_score` is a mean NLL over all residues, so a single substitution moves it by roughly
1/L. Signal and noise shrink in step, which is exactly why ICC is flat in length and why
reliability gives no hint that anything changes.

The consequence for the pipeline: **absolute `Δscore` is not comparable across proteins of
different size.** Any threshold, accept/reject cutoff or "meaningful difference" heuristic
calibrated on these 47–79 residue domains is roughly an order of magnitude too coarse for a
400-residue target. Ranking within one backbone is unaffected. This belongs in the mutation
pipeline's own calibration, and it is not something Study 1 was designed to address — it just
fell out of having the components reported separately.

## 7. Backbone descriptors — exploratory, and that label is load-bearing

Spearman of ICC at K=8 across the 16 backbones:

| descriptor | Pearson | Spearman |
|---|--:|--:|
| mean CA–CA contact number (8 Å) | −0.780 | −0.753 |
| mean SASA per residue | +0.750 | +0.797 |
| fraction buried (SASA < 20 Å²) | −0.734 | −0.774 |
| chain length | −0.451 | −0.638 |

Read naively: tightly packed proteins are less self-consistent at fixed K. Three reasons not to
read it that way yet. The relationship is **not monotone in length** — 3GRS at 461 residues
(0.9930) beats 1PPN at 212 (0.9851). The descriptors are mutually collinear, so these are not
four independent findings but roughly one. And 1PPN is both the lowest ICC and the highest
contact number, so a single backbone carries much of the correlation. n = 16.

Declared exploratory in the pre-registration and it stays that way: a hypothesis worth a
designed test, not a result. The buried-fraction figure additionally uses an absolute 20 Å²
threshold rather than relative accessibility, because the harness deliberately carries no
per-residue-type max-ASA reference table.

## 8. Runtime and cost

Wall clock **6.01 h** against a predicted 4.7 h — my estimate was 22% low. The throughput
*model* was not the problem: refitting on each backbone's least-contended point gives
`0.000424·L + 0.0040` s/seq/order against the design's `0.0004·L + 0.0040`, a 5% miss, which
projects 4.97 h on a quiet host. The extra hour is host contention.

Two of the 64 `s/seq/order` entries are unusable and should not be quoted: 3DKM at K=2 (0.402,
10.8× its own other points) and 1STN at K=4 (0.456, 7.7×). As in the K-sweep, wall-clock
measurement on a shared laptop is the fragile part. **The ICCs are unaffected** — reliability
is wall-clock-independent, which is what made it safe to run overnight with the host asleep.

Cost of the K=8 operating point, for planning: ≈0.0004·L + 0.004 s per sequence per order, so
scoring 100 variants at K=8 costs ~21 s on a 53-residue domain and ~163 s on a 461-residue one
(measured medians, not the model),
single-threaded on CPU (M3).

## 9. Limitations

- **Self-consistency only.** No experimental data is involved anywhere in Study 1. Every number
  here is the model agreeing with itself across re-draws of its own decoding orders.
- **One model checkpoint.** `v_48_020` only. Nothing here speaks to other ProteinMPNN weights.
- **Single mutants only.** The estimand is `f(mut) − f(WT)` for single substitutions. Doubles
  and the epistasis contrast are not covered; the contrast is ~4× noisier and needs K≈32 for
  comparable reliability, as the K-sweep measured.
- **X-ray, single-chain, standard residues.** Eligibility excludes NMR ensembles, gapped
  chains and nonstandard residues by construction. Nothing here transfers to a predicted or
  multi-chain backbone without re-measurement.
- **16 backbones is 16 backbones.** The rule passed with no near-misses, so the sufficiency
  claim is robust, but the descriptor question (§7) is badly underpowered at this n.
- **ICC(1,1) absorbs a systematically offset draw into its error term** rather than modelling a
  draw main effect, so it is biased *down* by such an effect — conservative for a retention
  rule. Detecting one needs a two-way model and more than five draws.

## Follow-ups

- Association-versus-K beyond three domains, which is the only thing that can decide whether
  K < 8 is safe (§3). Study 2 territory.
- `Δscore` calibration across protein size in the mutation pipeline (§6) — new, and not an
  ISSUES item yet.
- ISSUES I-23: both remaining actions are now answerable. K=8 is ratified on 16 backbones, the
  ≥32 wording is dead, and the "K<8 unsupported pending ≥3 draws" caveat is discharged.
- `epistasis_stats.meta_analyze` and D-draw support in `benchmark_score_only_epistasis.py`
  remain unwritten; both are Study 2 only and neither blocked this study.

## Artifacts and provenance

- `benchmarks/benchmark_score_reliability.py` — the harness. `--dry-run` prints the plan and
  cost without scoring; `--resume` reuses completed (backbone, K) points and refuses to mix
  parameters; artifacts are parameter-stamped and overwrite-guarded, with D and the seed list
  in the stamp so a D=5 run can neither reuse nor overwrite a D=2 one. Written after every
  backbone, so an interrupted run keeps everything finished so far.
- `benchmarks/score_reliability_n100_k2-4-8-16_d5_v2.{json,md}` — config, per-backbone
  provenance (PDB SHA-256, resolution, the exact 100 mutant labels, descriptors), all 64 points
  with every interval, the noise-scaling fits, the decision rule and the exploratory
  correlations.
- `benchmarks/epistasis_stats.py::icc_reliability` — one-way random-effects ICC(1,1) with the F
  interval, variant-row bootstrap and leave-one-draw-out jackknife. `es.reliability` is
  untouched, so the committed two-draw artifacts keep reproducing their published numbers; at
  D = 2 the two estimators agree to 8e-05.
- `benchmarks/screen_backbones.py` — the eligibility screen, and the provenance for the
  pre-registered backbone list. 16/16 proposed backbones eligible; 12/45 Tsuboyama domains
  eligible at ≤ 2.7 Å.
- `tests/test_score_reliability.py` (16 tests), `tests/test_epistasis_stats.py` (36 tests,
  9 of them for `icc_reliability`). Both run without weights, GPU or the ProteinGym parquet.
- Model: ProteinMPNN `v_48_020`, clone pinned at `8907e6671bfbfc92303b5f79c4b5e6ce47cdef57`,
  CPU (M3), `augment_eps = 0`. Structures from RCSB, cached under
  `benchmarks/epistasis_structures/`.
