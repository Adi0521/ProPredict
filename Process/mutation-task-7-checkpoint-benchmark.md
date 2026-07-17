# Task 7 (open question #1): ProteinMPNN checkpoint benchmark

**Date:** 2026-07-15 (original run) · **updated 2026-07-16** with the seeded + averaged
scorer after the determinism fix (see `Process/mutation-determinism-fix.md`).
**Source:** Master-plan "Still-open questions" #1 — pick the ProteinMPNN `model_name`
checkpoint empirically (`v_48_020` default vs higher-noise `v_48_030`, since this
pipeline always re-folds afterward, which in theory could favor a higher-noise model).
**Decision: keep `v_48_020` (no code change).**

> **Note on the 2026-07-16 update.** The original run scored each checkpoint **once with a
> random decoding order** — the `--seed 0` non-determinism bug (ProteinMPNN treats seed 0
> as unset). Its per-cell figures were not reproducible and it could not separate a real
> checkpoint effect from decoding-order noise. This write-up now reports the re-run with
> the fixed scorer: **fixed non-zero seed, scores averaged over 8 decoding orders, and each
> checkpoint repeated across 3 seeds (37/38/39) reported as mean ± std.** The decision is
> unchanged; the *reasoning* is corrected (see Interpretation). Original single-seed numbers
> are preserved at the bottom for the record.

## Method
`benchmarks/benchmark_proteinmpnn_checkpoints.py` — reruns the Task-2 ProteinGym
validation methodology across all four vanilla checkpoints on the same three assays.
Per assay: fold `target_seq` **once** via ESMFold-local (structure is checkpoint- and
seed-independent), then run `--conditional_probs_only` per checkpoint, score every DMS
mutant additively (`sum of log_p[to] - log_p[from]` over `:`-split subs), Spearman vs
`DMS_score`. Each cell is repeated across **seeds 37/38/39** (`--num_seq_per_target 8`, so
each score is itself a mean over 8 decoding orders) and reported as mean ± std over the 3
seeds. Data: `s3://proteingym/DMS_substitutions.parquet` (no-sign-request). 0 mutants
skipped in any cell (clean sequence/structure alignment throughout).

## Results — Spearman ρ vs DMS_score (mean ± std over seeds 37/38/39)

| Assay (category, L) | v_48_002 | v_48_010 | v_48_020 | v_48_030 | n |
|---|---|---|---|---|---|
| TCRG1_MOUSE (Stability, 37) | **+0.7887 ±0.0003** | +0.7052 ±0.0017 | +0.7559 ±0.0006 | +0.7240 ±0.0028 | 1058 |
| ESTA_BACSU (Stability, 212) | +0.4070 ±0.0006 | +0.4238 ±0.0005 | +0.4340 ±0.0006 | **+0.4375 ±0.0011** | 2172 |
| CCDB_ECOLI (Activity, 101) | +0.3162 ±0.0007 | **+0.3347 ±0.0011** | +0.3273 ±0.0006 | +0.2883 ±0.0006 | 1176 |
| **MEAN** | +0.5040 | +0.4879 | **+0.5057** | +0.4833 | — |

Predicted-structure quality was solid (mean pLDDT 83.1 / 91.3 / 89.0), so the comparison
is on real structures this pipeline would actually produce, not crystal structures.

## Interpretation
- **Seed noise is now tiny and the checkpoint differences are real — not "within noise."**
  Per-cell std is 0.0003–0.0028, an order of magnitude below the per-assay gaps between
  checkpoints (0.01–0.03). So the checkpoints are genuinely distinguishable *per assay*;
  the original write-up's "all within noise" framing was an artifact of scoring each cell
  once with a random decoding order. What's actually going on is **assay-dependence**, not
  noise.
- **The best checkpoint depends on the assay — no checkpoint dominates.** v_48_002 wins the
  37-mer TCRG1 (+0.789) by a wide 0.03 margin; v_48_030 wins the 212-aa ESTA; v_48_010 wins
  CCDB. The low-noise checkpoint is best on the tiny stability case, the higher-noise ones
  on the larger proteins.
- **On the 3-assay MEAN, v_48_020 (+0.5057) and v_48_002 (+0.5040) are a tie.** The 0.0017
  gap clears seed noise (std of the mean ≈ 0.0006) but is *not* robust to which assays were
  sampled: with per-assay ρ ranging 0.29→0.79, three assays cannot reliably rank the top
  two. v_48_010 (+0.4879) and v_48_030 (+0.4833) sit ~0.02 below the mean and are more
  clearly behind — but still on only three assays.
- **The "higher noise helps because we re-fold" hypothesis is not supported in general.**
  It holds on exactly one assay (ESTA is monotonic in noise, v_48_030 best) and is
  *reversed* on the other two (TCRG1: lowest-noise v_48_002 best; CCDB: v_48_030 worst).
- **The Activity assay (CCDB) is the expected blind spot** — ρ ≈ 0.29–0.33 across all
  checkpoints, well below the two Stability assays (0.40–0.79). This is the known limit of
  a purely structural-compatibility signal, consistent with Task 2, not a regression.

## Decision & rationale
Keep **`v_48_020`** (already the default in `config.py` / `.env.example`; ProteinMPNN's
own default). It is tied for the best mean ρ (+0.5057), it's the middle-noise generalist
(never worst, and never badly beaten, on any assay), and no checkpoint dominates the panel.
The honest justification is **not** "checkpoints are indistinguishable within noise" — the
per-assay effects are real and assay-dependent — but "on this small panel the top two are
statistically tied and no single checkpoint is best everywhere, so the incumbent default
stands." No code change required.

## Reproduce
```bash
conda activate ProPredict
# Rigorous (seeded, mean±std): the run this write-up reports
python -m benchmarks.benchmark_proteinmpnn_checkpoints --seeds 37,38,39
# Quick single-seed reproducible run (prints a NOTE that it can't separate checkpoint
# effect from decoding-order noise)
python -m benchmarks.benchmark_proteinmpnn_checkpoints
python -m benchmarks.benchmark_proteinmpnn_checkpoints --checkpoints v_48_020,v_48_030
```
Full numbers (incl. per-seed ρ, seeds, num_decoding_orders) saved to
`benchmarks/proteinmpnn_checkpoint_results.json`. Parquet cached in
`benchmarks/proteingym_cache/` (gitignored).

## Notes / follow-ups
- This is a *mutation-scoring* benchmark (Spearman vs DMS), a different axis from
  `benchmark_modal.py` (Boltz-2 structure prediction, TM-score/RMSD). Kept in `Process/`
  + a results JSON rather than `BENCHMARKS.md`, which is scoped to structure-prediction runs.
- Open question #1 is now resolved. Remaining plan items: combinatorial/multi-site
  mutation search (Phase 2, out of scope) and open question #2's docker-compose `.env`
  precedence caveat (documented in Task 6).

---

### Original single-seed run (2026-07-15, superseded — unseeded scorer)
Kept for the record. These cells were each scored once with a *random* decoding order
(the `--seed 0` bug), so they are not reproducible and the mean margins below were within
the un-measured seed noise:

| Assay | v_48_002 | v_48_010 | v_48_020 | v_48_030 |
|---|---|---|---|---|
| TCRG1_MOUSE | +0.7891 | +0.7056 | +0.7538 | +0.7199 |
| ESTA_BACSU | +0.4057 | +0.4228 | +0.4318 | +0.4414 |
| CCDB_ECOLI | +0.3164 | +0.3332 | +0.3322 | +0.2871 |
| **MEAN** | +0.5037 | +0.4872 | +0.5059 | +0.4828 |

Compare v_48_020 on TCRG1: +0.7538 here vs +0.7559 ±0.0006 seeded — within the old
single-sample noise, exactly as the determinism plan predicted.
