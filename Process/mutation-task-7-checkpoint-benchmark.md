# Task 7 (open question #1): ProteinMPNN checkpoint benchmark

**Date:** 2026-07-15
**Source:** Master-plan "Still-open questions" #1 — pick the ProteinMPNN `model_name`
checkpoint empirically (`v_48_020` default vs higher-noise `v_48_030`, since this
pipeline always re-folds afterward, which in theory could favor a higher-noise model).
**Decision: keep `v_48_020` (no code change).**

## Method
`benchmarks/benchmark_proteinmpnn_checkpoints.py` — reruns the Task-2 ProteinGym
validation methodology across all four vanilla checkpoints on the same three assays.
Per assay: fold `target_seq` **once** via ESMFold-local (structure is
checkpoint-independent), then run `--conditional_probs_only` per checkpoint, score every
DMS mutant additively (`sum of log_p[to] - log_p[from]` over `:`-split subs), Spearman
vs `DMS_score`. Data: `s3://proteingym/DMS_substitutions.parquet` (no-sign-request).
0 mutants skipped in any cell (clean sequence/structure alignment throughout).

## Results — Spearman ρ vs DMS_score (ESMFold-local predicted structures)

| Assay (category, L) | v_48_002 | v_48_010 | v_48_020 | v_48_030 | n |
|---|---|---|---|---|---|
| TCRG1_MOUSE (Stability, 37) | **+0.7891** | +0.7056 | +0.7538 | +0.7199 | 1058 |
| ESTA_BACSU (Stability, 212) | +0.4057 | +0.4228 | +0.4318 | **+0.4414** | 2172 |
| CCDB_ECOLI (Activity, 101) | +0.3164 | **+0.3332** | +0.3322 | +0.2871 | 1176 |
| **MEAN** | +0.5037 | +0.4872 | **+0.5059** | +0.4828 | — |

Predicted-structure quality was solid (mean pLDDT 83.1 / 91.3 / 89.0), so the comparison
is on real structures this pipeline would actually produce, not crystal structures.

## Interpretation
- **The choice is within noise.** All four checkpoints' means fall in a 0.023-wide band
  (0.483–0.506). No checkpoint wins more than one assay: v_48_002 wins TCRG1, v_48_030
  wins ESTA, v_48_010 wins CCDB. There is no dominant checkpoint.
- **The "higher noise helps because we re-fold" hypothesis is not supported in general.**
  It holds on exactly one assay (ESTA is monotonic in noise, v_48_030 best) and is
  *reversed* on the other two (TCRG1: lowest-noise v_48_002 best; CCDB: v_48_030 worst).
  So higher noise is not a free win here.
- **The Activity assay (CCDB) is the expected blind spot** — ρ ≈ 0.29–0.33 across all
  checkpoints, well below the two Stability assays (0.40–0.79). This is the known limit
  of a purely structural-compatibility signal, consistent with Task 2, not a regression.

## Decision & rationale
Keep **`v_48_020`** (already the default in `config.py` / `.env.example`; ProteinMPNN's
own default). It has the best mean ρ (+0.5059), it's the middle-noise generalist (never
worst on any assay), and the gaps to the others are within single-predicted-structure
noise. Switching checkpoints would not be a defensible improvement on this evidence.
No code change required — the benchmark **confirms** the existing default.

## Reproduce
```bash
conda activate ProPredict
python -m benchmarks.benchmark_proteinmpnn_checkpoints                       # all 4, all 3 assays
python -m benchmarks.benchmark_proteinmpnn_checkpoints --checkpoints v_48_020,v_48_030
```
Full numbers saved to `benchmarks/proteinmpnn_checkpoint_results.json`. Parquet cached in
`benchmarks/proteingym_cache/` (gitignored).

## Notes / follow-ups
- This is a *mutation-scoring* benchmark (Spearman vs DMS), a different axis from
  `benchmark_modal.py` (Boltz-2 structure prediction, TM-score/RMSD). Kept in `Process/`
  + a results JSON rather than `BENCHMARKS.md`, which is scoped to structure-prediction runs.
- Open question #1 is now resolved. Remaining plan items: combinatorial/multi-site
  mutation search (Phase 2, out of scope) and open question #2's docker-compose `.env`
  precedence caveat (documented in Task 6).
