# Mutation Master Plan — Task 2: Validation gate (ProteinGym)

**Date:** 2026-07-11
**Source:** `mutation-plans/Process-plan-master.md` Task 2 (hard gate before Task 3)
**Depends on:** Task 1 (`orchestrator/mutation_scan.py`), Task 0 (`PROTEINMPNN_PATH`).
**Verdict: PASS** — proceed to Task 3.

## Purpose

Before the ProteinMPNN scorer is referenced anywhere downstream, confirm it produces a
*sane* signal on **this pipeline's own predicted structures** (not crystal structures,
which ProteinMPNN was more likely validated on) by correlating its log-odds against real
deep-mutational-scanning measurements from ProteinGym (Notin et al., 2023/2024).

## Setup

- **Deps added to `ProPredict` conda env:** `pandas`, `scipy`, `pyarrow`.
- **`PROTEINMPNN_PATH`** set in `.env` → `/Users/adi-kewalram/ProteinMPNN`
  (clone confirmed: `vanilla_model_weights/v_48_020.pt` present).
- **ProteinGym data** (fetched to scratchpad, not committed — parquet is 88 MB):
  - reference: `reference_files/DMS_substitutions.csv` (217 rows) via raw GitHub.
  - measurements: `s3://proteingym/DMS_substitutions.parquet`
    (`aws s3 cp --no-sign-request`, no credentials). 2.47M rows, columns
    `DMS_score / mutant / target_seq / DMS_id / ...`. Filtered to the three assays.
- **Structures:** re-predicted each assay's `target_seq` with the pipeline's own
  `call_esmfold_local` (`facebook/esmfold_v1`, weights already cached). Residue
  numbering folds 1:1 with `target_seq`, so DMS `mutant` positions align directly —
  confirmed by `log_p.shape[0]` matching `seq_len` exactly for all three (0 rows skipped).

## Method

One `_run_proteinmpnn_conditional_probs` call per assay → `log_p` `[L, 21]`. Each DMS
`mutant` scored as the **sum of per-substitution log-odds**
`log_p[idx, mut] - log_p[idx, wt]` (additive over `:`-separated substitutions — standard
for site-independent models; only the 37-mer had multi-substitution rows, 437 of 1058).
Each substitution's wild-type letter was verified against the folded sequence before
scoring. Spearman correlation vs `DMS_score` (`scipy.stats.spearmanr`).

Script: `scratchpad/proteingym/validate_proteingym.py` (throwaway — not repo code; it
imports `orchestrator.mutation_scan` and `orchestrator.backends.esmfold`).

## Results

| DMS_id | Category | Len | n scored | mean pLDDT | Spearman ρ | p-value |
|---|---|---|---|---|---|---|
| `TCRG1_MOUSE_Tsuboyama_2023_1E0L` | Stability | 37 | 1058 | 83.1 | **+0.7461** | 9.6e-189 |
| `ESTA_BACSU_Nutschel_2020` | Stability | 212 | 2172 | 91.3 | **+0.4282** | 1.5e-97 |
| `CCDB_ECOLI_Adkar_2012` | Activity | 101 | 1176 | 89.0 | **+0.3305** | 2.3e-31 |

All positive, all highly significant, 0 rows skipped in any assay.

> **Annotation (2026-07-16): figures above are from the unseeded scorer.** These ρ were
> produced before the ProteinMPNN `--seed 0` determinism fix (`Process/mutation-determinism-fix.md`),
> i.e. a single random decoding order per assay — not reproducible. The **verdict (gate
> PASS) is unaffected**: the effect dwarfs the seed noise. For the record, the seeded +
> averaged scorer (seed 37, 8 decoding orders; from the Task-7 re-run on the same
> `v_48_020` default) gives:
> `TCRG1 +0.7559 ±0.0006` · `ESTA +0.4340 ±0.0006` · `CCDB +0.3273 ±0.0006` — each within
> ~0.01 of the values above, so every conclusion in this write-up stands. The exact table
> here was not re-generated in place; treat the seeded values as the reproducible ones.

## Interpretation

- **Gate cleared.** The plan's bar was "not near-zero / not negative on our own
  predicted structures." All three clear it comfortably.
- **Expected category ordering held:** the two Stability assays outrank the Activity
  assay. ProteinMPNN scores *structural compatibility*, which tracks fold stability more
  directly than catalytic function — so `CCDB` (activity) landing lowest is the
  informative-but-expected result the plan called out, not a failure. Worth carrying into
  any eventual paper as the scorer's known blind spot.
- **Sanity vs literature:** ProteinMPNN's published zero-shot ProteinGym average Spearman
  is ~0.29–0.31; these three sit in-range (CCDB) to well above (the small stability
  assays), consistent with structure-based scorers doing best on stability and small
  proteins.
- **Caveat on the 37-mer:** ρ=0.75 is unusually high and typical for very small
  single-domain stability assays — not evidence the scorer is this strong in general.
  The 212-residue `ESTA` (ρ=0.43) is the more representative real-protein data point.

## Notes / repro

- The validation process exits `-1` at teardown (spurious torch/MPS interpreter-shutdown
  signal) *after* printing the full summary and writing `validation_results.json` — the
  run itself succeeded. Not a scoring error.
- `AGENT_MODEL` in `.env` is still `deepseek-v4-flash` (unchanged, per Task 0 decision);
  irrelevant here — Task 2 touches no agent code.

## Next

Task 3 — `apply_mutation` agent tool in `orchestrator/agent.py`. Still standalone per the
plan (does **not** call `score_candidate_mutations` internally yet); the system-prompt
addition may now truthfully state that a *validated* structural-compatibility scorer
exists in the codebase.
