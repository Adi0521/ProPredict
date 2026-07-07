# Mutation Master Plan — Task 1: `orchestrator/mutation_scan.py`

**Date:** 2026-07-06
**Source:** `mutation-plans/Process-plan-master.md` Task 1 (from mutation-scan doc)
**Depends on:** Task 0 (`PROTEINMPNN_PATH` / `PROTEINMPNN_MODEL_NAME` config).

## What was done

Added `orchestrator/mutation_scan.py` — a standalone, structure-aware single-point
mutation scorer using ProteinMPNN's `--conditional_probs_only` mode (Dauparas et al.,
2022; MIT). Score is the log-likelihood-ratio

    score(pos, wt->mut) = log P(mut | backbone, rest of seq) - log P(wt | ...)

a **structural compatibility** score (not fitness/stability), per the HERMES method.

### Public API
- `score_candidate_mutations(pdb_string, sequence, positions=None, top_k=10,
  proteinmpnn_dir="", model_name="v_48_020") -> List[{position, from_aa, to_aa, score}]`
  sorted descending.
- `_run_proteinmpnn_conditional_probs(...)` — shells out to `protein_mpnn_run.py`,
  reads `conditional_probs_only/*.npz` `log_p` (shape `[1, L, 21]`), slices batch 0.
- `__main__` CLI: `python -m orchestrator.mutation_scan --pdb X.pdb --sequence MKT...`
  (reads `PROTEINMPNN_PATH` from env).

### Design notes
- **Standalone by design** — takes `proteinmpnn_dir`/`model_name` as params, does not
  import `config` or touch the agent loop. Keeps Task 3 (`apply_mutation`) independent.
- Git-clone tool, not pip-installable — same pattern as `insane.py` in `membrane.py`.
- **One addition over the plan's module text:** the position-skip branch now emits a
  `logger.warning` (out-of-range position usually means a stale sequence/structure pair
  upstream) instead of silently `continue`-ing — flagged in the plan's own test note.
- Early `RuntimeError` on: empty `proteinmpnn_dir`, missing `protein_mpnn_run.py`,
  subprocess non-zero exit (with truncated stdout/stderr), missing `.npz` output.

## Tests — `tests/test_mutation_scan.py` (new, fully mocked)
Unit cases patch `_run_proteinmpnn_conditional_probs` with a hand-checked `[2,21]`
`log_p` for sequence "AC" (A1D=+2.0, A1E=+1.0, C2G=+2.5): formula/sort, `top_k`
truncation, `positions` filter (19 = 20 std AA − wt), out-of-range skip + warning,
missing-dir guard. Subprocess/filesystem error paths test `_run_...` directly with a
stub script + patched `subprocess.run`. One inline-skip integration test gated on
`PROTEINMPNN_PATH` + `MPNN_TEST_PDB` + `MPNN_TEST_SEQ` (drift-catcher).

## Verification
`pytest tests/test_mutation_scan.py` in the `ProPredict` conda env: **8 passed,
1 skipped** (integration).

## Next
Task 2 — validation gate: sanity-check the scorer against 2-3 ProteinGym DMS assays on
this pipeline's *predicted* structures (Spearman correlation), write up pass/fail in
`Process/` before any `apply_mutation` work.
