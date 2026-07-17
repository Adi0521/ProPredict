# Determinism fix: ProteinMPNN scorer non-reproducibility

**Date:** 2026-07-16
**Source:** `mutation-plans/Process-plan-determinism-fix.md` (repo review against `b7bbd52`).
Bug re-verified against the live tree before implementing.
**Status: fixed.** Cheap, high-value fix done before any further mutation-scoring work.

## The bug

`orchestrator/mutation_scan.py::_run_proteinmpnn_conditional_probs` passed `--seed 0` to
ProteinMPNN. ProteinMPNN's `protein_mpnn_run.py:24` does:

```python
if args.seed:                       # 0 is falsy!
    seed = args.seed
else:
    seed = int(np.random.randint(0, high=999, size=1)[0])
torch.manual_seed(seed)
```

So `--seed 0` ≡ **no seed** → a random seed every run. `conditional_probs_only` is
stochastic (`protein_mpnn_run.py:291` draws `randn_1 = torch.randn(...)`, the random
decoding order), so each run produced a different `log_p` and different mutation scores.
The plan's sandbox proof: three `--seed 0` runs on `6MRR` gave max |Δ| = 0.447 on a single
mutation and a **different #1-ranked candidate every run** (R10V → H9Y → W2M).

This also explained an already-visible discrepancy in the repo: TCRG1 ρ = **+0.7461**
(task-2 write-up) vs **+0.7538** (task-7 write-up) — same assay/checkpoint/n/structure,
should have been bit-identical, differed only because the scorer wasn't reproducible.

### Verified against the live code before fixing
- `protein_mpnn_run.py:24` `if args.seed:` — confirmed (0 falls through to random).
- `protein_mpnn_run.py:290-294` loops `for j in range(NUM_BATCHES)`, fresh `randn_1` each,
  `concat_log_p` shape `[B, L, 21]` — confirms `--num_seq_per_target N` yields N
  decoding-order samples that can be averaged.
- Old `mutation_scan.py` passed `--seed 0`, `--num_seq_per_target 1`, returned `log_p[0]`.

## What was done

1. **`orchestrator/mutation_scan.py`**
   - `_run_proteinmpnn_conditional_probs` gains `seed=37` + `num_decoding_orders=8` params.
   - **Guard:** `seed == 0` raises `ValueError` (can't silently regress to the footgun).
   - CLI now passes `--seed <seed>` and `--num_seq_per_target <num_decoding_orders>`.
   - Returns `data["log_p"].mean(axis=0)` (averaged over decoding orders) instead of
     `log_p[0]`. Plan's measured spread: 8 samples cut the standard error ~sqrt(8) ≈ 2.8×
     for 8× the (seconds-on-CPU) compute.
   - Both params threaded through `score_candidate_mutations`; module docstring documents
     the reproducibility contract: identical only for a fixed
     `(seed, num_decoding_orders, model_name, structure)` tuple.

2. **`config.py` + `.env.example`** — `PROTEINMPNN_SEED=37`,
   `PROTEINMPNN_NUM_DECODING_ORDERS=8`, with the non-zero-seed warning inline.

3. **Call sites threaded**
   - `orchestrator/agent.py` (`scan_mutations` tool) passes both from config.
   - `benchmarks/benchmark_proteinmpnn_checkpoints.py` passes both, and additionally gained
     a `--seeds`/`--num-decoding-orders` interface: each checkpoint is now scored once per
     seed and reported as **mean ± std**. Single-seed runs print a NOTE that they can't
     separate checkpoint effect from decoding-order noise (see consequences below).

4. **Tests — `tests/test_mutation_scan.py`**
   - `seed=0` → `ValueError` (regression guard, fails before any FS work).
   - cmd construction: `--seed 37` / `--num_seq_per_target 3` asserted via mocked subprocess.
   - averaging: mocked `[3, L, 21]` npz → asserts return equals `mean(axis=0)`, not `[0]`.
   - integration test (env-gated, real binary) now asserts two same-seed runs are identical
     — the test that would actually have caught this.
   - `tests/test_agent.py::test_scan_mutations_happy_path` updated to expect the new kwargs.

**Test result:** `test_mutation_scan.py` 11 passed / 1 skipped (real-binary gate),
`test_agent.py` 18 passed, `test_boltz.py` 11 passed. Benchmark `--help` + config import OK.

## Consequences for prior work (honest, not quietly re-run)

- **task-2 validation gate** — verdict (PASS) stands: the effect (ρ ≈ 0.75/0.43/0.33) dwarfs
  seed noise (~0.008 on ρ). Exact figures aren't reproducible; should be re-run and the
  table annotated as originally-unseeded. *Not yet re-run.*
- **task-7 checkpoint benchmark** — decision (keep `v_48_020`) still right, but the write-up
  implied a measured preference (+0.5059 "best mean") the data can't support: each checkpoint
  was scored once with a random decoding order, and the 0.002 winning margin is smaller than
  the ~0.008 seed noise. Correct re-run: `--seeds 37,38,39`, report mean ± std per cell.
  *Not yet re-run* — the aborted run that prompted this fix should be redone this way.

## Follow-ups
- Re-run task-2 and task-7 benchmarks with the seeded + averaged scorer (`--seeds 37,38,39`)
  and update both write-ups.
- Note: no fold caching — each benchmark run re-folds all assays via ESMFold-local
  (~10 min/assay on M3). A sequence-keyed PDB cache would make iteration cheap.
