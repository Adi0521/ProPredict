# Determinism gate — end-to-end check that the ProteinMPNN scorer is actually reproducible

**Date:** 2026-10-02
**Status:** harness written, **run green on CPU**, committed in `bbef345`.
**Source:** the open follow-up in `Process/mutation-determinism-fix.md`.
**Reproduce:**
```bash
python -m benchmarks.check_determinism
python -m benchmarks.check_determinism --pdb path/to/single_chain.pdb --num-decoding-orders 16
```

## Why a harness when the fix is already tested

`Process/mutation-determinism-fix.md` (2026-07-16) built determinism *in*: `--seed 0` is
rejected, the CLI passes an explicit non-zero seed, and `log_p` is the mean over
`num_decoding_orders` random decoding orders. That is the right fix and it has unit tests.

What those tests cannot see is whether the guarantee survives **end-to-end against the real
binary** — GPU/torch nondeterminism, filesystem ordering, a seed silently unset somewhere in
the subprocess path. The mocked tests assert the *command* is built correctly; they do not
assert that two real runs produce the same array. This harness closes that gap, and adds a
measurement the tests never made.

Three checks, cheapest first:

| # | check | kind | needs weights |
|---|---|---|---|
| 0 | `seed=0` rejected before any subprocess | **gate** | no |
| 1 | two same-seed runs byte-identical | **gate** | yes |
| 2 | cross-seed per-residue spread | *report, not a gate* | yes |

Check 0 running without weights is the useful design touch: if the seed guard is broken, you
learn it instantly rather than after minutes of inference.

## Result (run 2026-10-02, CPU, M3)

```
[ok]   seed=0 rejected (would randomize decoding order)

=== same-seed determinism (seed=37, N=8) ===
  log_p shapes: (53, 21) vs (53, 21)   max|Δ| = 0.000e+00
  --> PASS (byte-identical)

=== cross-seed spread over seeds [37, 101, 271] ===
  mean per-residue std = 1.150e-02   max = 3.163e-02
```

**Gate 1 passes exactly, not approximately** — `max|Δ| = 0.000e+00` across the full `[53, 21]`
matrix. The determinism contract from the July fix holds against the real binary on this
platform. By the code path (`sys.exit(0 if (identical and ok_guard) else 1)`) both gates
passing exits 0, so this is CI-gateable as written.

**Check 2 is the number worth keeping.** A mean per-residue cross-seed std of **1.15e-02** at
`N=8` is the residual decoding-order noise that the mean-over-N is controlling but has not
eliminated. Determinism at a fixed seed does **not** mean the score is seed-independent — it
means it is reproducible. Those are different properties and this harness is the first thing
in the repo to report both.

**This corroborates Row B independently.** `research_plan/rowB-score-only-epistasis.md`
measured decoding-order noise a different way (epistasis noise std across two decoding seeds)
and found 8→32 orders cuts noise 0.016→0.008 and lifts per-pair SNR 1.7→3.0, recommending
`PROTEINMPNN_NUM_DECODING_ORDERS ≥ 32`. Two unrelated measurements agree that `N=8` leaves
real noise on the table. The recommendation stands and now has a second line of evidence.

## Scope and overlap, stated honestly

- **Gate 1 partly duplicates an existing test.** `mutation-determinism-fix.md` records that
  `tests/test_mutation_scan.py` already has an env-gated real-binary test asserting two
  same-seed runs are identical. The harness's added value is checks 0 and 2, the CI exit code,
  and being runnable on an arbitrary `--pdb` — not gate 1 on its own.
- **One platform, one structure, CPU only.** The green result is M3/CPU. The failure modes
  most worth fearing (torch nondeterminism) are GPU-side and remain unchecked; running this on
  the Modal GPU image would be the real test.
- **Check 2 uses only 3 seeds**, so the spread is a rough estimate, not a converged one.

## Defects found

1. **The docstring names the wrong default structure.** The module NOTE says *"The default
   structure below is one chain of HIV-PR"*, but `DEFAULT_PDB` points at
   `benchmarks/epistasis_structures/4G3O.raw.pdb` — the **AMFR_HUMAN domain** cached for the
   Row B epistasis study, not HIV protease. Verified: that file is a single chain A of 53
   residues, which matches the `L=53` in the run above. The file is a valid single-chain input
   so the harness works correctly; only the comment is wrong. (Note it is also the *raw* 53-residue
   structure, not the 47-residue DMS-trimmed slice Row B scores, so the two are not directly
   comparable.)
2. **A committed change undoes this fix's own follow-up.** `mutation-determinism-fix.md`
   closes with: *"Re-run task-2 and task-7 benchmarks with the seeded + averaged scorer
   (`--seeds 37,38,39`)"*. `999ef3e` (2026-07-17) set
   `benchmarks/proteinmpnn_checkpoint_results.json` to `seeds: [37, 38, 39]` — that re-run was
   done. **`bbef345` replaced it** with
   `seeds: [37]` and `spearman_std` collapsed from `0.0003` to `0.0`: a single-seed run
   overwrote the three-seed result and re-introduced exactly the flaw the July writeup called
   out — a checkpoint margin (0.002) smaller than the seed noise (~0.008), with no way to
   separate the two. It looks accidental. The three-seed data is recoverable from git history.

   This is the same failure mode as `benchmarks/score_only_epistasis.jsonl` in the same
   commit (see `Process/rowB-score-only-epistasis.md`): a results artifact overwritten by a
   lower-effort re-run at default flags. Both harnesses write unconditionally to a fixed
   filename, so the fix is shared — parameter-stamp the output name, or refuse to overwrite
   without `--force`.

## Follow-ups

- Run it inside the Modal GPU image; CPU-green does not cover torch GPU nondeterminism.
- Restore the three-seed `proteinmpnn_checkpoint_results.json` from history (defect 2), and
  parameter-stamp or write-protect results artifacts so this cannot recur.
- Fix the HIV-PR docstring NOTE (defect 1), or point `DEFAULT_PDB` at an actual HIV-PR chain
  so the comment becomes true — the affinity-invariance system is HIV-PR, so that may be the
  more useful default.
- Consider wiring gate 0 + gate 1 into CI; gate 0 alone is free (no weights) and catches the
  highest-severity regression.

## Artifacts

- `benchmarks/check_determinism.py` — the harness (seed=0 preflight, same-seed byte gate,
  cross-seed spread report, CI exit code).
- `Process/mutation-determinism-fix.md` — the July fix this validates.
- `research_plan/rowB-score-only-epistasis.md` — the independent decoding-order noise
  measurement that agrees with check 2.
