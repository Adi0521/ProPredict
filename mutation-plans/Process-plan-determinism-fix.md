# Plan — Fix ProteinMPNN scorer non-determinism (found during repo review)

**Status: real bug, empirically proven, cheap fix. Do this before any further
mutation-scoring work or benchmarking.**

**Repo state reviewed:** `b7bbd52` ("Benchmarked ProteinMPNN"), pulled fresh. Tasks 0–7
of `Process-plan-master.md` all landed. Test suite: **124 passed, 7 skipped, 0 real
failures** (3 boltz + 1 api error were missing `biopython`/`starlette.testclient` in the
reviewer's sandbox only, not repo problems).

---

## The bug

`orchestrator/mutation_scan.py::_run_proteinmpnn_conditional_probs` passes
`"--seed", "0"` (line 70). ProteinMPNN's `protein_mpnn_run.py` handles the seed like
this:

```python
if args.seed:
    seed=args.seed
else:
    seed=int(np.random.randint(0, high=999, size=1, dtype=int)[0])
torch.manual_seed(seed)
```

**`0` is falsy in Python.** So `--seed 0` takes the `else` branch and picks a *random*
seed every run — passing `--seed 0` is exactly equivalent to passing no seed at all.
This is a latent footgun in ProteinMPNN upstream, not in our code, but we hit it.

This matters because `conditional_probs_only` is stochastic:
```python
randn_1 = torch.randn(chain_M.shape, device=X.device)
log_conditional_probs = model.conditional_probs(X, S, ..., randn_1, ...)
```
`randn_1` is the **random decoding order**. Different seed → different decoding order →
different `log_p` → different mutation scores.

### Empirical proof (run in sandbox, real ProteinMPNN + real PDB `6MRR`)

Three identical invocations with `--seed 0`:
```
Run1 vs Run2 identical?  False
Run1 vs Run3 identical?  False
Max abs diff run1-run2:  0.447
  run1: E5L score = -0.8818
  run2: E5L score = -1.0566
  run3: E5L score = -0.8685
```

Two identical invocations with `--seed 37`:
```
--seed 37, two runs identical? True | max diff: 0.0
```

### Why it matters in practice

Top-10 candidate lists from `score_candidate_mutations` across three identical runs:
```
run1 top-10: ['R10V','W2M','H9Y','V36I','S3D','W2I','T4E','W2V','W2S','Q54K']
run2 top-10: ['H9Y','W2M','R10V','V36I','S3D','W2I','T4E','W2V','W2S','D30E']
run3 top-10: ['W2M','H9Y','R10V','V36I','W2I','S3D','W2V','T4E','W2S','Q54E']

run1 vs run2 overlap: 9/10    identical ordering? False
```
Set membership is fairly stable (9/10) — but **the #1 ranked candidate changes on every
run** (R10V → H9Y → W2M) and the 10th slot swaps entirely. Since `scan_mutations` exists
precisely to hand the agent a ranked shortlist, a non-reproducible #1 pick is a real
problem: the same protein scanned twice gives the agent different advice, and nothing in
a benchmark or a paper built on this would reproduce.

### It also explains a discrepancy already visible in the repo

`Process/mutation-task-2-validation-gate.md` reports TCRG1 ρ = **+0.7461** (v_48_020).
`Process/mutation-task-7-checkpoint-benchmark.md` reports TCRG1 ρ = **+0.7538** for the
*same assay, same checkpoint, same n=1058, same structure source, same mean pLDDT 83.1*.
Those should have been bit-identical. They differ because of this bug. Neither write-up
is wrong about its own run — the scorer just wasn't reproducible.

---

## Fix

### 1. `orchestrator/mutation_scan.py`

Two changes to `_run_proteinmpnn_conditional_probs`:

**(a) Non-zero seed.** Line 70, `"--seed", "0"` → use a nonzero default, and make it a
parameter so callers can vary it deliberately:

```python
def _run_proteinmpnn_conditional_probs(
    pdb_string: str,
    tmpdir: str,
    proteinmpnn_dir: str,
    model_name: str = "v_48_020",
    seed: int = 37,              # NOTE: must be non-zero — ProteinMPNN's
                                 # `if args.seed:` treats 0 as "pick a random seed".
    num_decoding_orders: int = 8,
) -> np.ndarray:
```
and in `cmd`:
```python
    "--seed", str(seed),
    "--num_seq_per_target", str(num_decoding_orders),
```
Add a guard so this can't silently regress:
```python
    if seed == 0:
        raise ValueError(
            "seed=0 is unusable: ProteinMPNN's `if args.seed:` check treats 0 as unset "
            "and picks a random seed, making scores non-reproducible. Use any non-zero int."
        )
```

**(b) Average over decoding orders instead of taking one arbitrary sample.** Line 86 is
currently:
```python
return data["log_p"][0]  # [L, 21] — first (only) batch element
```
With `--num_seq_per_target N`, ProteinMPNN returns `log_p` of shape `[N, L, 21]` — N
independent decoding-order samples (verified: `--num_seq_per_target 8` → shape
`(8, 68, 21)`). A single sample is noisy; the mean is a much better estimate:
```python
return data["log_p"].mean(axis=0)  # [L, 21] — averaged over decoding orders
```

Verified sample spread on one mutation (E5L, 8 decoding orders, `--seed 37`):
```
per-sample: -0.826 -1.067 -0.786 -0.899 -1.210 -0.983 -1.026 -0.926
spread: 0.424   mean: -0.9653   std: 0.128
```
So averaging over 8 cuts the standard error by ~sqrt(8) ≈ 2.8x versus the current
single-sample approach, for 8x the compute on a step that already takes seconds on CPU.

**Docstring:** the module docstring should state that scores are the mean log-odds over
`num_decoding_orders` random decoding orders at a fixed seed, and are reproducible only
for a fixed `(seed, num_decoding_orders, model_name, structure)` tuple.

### 2. `config.py` / `.env.example`

```python
PROTEINMPNN_SEED = int(os.getenv("PROTEINMPNN_SEED", 37))
PROTEINMPNN_NUM_DECODING_ORDERS = int(os.getenv("PROTEINMPNN_NUM_DECODING_ORDERS", 8))
```
```
# ProteinMPNN scoring reproducibility. SEED MUST BE NON-ZERO — ProteinMPNN treats
# seed=0 as "unset" and randomizes, making scores non-reproducible.
PROTEINMPNN_SEED=37
# Number of random decoding orders averaged per scan. Higher = less noise, linearly
# more compute. 8 is a reasonable default; 1 reproduces the old single-sample behavior.
PROTEINMPNN_NUM_DECODING_ORDERS=8
```
Thread both through `score_candidate_mutations` → `_run_proteinmpnn_conditional_probs`,
and through the `scan_mutations` agent tool call site in `orchestrator/agent.py`.

### 3. Tests — `tests/test_mutation_scan.py`

Add:
- `seed=0` → raises `ValueError` with the explanation (guards the regression directly)
- the `cmd` list contains `--seed <nonzero>` and `--num_seq_per_target <N>` matching the
  passed args (mock `subprocess.run`, assert on the command construction)
- averaging: mock the `.npz` load to return a known `[3, L, 21]` array, assert the
  returned matrix equals the mean over axis 0, not `[0]`
- **the existing env-gated real-binary integration test should assert determinism**: call
  `score_candidate_mutations` twice with the same seed on the same PDB, assert identical
  output. That's the test that actually would have caught this.

---

## Consequences for work already done (be honest about these, don't quietly re-run)

### `Process/mutation-task-2-validation-gate.md` — verdict stands, numbers should be re-run
The gate's conclusion (**PASS**) is not in danger: the effect being measured (ρ ≈ 0.75 /
0.43 / 0.33) is orders of magnitude larger than the seed noise (~0.008 on ρ). But the
exact reported figures aren't reproducible. Re-run with the fix and update the table, and
add a line noting the original run used the unseeded scorer.

### `Process/mutation-task-7-checkpoint-benchmark.md` — conclusion is right, reasoning is under-supported
This one needs more care. The write-up concludes "the choice is within noise" from a
0.023-wide band across four checkpoints, and keeps `v_48_020` on a **+0.5059 vs +0.5037**
mean margin over `v_48_002`. But:

- Each checkpoint was scored **once**, with a random (unseeded) decoding order.
- Observed run-to-run ρ noise from the seed alone is ~0.008 on a single assay — same
  order as the 0.002 margin the decision rests on.
- So the benchmark as run **cannot separate checkpoint effect from seed noise**. The
  ranking within that band is partly random.

The *decision* (keep `v_48_020`) is still the right call and doesn't need reversing —
it's the default, it's never worst, and the honest summary is "no checkpoint is
distinguishable on this evidence." But the write-up currently implies a measured
preference (+0.5059 is "best mean") that the data doesn't actually support. Recommended
fix: re-run with the seeded + averaged scorer, and **repeat each checkpoint across k≥3
seeds**, reporting mean ± std per cell. Then either a real difference emerges, or the
"indistinguishable" conclusion is properly earned rather than assumed. Given your stated
bar of "rigorous benchmark writeup with proper stats," this is exactly the kind of thing
a reviewer would catch.

---

## What the review found that's *good* (not everything needs fixing)

- `apply_mutation` matches the spec exactly, including the rollback-on-failure semantics
  and the cap check — the tricky parts are right.
- **Task 4 was implemented more correctly than the plan specified.** The plan claimed both
  `run_rosetta_relax` and `run_simulation` mutate `state["current_pdb"]` and both needed a
  clash recount. That was wrong: `run_simulation` only reads `current_pdb` and stores
  `sim_result` — the simulation backends don't return a structure at all. The
  implementation correctly added the recount only to `run_rosetta_relax` (and to
  `run_boltz_prediction`/`apply_mutation`, which do reassign the PDB) and correctly did
  *not* add a spurious one to `run_simulation`. Good catch against a wrong spec.
- The Task 2 write-up's caveat that the 37-mer's ρ=0.75 is "unusually high and typical for
  very small single-domain stability assays — not evidence the scorer is this strong in
  general" is exactly the right instinct, and the ESTA/CCDB framing as the representative
  and blind-spot cases respectively is well judged.
- `scan_mutations` correctly surfaces the "not a function/stability/fitness proxy" caveat
  in its tool output, so the agent can't over-read the number.
- Test coverage went from 0 → 27 (ligands) / 15 (membrane) / 16 (agent) / 9
  (mutation_scan). `pytest.ini`'s `--continue-on-collection-errors` is a sensible call
  given the optional-dep-heavy layout.