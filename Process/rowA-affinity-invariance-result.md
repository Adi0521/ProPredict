# Row A — Result: Boltz-2's affinity head is blind to HIV-PR resistance mutations

**Date:** 2026-10-02
**Source:** `research_plan/rowA-boltz-affinity-invariance.md` (the pre-registration)
**Deliverable:** run the pre-registered invariance experiment on Modal and call the verdict.
**Verdict: H_invariant.** Pooled slope **+0.013**, predicted Δ spread **26%** of experimental.
**Reproduce:**
```bash
modal run benchmark_affinity_invariance_modal.py            # 164 GPU jobs, resumable
python benchmarks/analyze_affinity_invariance.py benchmarks/affinity_invariance_results.jsonl
```

## What ran

164 Boltz-2 jobs on Modal A10G — 4 WT reference runs + 160 clinical isolates across DRV /
NFV / SQV / IDV, one job per `(isolate, drug)` at seed 37, `diffusion_samples=1`,
`sampling_steps=200`. **MSA-off arm only.** Every job returned: 164/164 `ok=True`, zero
failures, zero duplicate keys. The system is the HIV-1 protease **homodimer** (`id: [A, B]`)
plus the PI ligand, which only became expressible after the Bug-2 fix the pre-registration
specified.

The MSA-off arm is the headline (not an ablation) for the reason stated in the local
entrypoint: an HIV-protease MSA contains WT *and* every resistant variant, so an MSA-on model
can copy resistance from homologs. MSA-off is the clean test of whether the model reasons
about the mutation itself.

## Result (MSA=off, censored rows excluded, n=148)

| drug | n | spearman | slope | sd(pred) | sd(exp) | ratio |
|---|--:|--:|--:|--:|--:|--:|
| DRV | 37 | +0.351 | +0.168 | 0.180 | 0.513 | 0.35 |
| IDV | 38 | +0.226 | +0.067 | 0.154 | 0.700 | 0.22 |
| NFV | 37 | +0.069 | −0.002 | 0.182 | 0.752 | 0.24 |
| SQV | 36 | +0.067 | −0.004 | 0.137 | 0.710 | 0.19 |
| **ALL** | **148** | **+0.059** | **+0.013** | **0.182** | **0.693** | **0.26** |

**Read the slope, not the ρ.** A slope of +0.013 means a **10× experimental shift in
resistance moves the predicted log10(IC50) by 0.013 — about 3%.** The experimental data spans
0.69 log10 units of spread; the prediction spans 0.182, or 26% of it, and most of that 26% is
not aligned with the experiment (ρ = +0.059). This is the pre-registered definition of
practical blindness: not "no correlation survives significance," but "the response is ~2
orders of magnitude too small to rank anything."

## Two numbers that disagree, and which one governs

The Modal entrypoint prints an inline read when the sweep finishes; the standalone analyzer
prints a different one. Both were reproduced from the committed jsonl, and the difference is
entirely accounted for:

| | inline `_summarize` | `analyze_affinity_invariance.py` |
|---|---|---|
| n | 160 (censored **included**) | 148 (censored **excluded**) |
| spread stat | `max − min` = **1.394** | sample **sd** = **0.182** |
| pooled ρ | +0.023 | +0.059 |
| pooled slope | +0.0011 | +0.013 |
| verdict gate | `|ρ|<0.2` **OR** `|slope|<0.05` | `|slope|<0.1` **AND** `|ρ|<0.2` |

The analyzer governs — it excludes the 12 rows censored at the PhenoSense ceiling of 100
(whose true fold-change is unknown and only bounded below), and it reports a standard
deviation rather than a range, which a single outlier cannot inflate. The inline read is a
convenience so the sweep doesn't finish silent; its verdict gate is also the looser of the two
(an `OR`). **The conclusion is robust to the choice:** including the censored rows moves the
pooled slope to +0.001 and ρ to +0.023 — if anything *more* invariant, not less.

## The DRV number is the trap the pre-registration called in advance

DRV shows ρ = +0.351, which in isolation looks like signal. It is not safe to read that way,
and the reason was written down **before** the data existed: validating the analyzer on pure
synthetic noise produced per-drug ρ ranging from **−0.349 (NFV) to +0.231 (SQV)** at n≈37.
DRV's +0.351 sits at the edge of that envelope, it is one of four drugs tested, and no
multiple-comparison correction has been applied. The pre-registration's conclusion — *per-drug
ρ is not trustworthy at this sample size; report the pooled slope and the spread ratio* —
holds, and the fact that it was declared in advance is what makes discarding DRV honest rather
than convenient.

Even taken entirely at face value, DRV's slope of +0.168 means a 10× resistance shift moves
the prediction by 0.17 log10 units — still about one sixth of the experimental effect, still
not usable for ranking. The strongest drug in the panel does not reach practical utility.

## What this buys, and what it costs

1. **The research plan's central premise is confirmed, and quantified in the ligand domain for
   the first time.** Feldman et al. (2026) showed structural/confidence invariance on *apo*
   structures; this extends it to a **ligand-bound affinity head**, which nobody had measured.
   Cost: one afternoon of A10G time.
2. **It is a generous test that the model still fails.** These isolates carry a median
   mutation load around 8 of 99 residues and span 3 log10 units of real resistance, including
   100×-resistant variants. The model does not separate them from WT.
3. **Leakage cuts toward the null, so it does not threaten this result.** HIV-PR/PI
   co-crystals are certainly in Boltz-2's training set, which would push the model toward
   reproducing memorized WT-like complexes regardless of mutation — exactly the observed
   behavior. A *positive* result would have needed a leakage caveat; a null does not.
4. **The tier-3 gate has fired.** `orchestrator/qmmm_oracle.py` was written to be wired in
   "only if the affinity-invariance experiment returns H_invariant." It has. The physics tier
   now has a measured gap to fill rather than an assumed one. (Writeup: next in `Process/`.)

## Honest limits

- **MSA-off only.** The mechanistic ablation — does invariance *relax* when the MSA is removed,
  implicating MSA-washout as the cause — is not answered, because only the MSA-off arm ran.
  `--both-msa-arms` adds it, and the jsonl is resumable, so the MSA-on arm costs only its own
  164 jobs. Until then this is a measured phenomenon without a mechanism.
- **One seed, one system, one assay.** Single seed (37) per job, so within-model sampling
  noise is unmeasured; one protein family; PhenoSense fold-change only.
- **Boltz-2 claims near-FEP affinity performance** (Passaro et al. 2025). That sits in direct
  tension with this null and should be confronted, not sidestepped: the likely reconciliation
  is that its benchmarks measure *cross-ligand* ranking against a fixed target, whereas this
  measures *cross-mutant* ranking against a fixed ligand. Those are different generalization
  axes and the model is only trained for one. That reading is plausible but **not tested here.**

## Two defects found in the process

- **Modal mount/import mismatch (fixed).** `benchmark_affinity_invariance_modal.py` mounts the
  harness flat at `/root/benchmark_affinity_invariance.py` but imported it as
  `from benchmarks.benchmark_affinity_invariance import run_boltz`. Every GPU worker died with
  `ModuleNotFoundError` on the first full launch. Fixed to the top-level import; the 6 jobs
  from the earlier smoke test survived in the jsonl and the resume logic skipped them.
- **`analyze_affinity_invariance.py` docstring over-promises (open).** It states *"Censored
  rows are reported both included (rank-safe) and excluded"* — only the excluded branch is
  implemented. The included-censored numbers in the table above were computed ad hoc. Either
  implement the branch or correct the docstring; the numbers are unaffected.

## Artifacts

- `benchmark_affinity_invariance_modal.py` — Modal GPU wrapper (`.starmap` over A10G, resumable
  jsonl, inline verdict). Reuses `run_boltz` from the CLI harness verbatim — no logic fork.
- `benchmarks/benchmark_affinity_invariance.py` — standalone harness (YAML build, affinity-key
  parsing, timing).
- `benchmarks/analyze_affinity_invariance.py` — the definitive analysis (censoring-aware).
- `benchmarks/affinity_invariance_results.jsonl` — the 164-row result set (committed in `bbef345`).
- `benchmarks/hiv_pr_resistance_dataset.json` / `build_hiv_pr_dataset.py` — the stratified
  dataset and its builder.
