# Benchmark Notebook

Tracking Boltz-2 prediction quality across changes. Each run is logged to `results.jsonl` with full config and metrics; this file captures the human-readable takeaways.

## Boltz-2 build provenance (read before comparing runs across dates)

Until 2026-07-21 the Modal image installed Boltz-2 from **unpinned git HEAD**, and Modal
cached that layer — so the version behind any given run was whatever HEAD happened to be
when the image was last built, and it was recorded nowhere. That has now been pinned to
commit **`b1ebfc46`** (2026-05-29) in both `modal_app.py` and `requirements-gpu.txt`.

`b1ebfc46` reports version string `2.2.1` but is **6 commits ahead of the `v2.2.1` tag**,
including two numerics fixes (autocast device type, cpu float32 precision) — so
`boltz==2.2.1` from PyPI is a *different* build and is not what produced these numbers.

**Consequence for the record below:** Run 001 (2026-05-19) predates commit `b1ebfc46`
(2026-05-29), so it ran on an older Boltz build than Runs 002–011. Runs from 002 onward are
consistent with each other and with the pin. Any future version bump should be treated as a
new baseline and re-benchmarked, not compared directly against these rows.

---

## Run 001 — Baseline Boltz-2 (CASP15)

**Date:** 2026-05-19
**Commit:** `1731dcc` — added diagram to readme
**Backend:** Boltz-2, single sample, 200 diffusion steps, no MSA
**Targets:** 88 CASP15 (74 succeeded, 14 failed)

| Metric | Mean | Median |
|--------|------|--------|
| TM-score | 0.5025 | 0.4005 |
| RMSD (A) | 11.78 | 10.35 |
| pLDDT | 68.9 | — |

**TM-score distribution:** 32/74 >= 0.5, 25/74 >= 0.7

**Notes:** Baseline run with default settings. No MSA server, no refinement, no ensemble. 14 failures mostly from oversized targets (>1000 aa) or missing PDB files. The long-tail RMSD is dragged up by a few very poor predictions on large multi-domain proteins.

> **Boltz build caveat (added 2026-07-21).** This run predates commit `b1ebfc46` (2026-05-29),
> the build pinned in `modal_app.py` today, so it ran on an **older, unrecorded** Boltz-2.
> The exact build cannot be recovered — the image layer it used has since been replaced.
> This makes the Run 001 ↔ Run 006 agreement below a *cross-version* reproduction rather
> than a same-build one. That is arguably a stronger result (the pipeline was stable across
> a Boltz change), but it is not what "reproducible" was originally claiming here.

**Takeaways:**
- Median TM-score of 0.40 suggests many targets are near the noise floor — MSA and ensemble seeds should help here
- The 25 targets with TM >= 0.7 are mostly compact single-domain proteins under 300 aa
- pLDDT of 68.9 is below the typical "confident" threshold of 75 — room to improve

---

## Run 002–005 — W&B Integration & Harness Debugging

**Date:** 2026-06-23
**Commit:** `8af510e` — finished benchmarking setup
**Backend:** Boltz-2, single sample, 200 diffusion steps, no MSA
**Targets:** 5 CASP15 (subset)

These runs were iterative development runs while wiring up Weights & Biases logging and the local benchmark harness (Mac → Modal GPU dispatch). Run 002 failed entirely (MSA server error on Modal despite MSA=false). Runs 003–005 each succeeded on 4/5 targets with nearly identical results (TM mean ~0.745, pLDDT ~77), confirming the harness was working. Run 004 notes say "MSA enabled test" but config still had `BOLTZ_USE_MSA=false` — the flag wasn't plumbed to the YAML yet.

---

## Run 006 — Baseline Reproduction (Full Harness)

**Date:** 2026-06-23
**Commit:** `97a0816` — wired up msa setup, finished logging with W&B
**Backend:** Boltz-2, single sample, 200 diffusion steps, no MSA
**Targets:** 88 CASP15 (74 succeeded, 14 failed)
**Changes from previous:** New benchmark harness with W&B logging, local Mac dispatch to Modal A10G

| Metric | Mean | Median |
|--------|------|--------|
| TM-score | 0.5028 | 0.4007 |
| RMSD (A) | 11.78 | 10.35 |
| pLDDT | 69.2 | 71.5 |

**TM-score distribution:** 32/74 >= 0.5, 25/74 >= 0.7

**Notes:** First full run through the new benchmark harness (W&B + structured JSONL logging). Results are within noise of Run 001 (TM mean 0.5025 vs 0.5028), confirming reproducibility. Duration: ~580s total.

**Takeaways:**
- Baseline is reproducible across runs — stochasticity in Boltz-2 diffusion sampling is minimal at 200 steps
- New harness is working correctly with full CASP15 coverage

> **Amended 2026-07-21:** this reproduction was **cross-version**, not same-build — Run 001
> ran on a pre-`b1ebfc46` Boltz (see the caveat under Run 001). Matching Run 001 to within
> 0.0003 TM across both a harness change *and* a Boltz change is a stronger stability result
> than originally claimed, but the "stochasticity is minimal" conclusion is now confounded
> with version drift and should not be read as a clean seed-noise measurement.

---

## Run 007–009 — MSA Wiring Attempts

**Date:** 2026-06-23
**Commit:** `97a0816` (dirty)
**Backend:** Boltz-2, single sample, 200 diffusion steps, MSA flag being debugged
**Targets:** 10 CASP15 (subset)

Iterative attempts to get MSA working through the ColabFold server. Runs 007–008 had `BOLTZ_USE_MSA=false` in the recorded config despite notes saying "MSA actually enabled" — the flag was being set in `.env` but not propagated to the Boltz YAML input. TM mean ~0.60 on 8/10 succeeded, consistent with the no-MSA baseline on those targets. Run 009 set `BOLTZ_USE_MSA=true` in config and all 10 targets failed with "MSA file server not found" — the ColabFold MSA server wasn't reachable from the Modal container.

---

## Run 010 — MSA Enabled (ColabFold Server Working)

**Date:** 2026-06-23
**Commit:** `97a0816` (dirty)
**Backend:** Boltz-2, single sample, 200 diffusion steps, **MSA enabled via ColabFold**
**Targets:** 10 CASP15 (7 succeeded, 3 failed)
**Changes from previous:** MSA server connectivity fixed; Boltz-2 now receives MSA alignments

| Metric | Mean | Median |
|--------|------|--------|
| TM-score | 0.8711 | 0.9671 |
| RMSD (A) | 3.14 | 1.19 |
| pLDDT | 90.4 | 94.0 |

**TM-score distribution:** 7/7 >= 0.5, 5/7 >= 0.7

**Head-to-head vs Run 006 (no MSA), same targets:**

| Target | TM (no MSA) | TM (MSA) | Delta |
|--------|-------------|----------|-------|
| 7TY4 | 0.223 | 0.989 | **+0.766** |
| 7UL4 | 0.961 | 0.967 | +0.006 |
| 7UL5 | 0.922 | 0.975 | +0.053 |
| 7UXB | 0.878 | 0.995 | +0.117 |
| 7V3E | 0.876 | 0.898 | +0.022 |
| 7V3F | 0.151 | 0.650 | **+0.499** |
| 7VDM | 0.424 | 0.624 | **+0.200** |

**Notes:** MSA provides enormous gains on difficult targets. 7TY4 (515 aa) jumps from near-random (0.22) to near-perfect (0.99). 7V3F (495 aa multi-domain) improves from 0.15 to 0.65. Already-good targets (7UL4, 7UL5, 7UXB) see modest gains. pLDDT jumps from ~77 to ~90 across the board. Duration ~613s (vs ~226s without MSA) — MSA server queries add ~2.5x overhead. 7VDL was lost to a failure this run but succeeded without MSA.

**Takeaways:**
- MSA is transformative: mean TM 0.60 → 0.87 on the same 10-target subset, pLDDT 77 → 90
- The biggest gains are on hard targets where single-sequence mode struggles (large/multi-domain proteins)
- Next step: full 88-target run with MSA to get comparable numbers against the Run 001/006 baseline
- 3 failures (7TY5 = PDB 404 as always, 7UXC = Boltz stderr error, 7VDL = new failure to investigate)
- ~2.5x slower with MSA — acceptable tradeoff given the quality improvement

---

## Run 011 — Affinity capability verification (NOT a quality run)

**Date:** 2026-07-21
**Commit:** `3206d7e` — "Fixed bug in Boltz Affinity score always showing 0" (+ uncommitted glob anchor)
**Backend:** Boltz-2, 1 diffusion sample, 200 sampling steps, no MSA, A10G
**Target:** none — a 33-aa synthetic peptide + ethanol (`CCO`), no reference structure
**Changes from previous:** affinity JSON key fix (`affinity` → `affinity_pred_value`)

| Metric | Value |
|---|---|
| TM-score | **N/A** — no reference structure |
| RMSD | **N/A** — no reference structure |
| pLDDT | 91.2 |
| `affinity_pred_value` | +1.216 → IC50 ≈ 16 µM |
| `affinity_probability_binary` | 0.166 |

**This is deliberately not a quality benchmark and must not be read as one.** There is no
reference structure, the "target" is a synthetic peptide with no binding pocket, and the
ligand is ethanol. It is logged here for one reason: it is the **first run in this project's
history that produced an affinity number at all**, and it establishes what correct affinity
plumbing looks like. It is not appended to `results.jsonl` — that schema is for target-based
quality runs and this has no TM/RMSD to record.

Run via `modal run modal_app.py::test_boltz_affinity_gpu`, which reads the raw Boltz-2 output
directly (not through `call_boltz`) to get ground truth on filenames and JSON keys, then
checks our parser against it.

**Notes:**
- Ground truth: Boltz writes exactly one affinity file, `affinity_<record_id>.json`
  (here `affinity_input.json`), with keys `affinity_pred_value`,
  `affinity_probability_binary`, and `*1`/`*2` ensemble-member variants. We read the
  un-suffixed pair.
- Values are directionally sane: ethanol against a pocket-less peptide should be weak
  (16 µM) and improbable as a binder (p=0.17). That is the only signal being claimed here.
- `affinity_pred_value` is **log10(IC50), IC50 in µM — not kcal/mol**, and lower = tighter.
  Every prior label in the codebase said kcal/mol.

**Takeaways:**
- **Every affinity number in this repo's history before this run was `None`.** The backend
  read a JSON key Boltz never writes, so `StructurePrediction.affinity_score` was always
  null. No previously recorded benchmark is affected — none of Runs 001–010 measured
  affinity — but any earlier *reasoning* that assumed affinity was available was operating
  on nothing.
- The unit error is the more dangerous half: the agent reasons over this number, and
  "−8.4 kcal/mol" vs "1.2 log10(IC50 µM)" are opposite claims about binding strength.
- A real affinity benchmark still needs a system with measured binding data. The obvious
  candidate (HIV-1 protease, `benchmarks/hiv_pr_resistance_dataset.json`) is **blocked**:
  `call_boltz` builds one protein chain, and HIV-PR is an obligate homodimer whose active
  site forms at the dimer interface — a monomer has no pocket. See
  `research_plan/rowA-boltz-affinity-invariance.md` Bug 2.
- Reproducibility caveat surfaced while doing this: the Modal image installs boltz from
  **unpinned git HEAD** (`modal_app.py:42`), so the version behind Runs 001–011 is whatever
  HEAD was at first image build and is not recorded. Worth pinning before the next
  quality run.

---

<!-- Template for new entries:

## Run XXX — [Short description]

**Date:** YYYY-MM-DD
**Commit:** `abc1234` — commit message
**Backend:** [backend + key config]
**Targets:** [count] [source] ([succeeded] succeeded, [failed] failed)
**Changes from previous:** [what changed and why]

| Metric | Mean | Median |
|--------|------|--------|
| TM-score | | |
| RMSD (A) | | |
| pLDDT | | |

**Notes:**

**Takeaways:**

-->
