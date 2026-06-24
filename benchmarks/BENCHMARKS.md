# Benchmark Notebook

Tracking Boltz-2 prediction quality across changes. Each run is logged to `results.jsonl` with full config and metrics; this file captures the human-readable takeaways.

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
