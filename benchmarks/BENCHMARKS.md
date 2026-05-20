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
