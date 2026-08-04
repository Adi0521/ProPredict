# Pipeline-vs-Boltz A/B/C/E benchmark

**Date:** 2026-07-30 (clean run) · first run 2026-07-29 (contaminated, appendix below)
**Harness:** `benchmark_pipeline_modal.py` · logs to `benchmarks/pipeline_vs_boltz.jsonl`
**Motivation:** Every quality number in this repo (`benchmarks/BENCHMARKS.md`, `results.jsonl`)
is a *Boltz-2 backend* run. The project had no data on whether the **pipeline**
(`_run_prediction_core`: multi-model ensemble + iterative refinement + Rosetta relax) adds
anything over the Boltz call it wraps, or whether any gain is just the effect of drawing more
Boltz samples. This benchmark answers that, and — via arm E — attributes the gain to a specific
stage.

## Headline

**The pipeline beats raw Boltz significantly, but essentially all of the gain is ESMFold being
in the ensemble — the pipeline's cross-model pLDDT *selection* adds nothing over just always
using ESMFold, and its refinement/relax stages were effectively never exercised.**

## Design

Same sequence, same experimental reference, scored across four arms (Kabsch TM/RMSD, reused
from `benchmark_modal.score_structures`):

| Arm | What ran | Isolates |
|---|---|---|
| **A** | `call_boltz(seq, seed=0)`, single sample | today's baseline |
| **B** | best-of-N raw Boltz over seeds `1..N`, `N` = Boltz calls arm C made, best pLDDT | does *more sampling alone* help? |
| **C** | `_run_prediction_core({context:{}, priority:"accurate"})`, `AGENT_ENABLED=False` | the pipeline |
| **E** | the ESMFold structure the pipeline already folded inside arm C (zero extra compute) | how much of C is *just* ESMFold |

Analysis is **paired** (same targets): per-target ΔTM/ΔRMSD, Wilcoxon signed-rank, W/L/T. Arm C
records `winner_model` so a C win is attributed to the model that produced it, not inferred.

**Run config:** 20-target CASP15 subset, `ensemble_seeds=3`, agent OFF, **A100 (80 GB observed
in run logs)**, Boltz build `2.2.1@b1ebfc46ecf5`, ~25 min. 15/20 paired; 5 excluded: 7TY5 /
7WV6 / 7WV7 (PDB 404s), 7UXC (2188 aa — ESMFold O(L²) attention tried to allocate 156 GiB,
inherent), and one target cancelled by a local client disconnect (re-runnable).

## Results (n=15)

Paired stats:

| Comparison | mean ΔTM | median ΔTM | W/L/T | Wilcoxon p |
|---|---|---|---|---|
| C vs A (pipeline vs raw single) | +0.234 | +0.109 | 10/5/0 | 0.012 |
| **C vs B (pipeline vs compute-matched)** | **+0.235** | +0.119 | 11/4/0 | **0.004** |
| **E vs B (ESMFold-only vs compute-matched)** | **+0.241** | +0.119 | 11/4/0 | **0.022** |
| **C vs E (pLDDT selection vs always-ESMFold)** | **−0.006** | 0.000 | 6/2/7 | **0.547** |
| C vs A — RMSD (Å, lower better) | −6.08 | −2.06 | 12/3/0 | 0.004 |
| C vs B — RMSD | −5.81 | −2.23 | 11/4/0 | 0.008 |

Arm-C winners: **ESMFold 7, Boltz 8** · Boltz calls total A 15 / B 46 / C 46.

Per-target, with the selection decomposition (`C − E`):

| PDB | L | decision | C winner | TM_A | TM_B | TM_C | TM_E | C−E |
|---|---|---|---|---|---|---|---|---|
| 7XPU | 371 | accept | esmfold | 0.381 | 0.276 | **0.986** | 0.986 | 0.000 |
| 7XPT | 375 | accept | esmfold | 0.289 | 0.206 | **0.978** | 0.978 | 0.000 |
| 7UXB | 246 | accept | esmfold | 0.873 | 0.865 | **0.983** | 0.983 | 0.000 |
| 7TY4 | 515 | accept | esmfold | 0.225 | 0.340 | **0.961** | 0.961 | 0.000 |
| 7VQ7 | 389 | accept | esmfold | 0.201 | 0.269 | **0.937** | 0.937 | 0.000 |
| 7VDL | 215 | accept | esmfold | 0.395 | 0.560 | 0.765 | 0.765 | 0.000 |
| 7VQ6 | 175 | accept | esmfold | 0.245 | 0.197 | 0.404 | 0.404 | 0.000 |
| 7WGP | 269 | accept | boltz2 | 0.908 | 0.960 | 0.958 | 0.875 | +0.082 |
| 7WGQ | 277 | accept | boltz2 | 0.949 | 0.948 | 0.947 | 0.875 | +0.072 |
| 7X26 | 221 | accept | boltz2 | 0.841 | 0.693 | 0.751 | 0.701 | +0.051 |
| 7V3E | 280 | accept | boltz2 | 0.876 | 0.869 | 0.872 | 0.831 | +0.041 |
| 7UL4 | 286 | accept | boltz2 | 0.961 | 0.966 | 0.961 | 0.945 | +0.016 |
| 7V3F | 495 | **refine** | boltz2 | 0.146 | 0.225 | 0.255 | 0.242 | +0.012 |
| 7UL5 | 285 | accept | boltz2 | 0.922 | 0.913 | 0.909 | 0.970 | **−0.061** |
| 7VDM | 217 | accept | boltz2 | 0.412 | 0.327 | 0.472 | 0.768 | **−0.297** |

## Interpretation

1. **The pipeline genuinely beats raw Boltz.** C > B at p=0.004 (and C > A at p=0.012), on both
   TM and RMSD. Beating the *compute-matched* arm B means it isn't just "more Boltz samples."
2. **The entire effect is ESMFold in the ensemble.** Arm E (ESMFold alone) beats B by +0.241 —
   the *same* margin as C's +0.235 — and **C vs E is a statistical wash (Δ −0.006, p=0.55, 7 of
   15 exact ties).** The wins come from targets where single-sequence Boltz collapses
   (7XPT 0.29→0.98, 7XPU 0.38→0.99, 7VQ7 0.20→0.94, 7TY4 0.22→0.96) and ESMFold, being in the
   ensemble, rescues them. This is a real model-diversity benefit — but it is the *presence of a
   second model*, not any refinement logic.
3. **The cross-model pLDDT selection is net-neutral and occasionally harmful.** On the 7
   ESMFold-win targets C==E (selection irrelevant). On the 8 Boltz-win targets the selector
   helps 4× (correctly keeping Boltz where Boltz is better, e.g. 7WGP +0.082) but **mis-picks
   2×**, once catastrophically: **7VDM — it chose Boltz (0.472) over ESMFold (0.768), a −0.30 TM
   error** — because "highest mean pLDDT" compares two models' uncalibrated confidence scales.
   Net over the panel: zero. Always-take-ESMFold would have scored the same or slightly better
   here.
4. **Refinement/relax are effectively untested.** 14/15 targets accepted on the first ensemble
   (`boltz_calls == ensemble_seeds == 3`). Only **7V3F** triggered refinement (one extra seed;
   0.242→0.255, negligible). This benchmark measures the *ensemble + selection* stage; it says
   almost nothing about the refinement loop, Rosetta relax, or the agent.

## Caveats

- **n=15, one easy-ish CASP15 subset.** Directionally strong (p=0.004) but not a full-panel
  number. The excluded 7UXC (2188 aa) and the disconnect-cancelled target are not in it.
- **Absolute TM is soft.** `score_structures` uses RMSD-optimal (Kabsch) superposition on the
  first min-length CA atoms with no sequence alignment, so absolute TM slightly underestimates
  TM-align. Applied identically to every arm → the paired comparison is valid; absolute values
  are approximate.
- **The pLDDT selector carries a real regression risk** (7VDM). Worth a follow-up: gate
  cross-model selection on same-model pLDDT calibration, or prefer ESMFold when Boltz pLDDT is
  low, rather than comparing raw pLDDT across models.

## Follow-ups

- **Test refinement on purpose.** As configured (`PLDDT_ACCEPT_THRESHOLD=75`) almost everything
  accepts immediately. Lower the threshold or pick targets that stay sub-threshold after the
  ensemble to actually exercise the refinement/relax loop.
- **Fix or justify cross-model pLDDT selection** (see 7VDM above).
- Optional: re-run the one disconnect-cancelled target to reach n=16; 7UXC/404s will never pair.

## Reproduce

```bash
modal run benchmark_pipeline_modal.py                       # 20-target CASP15 subset, A100
modal run benchmark_pipeline_modal.py --pdb-ids 8D8D,8BCZ   # cheap 2-target smoke test
```
Per-target output → `pipeline_vs_boltz_results.json`; one summary row per run appended to
`benchmarks/pipeline_vs_boltz.jsonl` (kept out of `results.jsonl`, which is single-backend and
auto-numbered by line count).

---

## Appendix — superseded first run (2026-07-29, contaminated)

The first run (A10G, 24 GB) is **not valid** and its numbers must not be cited. ESMFold-local
stays resident in GPU memory (lazy singleton) while Boltz's subprocess also needs the GPU; on a
24 GB card that co-residency OOM'd on larger proteins, excluding **9/20** targets non-randomly
(biased toward large ones) and silently degrading arm C to ESMFold-only when its Boltz seeds
died. It reported C vs B mean ΔTM +0.211 (p=0.042) on n=11, and it surfaced three harness bugs
now fixed:

- **GPU A10G → A100** (`+ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) so ESMFold + Boltz
  fit. Very large targets (>~1000 aa) still OOM ESMFold's O(L²) attention — inherent, dropped
  not chased (7UXC).
- **Arm C records `winner_model`** — direct attribution instead of pLDDT inference.
- **Added arm E (ESMFold-only), zero extra compute** — which is what made the decomposition above
  (E vs B, C vs E) possible and revealed that the gain is ESMFold, not the selection.

The contaminated summary row was not kept in `pipeline_vs_boltz.jsonl`.
