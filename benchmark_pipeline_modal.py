"""
Pipeline-vs-Boltz A/B/C benchmark.

Answers the question the existing `benchmark_modal.py` cannot: does the ProPredict
*pipeline* (multi-model ensemble + iterative refinement + Rosetta relax) add anything
on top of the raw Boltz-2 backend it wraps, or is any gain just the result of drawing
more Boltz samples?

Three arms, all scored on the SAME sequence against the SAME experimental structure:

    A  raw single      call_boltz(seq, seed=0)                         (today's baseline)
    B  compute-matched  best-of-N raw Boltz over seeds 1..N,           (control: does more
                        N = Boltz calls arm C actually made, best pLDDT  sampling alone help?)
    C  full pipeline    _run_prediction_core(...), AGENT_ENABLED=False  (ensemble+refine+relax)
    E  ESMFold-only     the ESMFold structure the pipeline already      (attribution: how much
                        folded inside arm C — scored, zero extra cost    of C is just ESMFold?)

The headline comparisons are C vs B (does the pipeline *logic* beat spending the same
compute on raw sampling?) and C vs A (the pipeline as shipped vs the naive baseline).
Arm E decomposes the source of any C win: E vs B shows how much is ESMFold's ensemble
contribution, and C vs E shows whether C's best-by-pLDDT selection helps beyond it.
Analysis is PAIRED (same targets in every arm): per-target ΔTM / ΔRMSD, a paired
significance test, win/loss/tie counts, and the Boltz-call cost of each arm.

CAVEAT — scope. CASP15 targets carry no membrane/ligand context, so the MD, PropKa,
membrane, and ligand stages never fire. This harness therefore measures ONLY the
ensemble + refinement + relax path. Testing the environmental value-add needs targets
with known ligands/membranes and is a separate exercise.

Usage:
    modal run benchmark_pipeline_modal.py                          # 20-target CASP15 subset
    modal run benchmark_pipeline_modal.py --max-targets 5          # quick smoke test
    modal run benchmark_pipeline_modal.py --ensemble-seeds 3       # Boltz seeds in arm C's ensemble
    modal run benchmark_pipeline_modal.py --pdb-ids 1UBQ,1VII      # specific PDB IDs
"""

import json
import time
import math
import numpy as np

from modal_app import app, image

# Modal auto-mounts the entrypoint module (this file) into the container, but NOT sibling
# local modules. `benchmark_modal` is imported below (and again inside the GPU function via
# `score_structures`), so it must be present at /root remotely. Extend the image *here only*
# — rebinding the module-level `image` name leaves modal_app's own functions on the base
# image, so their build hash and reproducibility are unaffected.
image = image.add_local_file("benchmark_modal.py", remote_path="/root/benchmark_modal.py")

# Reuse the existing harness — target fetching and scoring are identical to the
# single-backend benchmark, so they must not be reimplemented here.
from benchmark_modal import (
    fetch_casp15_targets,
    fetch_rcsb_quality_targets,
    score_structures,
)


# ---------------------------------------------------------------------------
# Sequence extraction — same 3-to-1 logic benchmark_one uses, factored out
# ---------------------------------------------------------------------------

def _extract_sequence(true_pdb: str, chain: str):
    """Return (sequence, resolved_chain_id) from an experimental PDB string.

    Mirrors benchmark_modal.benchmark_one: prefer the requested chain, fall back
    to the first chain present, build the sequence from standard amino-acid ATOM
    records only.
    """
    import io
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa, protein_letters_3to1

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("t", io.StringIO(true_pdb))
    try:
        chain_obj = struct[0][chain]
    except KeyError:
        chains = list(struct[0].get_chains())
        if not chains:
            raise ValueError("No chains found in PDB")
        chain_obj = chains[0]
        chain = chain_obj.id

    seq = "".join(
        protein_letters_3to1.get(res.get_resname(), "X")
        for res in chain_obj
        if is_aa(res, standard=True)
    )
    return seq, chain


# ---------------------------------------------------------------------------
# Per-target Modal function — runs all three arms
# ---------------------------------------------------------------------------

# A100-40GB, not A10G: arm C holds the ESMFold-local model resident in GPU memory
# (a lazy singleton) while Boltz's subprocess also needs the GPU. On a 24 GB A10G that
# co-residency OOM'd on larger targets — knocking out ~half the set non-randomly and
# silently degrading arm C to ESMFold-only when its Boltz seeds died. 40 GB gives both
# models room; expandable_segments (set in-function) further cuts fragmentation OOMs.
# Very large targets (>~1000 aa) may still OOM ESMFold's O(L^2) attention — inherent.
@app.function(timeout=3600, gpu="A100-40GB", image=image)
def benchmark_pipeline_one(target: dict) -> dict:
    """Run arms A, B, C (+ ESMFold-only arm E) on one target and return paired scores.

    Slow by design: arm C runs ESMFold + up to N Boltz seeds + relax, and arm B
    re-runs N raw Boltz seeds to match C's compute — hence the 1-hour timeout.
    """
    import os
    import requests

    cfg = target.get("_cfg", {})
    os.environ.update({
        "BOLTZ_ENABLED": "True",
        "MODAL_ENABLED": "True",
        # Reduce CUDA fragmentation OOMs when ESMFold + Boltz share the GPU.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # Agent OFF: deterministic threshold refinement loop, reproducible, no API calls.
        "AGENT_ENABLED": "False",
        # How many Boltz seeds arm C's initial ensemble draws (refinement may add more).
        "ENSEMBLE_NUM_SEEDS": str(cfg.get("ENSEMBLE_NUM_SEEDS", 3)),
        "BOLTZ_DIFFUSION_SAMPLES": str(cfg.get("BOLTZ_DIFFUSION_SAMPLES", 1)),
        "BOLTZ_SAMPLING_STEPS": str(cfg.get("BOLTZ_SAMPLING_STEPS", 200)),
        "BOLTZ_USE_MSA": str(cfg.get("BOLTZ_USE_MSA", "False")),
        "BOLTZ_MSA_SERVER_URL": str(cfg.get("BOLTZ_MSA_SERVER_URL", "https://api.colabfold.com")),
    })

    from orchestrator.backends.boltz import call_boltz
    from orchestrator import tasks as _tasks

    # The pipeline does a Redis GET/SET around the whole run. There is no Redis in this
    # container and _get_redis() has no socket timeout, so a live lookup would hang.
    # Install a no-op client: guarantees no network and no cross-arm cache reuse.
    class _NoRedis:
        def get(self, *a, **k):
            return None
        def setex(self, *a, **k):
            return None
    _tasks._redis_client = _NoRedis()

    pdb_id, chain = target["pdb_id"], target["chain"]

    # --- Fetch experimental structure + sequence ---
    try:
        resp = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=30)
        resp.raise_for_status()
    except Exception as e:
        return {"pdb_id": pdb_id, "error": f"PDB fetch failed: {e}"}
    true_pdb = resp.text

    try:
        seq, chain = _extract_sequence(true_pdb, chain)
    except Exception as e:
        return {"pdb_id": pdb_id, "error": f"Sequence extraction failed: {e}"}
    if len(seq) < 10:
        return {"pdb_id": pdb_id, "error": f"Sequence too short ({len(seq)} aa)"}

    out = {"pdb_id": pdb_id, "name": target.get("name", pdb_id), "length": len(seq)}

    # --- Arm A: raw single Boltz (seed 0) ---
    try:
        a = call_boltz(seq, seed=0)
        out["A"] = {
            **score_structures(a.structure_pdb, true_pdb, chain),
            "plddt": round(a.mean_plddt, 2),
            "boltz_calls": 1,
        }
        out["_backend_version"] = a.backend_version
    except Exception as e:
        out["A"] = {"error": f"Boltz failed: {e}"}

    # --- Arm C: full pipeline (agent off) ---
    c_boltz_calls = 0
    try:
        res = _tasks._run_prediction_core(
            {"sequence": seq, "context": {}, "priority": "accurate"}
        )
        if res.get("status") != "completed":
            out["C"] = {"error": res.get("error_message", "pipeline did not complete")}
        else:
            ens = res["ensemble_result"]
            preds = res.get("predictions", [])
            # True Boltz cost = only the predictions the Boltz backend produced. The ensemble
            # also always holds one deterministic ESMFold structure (model_name esmfold /
            # esmfold_local) and, if enabled, stub-backend entries — none of which are Boltz
            # calls. total_seeds_tried counts ALL predictions, so it overstates Boltz work by
            # the ESMFold (+ stub) entries. Count model_name == "boltz2" instead.
            c_boltz_calls = sum(1 for p in preds if p.get("model_name") == "boltz2")
            out["C"] = {
                **score_structures(ens["structure_pdb"], true_pdb, chain),
                "plddt": round(ens["mean_plddt"], 2),
                "decision": res.get("post_processing", {}).get("decision"),
                "n_models": res.get("n_models_used"),
                "total_predictions": int(res.get("total_seeds_tried", len(preds))),
                "boltz_calls": c_boltz_calls,
                # Which backend's structure the best-by-pLDDT selector actually returned.
                # Recorded (not inferred from pLDDT) so a C win can be attributed cleanly:
                # "esmfold*" => the win is ensemble/model-diversity; "boltz2" => Boltz sampling.
                "winner_model": ens.get("model_name"),
            }
            # Arm E: ESMFold-only. The pipeline already folded ESMFold as part of the
            # ensemble, so we score THAT exact structure — zero extra compute. Isolates the
            # ensemble's active ingredient (model diversity) from Boltz sampling and from the
            # pLDDT selection C applies on top. C vs E then asks whether the selection helps
            # or hurts relative to just always taking ESMFold.
            esm = next(
                (p for p in preds if str(p.get("model_name", "")).startswith("esmfold")),
                None,
            )
            if esm:
                out["E"] = {
                    **score_structures(esm["structure_pdb"], true_pdb, chain),
                    "plddt": round(esm["mean_plddt"], 2),
                    "model": esm.get("model_name"),
                }
    except Exception as e:
        out["C"] = {"error": f"Pipeline failed: {e}"}

    # --- Arm B: compute-matched raw best-of-N (N = Boltz calls arm C actually made) ---
    # Seeds 1..N are disjoint from arm A's seed 0, so A and B never share a draw.
    if c_boltz_calls < 1:
        out["B"] = {"error": "pipeline made no Boltz calls to compute-match"}
    else:
        try:
            cands = [call_boltz(seq, seed=s) for s in range(1, c_boltz_calls + 1)]
            best = max(cands, key=lambda p: p.mean_plddt)
            out["B"] = {
                **score_structures(best.structure_pdb, true_pdb, chain),
                "plddt": round(best.mean_plddt, 2),
                "boltz_calls": c_boltz_calls,
            }
        except Exception as e:
            out["B"] = {"error": f"Boltz best-of-N failed: {e}"}

    return out


# ---------------------------------------------------------------------------
# Paired statistics — no scipy dependency required
# ---------------------------------------------------------------------------

def _sign_test_p(deltas):
    """Two-sided exact binomial sign test on paired deltas (zeros dropped).

    Used as a dependency-free fallback / cross-check for the Wilcoxon test. Tests
    H0: P(delta > 0) = 0.5, i.e. the two arms win equally often.
    """
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0, pos, neg
    k = min(pos, neg)
    # Two-sided: P(X <= k) + P(X >= n-k) under Binomial(n, 0.5)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = min(1.0, 2 * tail)
    return p, pos, neg


def _paired_summary(good, arm_x, arm_y, metric, higher_is_better):
    """Paired comparison of arm_x vs arm_y on one metric over targets where both scored."""
    deltas = []
    for r in good:
        rx, ry = r.get(arm_x, {}), r.get(arm_y, {})
        if metric in rx and metric in ry and "error" not in rx and "error" not in ry:
            deltas.append(rx[metric] - ry[metric])
    if not deltas:
        return None

    arr = np.array(deltas, dtype=float)
    # Orient "win" toward whichever direction is better for this metric.
    wins = int(np.sum(arr > 0)) if higher_is_better else int(np.sum(arr < 0))
    losses = int(np.sum(arr < 0)) if higher_is_better else int(np.sum(arr > 0))
    ties = int(np.sum(arr == 0))

    summary = {
        "n_paired": len(deltas),
        "mean_delta": round(float(arr.mean()), 4),
        "median_delta": round(float(np.median(arr)), 4),
        "wins": wins, "losses": losses, "ties": ties,
    }

    # Prefer Wilcoxon signed-rank; fall back to the exact sign test if scipy is absent.
    try:
        from scipy.stats import wilcoxon
        nonzero = arr[arr != 0]
        if len(nonzero) > 0:
            _, p = wilcoxon(nonzero)
            summary["p_value"] = round(float(p), 5)
            summary["test"] = "wilcoxon-signed-rank"
    except Exception:
        p, _, _ = _sign_test_p(deltas)
        summary["p_value"] = round(p, 5)
        summary["test"] = "sign-test (scipy unavailable)"
    return summary


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def run_pipeline_benchmark(
    source: str = "casp15",
    pdb_ids: str = "",
    max_targets: int = 20,
    ensemble_seeds: int = 3,
    out: str = "pipeline_vs_boltz_results.json",
    notes: str = "",
):
    """
    source:         casp15 | rcsb | custom  (default: casp15)
    pdb-ids:        comma-separated PDB IDs, implies source=custom
    max-targets:    cap the target count (default 20; the arms make this ~(2N+1)x
                    costlier than the single-backend benchmark)
    ensemble-seeds: Boltz seeds in arm C's initial ensemble (refinement may add more)
    out:            per-target results JSON
    notes:          free-text note recorded with the run

    Examples:
        modal run benchmark_pipeline_modal.py
        modal run benchmark_pipeline_modal.py --max-targets 5
        modal run benchmark_pipeline_modal.py --pdb-ids 1UBQ,1VII,1GB1
    """
    if pdb_ids:
        targets = [
            {"pdb_id": p.strip().upper(), "chain": "A", "name": p.strip().upper()}
            for p in pdb_ids.split(",")
        ]
        source = "custom"
    elif source == "rcsb":
        print("Querying RCSB for high-quality single-chain proteins deposited ≥ 2023...")
        targets = fetch_rcsb_quality_targets()
    else:
        print("Fetching CASP15 targets from RCSB...")
        targets = fetch_casp15_targets()

    if max_targets and max_targets < len(targets):
        targets = targets[:max_targets]

    cfg = {"ENSEMBLE_NUM_SEEDS": ensemble_seeds}
    for t in targets:
        t["_cfg"] = cfg

    print(
        f"\nPipeline-vs-Boltz benchmark on {len(targets)} targets "
        f"(agent OFF, ensemble_seeds={ensemble_seeds})\n"
        f"  A = raw single Boltz | B = compute-matched best-of-N Boltz\n"
        f"  C = full pipeline    | E = ESMFold-only (the pipeline's own ESMFold structure)\n"
    )

    t0 = time.time()
    results = list(benchmark_pipeline_one.map(targets, return_exceptions=True))
    duration_seconds = time.time() - t0

    # --- Per-target table ---
    header = (
        f"{'PDB':>6}  {'Len':>4}  {'Bcalls':>6}  "
        f"{'TM_A':>7}  {'TM_B':>7}  {'TM_C':>7}  {'TM_E':>7}  {'ΔTM(C-B)':>9}  {'C_winner':>13}"
    )
    print(header)
    print("-" * len(header))

    good, errors = [], []
    for r in results:
        if isinstance(r, Exception):
            errors.append(str(r))
            print(f"  EXCEPTION: {r}")
            continue
        # A target is usable for paired stats only if every arm scored (E included — it is
        # derived from arm C's own ESMFold run, so it is present whenever C completed).
        arms_ok = all(
            k in r and "error" not in r.get(k, {}) and "tm_score" in r.get(k, {})
            for k in ("A", "B", "C", "E")
        )
        if not arms_ok:
            errs = {k: r[k].get("error") for k in ("A", "B", "C", "E")
                    if isinstance(r.get(k), dict) and "error" in r[k]}
            errors.append({"pdb_id": r.get("pdb_id"), "arm_errors": errs})
            print(f"{r.get('pdb_id',''):>6}  incomplete: {errs}")
            continue

        good.append(r)
        d_cb = r["C"]["tm_score"] - r["B"]["tm_score"]
        print(
            f"{r['pdb_id']:>6}  {r['length']:>4}  {r['C'].get('boltz_calls','?'):>6}  "
            f"{r['A']['tm_score']:>7.4f}  {r['B']['tm_score']:>7.4f}  {r['C']['tm_score']:>7.4f}  "
            f"{r['E']['tm_score']:>7.4f}  {d_cb:>+9.4f}  {str(r['C'].get('winner_model','')):>13}"
        )

    # --- Paired analysis ---
    analysis = {}
    if good:
        # winner_model tally for arm C — the direct attribution of where C's structures came from.
        winners = {}
        for r in good:
            w = r["C"].get("winner_model", "?")
            winners[w] = winners.get(w, 0) + 1

        analysis = {
            "n_paired": len(good),
            "tm": {
                # Pipeline vs the two raw-Boltz baselines.
                "C_vs_A": _paired_summary(good, "C", "A", "tm_score", higher_is_better=True),
                "C_vs_B": _paired_summary(good, "C", "B", "tm_score", higher_is_better=True),
                # ESMFold-only vs the same baselines — how much of C's edge is just ESMFold.
                "E_vs_A": _paired_summary(good, "E", "A", "tm_score", higher_is_better=True),
                "E_vs_B": _paired_summary(good, "E", "B", "tm_score", higher_is_better=True),
                # Does C's pLDDT selection beat always-take-ESMFold? >0 => selection helps.
                "C_vs_E": _paired_summary(good, "C", "E", "tm_score", higher_is_better=True),
            },
            "rmsd": {
                "C_vs_A": _paired_summary(good, "C", "A", "rmsd", higher_is_better=False),
                "C_vs_B": _paired_summary(good, "C", "B", "rmsd", higher_is_better=False),
                "C_vs_E": _paired_summary(good, "C", "E", "rmsd", higher_is_better=False),
            },
            "boltz_calls": {
                "A": sum(r["A"].get("boltz_calls", 1) for r in good),
                "B": sum(r["B"].get("boltz_calls", 0) for r in good),
                "C": sum(r["C"].get("boltz_calls", 0) for r in good),
            },
            "C_winner_models": winners,
        }

        def _fmt(label, s):
            if not s:
                return f"  {label}: no paired data"
            return (
                f"  {label}: mean Δ={s['mean_delta']:+.4f}  median Δ={s['median_delta']:+.4f}  "
                f"W/L/T={s['wins']}/{s['losses']}/{s['ties']}  "
                f"p={s.get('p_value','?')} ({s.get('test','')})"
            )

        print(
            f"\n  Paired on {len(good)} targets "
            f"({len(errors)} excluded for an incomplete/failed arm)\n"
            f"  Arm C winners: {winners}\n"
            f"\n  TM-score (higher better):\n"
            f"{_fmt('C vs A  (pipeline vs raw single)', analysis['tm']['C_vs_A'])}\n"
            f"{_fmt('C vs B  (pipeline vs compute-matched)', analysis['tm']['C_vs_B'])}\n"
            f"{_fmt('E vs B  (ESMFold-only vs compute-matched)', analysis['tm']['E_vs_B'])}\n"
            f"{_fmt('C vs E  (pLDDT selection vs always-ESMFold)', analysis['tm']['C_vs_E'])}\n"
            f"\n  RMSD Å (lower better):\n"
            f"{_fmt('C vs A', analysis['rmsd']['C_vs_A'])}\n"
            f"{_fmt('C vs B', analysis['rmsd']['C_vs_B'])}\n"
            f"\n  Boltz calls total — A:{analysis['boltz_calls']['A']}  "
            f"B:{analysis['boltz_calls']['B']}  C:{analysis['boltz_calls']['C']}\n"
        )

    # --- Persist ---
    with open(out, "w") as f:
        json.dump(
            {"targets": results, "errors": errors, "analysis": analysis},
            f, indent=2, default=str,
        )
    print(f"  Per-target results saved to {out}")

    # Backend build reported from inside the workers (see benchmark_modal for rationale).
    reported = {
        r.get("_backend_version") for r in results
        if isinstance(r, dict) and r.get("_backend_version")
    }
    if len(reported) > 1:
        backend_build = "MIXED:" + ",".join(sorted(reported))
        print(f"  WARNING: targets ran on differing backend builds: {sorted(reported)}")
    else:
        backend_build = next(iter(reported), None)

    # Kept OUT of benchmarks/results.jsonl on purpose: that file is single-backend and
    # auto-numbered by line count (log_benchmark._next_run_id). This is a paired A/B/C
    # record with a different schema, so it gets its own append-only log.
    summary_row = {
        "timestamp": None,  # stamped below (Modal entrypoint has a real clock)
        "harness": "pipeline_vs_boltz",
        "source": source,
        "agent_enabled": False,
        "ensemble_seeds": ensemble_seeds,
        "backend_build": backend_build,
        "duration_seconds": round(duration_seconds, 1),
        "n_targets": len(targets),
        "n_paired": len(good),
        "notes": notes,
        "analysis": analysis,
    }
    import datetime as _dt
    summary_row["timestamp"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    with open("benchmarks/pipeline_vs_boltz.jsonl", "a") as f:
        f.write(json.dumps(summary_row, default=str) + "\n")
    print("  Summary row appended to benchmarks/pipeline_vs_boltz.jsonl")
