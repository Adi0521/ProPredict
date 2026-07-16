"""
Benchmark: ProteinMPNN checkpoint sweep for mutation-effect scoring quality.

Reruns the Task-2 ProteinGym validation-gate methodology across ProteinMPNN noise-level
checkpoints (v_48_002 / v_48_010 / v_48_020 / v_48_030) to answer master-plan open
question #1: which checkpoint best correlates with real deep-mutational-scanning (DMS)
fitness on THIS pipeline's own predicted structures.

Metric: Spearman correlation between the ProteinMPNN structural log-odds score and the
experimental DMS_score, per assay, per checkpoint. This is mutation-effect RANKING
quality — NOT structure prediction (see benchmark_modal.py for that; a different axis).

Method (per assay):
  1. Fold target_seq ONCE via ESMFold local — the structure is checkpoint-independent,
     so we never re-fold per checkpoint.
  2. For each checkpoint: run ProteinMPNN --conditional_probs_only -> log_p [L, 21],
     score every DMS mutant additively (sum over ':'-split single subs of
     log_p[to] - log_p[from]), then Spearman vs DMS_score.

Higher (more positive) log-odds = ProteinMPNN considers the substitution more
structurally compatible than the wild-type. A positive Spearman means that ordering
tracks the measured fitness. NOTE: this is a structural-compatibility signal, expected
to work best on stability assays and worse on activity assays.

Data (ProteinGym, Notin et al. 2023/2024) — auto-downloaded once into
benchmarks/proteingym_cache/ (gitignored):
  s3://proteingym/DMS_substitutions.parquet (~89 MB, all 217 assays combined,
  --no-sign-request). Columns used: DMS_id, mutant, DMS_score, target_seq.

Run (ProPredict conda env; needs PROTEINMPNN_PATH + ESMFOLD_LOCAL=True in .env):
    python -m benchmarks.benchmark_proteinmpnn_checkpoints
    python -m benchmarks.benchmark_proteinmpnn_checkpoints --checkpoints v_48_020,v_48_030
    python -m benchmarks.benchmark_proteinmpnn_checkpoints --assays CCDB_ECOLI_Adkar_2012
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import PROTEINMPNN_PATH
from orchestrator.backends.esmfold import call_esmfold_local
from orchestrator.mutation_scan import _ALPHABET, _run_proteinmpnn_conditional_probs


CACHE_DIR = os.path.join(os.path.dirname(__file__), "proteingym_cache")
PARQUET_S3 = "s3://proteingym/DMS_substitutions.parquet"

# Same three assays as the Task-2 validation gate: two Stability (best case for a
# structural scorer) + one Activity (contrast category — tests generalization).
DEFAULT_ASSAYS = [
    "TCRG1_MOUSE_Tsuboyama_2023_1E0L",   # Stability, 37 aa
    "ESTA_BACSU_Nutschel_2020",          # Stability, 212 aa
    "CCDB_ECOLI_Adkar_2012",             # Activity, 101 aa
]
# ProteinMPNN vanilla checkpoints, in ascending training-noise order.
DEFAULT_CHECKPOINTS = ["v_48_002", "v_48_010", "v_48_020", "v_48_030"]


def _ensure_parquet() -> str:
    """Download the combined ProteinGym substitutions parquet once, then cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "DMS_substitutions.parquet")
    if not os.path.isfile(path):
        print(f"Downloading ProteinGym parquet (~89 MB) -> {path}")
        subprocess.run(
            ["aws", "s3", "cp", "--no-sign-request", PARQUET_S3, path],
            check=True,
        )
    return path


def _score_mutant(mutant: str, log_p: np.ndarray, target_seq: str) -> Optional[float]:
    """
    Additive ProteinMPNN log-odds over ':'-separated single substitutions, e.g.
    "A14D" or "A14D:G20S". Position is 1-indexed against target_seq.

    Returns None (row skipped) if any sub is malformed, out of range, references a
    non-standard residue, or its wild-type letter does not match target_seq at that
    position (an alignment mismatch — better to skip than score garbage).
    """
    total = 0.0
    for sub in mutant.split(":"):
        sub = sub.strip()
        if len(sub) < 3:
            return None
        wt, mut = sub[0], sub[-1]
        try:
            pos = int(sub[1:-1])
        except ValueError:
            return None
        idx = pos - 1
        if idx < 0 or idx >= len(target_seq) or idx >= log_p.shape[0]:
            return None
        if target_seq[idx] != wt:
            return None
        if wt not in _ALPHABET or mut not in _ALPHABET:
            return None
        total += float(log_p[idx, _ALPHABET.index(mut)] - log_p[idx, _ALPHABET.index(wt)])
    return total


def benchmark_assay(dms_id: str, checkpoints: List[str], parquet_path: str) -> Optional[Dict]:
    df = pd.read_parquet(
        parquet_path, columns=["DMS_id", "mutant", "DMS_score", "target_seq"]
    )
    df = df[df["DMS_id"] == dms_id]
    if df.empty:
        print(f"[warn] no rows for {dms_id} — skipping")
        return None

    target_seq = df["target_seq"].iloc[0]
    mutants = df["mutant"].tolist()
    dms_scores = df["DMS_score"].tolist()
    print(f"\n=== {dms_id}  (L={len(target_seq)}, {len(df)} mutants) ===")

    # Fold ONCE — the predicted structure is identical for every checkpoint.
    print("  Folding target_seq via ESMFold local (one-time, checkpoint-independent)...")
    pred = call_esmfold_local(target_seq, seed=0)
    print(f"  Folded: mean pLDDT {pred.mean_plddt:.1f}")

    per_ckpt: Dict[str, Dict] = {}
    for ckpt in checkpoints:
        with tempfile.TemporaryDirectory() as td:
            log_p = _run_proteinmpnn_conditional_probs(
                pred.structure_pdb, td, PROTEINMPNN_PATH, model_name=ckpt
            )
        scores, truths, skipped = [], [], 0
        for mutant, dms in zip(mutants, dms_scores):
            s = _score_mutant(mutant, log_p, target_seq)
            if s is None or dms is None or (isinstance(dms, float) and np.isnan(dms)):
                skipped += 1
                continue
            scores.append(s)
            truths.append(float(dms))
        if len(scores) > 2:
            rho, pval = spearmanr(scores, truths)
        else:
            rho, pval = float("nan"), float("nan")
        per_ckpt[ckpt] = {
            "spearman": round(float(rho), 4),
            "p_value": float(pval),
            "n_scored": len(scores),
            "n_skipped": skipped,
        }
        print(f"  {ckpt}: rho={rho:+.4f}  (n={len(scores)}, skipped={skipped})")

    return {
        "dms_id": dms_id,
        "seq_len": len(target_seq),
        "mean_plddt": round(pred.mean_plddt, 2),
        "checkpoints": per_ckpt,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", default=",".join(DEFAULT_CHECKPOINTS),
                    help="comma-separated ProteinMPNN checkpoints (default: all four)")
    ap.add_argument("--assays", default=",".join(DEFAULT_ASSAYS),
                    help="comma-separated ProteinGym DMS_ids")
    ap.add_argument("--out",
                    default=os.path.join(os.path.dirname(__file__),
                                         "proteinmpnn_checkpoint_results.json"))
    args = ap.parse_args()

    if not PROTEINMPNN_PATH:
        sys.exit("ERROR: PROTEINMPNN_PATH not set — point it at your ProteinMPNN clone in .env")

    checkpoints = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    assays = [a.strip() for a in args.assays.split(",") if a.strip()]

    parquet_path = _ensure_parquet()

    results = []
    for dms_id in assays:
        r = benchmark_assay(dms_id, checkpoints, parquet_path)
        if r:
            results.append(r)

    # Summary table
    print("\n" + "=" * 72)
    print("SUMMARY — Spearman rho vs DMS_score (ESMFold-local predicted structures)")
    print("=" * 72)
    header = f"{'assay':<34}" + "".join(f"{c:>10}" for c in checkpoints)
    print(header)
    print("-" * len(header))
    for r in results:
        row = f"{r['dms_id']:<34}" + "".join(
            f"{r['checkpoints'][c]['spearman']:>+10.4f}" for c in checkpoints
        )
        print(row)
    if results:
        print("-" * len(header))
        means = [
            sum(r["checkpoints"][c]["spearman"] for r in results) / len(results)
            for c in checkpoints
        ]
        print(f"{'MEAN':<34}" + "".join(f"{m:>+10.4f}" for m in means))
        best_idx = int(np.argmax(means))
        print(f"\nBest mean checkpoint: {checkpoints[best_idx]} (mean rho {means[best_idx]:+.4f})")

    with open(args.out, "w") as f:
        json.dump({"checkpoints": checkpoints, "assays": results}, f, indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
