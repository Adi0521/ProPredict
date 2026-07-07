"""
Structure-aware mutation scoring via ProteinMPNN (Dauparas et al., 2022, Science;
MIT license, https://github.com/dauparas/ProteinMPNN).

Scores candidate single-point substitutions using the log-likelihood-ratio method
validated in HERMES (Visani et al.) and related zero-shot mutation-effect literature:

    score(pos, wt -> mut) = log P(mut | backbone, rest of sequence)
                           - log P(wt  | backbone, rest of sequence)

Positive score = ProteinMPNN considers the substitution more structurally compatible
than the wild-type at this position. This is a STRUCTURAL COMPATIBILITY score, not a
direct proxy for function, stability, or fitness — say so in any output/UI that
surfaces these numbers.

Standalone module: takes proteinmpnn_dir / model_name as parameters, not wired to the
agent loop or config.py. Run manually via:
    python -m orchestrator.mutation_scan --pdb myprotein.pdb --sequence MKT...
"""
import logging
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Verified against orchestrator's cloned ProteinMPNN source, protein_mpnn_utils.py:50
_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
_STANDARD_AA = _ALPHABET[:20]  # exclude X (unknown) from candidate substitutions


def _run_proteinmpnn_conditional_probs(
    pdb_string: str,
    tmpdir: str,
    proteinmpnn_dir: str,
    model_name: str = "v_48_020",
) -> np.ndarray:
    """
    Run ProteinMPNN --conditional_probs_only on a single-chain PDB and return the
    [L, 21] log-probability matrix (first/only batch element already sliced out).
    """
    pdb_path = os.path.join(tmpdir, "structure.pdb")
    with open(pdb_path, "w") as fh:
        fh.write(pdb_string)

    out_dir = os.path.join(tmpdir, "mpnn_out")
    os.makedirs(out_dir, exist_ok=True)

    run_script = os.path.join(proteinmpnn_dir, "protein_mpnn_run.py")
    weights_dir = os.path.join(proteinmpnn_dir, "vanilla_model_weights")

    if not os.path.isfile(run_script):
        raise RuntimeError(
            f"protein_mpnn_run.py not found at {run_script}. Check PROTEINMPNN_PATH "
            "points at a clone of https://github.com/dauparas/ProteinMPNN."
        )

    cmd = [
        sys.executable, run_script,
        "--pdb_path", pdb_path,
        "--out_folder", out_dir,
        "--path_to_model_weights", weights_dir,
        "--model_name", model_name,
        "--conditional_probs_only", "1",
        "--num_seq_per_target", "1",
        "--seed", "0",
        "--batch_size", "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=proteinmpnn_dir)
    if result.returncode != 0:
        raise RuntimeError(
            f"ProteinMPNN failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout[-800:]}\nstderr: {result.stderr[-800:]}"
        )

    npz_dir = os.path.join(out_dir, "conditional_probs_only")
    npz_files = [f for f in os.listdir(npz_dir) if f.endswith(".npz")] if os.path.isdir(npz_dir) else []
    if not npz_files:
        raise RuntimeError(f"ProteinMPNN completed but no .npz output found in {npz_dir}")

    data = np.load(os.path.join(npz_dir, npz_files[0]))
    return data["log_p"][0]  # [L, 21] — first (only) batch element


def score_candidate_mutations(
    pdb_string: str,
    sequence: str,
    positions: Optional[List[int]] = None,
    top_k: int = 10,
    proteinmpnn_dir: str = "",
    model_name: str = "v_48_020",
) -> List[Dict[str, Any]]:
    """
    Score candidate single-point substitutions using ProteinMPNN structural log-odds.

    Parameters
    ----------
    pdb_string : the current predicted structure (PDB format)
    sequence : the current sequence (must match the structure's residue count/order)
    positions : 1-indexed residue positions to consider; None = scan every position
    top_k : return at most this many candidates, sorted by score descending
    proteinmpnn_dir : path to a clone of https://github.com/dauparas/ProteinMPNN
    model_name : one of v_48_002 / v_48_010 / v_48_020 / v_48_030
                 (lower noise = higher native-sequence recovery; higher noise =
                 designs that fold more reliably when re-predicted — v_48_020 is
                 ProteinMPNN's own default)

    Returns
    -------
    List of dicts, sorted by score descending:
        {"position": int, "from_aa": str, "to_aa": str, "score": float}
    """
    if not proteinmpnn_dir:
        raise RuntimeError(
            "proteinmpnn_dir not provided. Set PROTEINMPNN_PATH in .env to a clone of "
            "https://github.com/dauparas/ProteinMPNN (weights are included in the clone)."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        log_p = _run_proteinmpnn_conditional_probs(pdb_string, tmpdir, proteinmpnn_dir, model_name)

    candidates: List[Dict[str, Any]] = []
    scan_positions = positions if positions is not None else range(1, len(sequence) + 1)
    for pos in scan_positions:
        idx = pos - 1
        if idx < 0 or idx >= len(sequence) or idx >= log_p.shape[0]:
            # Out-of-range usually means a stale sequence/structure pair upstream —
            # skip rather than crash, but surface it so the mismatch isn't silent.
            logger.warning(
                "skipping position %d: out of range for sequence length %d / log_p length %d",
                pos, len(sequence), log_p.shape[0],
            )
            continue
        wt_aa = sequence[idx]
        if wt_aa not in _STANDARD_AA:
            continue
        wt_idx = _ALPHABET.index(wt_aa)
        for mut_idx, mut_aa in enumerate(_STANDARD_AA):
            if mut_aa == wt_aa:
                continue
            score = float(log_p[idx, mut_idx] - log_p[idx, wt_idx])
            candidates.append({
                "position": pos, "from_aa": wt_aa, "to_aa": mut_aa,
                "score": round(score, 4),
            })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:top_k]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True, help="Path to a PDB file")
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--proteinmpnn-dir", default=os.getenv("PROTEINMPNN_PATH", ""))
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()
    with open(args.pdb) as fh:
        pdb_str = fh.read()
    results = score_candidate_mutations(
        pdb_str, args.sequence, top_k=args.top_k, proteinmpnn_dir=args.proteinmpnn_dir
    )
    for r in results:
        print(f"{r['from_aa']}{r['position']}{r['to_aa']}: {r['score']:+.4f}")
