"""
Fitness oracles for combinatorial / multi-site mutation search (AdaLead).

Two tiers of signal for a candidate multi-mutant, from cheap to expensive:

  additive_oracle    — sum of single-site ProteinMPNN structural log-odds from ONE
                       WT-structure conditional-probs pass. ~free, but assumes site
                       independence: it CANNOT see epistasis (the optimum of a sum of
                       independent per-site terms is just the best substitution at each
                       site — no search required). Use for fast seeding / pre-filtering.

  score_only_oracle  — thread each mutant sequence onto the WT backbone and run
                       ProteinMPNN `--score_only --path_to_fasta`, reading the per-sequence
                       `global_score` (mean negative log-likelihood of the whole
                       structure-sequence). Epistasis-aware (autoregressive sequence
                       context) and BATCHABLE: one model load scores an entire round's
                       candidate fasta. This is the real search-loop oracle.

A third tier — re-folding the top candidates and scoring by pLDDT/clashes/affinity — is
wired later in the two-stage funnel; it is not in this module.

Fitness convention: HIGHER = better for every oracle. ProteinMPNN's `global_score` is a
mean NLL (lower = better fit), so score_only_oracle returns its NEGATION.

Reproducibility mirrors mutation_scan.py: scores are the mean over `num_decoding_orders`
random decoding orders at a FIXED, non-zero seed. seed=0 is rejected — ProteinMPNN's CLI
does `if args.seed:` and treats 0 as unset, randomizing the seed.

Standalone module (params, not config.py wiring). The AdaLead search that consumes these
oracles is built on top in a later task.
"""
import glob
import logging
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np

# Reuse mutation_scan's verified alphabet and conditional-probs runner so the two modules
# can never drift (same [L,21] ordering, same seed guard, same decoding-order averaging).
from orchestrator.mutation_scan import (
    _ALPHABET,
    _STANDARD_AA,
    _run_proteinmpnn_conditional_probs,
)

logger = logging.getLogger(__name__)

# "A12V" -> ("A", 12, "V"): <wt-aa><1-indexed position><mut-aa>
_MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


# ---------------------------------------------------------------------------
# Mutation representation helpers (shared by both oracles and, later, AdaLead)
# ---------------------------------------------------------------------------

def parse_mutation(mutation: str) -> Tuple[str, int, str]:
    """'A12V' -> ('A', 12, 'V'). Position is 1-indexed. Raises ValueError on bad format."""
    m = _MUTATION_RE.match(mutation)
    if not m:
        raise ValueError(
            f"malformed mutation {mutation!r}: expected <wt-aa><1-indexed-pos><mut-aa>, "
            "e.g. 'A12V'."
        )
    wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
    if wt not in _STANDARD_AA or mut not in _STANDARD_AA:
        raise ValueError(
            f"mutation {mutation!r} uses a non-standard amino acid (allowed: {_STANDARD_AA})."
        )
    if pos < 1:
        raise ValueError(f"mutation {mutation!r} has non-positive position {pos}.")
    return wt, pos, mut


def format_mutation(wt: str, pos: int, mut: str) -> str:
    """('A', 12, 'V') -> 'A12V'."""
    return f"{wt}{pos}{mut}"


def apply_mutations(sequence: str, mutations: List[str]) -> str:
    """
    Apply a list of substitutions to `sequence`, returning the mutant sequence.

    Validates that each mutation's stated wild-type residue matches `sequence` at that
    position — a mismatch means the mutation list and sequence are out of sync (e.g. a
    stale sequence upstream), which is raised rather than silently mis-mutated.
    """
    chars = list(sequence)
    for mutation in mutations:
        wt, pos, mut = parse_mutation(mutation)
        idx = pos - 1
        if idx >= len(chars):
            raise ValueError(
                f"mutation {mutation!r} position {pos} is past the end of a "
                f"{len(chars)}-residue sequence."
            )
        if chars[idx] != wt:
            raise ValueError(
                f"mutation {mutation!r} expects {wt} at position {pos} but sequence has "
                f"{chars[idx]}. Sequence and mutation list are out of sync."
            )
        chars[idx] = mut
    return "".join(chars)


def mutations_from_sequences(wild_type: str, mutant: str) -> List[str]:
    """
    Diff two equal-length sequences into a list of 'A12V'-style mutation strings, ordered
    by position. Used to describe an AdaLead candidate (a raw sequence) in mutation terms.
    """
    if len(wild_type) != len(mutant):
        raise ValueError(
            f"length mismatch: wild_type is {len(wild_type)}, mutant is {len(mutant)}. "
            "Mutation search only produces substitutions (equal length)."
        )
    return [
        format_mutation(wt, i + 1, mt)
        for i, (wt, mt) in enumerate(zip(wild_type, mutant))
        if wt != mt
    ]


# ---------------------------------------------------------------------------
# Tier 1 — additive oracle (pure; seeding / pre-filter only)
# ---------------------------------------------------------------------------

def additive_oracle(log_p: np.ndarray, mutations: List[str]) -> float:
    """
    Additive fitness of a multi-mutant: sum over its substitutions of the single-site
    structural log-odds  log_p[pos, mut] - log_p[pos, wt].

    Same math as mutation_scan.score_candidate_mutations, summed across sites. `log_p` is
    the [L, 21] mean log-prob matrix from _run_proteinmpnn_conditional_probs on the WT
    structure. Pure and independent-by-construction: it CANNOT capture epistasis — that is
    exactly why it is a seed/pre-filter, not the search-loop oracle. Higher = better.
    """
    total = 0.0
    for mutation in mutations:
        wt, pos, mut = parse_mutation(mutation)
        idx = pos - 1
        if idx < 0 or idx >= log_p.shape[0]:
            raise ValueError(
                f"mutation {mutation!r} position {pos} is out of range for a "
                f"{log_p.shape[0]}-residue log_p."
            )
        total += float(log_p[idx, _ALPHABET.index(mut)] - log_p[idx, _ALPHABET.index(wt)])
    return total


# ---------------------------------------------------------------------------
# Tier 2 — score_only oracle (epistasis-aware; batched; the search-loop oracle)
# ---------------------------------------------------------------------------

def _run_proteinmpnn_score_only(
    pdb_string: str,
    sequences: List[str],
    tmpdir: str,
    proteinmpnn_dir: str,
    model_name: str = "v_48_020",
    seed: int = 37,
    num_decoding_orders: int = 8,
) -> List[float]:
    """
    Run ProteinMPNN `--score_only 1 --path_to_fasta` once over a fasta of `sequences`
    (threaded onto the single-chain WT backbone in `pdb_string`) and return the per-sequence
    mean `global_score` (mean NLL over `num_decoding_orders` random decoding orders),
    aligned to input order.

    One model load scores the whole fasta: ProteinMPNN loops over the sequences
    (protein_mpnn_run.py:244) writing '{name}_fasta_{N}.npz' for the N-th (1-indexed) fasta
    sequence, each holding a `global_score` array over the decoding-order samples.
    """
    if seed == 0:
        raise ValueError(
            "seed=0 is unusable: ProteinMPNN's `if args.seed:` check treats 0 as unset and "
            "randomizes the seed, making scores non-reproducible. Use any non-zero int."
        )

    pdb_path = os.path.join(tmpdir, "structure.pdb")
    with open(pdb_path, "w") as fh:
        fh.write(pdb_string)

    # ProteinMPNN threads each fasta sequence into S[:, :len(seq)]; a length mismatch would
    # silently score a chimera (short) or crash (long). Require equal, structure-sized seqs.
    fasta_path = os.path.join(tmpdir, "candidates.fasta")
    with open(fasta_path, "w") as fh:
        for i, seq in enumerate(sequences):
            fh.write(f">cand_{i}\n{seq}\n")

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
        "--score_only", "1",
        "--path_to_fasta", fasta_path,
        "--num_seq_per_target", str(num_decoding_orders),
        "--seed", str(seed),
        "--batch_size", "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=proteinmpnn_dir)
    if result.returncode != 0:
        raise RuntimeError(
            f"ProteinMPNN score_only failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout[-800:]}\nstderr: {result.stderr[-800:]}"
        )

    # Outputs: score_only/{name}_fasta_{N}.npz, N=1-indexed fasta order. Native ({name}_pdb)
    # is skipped. Map the trailing integer back to the 0-indexed input position.
    score_dir = os.path.join(out_dir, "score_only")
    fasta_npz = glob.glob(os.path.join(score_dir, "*_fasta_*.npz"))
    scores: List[Optional[float]] = [None] * len(sequences)
    for path in fasta_npz:
        m = re.search(r"_fasta_(\d+)\.npz$", os.path.basename(path))
        if not m:
            continue
        input_idx = int(m.group(1)) - 1
        if 0 <= input_idx < len(scores):
            scores[input_idx] = float(np.load(path)["global_score"].mean())

    missing = [i for i, s in enumerate(scores) if s is None]
    if missing:
        raise RuntimeError(
            f"ProteinMPNN score_only produced no output for candidate indices {missing} "
            f"(expected {len(sequences)} '*_fasta_*.npz' files in {score_dir})."
        )
    return [s for s in scores]  # type: ignore[misc]  # all non-None (checked above)


def score_only_oracle(
    pdb_string: str,
    sequences: List[str],
    proteinmpnn_dir: str = "",
    model_name: str = "v_48_020",
    seed: int = 37,
    num_decoding_orders: int = 8,
) -> List[float]:
    """
    Epistasis-aware fitness for a batch of mutant `sequences`, threaded onto the WT backbone
    in `pdb_string`. Returns one fitness per input sequence, aligned to input order, where
    HIGHER = better (the negated ProteinMPNN `global_score` mean NLL).

    Batched: a single ProteinMPNN model load scores the whole list — call it once per
    AdaLead round with that round's candidates, not once per candidate.

    All sequences must be the same length as the WT backbone (mutation search produces only
    substitutions); ProteinMPNN threads each into S[:, :len(seq)], so a wrong length would
    silently score a chimera or crash.
    """
    if not proteinmpnn_dir:
        raise RuntimeError(
            "proteinmpnn_dir not provided. Set PROTEINMPNN_PATH in .env to a clone of "
            "https://github.com/dauparas/ProteinMPNN (weights are included in the clone)."
        )
    if not sequences:
        return []
    lengths = {len(s) for s in sequences}
    if len(lengths) != 1:
        raise ValueError(
            f"score_only_oracle requires equal-length sequences (got lengths {sorted(lengths)}); "
            "mutation search only produces substitutions."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        global_scores = _run_proteinmpnn_score_only(
            pdb_string, sequences, tmpdir, proteinmpnn_dir, model_name,
            seed=seed, num_decoding_orders=num_decoding_orders,
        )
    # global_score is a mean NLL (lower = better fit); negate so higher = better everywhere.
    return [-g for g in global_scores]
