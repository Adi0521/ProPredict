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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from models.schemas import MutationCandidate, MutationSearchResult, StructurePrediction
# count_clashes lives in scoring.py, which only imports config + schemas at module level
# (BioPython is lazy-imported inside it) — safe to import here without pulling heavy deps.
from orchestrator.scoring import count_clashes

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


# ---------------------------------------------------------------------------
# AdaLead-lite — hand-rolled combinatorial search over an oracle
# ---------------------------------------------------------------------------
#
# This is AdaLead-INSPIRED, not textbook AdaLead. Two deliberate deviations, both
# documented here because both are load-bearing for the epistasis result:
#
#   (1) No greedy rollout. Textbook AdaLead grows each candidate by mutating one residue
#       at a time and keeping the step only if it improves — scored against a cheap
#       SURROGATE MODEL (`model.get_fitness`), which is what makes a query-per-step
#       affordable. This search is MODEL-FREE: ProteinMPNN is simultaneously the scorer and
#       the thing being optimized, so a rollout would necessarily hit the expensive oracle
#       once per mutation step. Dropping it is therefore the honest consequence of having no
#       surrogate, not merely a subprocess-saving trick. The cost is sample efficiency: we
#       lose intra-round exploitation and lean harder on λ, rounds, and crossover to
#       assemble multi-site combos. Between-round elitism (the elite band re-seeds each
#       round's parent pool) recovers the "greedy-around-best" behavior across rounds — the
#       correct model-free approximation. What remains is essentially an elitist GA with
#       uniform crossover and capped random mutation.
#
#   (2) Range-normalized elite band. FLEXS thresholds the parent pool multiplicatively:
#       `fitness >= top * (1 - sign(top)*kappa)` (default kappa=0.05). That degenerates when
#       the max fitness is ~0 — the band collapses to [0, 0] and excludes every negative
#       sequence, so an epistatic pair whose two halves each score BELOW the current best
#       could never enter the recombination pool to be fused. We instead use a band
#       normalized by the observed fitness RANGE: `fitness >= max - kappa*(max - min)`, which
#       is sign-agnostic and keeps below-best singles eligible as parents. Because the knob
#       geometry differs, `kappa` here is NOT comparable to FLEXS's 0.05: kappa=0.5 means
#       "top half of the observed fitness range," a deliberately wide default, described on
#       its own terms.
#
# Determinism: every random draw goes through np.random.default_rng(seed) AND samples only
# from ORDERED structures (the insertion-ordered `measured` dict, or lists). Never sample by
# iterating a Python set — set iteration order is subject to hash randomization across
# processes, which would make "same seed -> same result" flake in a fresh interpreter.


def _elite_parents(measured: Dict[str, float], kappa: float) -> List[str]:
    """
    The recombination/mutation parent pool: sequences whose fitness lies in the top `kappa`
    fraction of the observed fitness RANGE (see deviation (2) above). Returned as an ordered
    list (from `measured`'s insertion order) so downstream sampling is reproducible.
    Guaranteed non-empty — the max-fitness sequence always qualifies.
    """
    fits = list(measured.values())
    hi, lo = max(fits), min(fits)
    cutoff = hi - kappa * (hi - lo)   # hi==lo -> cutoff==hi -> every seq qualifies
    return [seq for seq, fit in measured.items() if fit >= cutoff]


def _recombine(parent_a: str, parent_b: str, rng: np.random.Generator) -> str:
    """Uniform crossover: each position independently taken from one parent (0.5 each).
    recombine(WT, WT) == WT, so a singleton round-1 pool is safe."""
    return "".join(
        parent_a[i] if rng.random() < 0.5 else parent_b[i]
        for i in range(len(parent_a))
    )


def _mutate(sequence: str, n_mutations: int, rng: np.random.Generator) -> str:
    """Apply `n_mutations` random point substitutions (each to a residue different from the
    current one). Positions may repeat across draws; that only reduces the effective count,
    which is fine."""
    chars = list(sequence)
    length = len(chars)
    for _ in range(n_mutations):
        idx = int(rng.integers(length))
        choices = [aa for aa in _STANDARD_AA if aa != chars[idx]]
        chars[idx] = choices[int(rng.integers(len(choices)))]
    return "".join(chars)


def _enforce_k_cap(
    sequence: str, wild_type: str, max_sites: int, rng: np.random.Generator
) -> str:
    """Revert random excess mutations back to WT so at most `max_sites` positions differ.
    Reverting to WT (not to a parent) keeps the operation self-contained and unbiased."""
    diff = [i for i in range(len(sequence)) if sequence[i] != wild_type[i]]
    if len(diff) <= max_sites:
        return sequence
    n_revert = len(diff) - max_sites
    revert_idx = rng.choice(diff, size=n_revert, replace=False)  # diff is ordered -> reproducible
    chars = list(sequence)
    for i in revert_idx:
        chars[int(i)] = wild_type[int(i)]
    return "".join(chars)


def adalead_search(
    wild_type: str,
    oracle: Callable[[List[str]], List[float]],
    rounds: int,
    candidates_per_round: int,
    max_sites: int,
    seed: int,
    initial_sequences: Optional[List[str]] = None,
    oracle_name: str = "score_only",
    kappa: float = 0.5,
    mutations_per_child: int = 1,
    recombination_rate: float = 0.5,
    top_k: int = 10,
) -> MutationSearchResult:
    """
    AdaLead-inspired combinatorial mutation search (see the module-level note for the two
    deviations from textbook AdaLead and the determinism contract).

    Parameters
    ----------
    wild_type : the reference sequence; all candidates are substitutions of it (equal length)
    oracle : batched fitness function, `List[str] -> List[float]`, HIGHER = better. Called
             once per round on that round's new candidates (plus once on `initial_sequences`).
    rounds : number of search rounds (oracle is called `1 + rounds` times total)
    candidates_per_round : λ — max NEW candidates proposed & evaluated per round
    max_sites : k-cap on simultaneous mutations per candidate
    seed : RNG seed; identical (inputs, seed) -> identical result
    initial_sequences : starting measured set (default [wild_type])
    oracle_name : label recorded on each returned candidate ("additive"/"score_only"/"refold")
    kappa : elite-band width as a fraction of the observed fitness RANGE (0.5 = top half)
    mutations_per_child : random point substitutions added per generated child
    recombination_rate : probability a child is bred by crossover of two parents (else it is
             bred by mutating a single parent)
    top_k : number of ranked candidates to return

    Returns
    -------
    MutationSearchResult with candidates ranked best-first (WT itself excluded), the total
    number of distinct sequences scored, and rounds run. `refolds_used` stays 0 — tier-3
    re-fold validation is wired in a later task.
    """
    rng = np.random.default_rng(seed)
    length = len(wild_type)

    seeds = initial_sequences if initial_sequences is not None else [wild_type]
    seeds = list(dict.fromkeys(seeds))  # de-dup, preserve order
    # measured is insertion-ordered (dict) so all later sampling from it is reproducible.
    measured: Dict[str, float] = {}
    seed_fits = oracle(seeds)
    for s, f in zip(seeds, seed_fits):
        measured[s] = f

    max_attempts = candidates_per_round * 25  # cap generation so dedup starvation can't spin
    for _ in range(rounds):
        parents = _elite_parents(measured, kappa)
        batch: List[str] = []
        seen_this_round: set = set()
        attempts = 0
        while len(batch) < candidates_per_round and attempts < max_attempts:
            attempts += 1
            if recombination_rate > 0 and rng.random() < recombination_rate:
                a = parents[int(rng.integers(len(parents)))]
                b = parents[int(rng.integers(len(parents)))]
                child = _recombine(a, b, rng)
            else:
                child = parents[int(rng.integers(len(parents)))]
            child = _mutate(child, mutations_per_child, rng)
            child = _enforce_k_cap(child, wild_type, max_sites, rng)
            if child == wild_type or child in measured or child in seen_this_round:
                continue
            seen_this_round.add(child)
            batch.append(child)

        if not batch:
            continue  # pool exhausted under the k-cap; nothing new to score this round
        fits = oracle(batch)
        for s, f in zip(batch, fits):
            measured[s] = f

    # Rank all scored mutants (exclude WT), best-first; tie-break by sequence for determinism.
    ranked = sorted(
        ((s, f) for s, f in measured.items() if s != wild_type),
        key=lambda sf: (-sf[1], sf[0]),
    )
    candidates = [
        MutationCandidate(
            mutations=mutations_from_sequences(wild_type, s),
            sequence=s,
            score=round(f, 4),
            oracle=oracle_name,
        )
        for s, f in ranked[:top_k]
    ]
    return MutationSearchResult(
        wild_type_sequence=wild_type,
        candidates=candidates,
        oracle=oracle_name,
        rounds=rounds,
        total_evaluated=len(measured),  # distinct sequences scored, incl. seeds and WT
        refolds_used=0,
    )


# ---------------------------------------------------------------------------
# Tier 3 — re-fold validation funnel (connects the cheap search to the real pipeline)
# ---------------------------------------------------------------------------
#
# The cheap tier-2 score_only search ranks by ProteinMPNN structural compatibility, which is
# NOT a fold/stability/fitness guarantee. The funnel re-folds the top handful of candidates
# through the real prediction backend (ESMFold/Boltz) and re-ranks them by a metric grounded
# in the actual predicted structure: `mean_plddt - 5*num_clashes` — the SAME formula
# scoring.compute_post_processing uses for accept/refine/escalate, so the funnel and the main
# pipeline agree on what "structurally good" means.
#
# Affinity is deliberately NOT a ranking input. Boltz-2's affinity head may be blind to point
# mutations (open question — research_plan/rowA-boltz-affinity-invariance.md), so ranking
# mutants by it would be unsound today. Both affinity_score (log10 IC50 uM) and
# affinity_probability are RECORDED on the candidate as metadata for later analysis.
#
# Re-folds are the only expensive step (minutes/candidate), so the count is hard-capped by
# MUTATION_SEARCH_MAX_REFOLDS at the call site.

RefoldFn = Callable[[str, Optional[Dict[str, Any]], int], StructurePrediction]


def _default_refold_fn(
    sequence: str, context: Optional[Dict[str, Any]], seed: int
) -> StructurePrediction:
    """Re-fold via the same backend choice as agent.apply_mutation: Boltz when enabled
    (carries ligand/membrane context + affinity), else ESMFold. Backends are imported lazily
    so this module never pulls torch/boltz at import time (optional-dep convention)."""
    from config import BOLTZ_ENABLED
    if BOLTZ_ENABLED:
        from orchestrator.backends.boltz import call_boltz
        return call_boltz(sequence, context=context, seed=seed)
    from orchestrator.backends.esmfold import call_esmfold_api
    return call_esmfold_api(sequence, seed=seed)


def _refold_one(cand: MutationCandidate, pred: StructurePrediction) -> MutationCandidate:
    """Attach re-fold metrics to a candidate. refold_score = plddt - 5*clashes (higher =
    better), matching scoring.compute_post_processing. Affinity fields are metadata only."""
    clashes = count_clashes(pred.structure_pdb)
    refold_score = pred.mean_plddt - 5.0 * clashes
    return cand.model_copy(update={
        "refold_plddt": round(pred.mean_plddt, 2),
        "refold_num_clashes": clashes,
        "refold_score": round(refold_score, 2),
        "refold_affinity": pred.affinity_score,
        "refold_affinity_probability": pred.affinity_probability,
    })


def refold_validate(
    wild_type: str,
    candidates: List[MutationCandidate],
    context: Optional[Dict[str, Any]],
    max_refolds: int,
    refold_fn: RefoldFn,
    seed: int = 0,
) -> Tuple[List[MutationCandidate], int]:
    """
    Re-fold the top `max_refolds` cheap-oracle candidates and re-rank them by predicted
    structural quality (refold_score = plddt - 5*clashes, higher = better).

    Returns `(reordered_candidates, refolds_used)`:
      - re-folded candidates (with refold_* populated) sorted best-first, then the remaining
        un-refolded candidates in their original cheap-oracle order;
      - `refolds_used` = number of SUCCESSFUL re-folds (<= max_refolds).

    A re-fold that raises (backend hiccup) is logged and skipped — that candidate stays in the
    un-refolded remainder rather than crashing the funnel, mirroring apply_mutation's
    "re-prediction failed" handling.
    """
    n = max(0, min(max_refolds, len(candidates)))
    validated: List[MutationCandidate] = []
    for cand in candidates[:n]:
        try:
            pred = refold_fn(cand.sequence, context, seed)
        except Exception as e:  # noqa: BLE001 — a bad re-fold must not sink the whole funnel
            logger.warning("re-fold failed for %s (%s) — skipping: %s",
                           cand.mutations, cand.sequence, e)
            continue
        validated.append(_refold_one(cand, pred))

    validated.sort(key=lambda c: c.refold_score, reverse=True)
    refolded_seqs = {c.sequence for c in validated}
    remainder = [c for c in candidates if c.sequence not in refolded_seqs]
    return validated + remainder, len(validated)


def search_and_validate(
    wild_type: str,
    oracle: Callable[[List[str]], List[float]],
    *,
    rounds: int,
    candidates_per_round: int,
    max_sites: int,
    seed: int,
    max_refolds: int = 0,
    refold_fn: Optional[RefoldFn] = None,
    context: Optional[Dict[str, Any]] = None,
    oracle_name: str = "score_only",
    **adalead_kwargs: Any,
) -> MutationSearchResult:
    """
    Two-stage funnel: cheap AdaLead-lite search, then (if `max_refolds > 0`) re-fold the top
    candidates through the real backend and re-rank by predicted structure quality.

    `refold_fn` defaults to `_default_refold_fn` (Boltz/ESMFold by flag); inject a stub for
    tests. When `max_refolds == 0` the funnel is a no-op and the cheap result is returned
    unchanged (refolds_used stays 0).
    """
    result = adalead_search(
        wild_type, oracle, rounds=rounds, candidates_per_round=candidates_per_round,
        max_sites=max_sites, seed=seed, oracle_name=oracle_name, **adalead_kwargs,
    )
    if max_refolds <= 0:
        return result

    validated, refolds_used = refold_validate(
        wild_type, result.candidates, context, max_refolds,
        refold_fn or _default_refold_fn, seed,
    )
    return result.model_copy(update={"candidates": validated, "refolds_used": refolds_used})
