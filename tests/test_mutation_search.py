"""
Tests for orchestrator/mutation_search.py (fitness oracles for multi-site mutation search).

Mirrors tests/test_mutation_scan.py: the additive oracle and mutation helpers are pure and
tested directly against synthetic arrays; the score_only oracle is mocked at the subprocess
boundary — the fake subprocess.run writes ProteinMPNN-shaped '*_fasta_N.npz' files so the
output-parsing / ordering logic is exercised without weights or a GPU.

The AdaLead-lite search (adalead_search) is tested against an in-memory oracle — no
ProteinMPNN — on a planted HIDDEN-SYNERGY landscape (see `_landscape` below). The headline
test is the ablation: the search fuses an epistatic pair that naive top-N single-site
stacking provably cannot. Because the search is stochastic, epistasis is asserted over a
FIXED set of seeds with a safe majority margin (deterministic: fixed seeds -> fixed count),
while the k-cap, per-round budget, and reproducibility properties get hard assertions.
"""
import os
from unittest.mock import patch

import numpy as np
import pytest

from models.schemas import MutationCandidate, StructurePrediction
from orchestrator.mutation_search import (
    adalead_search,
    additive_oracle,
    apply_mutations,
    format_mutation,
    mutations_from_sequences,
    parse_mutation,
    refold_validate,
    score_only_oracle,
    search_and_validate,
    _run_proteinmpnn_score_only,
)


# ---------------------------------------------------------------------------
# Mutation representation helpers
# ---------------------------------------------------------------------------

def test_parse_mutation_roundtrip():
    assert parse_mutation("A12V") == ("A", 12, "V")
    assert format_mutation("A", 12, "V") == "A12V"


@pytest.mark.parametrize("bad", ["", "12V", "AV", "A0V", "AXV", "A12B", "a12v"])
def test_parse_mutation_rejects_malformed(bad):
    # B and X are non-standard AAs; A0V is a non-positive position; lowercase is invalid.
    with pytest.raises(ValueError):
        parse_mutation(bad)


def test_apply_mutations_applies_in_place():
    assert apply_mutations("ACDEF", ["A1V", "F5G"]) == "VCDEG"


def test_apply_mutations_rejects_wt_mismatch():
    # Sequence has C at position 1, mutation claims A -> out of sync.
    with pytest.raises(ValueError, match="out of sync"):
        apply_mutations("CDEF", ["A1V"])


def test_apply_mutations_rejects_out_of_range():
    with pytest.raises(ValueError, match="past the end"):
        apply_mutations("AC", ["A9V"])


def test_mutations_from_sequences_diffs_by_position():
    assert mutations_from_sequences("ACDEF", "VCDEG") == ["A1V", "F5G"]
    assert mutations_from_sequences("ACDEF", "ACDEF") == []


def test_mutations_from_sequences_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        mutations_from_sequences("ACD", "ACDE")


# ---------------------------------------------------------------------------
# additive_oracle — pure, against a synthetic log_p
# ---------------------------------------------------------------------------

def _synthetic_log_p() -> np.ndarray:
    """
    [2, 21] log-prob matrix for sequence "AC" (same construction as test_mutation_scan).
    Alphabet ACDEFGHIKLMNPQRSTVWYX: A=0 C=1 D=2 E=3 G=5. Baseline -3.0.
      pos1 (wt A): D=-1.0 -> +2.0
      pos2 (wt C): G=-0.5 -> +2.5
    """
    lp = np.full((2, 21), -3.0)
    lp[0, 2] = -1.0   # A1D -> +2.0
    lp[1, 5] = -0.5   # C2G -> +2.5
    return lp


def test_additive_oracle_sums_single_site_logodds():
    lp = _synthetic_log_p()
    assert additive_oracle(lp, ["A1D"]) == pytest.approx(2.0)
    assert additive_oracle(lp, ["C2G"]) == pytest.approx(2.5)
    # Additivity: the pair is exactly the sum of the singles (no epistasis, by construction).
    assert additive_oracle(lp, ["A1D", "C2G"]) == pytest.approx(4.5)


def test_additive_oracle_empty_is_zero():
    assert additive_oracle(_synthetic_log_p(), []) == 0.0


def test_additive_oracle_position_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        additive_oracle(_synthetic_log_p(), ["A9D"])


# ---------------------------------------------------------------------------
# score_only_oracle — mocked at the subprocess boundary
# ---------------------------------------------------------------------------

def _make_fake_srun(global_scores_by_index, decoding_orders=3):
    """
    Build a fake subprocess.run that writes ProteinMPNN score_only output: for input index i
    (0-based), a '<name>_fasta_{i+1}.npz' with a `global_score` array whose mean equals
    global_scores_by_index[i]. The array is [decoding_orders] identical values so .mean()
    is exact.
    """
    def fake_srun(cmd, *args, **kwargs):
        out_folder = cmd[cmd.index("--out_folder") + 1]
        score_dir = os.path.join(out_folder, "score_only")
        os.makedirs(score_dir, exist_ok=True)
        for i, g in enumerate(global_scores_by_index):
            arr = np.full(decoding_orders, g, dtype=float)
            np.savez(os.path.join(score_dir, f"structure_fasta_{i + 1}.npz"), global_score=arr)

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""
        return _R()
    return fake_srun


@patch("orchestrator.mutation_search.subprocess.run")
def test_score_only_negates_and_preserves_order(mock_srun, tmp_path):
    (tmp_path / "protein_mpnn_run.py").write_text("# stub")
    # global_score is a mean NLL (lower = better); oracle returns the negation (higher = better).
    mock_srun.side_effect = _make_fake_srun([1.0, 0.5, 2.0])
    seqs = ["AAAA", "AACA", "AAGA"]
    out = score_only_oracle("PDBSTR", seqs, proteinmpnn_dir=str(tmp_path))
    assert out == pytest.approx([-1.0, -0.5, -2.0])
    # Ordering preserved: best (highest) fitness is the 0.5-NLL candidate at index 1.
    assert out.index(max(out)) == 1


@patch("orchestrator.mutation_search.subprocess.run")
def test_score_only_averages_over_decoding_orders(mock_srun, tmp_path):
    (tmp_path / "protein_mpnn_run.py").write_text("# stub")

    def fake_srun(cmd, *a, **k):
        out_folder = cmd[cmd.index("--out_folder") + 1]
        score_dir = os.path.join(out_folder, "score_only")
        os.makedirs(score_dir, exist_ok=True)
        # Three decoding-order samples averaging to 2.0 -> oracle returns -2.0.
        np.savez(os.path.join(score_dir, "structure_fasta_1.npz"),
                 global_score=np.array([1.0, 2.0, 3.0]))

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""
        return _R()

    mock_srun.side_effect = fake_srun
    out = score_only_oracle("PDBSTR", ["ACDE"], proteinmpnn_dir=str(tmp_path))
    assert out == pytest.approx([-2.0])


@patch("orchestrator.mutation_search.subprocess.run")
def test_score_only_passes_score_only_flags(mock_srun, tmp_path):
    (tmp_path / "protein_mpnn_run.py").write_text("# stub")
    mock_srun.side_effect = _make_fake_srun([1.0])
    score_only_oracle("PDBSTR", ["ACDE"], proteinmpnn_dir=str(tmp_path),
                      seed=37, num_decoding_orders=3)
    cmd = mock_srun.call_args[0][0]
    assert "--score_only" in cmd and cmd[cmd.index("--score_only") + 1] == "1"
    assert "--path_to_fasta" in cmd
    assert cmd[cmd.index("--seed") + 1] == "37"
    assert cmd[cmd.index("--num_seq_per_target") + 1] == "3"


@patch("orchestrator.mutation_search.subprocess.run")
def test_score_only_missing_output_raises(mock_srun, tmp_path):
    (tmp_path / "protein_mpnn_run.py").write_text("# stub")
    # Writes only fasta_1; index 1 (second candidate) missing -> error names it.
    mock_srun.side_effect = _make_fake_srun([1.0])
    with pytest.raises(RuntimeError, match=r"no output for candidate indices \[1\]"):
        score_only_oracle("PDBSTR", ["ACDE", "GCDE"], proteinmpnn_dir=str(tmp_path))


def test_score_only_empty_sequences_shortcircuits():
    # Returns [] without needing a proteinmpnn_dir subprocess.
    assert score_only_oracle("PDBSTR", [], proteinmpnn_dir="/fake") == []


def test_score_only_unequal_lengths_raises():
    with pytest.raises(ValueError, match="equal-length"):
        score_only_oracle("PDBSTR", ["ACDE", "ACD"], proteinmpnn_dir="/fake")


def test_score_only_missing_dir_raises():
    with pytest.raises(RuntimeError, match="proteinmpnn_dir not provided"):
        score_only_oracle("PDBSTR", ["ACDE"], proteinmpnn_dir="")


def test_score_only_seed_zero_raises(tmp_path):
    (tmp_path / "protein_mpnn_run.py").write_text("# stub")
    with pytest.raises(ValueError, match="seed=0"):
        _run_proteinmpnn_score_only("PDBSTR", ["ACDE"], str(tmp_path), str(tmp_path), seed=0)


def test_score_only_missing_run_script_raises(tmp_path):
    # tmp_path has no protein_mpnn_run.py -> isfile() is False, raised before any subprocess.
    with pytest.raises(RuntimeError, match="protein_mpnn_run.py not found"):
        _run_proteinmpnn_score_only("PDBSTR", ["ACDE"], str(tmp_path), str(tmp_path))


# ---------------------------------------------------------------------------
# adalead_search — against an in-memory planted landscape (no ProteinMPNN)
# ---------------------------------------------------------------------------

_WT = "AAAAAAAA"  # 8-residue wild type, fitness 0


def _landscape(seq: str) -> float:
    """
    Planted HIDDEN-SYNERGY landscape (fair + winnable, unlike a deceptive needle):
      pos1->C (+1.0), pos8->C (+1.0): the BEST singles, purely additive -> the decoys a
        naive top-N single-site stacker picks.
      pos3->D (+0.6), pos6->K (+0.6): individually DECENT (so they stay in the elite band)
        but not top-ranked; TOGETHER an extra +3.0 synergy bonus.
      any other substitution: -0.5
    The epistatic pair D3+K6 (+4.2) strictly beats any additive stack of the best singles
    (naive k=3 stack = +2.6); its components rank below the decoys, so naive stacking never
    fuses them. Global optimum under k=3 is D3+K6+one decoy = +5.2.
    """
    f = 0.0
    for i, (w, c) in enumerate(zip(_WT, seq)):
        if w == c:
            continue
        if i == 0 and c == "C":
            f += 1.0
        elif i == 7 and c == "C":
            f += 1.0
        elif i == 2 and c == "D":
            f += 0.6
        elif i == 5 and c == "K":
            f += 0.6
        else:
            f += -0.5
    if seq[2] == "D" and seq[5] == "K":
        f += 3.0
    return f


def _oracle(seqs):
    return [_landscape(s) for s in seqs]


# Naive top-N single-site stacker: the strawman the epistasis-aware search must beat. It
# scores every single mutation, greedily stacks the best distinct-position ones up to the
# k-cap, and evaluates the combination — the standard additive strategy.
_NAIVE_STACK_FITNESS = 2.6  # C1 + C8 + one decent single; never fuses D3+K6


def _has_epistatic_pair(candidate) -> bool:
    return candidate.sequence[2] == "D" and candidate.sequence[5] == "K"


def test_adalead_fuses_epistatic_pair_representative_seed():
    # Seed 0 is representative (7/8 of seeds 0..7 succeed), not lucky: it lands the global
    # optimum D3+K6+decoy and beats the naive additive stack.
    res = adalead_search(_WT, _oracle, rounds=25, candidates_per_round=30, max_sites=3,
                         seed=0, oracle_name="score_only")
    top = res.candidates[0]
    assert _has_epistatic_pair(top), f"expected D3+K6 fused, got {top.mutations}"
    assert top.score > _NAIVE_STACK_FITNESS
    assert set(top.mutations) >= {"A3D", "A6K"}
    assert res.oracle == "score_only" and res.rounds == 25 and res.refolds_used == 0


def test_adalead_beats_naive_stacking_over_fixed_seeds():
    # Deterministic (fixed seeds -> fixed count). Empirically 7/8 here; assert a safe
    # majority so a future tweak that makes it knife-edge trips this test.
    found = 0
    best_overall = float("-inf")
    for s in range(8):
        res = adalead_search(_WT, _oracle, rounds=25, candidates_per_round=30, max_sites=3,
                             seed=s, oracle_name="score_only")
        top = res.candidates[0]
        found += _has_epistatic_pair(top)
        best_overall = max(best_overall, top.score)
    assert found >= 6, f"epistatic pair found in only {found}/8 seeds — search regressed"
    assert best_overall > _NAIVE_STACK_FITNESS


def test_adalead_respects_k_cap():
    # No candidate may exceed max_sites mutations, across every returned candidate.
    res = adalead_search(_WT, _oracle, rounds=20, candidates_per_round=25, max_sites=2,
                         seed=0, top_k=25)
    assert res.candidates, "expected some candidates"
    assert all(len(c.mutations) <= 2 for c in res.candidates)


def test_adalead_respects_per_round_budget():
    # One batched oracle call per round (+1 for the initial seeds), each batch <= lambda.
    calls = []

    def counting_oracle(seqs):
        calls.append(len(seqs))
        return _oracle(seqs)

    rounds, lam = 10, 15
    adalead_search(_WT, counting_oracle, rounds=rounds, candidates_per_round=lam,
                   max_sites=3, seed=0)
    assert len(calls) == 1 + rounds            # initial seeds + one per round
    assert calls[0] == 1                        # default initial_sequences == [wild_type]
    assert all(n <= lam for n in calls[1:]), f"a round exceeded lambda={lam}: {calls}"


def test_adalead_is_deterministic():
    kw = dict(rounds=15, candidates_per_round=20, max_sites=3, seed=7, oracle_name="t")
    r1 = adalead_search(_WT, _oracle, **kw)
    r2 = adalead_search(_WT, _oracle, **kw)
    assert [(c.sequence, c.score) for c in r1.candidates] == \
           [(c.sequence, c.score) for c in r2.candidates]


def test_adalead_excludes_wild_type_from_candidates():
    res = adalead_search(_WT, _oracle, rounds=10, candidates_per_round=15, max_sites=3, seed=0)
    assert all(c.sequence != _WT for c in res.candidates)
    assert all(c.mutations for c in res.candidates)  # every candidate has >=1 mutation


def test_adalead_additive_landscape_stacks_to_k_cap():
    # Sanity: on a DENSE additive landscape (any mutation at positions 1-3 is +1.0, elsewhere
    # -0.5) the search cleanly stacks k-cap-many beneficial mutations. Dense so discovery
    # isn't itself a needle hunt — this checks stacking, not exploration. (A landscape with
    # two SPECIFIC point optima is actually harder here: without a synergy gradient pulling
    # them together, each must be independently discovered among 19*L substitutions.)
    def additive(seq):
        f = 0.0
        for i, (w, c) in enumerate(zip(_WT, seq)):
            if w == c:
                continue
            f += 1.0 if i in (0, 1, 2) else -0.5
        return f

    res = adalead_search(_WT, lambda ss: [additive(s) for s in ss],
                         rounds=15, candidates_per_round=20, max_sites=3, seed=0)
    top = res.candidates[0]
    positions = {int(m[1:-1]) for m in top.mutations}
    assert len(top.mutations) == 3 and positions <= {1, 2, 3}
    assert top.score == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# refold_validate / search_and_validate — the tier-3 re-fold funnel (mocked backend)
# ---------------------------------------------------------------------------

def _cand(muts, score):
    """A cheap-oracle candidate (no re-fold metrics yet)."""
    seq = apply_mutations(_WT, muts)
    return MutationCandidate(mutations=muts, sequence=seq, score=score, oracle="score_only")


# Minimal single-CA PDB: real count_clashes parses it and returns 0 (needs >=2 CAs to clash).
_MINIMAL_PDB = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"


def _pred(plddt, structure_pdb=_MINIMAL_PDB, affinity=None, affinity_prob=None):
    return StructurePrediction(
        structure_pdb=structure_pdb, plddt_scores=[plddt], mean_plddt=plddt, seed=0,
        affinity_score=affinity, affinity_probability=affinity_prob,
    )


def _refold_fn_from_map(plddt_by_seq, affinity_by_seq=None):
    """Build a stub refold_fn that returns a controlled pLDDT (and optional affinity) per
    sequence. structure_pdb is left empty so real count_clashes returns 0."""
    def fn(sequence, context, seed):
        aff = (affinity_by_seq or {}).get(sequence)
        prob = None if affinity_by_seq is None else affinity_by_seq.get(sequence)
        return _pred(plddt_by_seq[sequence], affinity=aff, affinity_prob=prob)
    return fn


def test_refold_validate_budget_cap():
    # 6 candidates, budget 3 -> exactly 3 re-folds, 3 refold_score set, 3 left None.
    cands = [_cand([f"A{i}C"], score=10 - i) for i in range(1, 7)]
    plddt = {c.sequence: 80.0 for c in cands}
    calls = []

    def counting_fn(sequence, context, seed):
        calls.append(sequence)
        return _pred(plddt[sequence])

    reordered, used = refold_validate(_WT, cands, {}, max_refolds=3, refold_fn=counting_fn)
    assert used == 3 and len(calls) == 3
    assert sum(c.refold_score is not None for c in reordered) == 3
    assert sum(c.refold_score is None for c in reordered) == 3


def test_refold_validate_reranks_by_structural_score():
    # Cheap top-1 (best score) folds poorly; a lower-ranked cheap candidate folds best and
    # must float to the front after validation.
    a = _cand(["A1C"], score=9.0)   # cheap #1 but low pLDDT
    b = _cand(["A2D"], score=8.0)
    c = _cand(["A3E"], score=7.0)   # cheap #3 but highest pLDDT
    fn = _refold_fn_from_map({a.sequence: 55.0, b.sequence: 70.0, c.sequence: 90.0})
    reordered, used = refold_validate(_WT, [a, b, c], {}, max_refolds=3, refold_fn=fn)
    assert used == 3
    assert reordered[0].sequence == c.sequence          # highest pLDDT wins
    assert reordered[0].refold_plddt == 90.0
    assert [r.refold_score for r in reordered] == sorted(
        [r.refold_score for r in reordered], reverse=True)


def test_refold_validate_clash_penalty(monkeypatch):
    # High pLDDT but many clashes must lose to a slightly-lower-pLDDT clean structure.
    # refold_score = plddt - 5*clashes: 95 - 5*4 = 75  <  80 - 5*0 = 80.
    a = _cand(["A1C"], score=9.0)
    b = _cand(["A2D"], score=8.0)
    plddt = {a.sequence: 95.0, b.sequence: 80.0}
    clashes = {a.sequence: 4, b.sequence: 0}

    def fn(sequence, context, seed):
        # encode clash count in the pdb string; the patched count_clashes decodes it
        return _pred(plddt[sequence], structure_pdb=sequence)

    monkeypatch.setattr("orchestrator.mutation_search.count_clashes",
                        lambda pdb: clashes[pdb])
    reordered, _ = refold_validate(_WT, [a, b], {}, max_refolds=2, refold_fn=fn)
    assert reordered[0].sequence == b.sequence          # clean structure wins
    assert reordered[0].refold_score == 80.0
    assert reordered[1].refold_score == 75.0


def test_refold_validate_affinity_recorded_not_ranked():
    # The candidate with the best (lowest) affinity but a worse structure must NOT rank first;
    # affinity is recorded as metadata only.
    a = _cand(["A1C"], score=9.0)   # great affinity, poor fold
    b = _cand(["A2D"], score=8.0)   # worse affinity, better fold
    fn = _refold_fn_from_map(
        {a.sequence: 60.0, b.sequence: 85.0},
        affinity_by_seq={a.sequence: -3.0, b.sequence: 1.0},  # lower affinity = "tighter"
    )
    reordered, _ = refold_validate(_WT, [a, b], {}, max_refolds=2, refold_fn=fn)
    assert reordered[0].sequence == b.sequence          # structure ranks, not affinity
    assert reordered[0].refold_affinity == 1.0          # affinity still recorded
    assert reordered[1].refold_affinity == -3.0


def test_refold_validate_failure_skipped():
    # One candidate's re-fold raises -> skipped, stays in the un-refolded remainder, and
    # refolds_used counts only successes.
    a = _cand(["A1C"], score=9.0)
    b = _cand(["A2D"], score=8.0)

    def fn(sequence, context, seed):
        if sequence == a.sequence:
            raise RuntimeError("backend blew up")
        return _pred(75.0)

    reordered, used = refold_validate(_WT, [a, b], {}, max_refolds=2, refold_fn=fn)
    assert used == 1
    validated = [c for c in reordered if c.refold_score is not None]
    assert len(validated) == 1 and validated[0].sequence == b.sequence
    # a survived un-refolded
    assert any(c.sequence == a.sequence and c.refold_score is None for c in reordered)


def test_refold_validate_more_budget_than_candidates():
    a = _cand(["A1C"], score=9.0)
    fn = _refold_fn_from_map({a.sequence: 80.0})
    reordered, used = refold_validate(_WT, [a], {}, max_refolds=5, refold_fn=fn)
    assert used == 1 and len(reordered) == 1


def test_search_and_validate_zero_budget_is_noop():
    res = search_and_validate(_WT, _oracle, rounds=10, candidates_per_round=15, max_sites=3,
                              seed=0, max_refolds=0)
    assert res.refolds_used == 0
    assert all(c.refold_score is None for c in res.candidates)


def test_search_and_validate_end_to_end():
    # Cheap search then funnel with a stub backend: refolds_used > 0 and the top candidate
    # carries re-fold metrics.
    search = adalead_search(_WT, _oracle, rounds=15, candidates_per_round=30, max_sites=3,
                            seed=0)
    plddt = {c.sequence: 70.0 + i for i, c in enumerate(search.candidates)}

    def fn(sequence, context, seed):
        return _pred(plddt.get(sequence, 70.0))

    res = search_and_validate(_WT, _oracle, rounds=15, candidates_per_round=30, max_sites=3,
                              seed=0, max_refolds=3, refold_fn=fn)
    assert res.refolds_used == 3
    assert res.candidates[0].refold_score is not None
    assert res.oracle == "score_only"  # top-level oracle still names the SEARCH oracle
