"""
Tests for orchestrator/mutation_search.py (fitness oracles for multi-site mutation search).

Mirrors tests/test_mutation_scan.py: the additive oracle and mutation helpers are pure and
tested directly against synthetic arrays; the score_only oracle is mocked at the subprocess
boundary — the fake subprocess.run writes ProteinMPNN-shaped '*_fasta_N.npz' files so the
output-parsing / ordering logic is exercised without weights or a GPU.
"""
import os
from unittest.mock import patch

import numpy as np
import pytest

from orchestrator.mutation_search import (
    additive_oracle,
    apply_mutations,
    format_mutation,
    mutations_from_sequences,
    parse_mutation,
    score_only_oracle,
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
