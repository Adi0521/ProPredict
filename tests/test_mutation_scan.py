"""
Tests for orchestrator/mutation_scan.py (ProteinMPNN structural mutation scoring).

Unit tests mock the ProteinMPNN subprocess entirely (synthetic log_p arrays,
patched subprocess.run) — no real weights or binary required. The integration test
at the bottom shells out to a real ProteinMPNN clone and is skipped unless
PROTEINMPNN_PATH (+ a test PDB/sequence) are provided. Mirrors tests/test_boltz.py.
"""
import logging
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from orchestrator.mutation_scan import (
    _run_proteinmpnn_conditional_probs,
    score_candidate_mutations,
)

_PATCH_TARGET = "orchestrator.mutation_scan._run_proteinmpnn_conditional_probs"


def _synthetic_log_p() -> np.ndarray:
    """
    Hand-constructed [2, 21] log-prob matrix for sequence "AC".

    Alphabet indices: A=0 C=1 D=2 E=3 G=5. Baseline -3.0 everywhere, so most
    substitutions score exactly 0.0 (log_p[mut] - log_p[wt] = -3 - -3). Bumped cells:
      pos1 (wt A, idx0): D=-1.0 -> score +2.0 ; E=-2.0 -> score +1.0
      pos2 (wt C, idx1): G=-0.5 -> score +2.5
    """
    lp = np.full((2, 21), -3.0)
    lp[0, 2] = -1.0   # A1D -> +2.0
    lp[0, 3] = -2.0   # A1E -> +1.0
    lp[1, 5] = -0.5   # C2G -> +2.5
    return lp


# ---------------------------------------------------------------------------
# score_candidate_mutations — formula, sort, truncation, filtering
# ---------------------------------------------------------------------------

@patch(_PATCH_TARGET)
def test_scores_formula_and_sort_order(mock_run):
    mock_run.return_value = _synthetic_log_p()
    res = score_candidate_mutations("PDBSTR", "AC", top_k=3, proteinmpnn_dir="/fake")
    assert res[0] == {"position": 2, "from_aa": "C", "to_aa": "G", "score": 2.5}
    assert res[1] == {"position": 1, "from_aa": "A", "to_aa": "D", "score": 2.0}
    assert res[2] == {"position": 1, "from_aa": "A", "to_aa": "E", "score": 1.0}
    assert len(res) == 3


@patch(_PATCH_TARGET)
def test_top_k_truncation(mock_run):
    mock_run.return_value = _synthetic_log_p()
    res = score_candidate_mutations("PDBSTR", "AC", top_k=2, proteinmpnn_dir="/fake")
    assert len(res) == 2
    assert [c["score"] for c in res] == [2.5, 2.0]


@patch(_PATCH_TARGET)
def test_positions_filter_restricts_scan(mock_run):
    mock_run.return_value = _synthetic_log_p()
    res = score_candidate_mutations(
        "PDBSTR", "AC", positions=[2], top_k=50, proteinmpnn_dir="/fake"
    )
    assert all(c["position"] == 2 and c["from_aa"] == "C" for c in res)
    assert res[0] == {"position": 2, "from_aa": "C", "to_aa": "G", "score": 2.5}
    assert len(res) == 19  # 20 standard AA minus the wild-type C


@patch(_PATCH_TARGET)
def test_out_of_range_position_skipped_with_warning(mock_run, caplog):
    mock_run.return_value = _synthetic_log_p()
    with caplog.at_level(logging.WARNING):
        res = score_candidate_mutations(
            "PDBSTR", "AC", positions=[5], proteinmpnn_dir="/fake"
        )
    assert res == []
    assert "out of range" in caplog.text


def test_missing_proteinmpnn_dir_raises_before_subprocess():
    # Empty dir must fail early — no patching of the runner needed.
    with pytest.raises(RuntimeError, match="proteinmpnn_dir not provided"):
        score_candidate_mutations("PDBSTR", "AC", proteinmpnn_dir="")


# ---------------------------------------------------------------------------
# _run_proteinmpnn_conditional_probs — subprocess / filesystem error paths
# ---------------------------------------------------------------------------

def test_missing_run_script_raises(tmp_path):
    # tmp_path has no protein_mpnn_run.py -> isfile() is False.
    with pytest.raises(RuntimeError, match="protein_mpnn_run.py not found"):
        _run_proteinmpnn_conditional_probs("PDBSTR", str(tmp_path), str(tmp_path))


@patch("orchestrator.mutation_scan.subprocess.run")
def test_subprocess_nonzero_exit_raises_with_output(mock_srun, tmp_path):
    (tmp_path / "protein_mpnn_run.py").write_text("# stub")
    mock_srun.return_value = MagicMock(returncode=1, stdout="boom-out", stderr="boom-err")
    with pytest.raises(RuntimeError) as ei:
        _run_proteinmpnn_conditional_probs("PDBSTR", str(tmp_path), str(tmp_path))
    msg = str(ei.value)
    assert "ProteinMPNN failed" in msg
    assert "boom-out" in msg and "boom-err" in msg


@patch("orchestrator.mutation_scan.subprocess.run")
def test_missing_npz_output_raises(mock_srun, tmp_path):
    (tmp_path / "protein_mpnn_run.py").write_text("# stub")
    # Exit 0 but the runner (mocked) never writes the conditional_probs_only/*.npz.
    mock_srun.return_value = MagicMock(returncode=0, stdout="", stderr="")
    with pytest.raises(RuntimeError, match="no .npz output"):
        _run_proteinmpnn_conditional_probs("PDBSTR", str(tmp_path), str(tmp_path))


# ---------------------------------------------------------------------------
# Integration — real ProteinMPNN clone (skipped unless explicitly configured)
# ---------------------------------------------------------------------------

def test_score_candidate_mutations_integration():
    """
    End-to-end against a real ProteinMPNN clone. Drift-catcher: fails if a future
    ProteinMPNN changes its CLI flags or .npz output layout.

    Set PROTEINMPNN_PATH (clone dir), MPNN_TEST_PDB (a PDB file), and MPNN_TEST_SEQ
    (its 1-letter sequence) to run.
    """
    mpnn_dir = os.getenv("PROTEINMPNN_PATH", "")
    pdb_file = os.getenv("MPNN_TEST_PDB", "")
    seq = os.getenv("MPNN_TEST_SEQ", "")
    if not (
        mpnn_dir
        and os.path.isfile(os.path.join(mpnn_dir, "protein_mpnn_run.py"))
        and pdb_file
        and os.path.isfile(pdb_file)
        and seq
    ):
        pytest.skip("set PROTEINMPNN_PATH, MPNN_TEST_PDB, MPNN_TEST_SEQ to run")

    with open(pdb_file) as fh:
        pdb_str = fh.read()
    res = score_candidate_mutations(pdb_str, seq, top_k=5, proteinmpnn_dir=mpnn_dir)
    assert isinstance(res, list) and len(res) <= 5
    for c in res:
        assert set(c) == {"position", "from_aa", "to_aa", "score"}
        assert isinstance(c["score"], float)
