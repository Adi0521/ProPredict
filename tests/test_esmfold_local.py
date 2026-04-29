"""
Tests for local ESMFold inference (Phase 1).

Fast tests mock the model/tokenizer; the integration test at the bottom
is skipped unless torch + transformers are installed AND a GPU/MPS is
available (too slow for CI on CPU with the full facebook/esmfold_v1 weights).
"""
import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ESMFold stores pLDDT as a 0–1 fraction in the B-factor column (positions 60–65).
# _parse_plddt_from_pdb multiplies by 100, so 0.85 → 85.0.
SAMPLE_PDB = (
    "ATOM      1  N   MET A   1       1.000   2.000   3.000  1.00  0.85           N\n"
    "ATOM      2  CA  MET A   1       1.500   2.500   3.500  1.00  0.85           C\n"
    "ATOM      3  N   ALA A   2       4.000   5.000   6.000  1.00  0.72           N\n"
    "ATOM      4  CA  ALA A   2       4.500   5.500   6.500  1.00  0.72           C\n"
    "END\n"
)

EXPECTED_PLDDT = [85.0, 72.0]


# ---------------------------------------------------------------------------
# Unit: _parse_plddt_from_pdb
# ---------------------------------------------------------------------------

def test_parse_plddt_from_pdb():
    from orchestrator.tasks import _parse_plddt_from_pdb

    scores = _parse_plddt_from_pdb(SAMPLE_PDB)
    assert scores == EXPECTED_PLDDT


def test_parse_plddt_empty_pdb():
    from orchestrator.tasks import _parse_plddt_from_pdb

    assert _parse_plddt_from_pdb("") == []
    assert _parse_plddt_from_pdb("HETATM  1  C   LIG A   1\n") == []


# ---------------------------------------------------------------------------
# Unit: call_esmfold_local — mock model/tokenizer
# ---------------------------------------------------------------------------

def _make_mock_model(pdb_output: str):
    """Return a mock EsmForProteinFolding-like object."""
    model = MagicMock()
    model.parameters.return_value = iter([MagicMock(device=MagicMock(__str__=lambda s: "cpu"))])
    model.output_to_pdb.return_value = [pdb_output]
    model.return_value = MagicMock()  # output of model(**inputs)
    return model


def _mock_torch():
    """Return a MagicMock that stands in for the torch module."""
    mock = MagicMock()
    # no_grad() must work as a context manager; MagicMock().__enter__/exit__ do that.
    mock.no_grad.return_value.__enter__ = lambda s: None
    mock.no_grad.return_value.__exit__ = lambda s, *a: False
    return mock


def test_call_esmfold_local_returns_structure_prediction():
    from orchestrator.tasks import call_esmfold_local

    mock_model = _make_mock_model(SAMPLE_PDB)
    mock_tokenizer = MagicMock(return_value={"input_ids": MagicMock(to=lambda d: MagicMock())})

    with patch.dict(sys.modules, {"torch": _mock_torch()}), \
         patch("orchestrator.tasks._esmfold_model", mock_model), \
         patch("orchestrator.tasks._esmfold_tokenizer", mock_tokenizer):
        result = call_esmfold_local("MKTAYIAK", seed=0)

    assert result.model_name == "esmfold_local"
    assert result.plddt_scores == EXPECTED_PLDDT
    assert abs(result.mean_plddt - sum(EXPECTED_PLDDT) / len(EXPECTED_PLDDT)) < 1e-6
    assert result.structure_pdb == SAMPLE_PDB
    assert result.seed == 0


def test_call_esmfold_local_raises_on_empty_pdb():
    from orchestrator.tasks import call_esmfold_local

    mock_model = _make_mock_model("REMARK no atoms\n")
    mock_tokenizer = MagicMock(return_value={"input_ids": MagicMock(to=lambda d: MagicMock())})

    with patch.dict(sys.modules, {"torch": _mock_torch()}), \
         patch("orchestrator.tasks._esmfold_model", mock_model), \
         patch("orchestrator.tasks._esmfold_tokenizer", mock_tokenizer):
        with pytest.raises(ValueError, match="No CA atoms"):
            call_esmfold_local("MKTAYIAK")


# ---------------------------------------------------------------------------
# Unit: call_esmfold_api — dispatch routing
# ---------------------------------------------------------------------------

def test_dispatch_routes_to_local_when_flag_true():
    with patch("orchestrator.tasks.ESMFOLD_LOCAL", True), \
         patch("orchestrator.tasks.call_esmfold_local") as mock_local, \
         patch("orchestrator.tasks._call_esmfold_remote") as mock_remote:
        from orchestrator.tasks import call_esmfold_api
        call_esmfold_api("MKTAYIAK", seed=1)
        mock_local.assert_called_once_with("MKTAYIAK", 1)
        mock_remote.assert_not_called()


def test_dispatch_routes_to_remote_when_flag_false():
    with patch("orchestrator.tasks.ESMFOLD_LOCAL", False), \
         patch("orchestrator.tasks.call_esmfold_local") as mock_local, \
         patch("orchestrator.tasks._call_esmfold_remote") as mock_remote:
        from orchestrator.tasks import call_esmfold_api
        call_esmfold_api("MKTAYIAK", seed=0)
        mock_remote.assert_called_once_with("MKTAYIAK", 0)
        mock_local.assert_not_called()


# ---------------------------------------------------------------------------
# Integration test (skipped unless model weights are available)
# ---------------------------------------------------------------------------

def test_call_esmfold_local_integration():
    """
    End-to-end test against the real facebook/esmfold_v1 weights.
    Skipped unless transformers imports cleanly. First run downloads ~2 GB.
    """
    try:
        from transformers import EsmForProteinFolding  # noqa: F401
    except Exception:
        pytest.skip("transformers not importable (missing or incompatible huggingface_hub)")

    from orchestrator.tasks import call_esmfold_local

    result = call_esmfold_local("MKTAYIAK", seed=0)
    assert result.model_name == "esmfold_local"
    assert len(result.plddt_scores) == 8  # 8-residue sequence
    assert 0.0 < result.mean_plddt <= 100.0
    assert "ATOM" in result.structure_pdb
