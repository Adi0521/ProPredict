"""
Tests for the apply_mutation agent tool in orchestrator/agent.py (Task 3).

Fully mocked — patches the prediction backends (call_boltz / call_esmfold_api),
count_clashes, and the BOLTZ_ENABLED / AGENT_MAX_MUTATIONS config values in the
orchestrator.agent namespace, then calls _execute_agent_tool() directly. No real
ESMFold/Boltz/BioPython needed. Mirrors tests/test_boltz.py style.
"""
import json
from unittest.mock import patch, MagicMock

import pytest

from orchestrator.agent import _execute_agent_tool, run_agent_refinement
from models.schemas import StructurePrediction


def _fake_pred(pdb="ATOM_MUT", plddt=None, mean=80.0, affinity=None):
    return StructurePrediction(
        structure_pdb=pdb,
        plddt_scores=plddt or [80.0, 80.0, 80.0],
        mean_plddt=mean,
        seed=0,
        model_name="esmfold_local",
        affinity_score=affinity,
    )


def _base_state(seq="ACDEF"):
    return {
        "current_pdb": "ATOM_ORIG",
        "plddt_scores": [50.0] * len(seq),
        "mean_plddt": 50.0,
        "num_clashes": 2,
        "context": {},
        "sequence": seq,
        "mutations_applied": [],
    }


def _apply(tool_input, state):
    return json.loads(_execute_agent_tool("apply_mutation", tool_input, state))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

@patch("orchestrator.agent.count_clashes", return_value=1)
@patch("orchestrator.agent.call_esmfold_api")
def test_valid_mutation_updates_state(mock_esm, mock_clash):
    with patch("orchestrator.agent.BOLTZ_ENABLED", False), \
         patch("orchestrator.agent.AGENT_MAX_MUTATIONS", 3):
        mock_esm.return_value = _fake_pred(pdb="ATOM_MUT", mean=88.0)
        state = _base_state("ACDEF")
        out = _apply({"position": 2, "from_aa": "C", "to_aa": "W"}, state)

    assert out["status"] == "completed"
    assert out["mutation"] == "C2W"
    assert out["mean_plddt"] == 88.0
    assert out["num_clashes"] == 1
    assert state["sequence"] == "AWDEF"
    assert state["current_pdb"] == "ATOM_MUT"
    assert state["mean_plddt"] == 88.0
    assert state["mutations_applied"] == ["C2W"]
    # backend was called with the mutated sequence
    mock_esm.assert_called_once()
    assert mock_esm.call_args.args[0] == "AWDEF"


@patch("orchestrator.agent.count_clashes", return_value=0)
@patch("orchestrator.agent.call_esmfold_api")
def test_lowercase_to_aa_is_normalized(mock_esm, mock_clash):
    with patch("orchestrator.agent.BOLTZ_ENABLED", False), \
         patch("orchestrator.agent.AGENT_MAX_MUTATIONS", 3):
        mock_esm.return_value = _fake_pred()
        state = _base_state("ACDEF")
        out = _apply({"position": 1, "to_aa": "g"}, state)

    assert out["mutation"] == "A1G"
    assert state["sequence"] == "GCDEF"


# ---------------------------------------------------------------------------
# Validation errors — state must stay unchanged, backend must not be called
# ---------------------------------------------------------------------------

@patch("orchestrator.agent.call_esmfold_api")
def test_from_aa_mismatch_errors_without_prediction(mock_esm):
    with patch("orchestrator.agent.BOLTZ_ENABLED", False), \
         patch("orchestrator.agent.AGENT_MAX_MUTATIONS", 3):
        state = _base_state("ACDEF")
        out = _apply({"position": 2, "from_aa": "G", "to_aa": "W"}, state)

    assert "mismatch" in out["error"]
    assert state["sequence"] == "ACDEF"
    assert state["mutations_applied"] == []
    mock_esm.assert_not_called()


@patch("orchestrator.agent.call_esmfold_api")
def test_invalid_to_aa_errors(mock_esm):
    with patch("orchestrator.agent.AGENT_MAX_MUTATIONS", 3):
        state = _base_state("ACDEF")
        out = _apply({"position": 2, "to_aa": "Z"}, state)

    assert "not a standard amino acid" in out["error"]
    assert state["sequence"] == "ACDEF"
    mock_esm.assert_not_called()


@pytest.mark.parametrize("pos", [0, -1, 99])
@patch("orchestrator.agent.call_esmfold_api")
def test_position_out_of_range_errors(mock_esm, pos):
    with patch("orchestrator.agent.AGENT_MAX_MUTATIONS", 3):
        state = _base_state("ACDEF")
        out = _apply({"position": pos, "to_aa": "W"}, state)

    assert "out of range" in out["error"]
    assert state["sequence"] == "ACDEF"
    mock_esm.assert_not_called()


@patch("orchestrator.agent.call_esmfold_api")
def test_missing_position_errors(mock_esm):
    with patch("orchestrator.agent.AGENT_MAX_MUTATIONS", 3):
        state = _base_state("ACDEF")
        out = _apply({"to_aa": "W"}, state)

    assert "required" in out["error"]
    mock_esm.assert_not_called()


# ---------------------------------------------------------------------------
# Rollback on backend failure
# ---------------------------------------------------------------------------

@patch("orchestrator.agent.call_esmfold_api", side_effect=RuntimeError("boom"))
def test_backend_failure_rolls_back(mock_esm):
    with patch("orchestrator.agent.BOLTZ_ENABLED", False), \
         patch("orchestrator.agent.AGENT_MAX_MUTATIONS", 3):
        state = _base_state("ACDEF")
        out = _apply({"position": 2, "to_aa": "W"}, state)

    assert "re-prediction failed" in out["error"]
    assert state["sequence"] == "ACDEF"          # unchanged
    assert state["current_pdb"] == "ATOM_ORIG"   # unchanged
    assert state["mutations_applied"] == []


# ---------------------------------------------------------------------------
# Backend routing
# ---------------------------------------------------------------------------

@patch("orchestrator.agent.count_clashes", return_value=0)
@patch("orchestrator.agent.call_boltz")
@patch("orchestrator.agent.call_esmfold_api")
def test_boltz_enabled_routes_to_boltz(mock_esm, mock_boltz, mock_clash):
    with patch("orchestrator.agent.BOLTZ_ENABLED", True), \
         patch("orchestrator.agent.AGENT_MAX_MUTATIONS", 3):
        mock_boltz.return_value = _fake_pred(mean=90.0, affinity=-8.4)
        state = _base_state("ACDEF")
        out = _apply({"position": 1, "to_aa": "G"}, state)

    mock_boltz.assert_called_once()
    assert mock_boltz.call_args.args[0] == "GCDEF"
    mock_esm.assert_not_called()
    assert out["affinity_kcal_mol"] == -8.4


# ---------------------------------------------------------------------------
# Cap enforcement
# ---------------------------------------------------------------------------

@patch("orchestrator.agent.call_boltz")
@patch("orchestrator.agent.call_esmfold_api")
def test_mutation_cap_enforced(mock_esm, mock_boltz):
    with patch("orchestrator.agent.BOLTZ_ENABLED", False), \
         patch("orchestrator.agent.AGENT_MAX_MUTATIONS", 2):
        state = _base_state("ACDEF")
        state["mutations_applied"] = ["A1V", "C2D"]  # already at the limit
        out = _apply({"position": 3, "to_aa": "W"}, state)

    assert "limit reached" in out["error"]
    assert state["mutations_applied"] == ["A1V", "C2D"]
    mock_esm.assert_not_called()
    mock_boltz.assert_not_called()


# ---------------------------------------------------------------------------
# Task 4 — num_clashes recomputed after refinement branches that reassign the PDB
# ---------------------------------------------------------------------------

@patch("orchestrator.agent.count_clashes", return_value=7)
@patch("orchestrator.agent.run_rosetta_relax", return_value=("RELAXED_PDB", -123.4))
def test_rosetta_relax_recomputes_clashes(mock_relax, mock_clash):
    with patch("orchestrator.agent.ROSETTA_ENABLED", True):
        state = _base_state("ACDEF")
        state["num_clashes"] = 2
        out = json.loads(_execute_agent_tool("run_rosetta_relax", {}, state))

    assert state["current_pdb"] == "RELAXED_PDB"
    assert state["num_clashes"] == 7          # recomputed on the relaxed structure
    assert out["num_clashes"] == 7            # and surfaced in the tool result
    mock_clash.assert_called_once_with("RELAXED_PDB")


@patch("orchestrator.agent.count_clashes", return_value=5)
@patch("orchestrator.agent.call_boltz")
def test_boltz_prediction_recomputes_clashes(mock_boltz, mock_clash):
    with patch("orchestrator.agent.BOLTZ_ENABLED", True):
        mock_boltz.return_value = _fake_pred(pdb="BOLTZ_PDB", mean=95.0)
        state = _base_state("ACDEF")
        state["num_clashes"] = 2
        out = json.loads(_execute_agent_tool("run_boltz_prediction", {"num_seeds": 3}, state))

    assert state["current_pdb"] == "BOLTZ_PDB"
    assert state["num_clashes"] == 5
    assert out["num_clashes"] == 5
    mock_clash.assert_called_with("BOLTZ_PDB")


def test_run_agent_refinement_reports_updated_clashes():
    """End-to-end: the reported PostProcessingResult.num_clashes must reflect the
    post-refinement structure, not the stale pre-loop count.

    run_agent_refinement does `import anthropic` internally; we inject a fake module
    into sys.modules so the test runs without the real SDK installed (the client is
    fully mocked anyway — no real anthropic types are exercised)."""
    def _tool_use(name, inp):
        blk = MagicMock()
        blk.type = "tool_use"
        blk.name = name
        blk.input = inp
        blk.id = f"tool_{name}"
        resp = MagicMock()
        resp.stop_reason = "tool_use"
        resp.content = [blk]
        return resp

    client = MagicMock()
    client.messages.create.side_effect = [
        _tool_use("run_rosetta_relax", {}),
        _tool_use("accept_structure", {"reasoning": "looks good"}),
    ]
    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic.return_value = client

    pred = _fake_pred(pdb="ORIG", plddt=[80.0, 80.0, 80.0], mean=80.0)

    with patch.dict("sys.modules", {"anthropic": fake_anthropic}), \
         patch("orchestrator.agent.ROSETTA_ENABLED", True), \
         patch("orchestrator.agent.AGENT_API_KEY", "sk-test"), \
         patch("orchestrator.agent.AGENT_BASE_URL", ""), \
         patch("orchestrator.agent.run_rosetta_relax", return_value=("RELAXED", -50.0)), \
         patch("orchestrator.agent.count_clashes", side_effect=[2, 9]):
        post_proc, updated_pdb = run_agent_refinement(pred, {}, "ACDEF")

    assert post_proc.decision == "accept"
    assert post_proc.num_clashes == 9        # post-relax count, not the pre-loop 2
    assert post_proc.score == 80.0 - 9 * 5.0
    assert updated_pdb == "RELAXED"
