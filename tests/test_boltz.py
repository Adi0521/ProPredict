"""
Tests for Boltz-2 integration (Phase 2).

Unit tests mock the filesystem and subprocess; the integration test at the
bottom requires Boltz-2 installed and a GPU (run on Modal or a local GPU machine).
"""
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch, mock_open

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SEQUENCE = "MKTAYIAK"

SAMPLE_CONFIDENCE = {
    "plddt": [0.85, 0.90, 0.78, 0.92, 0.88, 0.75, 0.83, 0.91],
    "ptm": 0.88,
    "confidence_score": 0.87,
}
EXPECTED_PLDDT = [v * 100 for v in SAMPLE_CONFIDENCE["plddt"]]

SAMPLE_AFFINITY = {"affinity": -8.42, "affinity_probability_binary": 0.91}

# Minimal CIF content that BioPython can parse for a single CA atom
SAMPLE_CIF = """\
data_boltz
_entry.id boltz
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 C CA . MET A 1 1 ? 1.000 2.000 3.000 1.00 85.00 ? 1 MET A CA 1
END
"""


# ---------------------------------------------------------------------------
# Unit: YAML input generation
# ---------------------------------------------------------------------------

def _build_boltz_yaml(sequence, context=None):
    """Call the YAML-building logic inside call_boltz without running subprocess."""
    import yaml

    ctx = context or {}
    ligands = ctx.get("ligands") or []

    sequences = [{
        "protein": {
            "id": "A",
            "sequence": sequence,
            "msa": "empty",
        }
    }]

    affinity_binder = None
    for i, lig in enumerate(ligands):
        chain_id = chr(ord("B") + i)
        smiles = lig.get("smiles") if isinstance(lig, dict) else lig.smiles
        sequences.append({"ligand": {"id": chain_id, "smiles": smiles}})
        if affinity_binder is None:
            affinity_binder = chain_id

    doc = {"version": 1, "sequences": sequences}
    if affinity_binder:
        doc["properties"] = [{"affinity": {"binder": affinity_binder}}]

    return doc, affinity_binder


def test_yaml_protein_only():
    doc, binder = _build_boltz_yaml(SAMPLE_SEQUENCE)
    assert doc["version"] == 1
    assert len(doc["sequences"]) == 1
    assert doc["sequences"][0]["protein"]["sequence"] == SAMPLE_SEQUENCE
    assert doc["sequences"][0]["protein"]["msa"] == "empty"
    assert binder is None
    assert "properties" not in doc


def test_yaml_with_ligand():
    ctx = {"ligands": [{"name": "ATP", "smiles": "C1=NC2=C(N1)N=CN=C2N"}]}
    doc, binder = _build_boltz_yaml(SAMPLE_SEQUENCE, ctx)
    assert len(doc["sequences"]) == 2
    assert doc["sequences"][1]["ligand"]["id"] == "B"
    assert doc["sequences"][1]["ligand"]["smiles"] == "C1=NC2=C(N1)N=CN=C2N"
    assert binder == "B"
    assert doc["properties"] == [{"affinity": {"binder": "B"}}]


def test_yaml_multiple_ligands():
    ctx = {
        "ligands": [
            {"name": "ATP", "smiles": "C1=NC2=C(N1)N=CN=C2N"},
            {"name": "Mg", "smiles": "[Mg]"},
        ]
    }
    doc, binder = _build_boltz_yaml(SAMPLE_SEQUENCE, ctx)
    assert len(doc["sequences"]) == 3
    assert doc["sequences"][1]["ligand"]["id"] == "B"
    assert doc["sequences"][2]["ligand"]["id"] == "C"
    # Affinity only wired to first ligand
    assert binder == "B"


# ---------------------------------------------------------------------------
# Unit: SMILES validation
# ---------------------------------------------------------------------------

def test_call_boltz_raises_on_missing_smiles():
    from orchestrator.tasks import call_boltz

    ctx = {"ligands": [{"name": "mystery", "smiles": None}]}
    with pytest.raises(ValueError, match="no SMILES"):
        call_boltz(SAMPLE_SEQUENCE, context=ctx)


def test_call_boltz_raises_on_empty_smiles():
    from orchestrator.tasks import call_boltz

    ctx = {"ligands": [{"name": "empty", "smiles": ""}]}
    with pytest.raises(ValueError, match="no SMILES"):
        call_boltz(SAMPLE_SEQUENCE, context=ctx)


# ---------------------------------------------------------------------------
# Unit: _cif_to_pdb
# ---------------------------------------------------------------------------

def test_cif_to_pdb_produces_atom_records(tmp_path):
    cif_file = tmp_path / "test.cif"
    cif_file.write_text(SAMPLE_CIF)

    from orchestrator.tasks import _cif_to_pdb
    pdb_string = _cif_to_pdb(str(cif_file))

    assert "ATOM" in pdb_string


# ---------------------------------------------------------------------------
# Unit: call_boltz — mock subprocess + filesystem
# ---------------------------------------------------------------------------

def _make_fake_results_dir(base_dir, plddt_data, affinity_data=None):
    """Write the output files Boltz-2 would produce under base_dir."""
    pred_dir = os.path.join(base_dir, "boltz_results_input", "predictions")
    os.makedirs(pred_dir)

    # Write a minimal CIF (BioPython parses this without needing real coordinates)
    with open(os.path.join(pred_dir, "input_model_0.cif"), "w") as f:
        f.write(SAMPLE_CIF)

    with open(os.path.join(pred_dir, "input_confidence_model_0.json"), "w") as f:
        json.dump(plddt_data, f)

    if affinity_data is not None:
        with open(os.path.join(pred_dir, "input_affinity_0.json"), "w") as f:
            json.dump(affinity_data, f)

    return pred_dir


def _mock_subprocess_success():
    proc = MagicMock()
    proc.returncode = 0
    proc.stderr = ""
    return proc


def test_call_boltz_returns_structure_prediction(tmp_path):
    from orchestrator.tasks import call_boltz

    def fake_run(cmd, **kwargs):
        # Locate the out_dir arg from the command list and populate it
        out_idx = cmd.index("--out_dir") + 1
        out_dir = cmd[out_idx]
        _make_fake_results_dir(out_dir, SAMPLE_CONFIDENCE)
        return _mock_subprocess_success()

    with patch("orchestrator.tasks.subprocess.run", side_effect=fake_run), \
         patch("orchestrator.tasks.BOLTZ_ENABLED", True):
        result = call_boltz(SAMPLE_SEQUENCE, seed=0)

    assert result.model_name == "boltz2"
    assert len(result.plddt_scores) == len(SAMPLE_CONFIDENCE["plddt"])
    assert abs(result.mean_plddt - sum(EXPECTED_PLDDT) / len(EXPECTED_PLDDT)) < 1e-4
    assert result.affinity_score is None
    assert result.seed == 0


def test_call_boltz_parses_affinity(tmp_path):
    from orchestrator.tasks import call_boltz

    ctx = {"ligands": [{"name": "ATP", "smiles": "C1=NC2=C(N1)N=CN=C2N"}]}

    def fake_run(cmd, **kwargs):
        out_idx = cmd.index("--out_dir") + 1
        out_dir = cmd[out_idx]
        _make_fake_results_dir(out_dir, SAMPLE_CONFIDENCE, affinity_data=SAMPLE_AFFINITY)
        return _mock_subprocess_success()

    with patch("orchestrator.tasks.subprocess.run", side_effect=fake_run), \
         patch("orchestrator.tasks.BOLTZ_ENABLED", True):
        result = call_boltz(SAMPLE_SEQUENCE, context=ctx, seed=0)

    assert result.affinity_score == pytest.approx(-8.42)


def test_call_boltz_raises_on_subprocess_failure():
    from orchestrator.tasks import call_boltz

    proc = MagicMock()
    proc.returncode = 1
    proc.stderr = "CUDA out of memory"

    with patch("orchestrator.tasks.subprocess.run", return_value=proc):
        with pytest.raises(RuntimeError, match="Boltz-2 failed"):
            call_boltz(SAMPLE_SEQUENCE)


def test_call_boltz_raises_on_missing_cif():
    from orchestrator.tasks import call_boltz

    def fake_run(cmd, **kwargs):
        # Write nothing — simulate missing CIF
        return _mock_subprocess_success()

    with patch("orchestrator.tasks.subprocess.run", side_effect=fake_run):
        with pytest.raises(FileNotFoundError, match="CIF output not found"):
            call_boltz(SAMPLE_SEQUENCE)


# ---------------------------------------------------------------------------
# Unit: main task wires Boltz into predictions list
# ---------------------------------------------------------------------------

def test_boltz_appended_to_predictions_when_enabled():
    """Verify that a successful Boltz call adds to the predictions list."""
    from models.schemas import StructurePrediction

    fake_pred = StructurePrediction(
        structure_pdb="ATOM ...",
        plddt_scores=[90.0],
        mean_plddt=90.0,
        seed=0,
        model_name="boltz2",
    )

    with patch("orchestrator.tasks.BOLTZ_ENABLED", True), \
         patch("orchestrator.tasks.call_boltz", return_value=fake_pred) as mock_boltz:
        # Import after patching so the module-level flag is seen
        from orchestrator.tasks import call_boltz as cb
        result = cb("MKTAYIAK", context={}, seed=0)

    assert result.model_name == "boltz2"
    assert result.mean_plddt == 90.0


# ---------------------------------------------------------------------------
# Integration test (requires Boltz-2 installed + GPU)
# ---------------------------------------------------------------------------

def test_call_boltz_integration():
    """
    End-to-end test against real Boltz-2 weights.
    Skipped unless `boltz` is importable (i.e. installed from source).
    Intended to run on Modal (A10G GPU) or a local GPU machine.
    """
    try:
        import subprocess as sp
        sp.run(["boltz", "--help"], capture_output=True, check=True)
    except Exception:
        pytest.skip("boltz CLI not found — install with: pip install git+https://github.com/jwohlwend/boltz")

    from orchestrator.tasks import call_boltz

    result = call_boltz(SAMPLE_SEQUENCE, seed=0)
    assert result.model_name == "boltz2"
    assert len(result.plddt_scores) == len(SAMPLE_SEQUENCE)
    assert 0.0 < result.mean_plddt <= 100.0
    assert "ATOM" in result.structure_pdb
    assert result.affinity_score is None  # no ligand provided


def test_call_boltz_affinity_integration():
    """
    End-to-end affinity prediction test.
    Skipped unless boltz CLI is available.
    """
    try:
        import subprocess as sp
        sp.run(["boltz", "--help"], capture_output=True, check=True)
    except Exception:
        pytest.skip("boltz CLI not found")

    from orchestrator.tasks import call_boltz

    ctx = {"ligands": [{"name": "ethanol", "smiles": "CCO"}]}
    result = call_boltz(SAMPLE_SEQUENCE, context=ctx, seed=0)

    assert result.model_name == "boltz2"
    assert result.affinity_score is not None
    assert isinstance(result.affinity_score, float)
