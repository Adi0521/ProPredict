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
    "plddt": [85.0, 90.0, 78.0, 92.0, 88.0, 75.0, 83.0, 91.0],
    "ptm": 0.88,
    "confidence_score": 0.87,
}
EXPECTED_PLDDT = SAMPLE_CONFIDENCE["plddt"]

# Real Boltz-2 affinity keys and values, captured from an actual A10G run
# (modal_app.py::test_boltz_affinity_gpu, 2026-07-21). The *1/*2 keys are the individual
# ensemble members; the un-suffixed pair is what we read. affinity_pred_value is
# log10(IC50) with IC50 in uM, so a sub-micromolar binder is negative.
#
# This fixture previously used a key called "affinity", which Boltz never writes — the mock
# agreed with the bug, so the suite stayed green while affinity_score was None on every real
# run. Keep this fixture pinned to observed output, never to what the code expects.
SAMPLE_AFFINITY = {
    "affinity_pred_value": -1.35,
    "affinity_probability_binary": 0.91,
    "affinity_pred_value1": -1.30,
    "affinity_probability_binary1": 0.89,
    "affinity_pred_value2": -1.40,
    "affinity_probability_binary2": 0.93,
}

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

# Test the REAL builder, not a copy. A previous version of this file reimplemented the
# YAML-building logic here; that duplication is precisely how the affinity-key bug survived
# (the test agreed with a copy, not with call_boltz). Import the actual function.
def _build_boltz_yaml(sequence, context=None, protein_copies=1):
    from orchestrator.backends.boltz import _build_boltz_input
    return _build_boltz_input(sequence, context=context, protein_copies=protein_copies,
                              use_msa=False)


def test_yaml_protein_only():
    doc, binder = _build_boltz_yaml(SAMPLE_SEQUENCE)
    assert doc["version"] == 1
    assert len(doc["sequences"]) == 1
    assert doc["sequences"][0]["protein"]["sequence"] == SAMPLE_SEQUENCE
    assert doc["sequences"][0]["protein"]["msa"] == "empty"
    # Monomer stays a scalar "A" — byte-identical to the pre-homomer YAML.
    assert doc["sequences"][0]["protein"]["id"] == "A"
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
# Unit: homo-oligomer (protein_copies) — the HIV-PR / homodimer fix
# ---------------------------------------------------------------------------

def test_yaml_homodimer_uses_id_list():
    doc, binder = _build_boltz_yaml(SAMPLE_SEQUENCE, protein_copies=2)
    assert len(doc["sequences"]) == 1
    # Boltz's homomer form: a list of chain IDs on one protein entry.
    assert doc["sequences"][0]["protein"]["id"] == ["A", "B"]
    assert doc["sequences"][0]["protein"]["sequence"] == SAMPLE_SEQUENCE
    assert binder is None


def test_yaml_homodimer_with_ligand_shifts_ligand_chain():
    """The collision fix: with 2 protein chains (A, B) the ligand must be C, not B."""
    ctx = {"ligands": [{"name": "inhibitor", "smiles": "CC(C)CN"}]}
    doc, binder = _build_boltz_yaml(SAMPLE_SEQUENCE, ctx, protein_copies=2)
    assert doc["sequences"][0]["protein"]["id"] == ["A", "B"]
    assert doc["sequences"][1]["ligand"]["id"] == "C"        # NOT "B"
    assert binder == "C"
    assert doc["properties"] == [{"affinity": {"binder": "C"}}]


def test_yaml_trimer_chain_ids():
    ctx = {"ligands": [{"name": "x", "smiles": "CCO"}]}
    doc, _ = _build_boltz_yaml(SAMPLE_SEQUENCE, ctx, protein_copies=3)
    assert doc["sequences"][0]["protein"]["id"] == ["A", "B", "C"]
    assert doc["sequences"][1]["ligand"]["id"] == "D"


def test_build_boltz_input_rejects_zero_copies():
    from orchestrator.backends.boltz import _build_boltz_input
    with pytest.raises(ValueError, match="protein_copies must be >= 1"):
        _build_boltz_input(SAMPLE_SEQUENCE, protein_copies=0)


def test_build_boltz_input_rejects_too_many_chains():
    from orchestrator.backends.boltz import _build_boltz_input
    ctx = {"ligands": [{"name": "x", "smiles": "CCO"}]}   # 26 proteins + 1 ligand = 27
    with pytest.raises(ValueError, match="too many chains"):
        _build_boltz_input(SAMPLE_SEQUENCE, context=ctx, protein_copies=26)


def test_call_boltz_reads_protein_copies_from_context(tmp_path):
    """context['protein_copies'] must reach the YAML with no explicit kwarg."""
    from orchestrator.backends import boltz as boltz_mod

    captured = {}

    def fake_run(cmd, **kwargs):
        out_idx = cmd.index("--out_dir") + 1
        # capture the YAML that was written
        yaml_path = cmd[2]
        import yaml
        with open(yaml_path) as fh:
            captured["doc"] = yaml.safe_load(fh)
        _make_fake_results_dir(cmd[out_idx], SAMPLE_CONFIDENCE)
        return _mock_subprocess_success()

    with patch("orchestrator.backends.boltz.subprocess.run", side_effect=fake_run), \
         patch("orchestrator.backends.boltz.get_boltz_build_info",
               return_value={"version": "2.2.1", "commit": None, "label": "2.2.1"}):
        boltz_mod.call_boltz(SAMPLE_SEQUENCE, context={"protein_copies": 2}, seed=0)

    assert captured["doc"]["sequences"][0]["protein"]["id"] == ["A", "B"]


def test_call_boltz_kwarg_overrides_context(tmp_path):
    from orchestrator.backends import boltz as boltz_mod

    captured = {}

    def fake_run(cmd, **kwargs):
        out_idx = cmd.index("--out_dir") + 1
        import yaml
        with open(cmd[2]) as fh:
            captured["doc"] = yaml.safe_load(fh)
        _make_fake_results_dir(cmd[out_idx], SAMPLE_CONFIDENCE)
        return _mock_subprocess_success()

    with patch("orchestrator.backends.boltz.subprocess.run", side_effect=fake_run), \
         patch("orchestrator.backends.boltz.get_boltz_build_info",
               return_value={"version": "2.2.1", "commit": None, "label": "2.2.1"}):
        # context says 3, explicit kwarg says 1 -> kwarg wins -> scalar "A"
        boltz_mod.call_boltz(SAMPLE_SEQUENCE, context={"protein_copies": 3},
                             seed=0, protein_copies=1)

    assert captured["doc"]["sequences"][0]["protein"]["id"] == "A"


# ---------------------------------------------------------------------------
# Unit: SMILES validation
# ---------------------------------------------------------------------------

def test_call_boltz_raises_on_missing_smiles():
    from orchestrator.backends.boltz import call_boltz

    ctx = {"ligands": [{"name": "mystery", "smiles": None}]}
    with pytest.raises(ValueError, match="no SMILES"):
        call_boltz(SAMPLE_SEQUENCE, context=ctx)


def test_call_boltz_raises_on_empty_smiles():
    from orchestrator.backends.boltz import call_boltz

    ctx = {"ligands": [{"name": "empty", "smiles": ""}]}
    with pytest.raises(ValueError, match="no SMILES"):
        call_boltz(SAMPLE_SEQUENCE, context=ctx)


# ---------------------------------------------------------------------------
# Unit: _cif_to_pdb
# ---------------------------------------------------------------------------

def test_cif_to_pdb_produces_atom_records(tmp_path):
    cif_file = tmp_path / "test.cif"
    cif_file.write_text(SAMPLE_CIF)

    from orchestrator.backends.boltz import _cif_to_pdb
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
        # Real filename, verified on an A10G run: affinity_<record_id>.json, NOT the
        # input_affinity_0.json this fixture used to invent. The fabricated name is why the
        # glob could not be anchored until the GPU run settled it.
        with open(os.path.join(pred_dir, "affinity_input.json"), "w") as f:
            json.dump(affinity_data, f)

    return pred_dir


def _mock_subprocess_success():
    proc = MagicMock()
    proc.returncode = 0
    proc.stderr = ""
    return proc


def test_call_boltz_returns_structure_prediction(tmp_path):
    from orchestrator.backends.boltz import call_boltz

    def fake_run(cmd, **kwargs):
        # Locate the out_dir arg from the command list and populate it
        out_idx = cmd.index("--out_dir") + 1
        out_dir = cmd[out_idx]
        _make_fake_results_dir(out_dir, SAMPLE_CONFIDENCE)
        return _mock_subprocess_success()

    with patch("orchestrator.backends.boltz.subprocess.run", side_effect=fake_run):
        result = call_boltz(SAMPLE_SEQUENCE, seed=0)

    assert result.model_name == "boltz2"
    assert len(result.plddt_scores) == len(SAMPLE_CONFIDENCE["plddt"])
    assert abs(result.mean_plddt - sum(EXPECTED_PLDDT) / len(EXPECTED_PLDDT)) < 1e-4
    assert result.affinity_score is None
    assert result.seed == 0


def test_call_boltz_parses_affinity(tmp_path):
    from orchestrator.backends.boltz import call_boltz

    ctx = {"ligands": [{"name": "ATP", "smiles": "C1=NC2=C(N1)N=CN=C2N"}]}

    def fake_run(cmd, **kwargs):
        out_idx = cmd.index("--out_dir") + 1
        out_dir = cmd[out_idx]
        _make_fake_results_dir(out_dir, SAMPLE_CONFIDENCE, affinity_data=SAMPLE_AFFINITY)
        return _mock_subprocess_success()

    with patch("orchestrator.backends.boltz.subprocess.run", side_effect=fake_run):
        result = call_boltz(SAMPLE_SEQUENCE, context=ctx, seed=0)

    # Regression guard: assert populated, not just equal. An `is not None` check here
    # would have caught the wrong-key bug on day one.
    assert result.affinity_score is not None
    assert result.affinity_score == pytest.approx(-1.35)
    assert result.affinity_probability == pytest.approx(0.91)


class TestBoltzBuildInfo:
    """
    get_boltz_build_info() identifies the build that produced a structure. The COMMIT is the
    load-bearing part: the version string "2.2.1" describes both the v2.2.1 tag and the
    commit 6 ahead of it that this project pins (Process/boltz-version-pin.md).
    """

    def _patch_md(self, version=None, direct_url=None):
        """
        Patch importlib.metadata's functions, NOT the module in sys.modules:
        `import importlib.metadata as md` binds via getattr on the already-imported
        `importlib` package, so a sys.modules swap is silently ignored.
        """
        from contextlib import ExitStack

        stack = ExitStack()
        if version is None:
            stack.enter_context(patch("importlib.metadata.version",
                                      side_effect=Exception("PackageNotFoundError")))
        else:
            stack.enter_context(patch("importlib.metadata.version", return_value=version))
        dist = MagicMock()
        dist.read_text.return_value = direct_url
        stack.enter_context(patch("importlib.metadata.distribution", return_value=dist))
        return stack

    def test_reports_version_and_commit_from_vcs_install(self):
        from orchestrator.backends.boltz import get_boltz_build_info

        direct_url = json.dumps({
            "url": "https://github.com/jwohlwend/boltz.git",
            "vcs_info": {"vcs": "git", "commit_id": "b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc"},
        })
        with self._patch_md("2.2.1", direct_url):
            info = get_boltz_build_info()

        assert info["version"] == "2.2.1"
        assert info["commit"] == "b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc"
        assert info["label"] == "2.2.1@b1ebfc46ecf5"

    def test_falls_back_to_bare_version_without_vcs_info(self):
        """A wheel/sdist install has no direct_url.json — record what we do know."""
        from orchestrator.backends.boltz import get_boltz_build_info

        with self._patch_md("2.2.1", None):
            info = get_boltz_build_info()

        assert info["version"] == "2.2.1"
        assert info["commit"] is None
        assert info["label"] == "2.2.1"

    def test_all_none_when_boltz_not_installed(self):
        """The normal case on a dev machine — must not raise."""
        from orchestrator.backends.boltz import get_boltz_build_info

        with self._patch_md(version=None):
            info = get_boltz_build_info()

        assert info == {"version": None, "commit": None, "label": None}

    def test_malformed_direct_url_does_not_crash_a_prediction(self):
        from orchestrator.backends.boltz import get_boltz_build_info

        with self._patch_md("2.2.1", "{not valid json"):
            info = get_boltz_build_info()

        assert info["version"] == "2.2.1"
        assert info["commit"] is None


def test_call_boltz_stamps_backend_version(tmp_path):
    """The stamp must reach the StructurePrediction, not just be computable."""
    from orchestrator.backends.boltz import call_boltz

    def fake_run(cmd, **kwargs):
        out_idx = cmd.index("--out_dir") + 1
        _make_fake_results_dir(cmd[out_idx], SAMPLE_CONFIDENCE)
        return _mock_subprocess_success()

    with patch("orchestrator.backends.boltz.subprocess.run", side_effect=fake_run), \
         patch("orchestrator.backends.boltz.get_boltz_build_info",
               return_value={"version": "2.2.1", "commit": "b1ebfc46" * 5,
                             "label": "2.2.1@b1ebfc46ecf5"}):
        result = call_boltz(SAMPLE_SEQUENCE, seed=0)

    assert result.backend_version == "2.2.1@b1ebfc46ecf5"


def test_call_boltz_ignores_pae_affinity_file(tmp_path):
    """A pae_affinity_*.json must not be mistaken for the affinity summary."""
    from orchestrator.backends.boltz import call_boltz

    ctx = {"ligands": [{"name": "ATP", "smiles": "C1=NC2=C(N1)N=CN=C2N"}]}

    def fake_run(cmd, **kwargs):
        out_idx = cmd.index("--out_dir") + 1
        out_dir = cmd[out_idx]
        pred_dir = _make_fake_results_dir(out_dir, SAMPLE_CONFIDENCE, affinity_data=SAMPLE_AFFINITY)
        # Sorts ahead of input_affinity_0.json and carries none of the real keys.
        with open(os.path.join(pred_dir, "input_pae_affinity_0.json"), "w") as f:
            json.dump({"pae": [[0.1]]}, f)
        return _mock_subprocess_success()

    with patch("orchestrator.backends.boltz.subprocess.run", side_effect=fake_run):
        result = call_boltz(SAMPLE_SEQUENCE, context=ctx, seed=0)

    assert result.affinity_score == pytest.approx(-1.35)


def test_call_boltz_raises_on_subprocess_failure():
    from orchestrator.backends.boltz import call_boltz

    proc = MagicMock()
    proc.returncode = 1
    proc.stderr = "CUDA out of memory"

    with patch("orchestrator.backends.boltz.subprocess.run", return_value=proc):
        with pytest.raises(RuntimeError, match="Boltz-2 failed"):
            call_boltz(SAMPLE_SEQUENCE)


def test_call_boltz_raises_on_missing_cif():
    from orchestrator.backends.boltz import call_boltz

    def fake_run(cmd, **kwargs):
        # Write nothing — simulate missing CIF
        return _mock_subprocess_success()

    with patch("orchestrator.backends.boltz.subprocess.run", side_effect=fake_run):
        with pytest.raises(FileNotFoundError, match="Boltz-2 produced no"):
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

    from orchestrator.backends.boltz import call_boltz

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

    from orchestrator.backends.boltz import call_boltz

    ctx = {"ligands": [{"name": "ethanol", "smiles": "CCO"}]}
    result = call_boltz(SAMPLE_SEQUENCE, context=ctx, seed=0)

    assert result.model_name == "boltz2"
    assert result.affinity_score is not None
    assert isinstance(result.affinity_score, float)
    # The probability head ships alongside the value in the same summary JSON.
    assert result.affinity_probability is not None
    assert 0.0 <= result.affinity_probability <= 1.0
