"""
Tests for orchestrator/membrane.py (Stage F — membrane embedding).

Fully mocked — no insane.py binary and no OpenMM/openmmforcefields needed locally.
insane.py resolution and the subprocess are mocked; the OpenMM path is exercised by
injecting fake `openmm` / `openmm.app` / `openmmforcefields` modules into sys.modules.
The pure helpers (_resolve_insane, _lipid_name) are tested directly.

The real-binary counterpart is modal_app.py::test_membrane_modal, which runs an actual
PDBFixer -> ForceField(charmm36 lipids) -> addMembrane build in the Modal image (run:
modal run modal_app.py::test_membrane_modal). insane.py is NOT in that image, so
embed_in_membrane_gromacs is covered here (mocked) only.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.membrane import (
    _resolve_insane,
    _lipid_name,
    embed_in_membrane_gromacs,
    embed_in_membrane_openmm,
)


# ---------------------------------------------------------------------------
# _resolve_insane
# ---------------------------------------------------------------------------

def test_resolve_insane_uses_config_path_when_file():
    with patch("orchestrator.membrane.os.path.isfile", return_value=True), \
         patch("orchestrator.membrane.shutil.which") as mock_which:
        assert _resolve_insane("/opt/insane.py") == "/opt/insane.py"
        mock_which.assert_not_called()  # PATH not consulted when config file exists


def test_resolve_insane_finds_insane_py_on_path():
    def _which(name):
        return "/usr/bin/insane.py" if name == "insane.py" else None
    with patch("orchestrator.membrane.os.path.isfile", return_value=False), \
         patch("orchestrator.membrane.shutil.which", side_effect=_which):
        assert _resolve_insane("") == "/usr/bin/insane.py"


def test_resolve_insane_falls_back_to_compiled_insane():
    def _which(name):
        return "/usr/bin/insane" if name == "insane" else None
    with patch("orchestrator.membrane.os.path.isfile", return_value=False), \
         patch("orchestrator.membrane.shutil.which", side_effect=_which):
        assert _resolve_insane("") == "/usr/bin/insane"


def test_resolve_insane_returns_none_when_absent():
    with patch("orchestrator.membrane.os.path.isfile", return_value=False), \
         patch("orchestrator.membrane.shutil.which", return_value=None):
        assert _resolve_insane("/nope/insane.py") is None


# ---------------------------------------------------------------------------
# _lipid_name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inp,expected", [
    (None, "POPC"),
    ("", "POPC"),
    ("popc", "POPC"),
    ("POPE", "POPE"),
    ("chol", "CHOL"),
    ("XYZ", "XYZ"),   # unknown -> uppercased passthrough
])
def test_lipid_name(inp, expected):
    assert _lipid_name(inp) == expected


# ---------------------------------------------------------------------------
# embed_in_membrane_gromacs
# ---------------------------------------------------------------------------

def test_gromacs_insane_not_found_raises(tmp_path):
    with patch("orchestrator.membrane._resolve_insane", return_value=None):
        with pytest.raises(RuntimeError, match="insane.py not found"):
            embed_in_membrane_gromacs("PDB", {"type": "POPC"}, str(tmp_path))


@patch("orchestrator.membrane.subprocess.run")
def test_gromacs_success_returns_paths(mock_run, tmp_path):
    out_dir = str(tmp_path)
    gro = os.path.join(out_dir, "membrane_system.gro")
    top = os.path.join(out_dir, "membrane_system.top")

    def _side(cmd, **kw):
        open(gro, "w").close()
        open(top, "w").close()
        return MagicMock(returncode=0, stdout="", stderr="")

    mock_run.side_effect = _side
    with patch("orchestrator.membrane._resolve_insane", return_value="/x/insane.py"):
        got_gro, got_top = embed_in_membrane_gromacs("PDB", {"type": "POPC"}, out_dir)

    assert got_gro == gro and got_top == top
    # lipid propagated into the -l argument
    cmd = mock_run.call_args.args[0]
    assert "POPC:1" in cmd


@patch("orchestrator.membrane.subprocess.run")
def test_gromacs_span_adds_center_flag(mock_run, tmp_path):
    out_dir = str(tmp_path)

    def _side(cmd, **kw):
        open(os.path.join(out_dir, "membrane_system.gro"), "w").close()
        open(os.path.join(out_dir, "membrane_system.top"), "w").close()
        return MagicMock(returncode=0, stdout="", stderr="")

    mock_run.side_effect = _side
    with patch("orchestrator.membrane._resolve_insane", return_value="/x/insane.py"):
        embed_in_membrane_gromacs("PDB", {"type": "POPC", "span": [10, 20]}, out_dir)

    assert "-center" in mock_run.call_args.args[0]


@patch("orchestrator.membrane.subprocess.run")
def test_gromacs_no_span_omits_center_flag(mock_run, tmp_path):
    out_dir = str(tmp_path)

    def _side(cmd, **kw):
        open(os.path.join(out_dir, "membrane_system.gro"), "w").close()
        open(os.path.join(out_dir, "membrane_system.top"), "w").close()
        return MagicMock(returncode=0, stdout="", stderr="")

    mock_run.side_effect = _side
    with patch("orchestrator.membrane._resolve_insane", return_value="/x/insane.py"):
        embed_in_membrane_gromacs("PDB", {"type": "POPC"}, out_dir)

    assert "-center" not in mock_run.call_args.args[0]


@patch("orchestrator.membrane.subprocess.run")
def test_gromacs_nonzero_exit_raises(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=1, stdout="ins-out", stderr="ins-err")
    with patch("orchestrator.membrane._resolve_insane", return_value="/x/insane.py"):
        with pytest.raises(RuntimeError) as ei:
            embed_in_membrane_gromacs("PDB", {"type": "POPC"}, str(tmp_path))
    msg = str(ei.value)
    assert "insane.py failed" in msg and "ins-out" in msg and "ins-err" in msg


@patch("orchestrator.membrane.subprocess.run")
def test_gromacs_missing_output_raises(mock_run, tmp_path):
    # exit 0 but no .gro/.top written
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    with patch("orchestrator.membrane._resolve_insane", return_value="/x/insane.py"):
        with pytest.raises(RuntimeError, match="output files are missing"):
            embed_in_membrane_gromacs("PDB", {"type": "POPC"}, str(tmp_path))


# ---------------------------------------------------------------------------
# embed_in_membrane_openmm — fake openmm / openmmforcefields
# ---------------------------------------------------------------------------

def _fake_openmm_modules():
    openmm_mod = MagicMock(name="openmm")
    openmm_mod.unit = MagicMock(name="unit")
    app_mod = MagicMock(name="openmm.app")
    app_mod.Modeller = MagicMock(name="Modeller")
    return {
        "openmm": openmm_mod,
        "openmm.app": app_mod,
        "openmmforcefields": MagicMock(name="openmmforcefields"),
    }


def test_openmm_missing_raises():
    modeller, ff = MagicMock(), MagicMock()
    with patch.dict(sys.modules, {"openmm": None}):
        with pytest.raises(RuntimeError, match="OpenMM is not installed"):
            embed_in_membrane_openmm(modeller, ff, {"type": "POPC"})


def test_openmm_missing_forcefields_raises():
    mods = _fake_openmm_modules()
    mods["openmmforcefields"] = None  # openmm present, openmmforcefields absent
    modeller, ff = MagicMock(), MagicMock()
    with patch.dict(sys.modules, mods):
        with pytest.raises(RuntimeError, match="openmmforcefields is not installed"):
            embed_in_membrane_openmm(modeller, ff, {"type": "POPC"})


def test_openmm_add_membrane_failure_wrapped():
    mods = _fake_openmm_modules()
    modeller, ff = MagicMock(), MagicMock()
    modeller.addMembrane.side_effect = Exception("bilayer boom")
    with patch.dict(sys.modules, mods):
        with pytest.raises(RuntimeError, match="addMembrane failed for lipid 'POPC'"):
            embed_in_membrane_openmm(modeller, ff, {"type": "POPC"})


def test_openmm_success_returns_modeller():
    mods = _fake_openmm_modules()
    modeller, ff = MagicMock(), MagicMock()
    modeller.topology.getNumAtoms.return_value = 4321
    with patch.dict(sys.modules, mods):
        out = embed_in_membrane_openmm(modeller, ff, {"type": "POPE"})
    assert out is modeller
    # lipid name normalised and passed through to addMembrane
    _, kwargs = modeller.addMembrane.call_args
    assert kwargs["lipidType"] == "POPE"
