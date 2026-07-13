"""
Tests for orchestrator/ligands.py (Stage F — ligand prep & docking).

Fully mocked — no real RDKit / Vina / meeko / GNINA / ACPYPE / OpenFF needed, so
these run anywhere (none of those are installed in the local dev env). RDKit-backed
functions are exercised by injecting fake `rdkit` / `rdkit.Chem` modules into
sys.modules; docking/parameterization binaries are mocked via shutil.which +
subprocess.run. The pure-Python PDB parsers are tested against synthetic PDB files.

The real-binary counterpart is modal_app.py::test_ligands_modal, which runs the actual
RDKit -> Vina -> OpenFF pipeline in the Modal image (run: modal run
modal_app.py::test_ligands_modal). GNINA and ACPYPE are absent from that image too, so
dock_gnina / parameterize_ligand_acpype are covered here (mocked) only.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.ligands import (
    smiles_to_3d,
    dock_gnina,
    dock_vina,
    parameterize_ligand_acpype,
    parameterize_ligand_openff,
    prepare_ligands,
    _ca_centroid,
    _all_ca_coords,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_rdkit(mol_from_smiles="__mol__", embed_result=0):
    """
    Build fake `rdkit` + `rdkit.Chem` modules for smiles_to_3d.

    `from rdkit import Chem`            -> sys.modules["rdkit"].Chem
    `from rdkit.Chem import AllChem`    -> sys.modules["rdkit.Chem"].AllChem
    """
    chem = MagicMock(name="Chem")
    chem.MolFromSmiles.return_value = mol_from_smiles
    chem.AddHs.side_effect = lambda m: m

    allchem = MagicMock(name="AllChem")
    allchem.ETKDGv3.return_value = MagicMock(name="ETKDGparams")
    allchem.EmbedMolecule.return_value = embed_result

    chem_mod = MagicMock(name="rdkit.Chem")
    chem_mod.MolFromSmiles = chem.MolFromSmiles
    chem_mod.AddHs = chem.AddHs
    chem_mod.SDWriter.return_value = MagicMock(name="SDWriter")
    chem_mod.AllChem = allchem

    rdkit_mod = MagicMock(name="rdkit")
    rdkit_mod.Chem = chem_mod

    return {"rdkit": rdkit_mod, "rdkit.Chem": chem_mod}, allchem


_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.000   2.000   3.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C
ATOM      4  CA  GLY A   2       3.000   6.000   9.000  1.00  0.00           C
ATOM      5  CA  SER A   3      11.000  10.000  10.000  1.00  0.00           C
HETATM    6  CA  CA  A 101       5.000   5.000   5.000  1.00  0.00          CA
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


# ---------------------------------------------------------------------------
# smiles_to_3d
# ---------------------------------------------------------------------------

def test_smiles_to_3d_rdkit_missing_raises(tmp_path):
    # rdkit genuinely isn't installed -> the internal import raises ImportError.
    with patch.dict(sys.modules, {"rdkit": None}):
        with pytest.raises(RuntimeError, match="RDKit is not installed"):
            smiles_to_3d("CCO", "ETH", str(tmp_path))


def test_smiles_to_3d_invalid_smiles_raises(tmp_path):
    fake, _ = _fake_rdkit(mol_from_smiles=None)  # MolFromSmiles -> None
    with patch.dict(sys.modules, fake):
        with pytest.raises(ValueError, match="could not parse SMILES"):
            smiles_to_3d("not_a_smiles", "ETH", str(tmp_path))


def test_smiles_to_3d_embed_failure_raises(tmp_path):
    fake, _ = _fake_rdkit(embed_result=-1)  # EmbedMolecule -> -1
    with patch.dict(sys.modules, fake):
        with pytest.raises(RuntimeError, match="ETKDG conformer generation failed"):
            smiles_to_3d("CCO", "ETH", str(tmp_path))


def test_smiles_to_3d_success_returns_path(tmp_path):
    fake, allchem = _fake_rdkit(embed_result=0)
    with patch.dict(sys.modules, fake):
        out = smiles_to_3d("CCO", "ETH", str(tmp_path))
    assert out == os.path.join(str(tmp_path), "ETH.sdf")
    allchem.MMFFOptimizeMolecule.assert_called_once()


# ---------------------------------------------------------------------------
# _ca_centroid / _all_ca_coords (pure Python)
# ---------------------------------------------------------------------------

def test_all_ca_coords_parses_only_ca_atoms(tmp_path):
    pdb = _write(tmp_path, "p.pdb", _PDB)
    coords = _all_ca_coords(pdb)
    # 3 ATOM CA rows; the HETATM "CA" (calcium ion) must be ignored.
    assert coords == [(1.0, 2.0, 3.0), (3.0, 6.0, 9.0), (11.0, 10.0, 10.0)]


def test_ca_centroid_over_subset(tmp_path):
    pdb = _write(tmp_path, "p.pdb", _PDB)
    # residues 1 and 2 -> mean of (1,2,3) and (3,6,9)
    cx, cy, cz = _ca_centroid(pdb, [1, 2])
    assert (round(cx, 3), round(cy, 3), round(cz, 3)) == (2.0, 4.0, 6.0)


def test_ca_centroid_falls_back_to_all_when_no_match(tmp_path):
    pdb = _write(tmp_path, "p.pdb", _PDB)
    # residue 999 doesn't exist -> falls back to the full-protein CA centroid
    cx, cy, cz = _ca_centroid(pdb, [999])
    assert (round(cx, 3), round(cy, 3), round(cz, 3)) == (5.0, 6.0, round(22.0 / 3, 3))


def test_ca_helpers_on_missing_file():
    assert _all_ca_coords("/no/such/file.pdb") == []
    assert _ca_centroid("/no/such/file.pdb", [1]) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# dock_gnina
# ---------------------------------------------------------------------------

def test_dock_gnina_binary_missing_raises(tmp_path):
    with patch("orchestrator.ligands.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="GNINA binary .* not found"):
            dock_gnina("lig.sdf", "rec.pdb", [1], str(tmp_path))


@patch("orchestrator.ligands.subprocess.run")
def test_dock_gnina_binding_site_command(mock_run, tmp_path):
    receptor = _write(tmp_path, "rec.pdb", _PDB)
    out_dir = str(tmp_path)

    def _side(cmd, **kw):
        # simulate GNINA writing its output
        open(os.path.join(out_dir, "docked.sdf"), "w").close()
        return MagicMock(returncode=0, stdout="", stderr="")

    mock_run.side_effect = _side
    with patch("orchestrator.ligands.shutil.which", return_value="/usr/bin/gnina"):
        out = dock_gnina("lig.sdf", receptor, [1, 2], out_dir)

    assert out == os.path.join(out_dir, "docked.sdf")
    cmd = mock_run.call_args.args[0]
    assert "--center_x" in cmd and "--size_x" in cmd
    assert "--autobox_ligand" not in cmd


@patch("orchestrator.ligands.subprocess.run")
def test_dock_gnina_blind_command(mock_run, tmp_path):
    receptor = _write(tmp_path, "rec.pdb", _PDB)
    out_dir = str(tmp_path)

    def _side(cmd, **kw):
        open(os.path.join(out_dir, "docked.sdf"), "w").close()
        return MagicMock(returncode=0, stdout="", stderr="")

    mock_run.side_effect = _side
    with patch("orchestrator.ligands.shutil.which", return_value="/usr/bin/gnina"):
        out = dock_gnina("lig.sdf", receptor, None, out_dir)

    cmd = mock_run.call_args.args[0]
    assert "--autobox_ligand" in cmd
    assert "--center_x" not in cmd


@patch("orchestrator.ligands.subprocess.run")
def test_dock_gnina_nonzero_exit_raises(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=1, stdout="oops-out", stderr="oops-err")
    with patch("orchestrator.ligands.shutil.which", return_value="/usr/bin/gnina"):
        with pytest.raises(RuntimeError) as ei:
            dock_gnina("lig.sdf", _write(tmp_path, "rec.pdb", _PDB), None, str(tmp_path))
    msg = str(ei.value)
    assert "GNINA docking failed" in msg and "oops-out" in msg and "oops-err" in msg


@patch("orchestrator.ligands.subprocess.run")
def test_dock_gnina_missing_output_raises(mock_run, tmp_path):
    # exit 0 but no docked.sdf written
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    with patch("orchestrator.ligands.shutil.which", return_value="/usr/bin/gnina"):
        with pytest.raises(RuntimeError, match="docked.sdf not found"):
            dock_gnina("lig.sdf", _write(tmp_path, "rec.pdb", _PDB), None, str(tmp_path))


# ---------------------------------------------------------------------------
# dock_vina — import guards only (real happy path lives in test_ligands_modal)
# ---------------------------------------------------------------------------

def test_dock_vina_missing_vina_raises(tmp_path):
    with patch.dict(sys.modules, {"vina": None}):
        with pytest.raises(RuntimeError, match="vina is not installed"):
            dock_vina("lig.sdf", "rec.pdb", None, str(tmp_path))


def test_dock_vina_missing_rdkit_raises(tmp_path):
    # vina present (fake) but rdkit absent -> RuntimeError from the RDKit guard
    fake_vina = MagicMock(name="vina")
    with patch.dict(sys.modules, {"vina": fake_vina, "rdkit": None}):
        with pytest.raises(RuntimeError, match="RDKit is required for Vina"):
            dock_vina("lig.sdf", "rec.pdb", None, str(tmp_path))


# ---------------------------------------------------------------------------
# parameterize_ligand_acpype
# ---------------------------------------------------------------------------

def test_acpype_missing_raises(tmp_path):
    with patch("orchestrator.ligands.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="acpype is not installed"):
            parameterize_ligand_acpype("docked.sdf", "LIG", str(tmp_path))


@patch("orchestrator.ligands.subprocess.run")
def test_acpype_success_collects_outputs(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    # ACPYPE writes into <out_dir>/<name>.acpype/
    acpype_dir = tmp_path / "LIG.acpype"
    acpype_dir.mkdir()
    (acpype_dir / "LIG_GMX.itp").write_text("itp")
    (acpype_dir / "LIG_GMX.gro").write_text("gro")
    (acpype_dir / "LIG.mol2").write_text("mol2")

    with patch("orchestrator.ligands.shutil.which", return_value="/usr/bin/acpype"):
        out = parameterize_ligand_acpype("docked.sdf", "LIG", str(tmp_path))

    assert set(out) == {"itp", "gro", "mol2"}
    assert out["itp"].endswith("LIG_GMX.itp")


@patch("orchestrator.ligands.subprocess.run")
def test_acpype_nonzero_exit_raises(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=2, stdout="a-out", stderr="a-err")
    with patch("orchestrator.ligands.shutil.which", return_value="/usr/bin/acpype"):
        with pytest.raises(RuntimeError) as ei:
            parameterize_ligand_acpype("docked.sdf", "LIG", str(tmp_path))
    assert "ACPYPE failed" in str(ei.value)


@patch("orchestrator.ligands.subprocess.run")
def test_acpype_success_but_outputs_missing(mock_run, tmp_path):
    # exit 0 but the .acpype dir / files never appear -> empty dict, no crash
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
    with patch("orchestrator.ligands.shutil.which", return_value="/usr/bin/acpype"):
        out = parameterize_ligand_acpype("docked.sdf", "LIG", str(tmp_path))
    assert out == {}


# ---------------------------------------------------------------------------
# parameterize_ligand_openff — import guard
# ---------------------------------------------------------------------------

def test_openff_missing_toolkit_raises(tmp_path):
    with patch.dict(sys.modules, {"openff.toolkit": None}):
        with pytest.raises(RuntimeError, match="openff-toolkit is not installed"):
            parameterize_ligand_openff("docked.sdf", "LIG", str(tmp_path))


# ---------------------------------------------------------------------------
# prepare_ligands — the GNINA -> Vina -> undocked fallback chain
# ---------------------------------------------------------------------------

def test_prepare_ligands_skips_ligand_without_smiles(tmp_path):
    out = prepare_ligands([{"name": "NOSMI"}], _PDB, str(tmp_path))
    assert out == []


@patch("orchestrator.ligands.smiles_to_3d", side_effect=RuntimeError("rdkit down"))
def test_prepare_ligands_skips_when_conformer_fails(mock_s2d, tmp_path):
    out = prepare_ligands([{"name": "LIG", "smiles": "CCO"}], _PDB, str(tmp_path))
    assert out == []


@patch("orchestrator.ligands.parameterize_ligand_acpype",
       return_value={"itp": "/x/LIG.itp", "gro": "/x/LIG.gro", "mol2": "/x/LIG.mol2"})
@patch("orchestrator.ligands.dock_gnina", return_value="/x/docked.sdf")
@patch("orchestrator.ligands.smiles_to_3d", return_value="/x/LIG.sdf")
def test_prepare_ligands_gnina_success_acpype(mock_s2d, mock_gnina, mock_acpype, tmp_path):
    out = prepare_ligands([{"name": "LIG", "smiles": "CCO", "binding_site": [1]}], _PDB, str(tmp_path))
    assert len(out) == 1
    e = out[0]
    assert e["docked_sdf"] == "/x/docked.sdf"
    assert e["parameterizer"] == "acpype"
    assert e["itp"] == "/x/LIG.itp"
    mock_acpype.assert_called_once()


@patch("orchestrator.ligands.parameterize_ligand_acpype", return_value={})
@patch("orchestrator.ligands.dock_vina", return_value="/x/docked_vina.sdf")
@patch("orchestrator.ligands.dock_gnina", side_effect=RuntimeError("no gnina"))
@patch("orchestrator.ligands.smiles_to_3d", return_value="/x/LIG.sdf")
def test_prepare_ligands_falls_back_to_vina(mock_s2d, mock_gnina, mock_vina, mock_acpype, tmp_path):
    out = prepare_ligands([{"name": "LIG", "smiles": "CCO"}], _PDB, str(tmp_path))
    assert out[0]["docked_sdf"] == "/x/docked_vina.sdf"
    mock_vina.assert_called_once()


@patch("orchestrator.ligands.parameterize_ligand_acpype", return_value={})
@patch("orchestrator.ligands.dock_vina", side_effect=RuntimeError("no vina"))
@patch("orchestrator.ligands.dock_gnina", side_effect=RuntimeError("no gnina"))
@patch("orchestrator.ligands.smiles_to_3d", return_value="/x/LIG.sdf")
def test_prepare_ligands_falls_back_to_undocked(mock_s2d, mock_gnina, mock_vina, mock_acpype, tmp_path):
    out = prepare_ligands([{"name": "LIG", "smiles": "CCO"}], _PDB, str(tmp_path))
    # both dockers failed -> undocked conformer is used as the pose
    assert out[0]["docked_sdf"] == "/x/LIG.sdf"


@patch("orchestrator.ligands.parameterize_ligand_openff", return_value={"xml": "/x/LIG.xml"})
@patch("orchestrator.ligands.dock_gnina", return_value="/x/docked.sdf")
@patch("orchestrator.ligands.smiles_to_3d", return_value="/x/LIG.sdf")
def test_prepare_ligands_use_openff(mock_s2d, mock_gnina, mock_openff, tmp_path):
    out = prepare_ligands(
        [{"name": "LIG", "smiles": "CCO"}], _PDB, str(tmp_path), use_openff=True
    )
    assert out[0]["parameterizer"] == "openff"
    assert out[0]["xml"] == "/x/LIG.xml"
    mock_openff.assert_called_once()


@patch("orchestrator.ligands.parameterize_ligand_acpype", side_effect=RuntimeError("acpype boom"))
@patch("orchestrator.ligands.dock_gnina", return_value="/x/docked.sdf")
@patch("orchestrator.ligands.smiles_to_3d", return_value="/x/LIG.sdf")
def test_prepare_ligands_parameterization_failure_is_none(mock_s2d, mock_gnina, mock_acpype, tmp_path):
    out = prepare_ligands([{"name": "LIG", "smiles": "CCO"}], _PDB, str(tmp_path))
    # entry is still returned, docked, but parameterizer falls back to "none"
    assert len(out) == 1
    assert out[0]["parameterizer"] == "none"
    assert out[0]["docked_sdf"] == "/x/docked.sdf"
