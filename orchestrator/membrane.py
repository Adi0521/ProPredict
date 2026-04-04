"""
Stage F — Membrane environment support.

Two backends:
  1. insane.py  — GROMACS workflow: embed protein in lipid bilayer, generate .gro + .top
  2. OpenMM     — addMembrane via openmmforcefields (CHARMM36m lipids)

Usage
-----
from orchestrator.membrane import embed_in_membrane_gromacs, embed_in_membrane_openmm

# GROMACS
gro_path, top_path = embed_in_membrane_gromacs(pdb_string, membrane_context, tmpdir)

# OpenMM
modeller = embed_in_membrane_openmm(modeller, ff, membrane_context)
"""

import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported lipid types and their insane.py / CHARMM36m names
# ---------------------------------------------------------------------------

_INSANE_LIPID_MAP: Dict[str, str] = {
    "POPC":  "POPC",
    "POPE":  "POPE",
    "DPPC":  "DPPC",
    "DMPC":  "DMPC",
    "POPG":  "POPG",
    "POPS":  "POPS",
    "CHOL":  "CHOL",
}

_DEFAULT_LIPID = "POPC"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_insane(insane_path: str) -> Optional[str]:
    """
    Return the path to the insane.py script (or None if not found).

    Looks in order:
      1. insane_path from config
      2. 'insane.py' on PATH
      3. 'insane' on PATH (compiled version)
    """
    if insane_path and os.path.isfile(insane_path):
        return insane_path
    for candidate in ("insane.py", "insane"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def _lipid_name(membrane_type: Optional[str]) -> str:
    """Normalise a membrane type string to an insane.py lipid name."""
    if not membrane_type:
        return _DEFAULT_LIPID
    upper = membrane_type.upper()
    return _INSANE_LIPID_MAP.get(upper, upper)


# ---------------------------------------------------------------------------
# GROMACS membrane embedding via insane.py
# ---------------------------------------------------------------------------


def embed_in_membrane_gromacs(
    pdb_string: str,
    membrane_context: Dict[str, Any],
    tmpdir: str,
    insane_path: str = "",
    membrane_ff: str = "charmm36m-ildn",
) -> Tuple[str, str]:
    """
    Embed a protein in a lipid bilayer for GROMACS using insane.py.

    Steps:
      1. Write the protein PDB to tmpdir.
      2. Run insane.py to build a combined protein + membrane .gro and .top.
      3. Return (gro_path, top_path) for further GROMACS pre-processing.

    Parameters
    ----------
    pdb_string : str
        Protein structure in PDB format (post-pdb2gmx if possible — insane
        expects a GROMACS .gro, but pdb is accepted by newer versions).
    membrane_context : dict
        Should contain at minimum {"type": "POPC"}.
        Optionally: {"span": [start_res, end_res]} for TM protein positioning.
    tmpdir : str
        Directory in which all intermediate files are written.
    insane_path : str
        Explicit path to insane.py; if empty, searched on PATH.
    membrane_ff : str
        GROMACS force-field directory name for pdb2gmx (default: charmm36m-ildn).

    Returns
    -------
    (gro_path, top_path) : Tuple[str, str]
        Absolute paths to the combined .gro and .top files ready for GROMACS.

    Raises
    ------
    RuntimeError
        If insane.py is not found or the subprocess fails.
    """
    script = _resolve_insane(insane_path)
    if script is None:
        raise RuntimeError(
            "insane.py not found. Download from the Tieleman lab "
            "(http://cgmartini.nl/index.php/tools2/proteins-and-bilayers) "
            "and set INSANE_PATH in .env, or place insane.py on PATH."
        )

    lipid = _lipid_name(membrane_context.get("type"))
    logger.info(f"insane.py: embedding protein in {lipid} bilayer...")

    pdb_path = os.path.join(tmpdir, "protein_for_insane.pdb")
    with open(pdb_path, "w") as fh:
        fh.write(pdb_string)

    gro_out = os.path.join(tmpdir, "membrane_system.gro")
    top_out = os.path.join(tmpdir, "membrane_system.top")

    # Build the insane.py command
    # -f  protein input (PDB or GRO)
    # -o  output coordinate file (.gro)
    # -p  output topology file (.top)
    # -pbc square — rectangular box by default
    # -l  lipid type and ratio (e.g. POPC:1)
    # -d  slab distance (bilayer) = 0 means auto
    # -z  box z-dimension (nm); set large enough
    insane_cmd = [
        "python", script,
        "-f", pdb_path,
        "-o", gro_out,
        "-p", top_out,
        "-pbc", "square",
        "-l", f"{lipid}:1",
        "-d", "0",
        "-z", "10",
    ]

    # TM protein positioning: centre the protein at z=0 (membrane midplane)
    span = membrane_context.get("span")
    if span and len(span) == 2:
        logger.info(f"insane.py: TM span residues {span[0]}–{span[1]}, centering in bilayer")
        # insane -center flag places the geometric centre at z=0
        insane_cmd += ["-center"]

    logger.debug(f"insane.py command: {' '.join(insane_cmd)}")
    result = subprocess.run(
        insane_cmd,
        capture_output=True,
        text=True,
        cwd=tmpdir,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"insane.py failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout[-800:]}\n"
            f"stderr: {result.stderr[-800:]}"
        )

    if not os.path.isfile(gro_out) or not os.path.isfile(top_out):
        raise RuntimeError(
            "insane.py completed but output files are missing. "
            f"Expected: {gro_out}, {top_out}"
        )

    logger.info(f"insane.py: membrane system written to {gro_out}")
    return gro_out, top_out


# ---------------------------------------------------------------------------
# OpenMM membrane embedding via openmmforcefields (CHARMM36m)
# ---------------------------------------------------------------------------


def embed_in_membrane_openmm(
    modeller: Any,
    ff: Any,
    membrane_context: Dict[str, Any],
) -> Any:
    """
    Embed a protein in a lipid bilayer for OpenMM using CHARMM36m lipid parameters.

    Requires:
        conda install -c conda-forge openmmforcefields
        pip install openmmforcefields

    The membrane is built by calling Modeller.addMembrane() which:
      - Places the protein in a pre-built POPC (or other) bilayer patch
      - Solvates the upper/lower leaflets and water above/below
      - Returns a system ready for force field parameterisation with CHARMM36m

    Parameters
    ----------
    modeller : openmm.app.Modeller
        Modeller that already has hydrogens added (via addHydrogens(pH=pH)).
        Positions should be in nm.
    ff : openmm.app.ForceField
        The ForceField object — must include CHARMM36m lipid XML.
        Load with:
            from openmmforcefields.generators import SystemGenerator
            ff = ForceField(
                "charmm36.xml",
                "charmm36/water.xml",
                "charmm36/lipids.xml",
            )
    membrane_context : dict
        Should contain {"type": "POPC"} (used to select the lipid residue name).

    Returns
    -------
    modeller : openmm.app.Modeller
        The updated Modeller with membrane + water added.

    Raises
    ------
    RuntimeError
        If openmm or openmmforcefields are not installed.
    ImportError
        (Re-raised) if required packages are absent.
    """
    try:
        from openmm import unit
        from openmm.app import Modeller
    except ImportError:
        raise RuntimeError(
            "OpenMM is not installed. "
            "Install: conda install -c conda-forge openmm"
        )

    try:
        # openmmforcefields provides CHARMM36m lipids
        import openmmforcefields  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "openmmforcefields is not installed. "
            "Install: conda install -c conda-forge openmmforcefields\n"
            "This package is required for CHARMM36m lipid parameters."
        )

    lipid = _lipid_name(membrane_context.get("type"))
    logger.info(f"OpenMM: embedding protein in {lipid} bilayer using CHARMM36m...")

    # addMembrane places the protein in a periodic lipid bilayer.
    # lipidType must be a residue name recognised by the CHARMM36m lipid force field.
    # minimumPadding adds water above/below the bilayer (nm).
    try:
        modeller.addMembrane(
            ff,
            lipidType=lipid,
            minimumPadding=1.0 * unit.nanometers,
            positiveIon="Na+",
            negativeIon="Cl-",
            ionicStrength=0.15 * unit.molar,
        )
    except Exception as e:
        raise RuntimeError(
            f"OpenMM addMembrane failed for lipid '{lipid}': {e}\n"
            "Check that the lipid type is supported by CHARMM36m "
            "and that openmmforcefields is installed."
        )

    n_atoms = modeller.topology.getNumAtoms()
    logger.info(f"OpenMM: membrane system has {n_atoms} atoms")
    return modeller
