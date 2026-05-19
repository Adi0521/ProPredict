import io
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Optional, Dict, Any, List, Tuple

from config import (
    GROMACS_BIN,
    GNINA_BIN,
    INSANE_PATH,
    MEMBRANE_FF,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PyRosetta relax (optional — requires: conda install -c rosettacommons pyrosetta)
# ---------------------------------------------------------------------------

def run_rosetta_relax(pdb_string: str) -> Tuple[str, float]:
    """
    Run Rosetta FastRelax on a structure using PyRosetta.

    Returns (relaxed_pdb_string, rosetta_score).
    Raises RuntimeError if PyRosetta is not installed.
    """
    try:
        import pyrosetta  # type: ignore
    except ImportError:
        raise RuntimeError(
            "PyRosetta is not installed. "
            "Install with: conda install -c rosettacommons pyrosetta"
        )

    logger.info("Initialising PyRosetta...")
    pyrosetta.init("-mute all")

    pose = pyrosetta.pose_from_pdbstring(pdb_string)
    scorefxn = pyrosetta.create_score_function("ref2015_cart")

    relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(scorefxn)
    relax.apply(pose)

    score = scorefxn(pose)
    logger.info(f"Rosetta FastRelax complete. Score: {score:.3f}")

    pdb_out = io.StringIO()
    pose.dump_pdb(pdb_out)
    return pdb_out.getvalue(), score


# ---------------------------------------------------------------------------
# Protonation state assignment via PropKa3 (optional — pip install propka)
# ---------------------------------------------------------------------------

_MODEL_PKA = {"HIS": 6.0, "ASP": 3.9, "GLU": 4.1, "LYS": 10.5, "TYR": 10.1, "CYS": 8.3}


def _run_propka(pdb_path: str) -> Dict[Tuple[int, str, str], float]:
    """
    Run PropKa3 on a PDB file and return per-residue pKa values.

    Key is (res_num, chain_id, res_name).  Returns an empty dict if propka
    is not installed or the run fails — callers fall back to model pKa values.
    """
    try:
        from propka.run import single as propka_single
    except ImportError:
        logger.warning(
            "PropKa3 not installed — protonation assignment will use model pKa values "
            "(pip install propka)"
        )
        return {}

    try:
        mol = propka_single(pdb_path, optargs=["--quiet"])

        conf = None
        for name in ("AVR", *mol.conformations.keys()):
            if name in mol.conformations:
                conf = mol.conformations[name]
                break
        if conf is None:
            return {}

        pkas: Dict[Tuple[int, str, str], float] = {}
        for group in conf.groups:
            if not hasattr(group, "pka_value") or group.pka_value is None:
                continue
            try:
                res_name = group.res_name.strip()
                res_num = int(group.res_num)
                chain = group.chain_id.strip()
            except AttributeError:
                try:
                    parts = group.label.split()
                    res_name = parts[0]
                    res_num = int(parts[1])
                    chain = parts[2] if len(parts) > 2 else "A"
                except (AttributeError, IndexError, ValueError):
                    continue

            if res_name in ("HIS", "ASP", "GLU", "LYS", "TYR", "CYS"):
                pkas[(res_num, chain, res_name)] = group.pka_value

        logger.info(f"PropKa3: computed pKa for {len(pkas)} titratable groups")
        return pkas

    except Exception as e:
        logger.warning(f"PropKa3 run failed: {e} — falling back to model pKa values")
        return {}


def _get_titratable_residues(
    pdb_string: str, res_names: set
) -> List[Tuple[int, str, str]]:
    """
    Return (res_num, chain_id, res_name) tuples for matching residues in PDB
    record order — the same order pdb2gmx will ask protonation questions.
    """
    seen: set = set()
    residues: List[Tuple[int, str, str]] = []
    for line in pdb_string.splitlines():
        if not line.startswith("ATOM"):
            continue
        res_name = line[17:20].strip()
        if res_name not in res_names:
            continue
        chain = line[21]
        try:
            res_num = int(line[22:26].strip())
        except ValueError:
            continue
        key = (res_num, chain)
        if key not in seen:
            seen.add(key)
            residues.append((res_num, chain, res_name))
    return residues


def _determine_protonation_states(
    pdb_string: str,
    pH: float,
    pka_dict: Dict[Tuple[int, str, str], float],
) -> Dict[str, List[int]]:
    """
    Determine pdb2gmx protonation state integers for HIS, ASP, GLU at a given pH.

    pdb2gmx integer encoding:
      HIS: 0 = HID (H on ND1), 1 = HIE (H on NE2), 2 = HIP (both, charged)
      ASP: 0 = deprotonated (standard), 1 = protonated (ASPH)
      GLU: 0 = deprotonated (standard), 1 = protonated (GLUH)
    """
    def _state(res_num: int, chain: str, res_name: str, charged_code: int, neutral_code: int) -> int:
        pka = pka_dict.get((res_num, chain, res_name), _MODEL_PKA[res_name])
        return charged_code if pka > pH else neutral_code

    his_residues = _get_titratable_residues(pdb_string, {"HIS"})
    asp_residues = _get_titratable_residues(pdb_string, {"ASP"})
    glu_residues = _get_titratable_residues(pdb_string, {"GLU"})

    his_states = [_state(n, c, r, 2, 1) for n, c, r in his_residues]
    asp_states = [_state(n, c, r, 1, 0) for n, c, r in asp_residues]
    glu_states = [_state(n, c, r, 1, 0) for n, c, r in glu_residues]

    charged_his = sum(1 for s in his_states if s == 2)
    protonated_asp = sum(asp_states)
    protonated_glu = sum(glu_states)
    logger.info(
        f"Protonation at pH {pH:.1f}: "
        f"{charged_his}/{len(his_states)} HIS charged, "
        f"{protonated_asp}/{len(asp_states)} ASP protonated, "
        f"{protonated_glu}/{len(glu_states)} GLU protonated"
    )
    return {"his": his_states, "asp": asp_states, "glu": glu_states}


# ---------------------------------------------------------------------------
# GROMACS energy minimisation
# ---------------------------------------------------------------------------

_GROMACS_EM_MDP = """\
; Steepest-descent energy minimisation
integrator  = steep
emtol       = 1000.0
emstep      = 0.01
nsteps      = 1000
nstlist     = 1
cutoff-scheme = Verlet
ns_type     = grid
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
"""


def run_gromacs_em(pdb_string: str, pH: float = 7.4) -> Dict[str, Any]:
    """
    Run GROMACS energy minimisation on a structure.

    Pipeline: PropKa -> pdb2gmx -> editconf -> solvate -> grompp -> mdrun -> energy
    Returns {"potential_energy": float, "protonation_summary": dict}.
    Raises RuntimeError if gmx binary is not found or any step fails.
    """
    gmx = shutil.which(GROMACS_BIN)
    if gmx is None:
        raise RuntimeError(
            f"GROMACS binary '{GROMACS_BIN}' not found. "
            "Install with: brew install gromacs  (M3 Mac)  "
            "or set GROMACS_BIN to the correct path."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "input.pdb")
        mdp_path = os.path.join(tmpdir, "em.mdp")

        with open(pdb_path, "w") as f:
            f.write(pdb_string)
        with open(mdp_path, "w") as f:
            f.write(_GROMACS_EM_MDP)

        def _gmx(*args, stdin_input: Optional[str] = None):
            """Run a gmx sub-command, raise on failure."""
            cmd = [gmx] + list(args)
            logger.debug(f"Running: {' '.join(cmd)}")
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                cwd=tmpdir,
                input=stdin_input,
                text=(stdin_input is not None),
            )

        logger.info(f"Running PropKa3 for protonation assignment at pH {pH:.1f}...")
        pka_dict = _run_propka(pdb_path)
        protonation = _determine_protonation_states(pdb_string, pH, pka_dict)

        pdb2gmx_args = [
            "pdb2gmx", "-f", "input.pdb", "-o", "processed.gro",
            "-p", "topol.top", "-ff", "amber99sb-ildn", "-water", "spc", "-ignh",
        ]
        stdin_lines: List[int] = []
        if protonation["his"]:
            pdb2gmx_args.append("-his")
            stdin_lines.extend(protonation["his"])
        if protonation["asp"]:
            pdb2gmx_args.append("-asp")
            stdin_lines.extend(protonation["asp"])
        if protonation["glu"]:
            pdb2gmx_args.append("-glu")
            stdin_lines.extend(protonation["glu"])

        stdin_input = "\n".join(str(x) for x in stdin_lines) + "\n" if stdin_lines else None

        logger.info("GROMACS: converting PDB to GROMACS format (with pH-aware protonation)...")
        _gmx(*pdb2gmx_args, stdin_input=stdin_input)

        logger.info("GROMACS: setting up simulation box...")
        _gmx("editconf", "-f", "processed.gro", "-o", "box.gro",
             "-c", "-d", "1.0", "-bt", "cubic")

        logger.info("GROMACS: solvating...")
        _gmx("solvate", "-cp", "box.gro", "-cs", "spc216.gro",
             "-o", "solvated.gro", "-p", "topol.top")

        logger.info("GROMACS: preparing energy minimisation run...")
        _gmx("grompp", "-f", "em.mdp", "-c", "solvated.gro",
             "-p", "topol.top", "-o", "em.tpr")

        logger.info("GROMACS: running energy minimisation...")
        _gmx("mdrun", "-v", "-deffnm", "em")

        logger.info("GROMACS: extracting potential energy...")
        energy_proc = subprocess.run(
            [gmx, "energy", "-f", "em.edr", "-o", "energy.xvg"],
            input="Potential\n",
            text=True,
            capture_output=True,
            cwd=tmpdir,
        )

        potential_energy = _parse_gromacs_energy(
            os.path.join(tmpdir, "energy.xvg")
        )
        logger.info(f"GROMACS EM complete. Potential energy: {potential_energy:.3f} kJ/mol")
        return {
            "potential_energy": potential_energy,
            "pH": pH,
            "protonation": protonation,
        }


def _parse_gromacs_energy(xvg_path: str) -> float:
    """Parse the last potential energy value from a GROMACS .xvg file."""
    last_value = None
    with open(xvg_path) as f:
        for line in f:
            if line.startswith(("#", "@")):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    last_value = float(parts[1])
                except ValueError:
                    pass
    if last_value is None:
        raise RuntimeError("Could not parse potential energy from GROMACS energy.xvg")
    return last_value


def _parse_gromacs_xvg(xvg_path: str) -> List[float]:
    """Parse all Y-column values from a GROMACS XVG file (e.g. RMSD, Rg)."""
    values: List[float] = []
    try:
        with open(xvg_path) as f:
            for line in f:
                if line.startswith(("#", "@")):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        values.append(float(parts[1]))
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return values


# ---------------------------------------------------------------------------
# GROMACS full MD pipeline (NVT -> NPT -> Production)
# ---------------------------------------------------------------------------

_GROMACS_IONS_MDP = """\
; Minimal MDP used only to generate a .tpr for ion placement
integrator    = steep
emtol         = 1000.0
nsteps        = 0
cutoff-scheme = Verlet
coulombtype   = PME
rcoulomb      = 1.0
rvdw          = 1.0
pbc           = xyz
"""


def _make_nvt_mdp(temperature_k: float, nsteps: int = 50000) -> str:
    """NVT equilibration MDP (~100 ps at dt=0.002 ps with default nsteps=50000)."""
    return f"""\
; NVT equilibration — {nsteps * 0.002:.0f} ps
integrator          = md
dt                  = 0.002
nsteps              = {nsteps}
nstenergy           = 500
nstlog              = 500
nstxout-compressed  = 500
continuation        = no
constraint_algorithm = lincs
constraints         = h-bonds
lincs_iter          = 1
lincs_order         = 4
cutoff-scheme       = Verlet
ns_type             = grid
nstlist             = 10
rcoulomb            = 1.0
rvdw                = 1.0
coulombtype         = PME
pme_order           = 4
fourierspacing      = 0.16
tcoupl              = V-rescale
tc-grps             = Protein Non-Protein
tau_t               = 0.1    0.1
ref_t               = {temperature_k:.2f}  {temperature_k:.2f}
pcoupl              = no
pbc                 = xyz
DispCorr            = EnerPres
gen_vel             = yes
gen_temp            = {temperature_k:.2f}
gen_seed            = -1
"""


def _make_npt_mdp(temperature_k: float, nsteps: int = 50000) -> str:
    """NPT equilibration MDP (~100 ps). Continues from NVT checkpoint."""
    return f"""\
; NPT equilibration — {nsteps * 0.002:.0f} ps
integrator          = md
dt                  = 0.002
nsteps              = {nsteps}
nstenergy           = 500
nstlog              = 500
nstxout-compressed  = 500
continuation        = yes
constraint_algorithm = lincs
constraints         = h-bonds
lincs_iter          = 1
lincs_order         = 4
cutoff-scheme       = Verlet
ns_type             = grid
nstlist             = 10
rcoulomb            = 1.0
rvdw                = 1.0
coulombtype         = PME
pme_order           = 4
fourierspacing      = 0.16
tcoupl              = V-rescale
tc-grps             = Protein Non-Protein
tau_t               = 0.1    0.1
ref_t               = {temperature_k:.2f}  {temperature_k:.2f}
pcoupl              = Parrinello-Rahman
pcoupltype          = isotropic
tau_p               = 2.0
ref_p               = 1.0
compressibility     = 4.5e-5
refcoord_scaling    = com
pbc                 = xyz
DispCorr            = EnerPres
gen_vel             = no
"""


def _make_production_mdp(temperature_k: float, nsteps: int) -> str:
    """Production MD MDP. Continues from NPT checkpoint."""
    return f"""\
; Production MD — {nsteps * 0.002 / 1000:.3f} ns
integrator          = md
dt                  = 0.002
nsteps              = {nsteps}
nstenergy           = 5000
nstlog              = 5000
nstxout-compressed  = 5000
continuation        = yes
constraint_algorithm = lincs
constraints         = h-bonds
lincs_iter          = 1
lincs_order         = 4
cutoff-scheme       = Verlet
ns_type             = grid
nstlist             = 10
rcoulomb            = 1.0
rvdw                = 1.0
coulombtype         = PME
pme_order           = 4
fourierspacing      = 0.16
tcoupl              = V-rescale
tc-grps             = Protein Non-Protein
tau_t               = 0.1    0.1
ref_t               = {temperature_k:.2f}  {temperature_k:.2f}
pcoupl              = Parrinello-Rahman
pcoupltype          = isotropic
tau_p               = 2.0
ref_p               = 1.0
compressibility     = 4.5e-5
pbc                 = xyz
DispCorr            = EnerPres
gen_vel             = no
"""


def _analyze_gromacs_trajectory(tmpdir: str, gmx: str) -> Dict[str, Any]:
    """Compute backbone RMSD and protein Rg from the production trajectory."""
    results: Dict[str, Any] = {}

    try:
        subprocess.run(
            [gmx, "rms", "-s", "prod.tpr", "-f", "prod.xtc", "-o", "rmsd.xvg"],
            input="Backbone\nBackbone\n",
            text=True,
            capture_output=True,
            cwd=tmpdir,
            check=True,
        )
        rmsd = _parse_gromacs_xvg(os.path.join(tmpdir, "rmsd.xvg"))
        if rmsd:
            results["rmsd_nm"] = rmsd
            results["rmsd_final_nm"] = rmsd[-1]
    except Exception as e:
        logger.warning(f"RMSD analysis failed: {e}")

    try:
        subprocess.run(
            [gmx, "gyrate", "-s", "prod.tpr", "-f", "prod.xtc", "-o", "gyrate.xvg"],
            input="Protein\n",
            text=True,
            capture_output=True,
            cwd=tmpdir,
            check=True,
        )
        rg = _parse_gromacs_xvg(os.path.join(tmpdir, "gyrate.xvg"))
        if rg:
            results["rg_nm"] = rg
    except Exception as e:
        logger.warning(f"Rg analysis failed: {e}")

    return results


def run_gromacs_md(
    pdb_string: str,
    pH: float = 7.4,
    temperature_c: float = 25.0,
    production_ns: float = 0.1,
    membrane_context: Optional[Dict[str, Any]] = None,
    ligand_contexts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Full GROMACS MD pipeline.

    Stages: PropKa -> pdb2gmx -> [insane.py membrane embed] ->
            editconf -> solvate -> [ligand topology merge] -> genion ->
            EM -> NVT (100 ps) -> NPT (100 ps) -> Production -> Analysis

    membrane_context : dict | None
        MembraneContext dict (keys: "type", "span"). When provided, insane.py
        embeds the protein in a lipid bilayer before solvation.
    ligand_contexts : list of dict | None
        List of LigandContext dicts (keys: "name", "smiles", "binding_site").
        When provided, GNINA docking + ACPYPE parameterization is run for each
        ligand with a SMILES string, and their .itp topologies are merged into
        the system before EM.

    Returns potential_energy, rmsd_nm, rg_nm, pH, temperature_c, protonation,
    and optionally membrane/ligand metadata.
    Raises RuntimeError if gmx binary is not found or any stage fails.
    """
    gmx = shutil.which(GROMACS_BIN)
    if gmx is None:
        raise RuntimeError(
            f"GROMACS binary '{GROMACS_BIN}' not found. "
            "Install: brew install gromacs  (M3 Mac) or see Dockerfile.celery."
        )

    temperature_k = temperature_c + 273.15
    production_steps = int(production_ns * 1_000_000 / 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb_path = os.path.join(tmpdir, "input.pdb")
        with open(pdb_path, "w") as f:
            f.write(pdb_string)

        for name, content in [
            ("ions.mdp",  _GROMACS_IONS_MDP),
            ("em.mdp",    _GROMACS_EM_MDP),
            ("nvt.mdp",   _make_nvt_mdp(temperature_k)),
            ("npt.mdp",   _make_npt_mdp(temperature_k)),
            ("prod.mdp",  _make_production_mdp(temperature_k, production_steps)),
        ]:
            with open(os.path.join(tmpdir, name), "w") as f:
                f.write(content)

        def _gmx(*args, stdin_input: Optional[str] = None):
            cmd = [gmx] + list(args)
            logger.debug(f"Running: {' '.join(cmd)}")
            subprocess.run(
                cmd, check=True, capture_output=True, cwd=tmpdir,
                input=stdin_input, text=(stdin_input is not None),
            )

        # --- Protonation ---
        pka_dict = _run_propka(pdb_path)
        protonation = _determine_protonation_states(pdb_string, pH, pka_dict)

        ff_name = MEMBRANE_FF if membrane_context else "amber99sb-ildn"

        pdb2gmx_args = [
            "pdb2gmx", "-f", "input.pdb", "-o", "processed.gro",
            "-p", "topol.top", "-ff", ff_name, "-water", "spc", "-ignh",
        ]
        stdin_lines: List[int] = []
        if protonation["his"]:
            pdb2gmx_args.append("-his")
            stdin_lines.extend(protonation["his"])
        if protonation["asp"]:
            pdb2gmx_args.append("-asp")
            stdin_lines.extend(protonation["asp"])
        if protonation["glu"]:
            pdb2gmx_args.append("-glu")
            stdin_lines.extend(protonation["glu"])
        pdb2gmx_stdin = "\n".join(str(x) for x in stdin_lines) + "\n" if stdin_lines else None

        logger.info("GROMACS MD: pdb2gmx (pH-aware protonation)...")
        _gmx(*pdb2gmx_args, stdin_input=pdb2gmx_stdin)

        # --- Membrane embedding (insane.py) ---
        membrane_meta: Dict[str, Any] = {}
        if membrane_context:
            try:
                from orchestrator.membrane import embed_in_membrane_gromacs
                gro_mem, top_mem = embed_in_membrane_gromacs(
                    pdb_string, membrane_context, tmpdir,
                    insane_path=INSANE_PATH, membrane_ff=MEMBRANE_FF,
                )
                shutil.copy(gro_mem, os.path.join(tmpdir, "processed.gro"))
                shutil.copy(top_mem, os.path.join(tmpdir, "topol.top"))
                membrane_meta = {
                    "membrane_type": membrane_context.get("type", "POPC"),
                    "membrane_embedded": True,
                }
                logger.info(f"GROMACS MD: membrane embedding complete ({membrane_meta})")
            except RuntimeError as e:
                logger.warning(f"Membrane embedding skipped: {e}")
                membrane_meta = {"membrane_embedded": False, "membrane_error": str(e)}

        # --- Box, solvation, ions ---
        _gmx("editconf", "-f", "processed.gro", "-o", "box.gro", "-c", "-d", "1.0", "-bt", "cubic")
        _gmx("solvate", "-cp", "box.gro", "-cs", "spc216.gro", "-o", "solvated.gro", "-p", "topol.top")

        # --- Ligand preparation and topology merge ---
        ligand_meta: List[Dict[str, Any]] = []
        if ligand_contexts:
            try:
                from orchestrator.ligands import prepare_ligands
                prepared = prepare_ligands(
                    ligand_contexts, pdb_string, tmpdir, gnina_bin=GNINA_BIN
                )
                for lig in prepared:
                    itp = lig.get("itp")
                    if itp and os.path.isfile(itp):
                        with open(os.path.join(tmpdir, "topol.top"), "a") as fh:
                            fh.write(f'\n; Ligand {lig["name"]}\n')
                            fh.write(f'#include "{itp}"\n')
                        ligand_meta.append({
                            "name": lig["name"],
                            "parameterizer": lig["parameterizer"],
                            "docked": lig["docked_sdf"] is not None,
                        })
                        logger.info(f"GROMACS MD: ligand '{lig['name']}' topology merged")
                    else:
                        logger.warning(
                            f"Ligand '{lig['name']}': no .itp produced — "
                            "skipping topology merge"
                        )
            except Exception as e:
                logger.warning(f"Ligand preparation failed: {e}")

        logger.info("GROMACS MD: adding neutralizing ions...")
        _gmx("grompp", "-f", "ions.mdp", "-c", "solvated.gro", "-p", "topol.top",
             "-o", "ions.tpr", "-maxwarn", "1")
        _gmx("genion", "-s", "ions.tpr", "-o", "neutralized.gro", "-p", "topol.top",
             "-pname", "NA", "-nname", "CL", "-neutral", stdin_input="SOL\n")

        # --- Energy minimization ---
        logger.info("GROMACS MD: energy minimization...")
        _gmx("grompp", "-f", "em.mdp", "-c", "neutralized.gro", "-p", "topol.top", "-o", "em.tpr")
        _gmx("mdrun", "-v", "-deffnm", "em", "-ntmpi", "1")

        subprocess.run(
            [gmx, "energy", "-f", "em.edr", "-o", "em_energy.xvg"],
            input="Potential\n", text=True, capture_output=True, cwd=tmpdir,
        )
        potential_energy = _parse_gromacs_energy(os.path.join(tmpdir, "em_energy.xvg"))
        logger.info(f"GROMACS MD: EM done. PE = {potential_energy:.1f} kJ/mol")

        # --- NVT equilibration ---
        logger.info("GROMACS MD: NVT equilibration (100 ps)...")
        _gmx("grompp", "-f", "nvt.mdp", "-c", "em.gro", "-r", "em.gro",
             "-p", "topol.top", "-o", "nvt.tpr")
        _gmx("mdrun", "-v", "-deffnm", "nvt", "-ntmpi", "1")

        # --- NPT equilibration ---
        logger.info("GROMACS MD: NPT equilibration (100 ps)...")
        _gmx("grompp", "-f", "npt.mdp", "-c", "nvt.gro", "-r", "nvt.gro",
             "-t", "nvt.cpt", "-p", "topol.top", "-o", "npt.tpr")
        _gmx("mdrun", "-v", "-deffnm", "npt", "-ntmpi", "1")

        # --- Production MD ---
        logger.info(f"GROMACS MD: production ({production_ns} ns)...")
        _gmx("grompp", "-f", "prod.mdp", "-c", "npt.gro", "-t", "npt.cpt",
             "-p", "topol.top", "-o", "prod.tpr")
        _gmx("mdrun", "-v", "-deffnm", "prod", "-ntmpi", "1")

        # --- Trajectory analysis ---
        analysis = _analyze_gromacs_trajectory(tmpdir, gmx)

        logger.info("GROMACS MD pipeline complete.")
        return {
            "potential_energy": potential_energy,
            "pH": pH,
            "temperature_c": temperature_c,
            "production_ns": production_ns,
            "protonation": protonation,
            "backend": "gromacs",
            **membrane_meta,
            **({"ligands": ligand_meta} if ligand_meta else {}),
            **analysis,
        }


# ---------------------------------------------------------------------------
# OpenMM simulation backend (optional — conda install -c conda-forge openmm)
# ---------------------------------------------------------------------------

def _compute_openmm_trajectory_metrics(
    frames: List[Any], ca_indices: List[int]
) -> Tuple[List[float], List[float]]:
    """
    Compute per-frame CA RMSD (vs frame 0) and radius of gyration from OpenMM frames.
    Positions are expected in nanometres (OpenMM native unit).
    """
    import numpy as np

    if not frames or not ca_indices:
        return [], []

    ref = frames[0][ca_indices]
    rmsd_list: List[float] = []
    rg_list: List[float] = []

    for positions in frames:
        ca = positions[ca_indices]
        diff = ca - ref
        rmsd_list.append(float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))))
        center = ca.mean(axis=0)
        rg_list.append(float(np.sqrt(np.mean(np.sum((ca - center) ** 2, axis=1)))))

    return rmsd_list, rg_list


def run_openmm_simulation(
    pdb_string: str,
    pH: float = 7.4,
    temperature_c: float = 25.0,
    production_ns: float = 0.1,
    membrane_context: Optional[Dict[str, Any]] = None,
    ligand_contexts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Full OpenMM MD simulation (Python-native, no subprocesses).

    Pipeline: add H at pH -> [membrane embed OR solvate] ->
              [ligand parameterization + topology merge] ->
              EM -> NVT (50 ps) -> NPT (50 ps) -> Production -> Analysis

    membrane_context : dict | None
        MembraneContext dict. When provided, Modeller.addMembrane() embeds
        the protein in a CHARMM36m lipid bilayer instead of plain solvation.
        Requires: conda install -c conda-forge openmmforcefields
    ligand_contexts : list of dict | None
        LigandContext dicts. When provided, GNINA docking + OpenFF SMIRNOFF
        parameterization runs for each ligand with a SMILES string, and the
        resulting OpenMM system XML is merged into the simulation.

    pH-aware protonation is handled natively by Modeller.addHydrogens(pH=pH).
    Raises RuntimeError if openmm is not installed.
    """
    try:
        import openmm as omm
        from openmm import unit
        from openmm.app import PDBFile, ForceField, Modeller, Simulation, PME, HBonds
    except ImportError:
        raise RuntimeError(
            "OpenMM is not installed. "
            "Install: conda install -c conda-forge openmm"
        )

    import numpy as np

    temperature = (temperature_c + 273.15) * unit.kelvin
    dt = 0.002 * unit.picoseconds
    production_steps = int(production_ns * 1_000_000 / 2)
    report_interval = max(1, production_steps // 100)

    logger.info("OpenMM: fixing PDB (missing terminals, heavy atoms)...")
    try:
        from pdbfixer import PDBFixer
        fixer = PDBFixer(pdbfile=io.StringIO(pdb_string))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixed_pdb_io = io.StringIO()
        PDBFile.writeFile(fixer.topology, fixer.positions, fixed_pdb_io)
        fixed_pdb_io.seek(0)
        pdb = PDBFile(fixed_pdb_io)
    except ImportError:
        logger.warning("pdbfixer not installed — skipping PDB fixing (pip install pdbfixer)")
        pdb = PDBFile(io.StringIO(pdb_string))

    logger.info(f"OpenMM: adding hydrogens at pH {pH:.1f}...")

    if membrane_context:
        try:
            ff = ForceField("charmm36.xml", "charmm36/water.xml", "charmm36/lipids.xml")
        except Exception:
            logger.warning("CHARMM36m XML not found — falling back to AMBER14")
            ff = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    else:
        ff = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(ff, pH=pH)

    membrane_meta: Dict[str, Any] = {}
    if membrane_context:
        try:
            from orchestrator.membrane import embed_in_membrane_openmm
            modeller = embed_in_membrane_openmm(modeller, ff, membrane_context)
            membrane_meta = {
                "membrane_type": membrane_context.get("type", "POPC"),
                "membrane_embedded": True,
            }
            logger.info("OpenMM: membrane embedding complete")
        except RuntimeError as e:
            logger.warning(f"OpenMM membrane embedding failed, falling back to solvation: {e}")
            membrane_meta = {"membrane_embedded": False, "membrane_error": str(e)}
            modeller.addSolvent(
                ff,
                model="tip3p",
                padding=1.0 * unit.nanometers,
                ionicStrength=0.15 * unit.molar,
            )
    else:
        modeller.addSolvent(
            ff,
            model="tip3p",
            padding=1.0 * unit.nanometers,
            ionicStrength=0.15 * unit.molar,
        )

    ligand_meta: List[Dict[str, Any]] = []
    if ligand_contexts:
        with tempfile.TemporaryDirectory() as lig_tmpdir:
            try:
                from orchestrator.ligands import prepare_ligands
                prepared = prepare_ligands(
                    ligand_contexts, pdb_string, lig_tmpdir,
                    gnina_bin=GNINA_BIN, use_openff=True,
                )
                for lig in prepared:
                    xml_path = lig.get("xml")
                    if xml_path and os.path.isfile(xml_path):
                        try:
                            ff.loadFile(xml_path)
                            ligand_meta.append({
                                "name": lig["name"],
                                "parameterizer": lig["parameterizer"],
                                "docked": lig["docked_sdf"] is not None,
                            })
                            logger.info(f"OpenMM: ligand '{lig['name']}' OpenFF XML loaded")
                        except Exception as e:
                            logger.warning(f"OpenMM: could not load ligand XML for '{lig['name']}': {e}")
                    else:
                        logger.warning(
                            f"Ligand '{lig['name']}': no OpenFF XML produced — "
                            "skipping force-field merge"
                        )
            except Exception as e:
                logger.warning(f"Ligand preparation (OpenFF) failed: {e}")

    n_atoms = modeller.topology.getNumAtoms()
    logger.info(f"OpenMM: {n_atoms} atoms after solvation/membrane setup")

    sim_system_pdb_str: Optional[str] = None
    try:
        _sim_pdb_io = io.StringIO()
        PDBFile.writeFile(modeller.topology, modeller.positions, _sim_pdb_io)
        sim_system_pdb_str = _sim_pdb_io.getvalue()
        logger.info("OpenMM: pre-simulation system PDB captured")
    except Exception as _e:
        logger.warning(f"OpenMM: could not capture pre-simulation PDB: {_e}")

    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * unit.nanometers,
        constraints=HBonds,
    )

    integrator = omm.LangevinMiddleIntegrator(
        temperature,
        1.0 / unit.picoseconds,
        dt,
    )

    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    logger.info("OpenMM: energy minimization...")
    simulation.minimizeEnergy()
    state = simulation.context.getState(getEnergy=True)
    potential_energy_kj = state.getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    logger.info(f"OpenMM: EM done. PE = {potential_energy_kj:.1f} kJ/mol")

    logger.info("OpenMM: NVT equilibration (50 ps)...")
    simulation.step(25_000)

    system.addForce(omm.MonteCarloBarostat(1.0 * unit.bar, temperature))
    simulation.context.reinitialize(preserveState=True)

    logger.info("OpenMM: NPT equilibration (50 ps)...")
    simulation.step(25_000)

    logger.info(f"OpenMM: production ({production_ns} ns, {production_steps} steps)...")
    frames: List[Any] = []
    n_chunks = production_steps // report_interval
    for _ in range(n_chunks):
        simulation.step(report_interval)
        state = simulation.context.getState(getPositions=True)
        frames.append(
            np.array(state.getPositions(asNumpy=True).value_in_unit(unit.nanometers))
        )

    ca_indices = [
        i for i, atom in enumerate(modeller.topology.atoms()) if atom.name == "CA"
    ]
    rmsd_nm, rg_nm = _compute_openmm_trajectory_metrics(frames, ca_indices)

    logger.info(f"OpenMM: complete. {len(frames)} frames, {len(ca_indices)} CA atoms.")
    return {
        "potential_energy": potential_energy_kj,
        "rmsd_nm": rmsd_nm,
        "rg_nm": rg_nm,
        "n_frames": len(frames),
        "production_ns": production_ns,
        "pH": pH,
        "temperature_c": temperature_c,
        "backend": "openmm",
        "simulation_pdb": sim_system_pdb_str,
        **membrane_meta,
        **({"ligands": ligand_meta} if ligand_meta else {}),
    }
