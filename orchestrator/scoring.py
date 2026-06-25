import io
import logging
import math
from typing import Any, Dict, List, Optional

from config import (
    PLDDT_ACCEPT_THRESHOLD,
    PLDDT_REFINE_THRESHOLD,
    SIM_RMSD_MAX_NM,
    SIM_RG_DIVERGENCE_FACTOR,
)
from models.schemas import StructurePrediction, PostProcessingResult

logger = logging.getLogger(__name__)


def count_clashes(pdb_string: str) -> int:
    """
    Count steric clashes between non-adjacent CA atoms closer than 3.8 A.

    Uses BioPython NeighborSearch. Adjacent residues (|i-j| <= 1) are excluded
    since peptide bond CA-CA distances are naturally ~3.8 A.
    Returns 0 and logs a warning if BioPython is not installed.
    """
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.NeighborSearch import NeighborSearch
    except ImportError:
        logger.warning("BioPython not installed — clash detection skipped (pip install biopython)")
        return 0

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pred", io.StringIO(pdb_string))
    ca_atoms = [atom for atom in structure.get_atoms() if atom.get_name() == "CA"]
    if len(ca_atoms) < 2:
        return 0

    ns = NeighborSearch(ca_atoms)
    seen: set = set()
    clashes = 0
    for atom in ca_atoms:
        res_i = atom.get_parent().get_id()[1]
        for other in ns.search(atom.coord, 3.8, "A"):
            if other is atom:
                continue
            res_j = other.get_parent().get_id()[1]
            if abs(res_i - res_j) <= 1:
                continue
            pair = (min(res_i, res_j), max(res_i, res_j))
            if pair not in seen:
                seen.add(pair)
                clashes += 1
    return clashes


def compute_post_processing(prediction: StructurePrediction) -> PostProcessingResult:
    """Compute post-processing scores and make an accept/refine/escalate decision."""
    mean_plddt = prediction.mean_plddt

    num_clashes = count_clashes(prediction.structure_pdb)
    score = mean_plddt - (num_clashes * 5.0)

    if mean_plddt >= PLDDT_ACCEPT_THRESHOLD:
        decision = "accept"
    elif mean_plddt >= PLDDT_REFINE_THRESHOLD:
        decision = "refine"
    else:
        decision = "escalate"

    return PostProcessingResult(
        num_clashes=num_clashes,
        score=score,
        decision=decision,
    )


def _coerce_float(value: Any) -> Optional[float]:
    """Best-effort float conversion; returns None for non-numeric input."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def validate_simulation_metrics(sim_result: Dict[str, Any]) -> Optional[str]:
    """
    Sanity-check an MD trajectory for signs of an unphysical/blown-up run.

    Accepts the dict returned by run_openmm_simulation / run_gromacs_md. Checks:
      1. Potential energy is finite (not NaN/inf).
      2. CA RMSD never exceeds SIM_RMSD_MAX_NM (blowup).
      3. Radius of gyration never diverges past SIM_RG_DIVERGENCE_FACTOR x its
         initial value (unfolding/explosion).

    Returns None when the trajectory looks sane, otherwise a human-readable reason
    string. Missing/empty metrics are treated as "can't check" (not a failure) so
    a partial result never produces a false escalation.
    """
    if not sim_result:
        return None

    # 1. Potential energy must be finite.
    energy = _coerce_float(sim_result.get("potential_energy"))
    if energy is not None and not math.isfinite(energy):
        return f"Potential energy is non-finite ({sim_result.get('potential_energy')})"

    # 2. RMSD blowup. Prefer the per-frame list; fall back to a final scalar.
    rmsd_series = sim_result.get("rmsd_nm")
    rmsd_values: List[float] = []
    if isinstance(rmsd_series, (list, tuple)):
        rmsd_values = [v for v in (_coerce_float(x) for x in rmsd_series) if v is not None]
    if not rmsd_values:
        final = _coerce_float(sim_result.get("rmsd_final_nm"))
        if final is not None:
            rmsd_values = [final]
    if rmsd_values:
        max_rmsd = max(rmsd_values)
        if max_rmsd > SIM_RMSD_MAX_NM:
            return f"RMSD blew up to {max_rmsd:.2f} nm (> {SIM_RMSD_MAX_NM:.2f} nm)"

    # 3. Rg divergence relative to the initial frame.
    rg_series = sim_result.get("rg_nm")
    if isinstance(rg_series, (list, tuple)):
        rg_values = [v for v in (_coerce_float(x) for x in rg_series) if v is not None]
        if len(rg_values) >= 2 and rg_values[0] > 0:
            max_rg = max(rg_values)
            if max_rg > SIM_RG_DIVERGENCE_FACTOR * rg_values[0]:
                return (
                    f"Radius of gyration diverged to {max_rg:.2f} nm "
                    f"(> {SIM_RG_DIVERGENCE_FACTOR:.1f}x initial {rg_values[0]:.2f} nm)"
                )

    return None
