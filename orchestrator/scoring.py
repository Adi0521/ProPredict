import io
import logging
from typing import List

from config import (
    PLDDT_ACCEPT_THRESHOLD,
    PLDDT_REFINE_THRESHOLD,
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
