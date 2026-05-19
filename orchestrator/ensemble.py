import io
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def align_and_compare_structures(pdb_strings: List[str]) -> Dict[str, Any]:
    """
    Align all structures to the first one and compute per-residue CA disagreement.

    Algorithm (star topology — all vs model 0):
      1. Parse CA atoms from each PDB; index by residue number.
      2. Find residue numbers common to ALL models.
      3. For each model i >= 1: BioPython Superimposer aligns to model 0,
         then record per-residue CA distance after alignment.
      4. Average distances across all (0,i) pairs -> per-residue disagreement (nm).
      5. Group stretches above DISAGREEMENT_THRESHOLD into regions.

    Returns {} if BioPython is missing or fewer than 2 structures are provided.
    """
    if len(pdb_strings) < 2:
        return {}

    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.Superimposer import Superimposer
    except ImportError:
        logger.warning("BioPython not available — inter-model comparison skipped")
        return {}

    DISAGREEMENT_THRESHOLD_NM = 0.3

    parser = PDBParser(QUIET=True)

    def _ca_by_resnum(pdb_str: str, label: str) -> Dict[int, Any]:
        struct = parser.get_structure(label, io.StringIO(pdb_str))
        ca: Dict[int, Any] = {}
        for atom in struct.get_atoms():
            if atom.get_name() == "CA":
                res_num = atom.get_parent().get_id()[1]
                ca[res_num] = atom
        return ca

    all_ca = [_ca_by_resnum(pdb, f"m{i}") for i, pdb in enumerate(pdb_strings)]
    common = sorted(set.intersection(*[set(d.keys()) for d in all_ca]))

    if len(common) < 3:
        logger.warning("Too few common residues for inter-model alignment")
        return {}

    ref_atoms = [all_ca[0][r] for r in common]
    sup = Superimposer()

    per_pair_dists: List[List[float]] = [[] for _ in common]

    for i in range(1, len(all_ca)):
        mobile = [all_ca[i][r] for r in common]
        sup.set_atoms(ref_atoms, mobile)
        sup.apply(mobile)

        for j, (ref_a, mob_a) in enumerate(zip(ref_atoms, mobile)):
            dist_nm = (ref_a.get_vector() - mob_a.get_vector()).norm() / 10.0
            per_pair_dists[j].append(dist_nm)

    mean_per_residue = [
        sum(ds) / len(ds) if ds else 0.0 for ds in per_pair_dists
    ]

    regions: List[Dict[str, Any]] = []
    in_region = False
    r_start = 0
    r_vals: List[float] = []

    for idx, (res_num, dist) in enumerate(zip(common, mean_per_residue)):
        if dist > DISAGREEMENT_THRESHOLD_NM:
            if not in_region:
                in_region, r_start, r_vals = True, res_num, []
            r_vals.append(dist)
        elif in_region:
            regions.append({
                "start": r_start,
                "end": common[idx - 1],
                "mean_disagreement_nm": round(sum(r_vals) / len(r_vals), 4),
            })
            in_region = False
    if in_region:
        regions.append({
            "start": r_start,
            "end": common[-1],
            "mean_disagreement_nm": round(sum(r_vals) / len(r_vals), 4),
        })

    mean_overall = sum(mean_per_residue) / len(mean_per_residue)
    logger.info(
        f"Inter-model comparison: {len(pdb_strings)} models, "
        f"{len(common)} common residues, "
        f"mean disagreement {mean_overall:.3f} nm, "
        f"{len(regions)} high-disagreement region(s)"
    )
    return {
        "per_residue_disagreement_nm": [round(d, 4) for d in mean_per_residue],
        "mean_disagreement_nm": round(mean_overall, 4),
        "disagreement_regions": regions,
        "n_models_compared": len(pdb_strings),
        "n_common_residues": len(common),
    }
