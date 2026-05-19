import glob
import io
import json
import logging
import os
import subprocess
import tempfile
from typing import Optional, Dict, Any, List

from config import (
    BOLTZ_DIFFUSION_SAMPLES,
    BOLTZ_SAMPLING_STEPS,
    BOLTZ_USE_MSA,
)
from models.schemas import StructurePrediction
from orchestrator.backends.esmfold import _parse_plddt_from_pdb

logger = logging.getLogger(__name__)


def _cif_to_pdb(cif_path: str) -> str:
    """Convert a Boltz-2 CIF output file to a PDB string via BioPython."""
    from Bio.PDB import MMCIFParser, PDBIO  # type: ignore
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("boltz", cif_path)
    pdbio = PDBIO()
    pdbio.set_structure(structure)
    out = io.StringIO()
    pdbio.save(out)
    return out.getvalue()


def call_boltz(
    sequence: str,
    context: Optional[Dict[str, Any]] = None,
    seed: int = 0,
) -> StructurePrediction:
    """
    Run Boltz-2 prediction via CLI subprocess.

    Writes a YAML input, calls `boltz predict`, parses the CIF output with
    BioPython, reads pLDDT from the confidence JSON, and optionally reads
    binding affinity when ligands are present.

    Install: pip install git+https://github.com/jwohlwend/boltz
    Then set BOLTZ_ENABLED=True.
    """
    import yaml  # pyyaml — guaranteed by requirements.txt

    ctx = context or {}
    ligands = ctx.get("ligands") or []

    for lig in ligands:
        lig_name = lig.get("name", "unknown") if isinstance(lig, dict) else lig.name
        lig_smiles = lig.get("smiles") if isinstance(lig, dict) else lig.smiles
        if not lig_smiles:
            raise ValueError(
                f"Ligand '{lig_name}' has no SMILES string. "
                "Boltz-2 requires SMILES for all ligands. "
                "Add smiles to the LigandContext or remove the ligand from context."
            )

    sequences: list = [{
        "protein": {
            "id": "A",
            "sequence": sequence,
            "msa": "server" if BOLTZ_USE_MSA else "empty",
        }
    }]

    affinity_binder: Optional[str] = None
    for i, lig in enumerate(ligands):
        chain_id = chr(ord("B") + i)
        smiles = lig.get("smiles") if isinstance(lig, dict) else lig.smiles
        sequences.append({"ligand": {"id": chain_id, "smiles": smiles}})
        if affinity_binder is None:
            affinity_binder = chain_id

    boltz_input: Dict[str, Any] = {"version": 1, "sequences": sequences}
    if affinity_binder:
        boltz_input["properties"] = [{"affinity": {"binder": affinity_binder}}]

    with tempfile.TemporaryDirectory() as tmpdir:
        yaml_path = os.path.join(tmpdir, "input.yaml")
        out_dir = os.path.join(tmpdir, "output")
        os.makedirs(out_dir)

        with open(yaml_path, "w") as fh:
            yaml.dump(boltz_input, fh, default_flow_style=False)

        cmd = [
            "boltz", "predict", yaml_path,
            "--out_dir", out_dir,
            "--diffusion_samples", str(BOLTZ_DIFFUSION_SAMPLES),
            "--sampling_steps", str(BOLTZ_SAMPLING_STEPS),
            "--seed", str(seed),
        ]
        logger.info(f"Running Boltz-2: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            raise RuntimeError(f"Boltz-2 failed (exit {proc.returncode}): {proc.stderr[-2000:]}")

        cif_hits = sorted(glob.glob(os.path.join(out_dir, "**", "*model_0.cif"), recursive=True))
        logger.info(f"Boltz-2 output tree: {glob.glob(os.path.join(out_dir, '**', '*'), recursive=True)}")
        if not cif_hits:
            raise FileNotFoundError(
                f"Boltz-2 produced no *model_0.cif under {out_dir}. "
                f"stderr: {proc.stderr[-1000:]}"
            )
        cif_path = cif_hits[0]
        results_dir = os.path.dirname(cif_path)
        pdb_string = _cif_to_pdb(cif_path)

        import importlib.metadata
        try:
            _boltz_major = int(importlib.metadata.version("boltz").split(".")[0])
        except importlib.metadata.PackageNotFoundError:
            _boltz_major = 2

        plddt_scores: List[float] = []
        conf_hits = glob.glob(os.path.join(out_dir, "**", "*confidence*model_0.json"), recursive=True)
        if conf_hits:
            with open(conf_hits[0]) as fh:
                conf = json.load(fh)
            raw = conf.get("plddt", [])
            plddt_scores = [v * 100.0 for v in raw] if _boltz_major < 2 else list(raw)

        if not plddt_scores:
            raw_bfactor = _parse_plddt_from_pdb(pdb_string)
            plddt_scores = raw_bfactor if _boltz_major < 2 else [v / 100.0 for v in raw_bfactor]

        if not plddt_scores:
            raise ValueError("No pLDDT scores found in Boltz-2 output")

        mean_plddt = sum(plddt_scores) / len(plddt_scores)

        affinity_score: Optional[float] = None
        if affinity_binder:
            aff_hits = glob.glob(os.path.join(out_dir, "**", "*affinity*.json"), recursive=True)
            aff_path = aff_hits[0] if aff_hits else None
            if aff_path and os.path.exists(aff_path):
                with open(aff_path) as fh:
                    aff_data = json.load(fh)
                affinity_score = aff_data.get("affinity")

        logger.info(
            f"Boltz-2 succeeded. Mean pLDDT: {mean_plddt:.2f}"
            + (f", affinity: {affinity_score:.3f} kcal/mol" if affinity_score is not None else "")
        )

        return StructurePrediction(
            structure_pdb=pdb_string,
            plddt_scores=plddt_scores,
            mean_plddt=mean_plddt,
            seed=seed,
            model_name="boltz2",
            affinity_score=affinity_score,
        )
