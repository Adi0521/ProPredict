import glob
import io
import logging
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from typing import Optional, Dict, Any, List, Tuple
from celery import Celery, Task
from datetime import datetime
import requests

from config import (
    CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND,
    ESMFOLD_LOCAL,
    ESMFOLD_MODEL_NAME,
    ESMFOLD_API_URL,
    ESMFOLD_TIMEOUT,
    ESMFOLD_RETRIES,
    PLDDT_ACCEPT_THRESHOLD,
    PLDDT_REFINE_THRESHOLD,
    ROSETTA_ENABLED,
    GROMACS_ENABLED,
    GROMACS_BIN,
    OPENMM_ENABLED,
    ROSETTAFOLD_ENABLED,
    OPENFOLD_ENABLED,
    BOLTZ_ENABLED,
    BOLTZ_DIFFUSION_SAMPLES,
    BOLTZ_SAMPLING_STEPS,
    BOLTZ_USE_MSA,
    MD_PRODUCTION_NS,
    AGENT_ENABLED,
    ANTHROPIC_API_KEY,
    AGENT_MODEL,
    AGENT_MAX_ITERATIONS,
    ENSEMBLE_NUM_SEEDS,
    REDIS_URL,
    CACHE_TTL,
    REDIS_CACHE_PREFIX,
    GNINA_BIN,
    INSANE_PATH,
    MEMBRANE_FF,
)
# membrane/ligand helpers imported lazily inside run_gromacs_md / run_openmm_simulation
# so the worker starts even when rdkit / openff / gnina are not installed.
from models.schemas import StructurePrediction, PostProcessingResult

# Configure Celery app
app = Celery(
    "propredict",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

logger = logging.getLogger(__name__)

import redis as redis_lib

_redis_client: Optional[redis_lib.Redis] = None


def _get_redis() -> redis_lib.Redis:
    """Return a module-level Redis client, initialised lazily."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis_lib.Redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


# ---------------------------------------------------------------------------
# Celery task base with webhook callbacks
# ---------------------------------------------------------------------------

class CallbackTask(Task):
    """Task that sends webhook callbacks on completion."""

    def on_success(self, retval, task_id, args, kwargs):
        request_data = args[0] if args else None
        if request_data and request_data.get("webhook_url"):
            send_webhook(request_data["webhook_url"], {"status": "completed", "result": retval})

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        request_data = args[0] if args else None
        if request_data and request_data.get("webhook_url"):
            send_webhook(request_data["webhook_url"], {"status": "failed", "error": str(exc)})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_cache_key(sequence: str, context: Dict[str, Any], pipeline: str = "esm_base") -> str:
    """Generate a deterministic cache key from sequence and context."""
    context_str = json.dumps(context, sort_keys=True)
    key_input = f"{sequence}:{context_str}:{pipeline}"
    return hashlib.sha256(key_input.encode()).hexdigest()


def send_webhook(webhook_url: str, payload: Dict[str, Any]) -> None:
    """Send webhook notification with exponential backoff."""
    for attempt in range(3):
        try:
            requests.post(webhook_url, json=payload, timeout=10)
            logger.info(f"Webhook sent to {webhook_url}")
            return
        except requests.exceptions.RequestException as e:
            logger.warning(f"Webhook send failed (attempt {attempt + 1}): {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                logger.error("Webhook failed after 3 attempts")


# ---------------------------------------------------------------------------
# ESMFold inference
# ---------------------------------------------------------------------------

_esmfold_model = None
_esmfold_tokenizer = None


def _get_esmfold_local():
    """Lazy-load the ESMFold model and tokenizer (heavy — only once per worker)."""
    global _esmfold_model, _esmfold_tokenizer
    if _esmfold_model is None:
        from transformers import EsmForProteinFolding, AutoTokenizer  # type: ignore
        import torch
        logger.info(f"Loading ESMFold model: {ESMFOLD_MODEL_NAME}")
        _esmfold_tokenizer = AutoTokenizer.from_pretrained(ESMFOLD_MODEL_NAME)
        _esmfold_model = EsmForProteinFolding.from_pretrained(ESMFOLD_MODEL_NAME)
        _esmfold_model.eval()
        if torch.cuda.is_available():
            _esmfold_model = _esmfold_model.cuda()
        elif torch.backends.mps.is_available():
            _esmfold_model = _esmfold_model.to("mps")
        logger.info("ESMFold model loaded.")
    return _esmfold_model, _esmfold_tokenizer


def _parse_plddt_from_pdb(pdb_string: str) -> List[float]:
    """
    Extract per-residue pLDDT from ESMFold PDB output.
    ESMFold stores pLDDT in the B-factor column of CA atoms as a 0–1 fraction.
    Multiply by 100 to normalise to the standard 0–100 scale used by thresholds.
    """
    scores: List[float] = []
    for line in pdb_string.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                scores.append(float(line[60:66].strip()) * 100)
            except ValueError:
                pass
    return scores


def _call_esmfold_remote(sequence: str, seed: int = 0) -> StructurePrediction:
    """
    Call the ESMFold REST API with retries (fallback when ESMFOLD_LOCAL=False).

    ESMFold endpoint accepts a raw amino acid sequence as the POST body
    (application/x-www-form-urlencoded) and returns a PDB string.
    pLDDT scores are embedded in the B-factor column of CA atoms.
    """
    for attempt in range(ESMFOLD_RETRIES):
        try:
            logger.info(f"ESMFold API call attempt {attempt + 1}/{ESMFOLD_RETRIES}")
            response = requests.post(
                ESMFOLD_API_URL,
                data=sequence,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=ESMFOLD_TIMEOUT,
            )
            response.raise_for_status()

            pdb_string = response.text
            plddt_scores = _parse_plddt_from_pdb(pdb_string)

            if not plddt_scores:
                raise ValueError("No CA atoms found in ESMFold PDB output — response may be malformed")

            mean_plddt = sum(plddt_scores) / len(plddt_scores)
            logger.info(f"ESMFold remote call succeeded. Mean pLDDT: {mean_plddt:.2f}")

            return StructurePrediction(
                structure_pdb=pdb_string,
                plddt_scores=plddt_scores,
                mean_plddt=mean_plddt,
                seed=seed,
                model_name="esmfold",
            )

        except requests.exceptions.RequestException as e:
            logger.warning(f"ESMFold API call failed (attempt {attempt + 1}): {e}")
            if attempt < ESMFOLD_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"ESMFold API failed after {ESMFOLD_RETRIES} attempts")
                raise


def call_esmfold_local(sequence: str, seed: int = 0) -> StructurePrediction:
    """
    Run ESMFold locally via HuggingFace Transformers.

    ESMFold is deterministic — the seed parameter does not affect output. When
    ENSEMBLE_NUM_SEEDS > 1, the ensemble loop will call this N times but receive
    identical structures. This is a known limitation of ESMFold, not a ProPredict bug.
    """
    import torch
    model, tokenizer = _get_esmfold_local()

    logger.info(f"ESMFold local inference on sequence of length {len(sequence)}")
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)

    pdb_string = model.output_to_pdb(output)[0]
    plddt_scores = _parse_plddt_from_pdb(pdb_string)

    if not plddt_scores:
        raise ValueError("No CA atoms found in ESMFold local output — model output may be malformed")

    mean_plddt = sum(plddt_scores) / len(plddt_scores)
    logger.info(f"ESMFold local inference succeeded. Mean pLDDT: {mean_plddt:.2f}")

    return StructurePrediction(
        structure_pdb=pdb_string,
        plddt_scores=plddt_scores,
        mean_plddt=mean_plddt,
        seed=seed,
        model_name="esmfold_local",
    )


def call_esmfold_api(sequence: str, seed: int = 0) -> StructurePrediction:
    """Dispatch to local model or remote API based on ESMFOLD_LOCAL flag."""
    if ESMFOLD_LOCAL:
        return call_esmfold_local(sequence, seed)
    return _call_esmfold_remote(sequence, seed)


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
# Multi-model prediction backends (Stage E)
# ---------------------------------------------------------------------------

def call_rosettafold2(sequence: str, seed: int = 0) -> StructurePrediction:
    """
    Run RoseTTAFold2 locally and return a StructurePrediction.

    DEPRECATED: Boltz-2 supersedes RoseTTAFold2 in accuracy and ease of install.
    Prefer BOLTZ_ENABLED=True. This stub is kept for future completion if needed.

    Installation:
        git clone https://github.com/baker-lab/RoseTTAFold2
        cd RoseTTAFold2 && conda env create -f environment.yaml
        conda activate RF2 && pip install -e .
    Then set ROSETTAFOLD_ENABLED=True.

    Raises RuntimeError if the package is not installed.
    Raises NotImplementedError — complete the body once RF2 is installed.
    """
    try:
        import rf2aa  # type: ignore  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "RoseTTAFold2 (rf2aa) is not installed. "
            "See: https://github.com/baker-lab/RoseTTAFold2"
        )

    # TODO: implement using the RF2AA runner once the conda env is active.
    # Example sketch (API may differ by version):
    #   from rf2aa.run_inference import run_inference
    #   pdb_string, plddt = run_inference(sequence)
    raise NotImplementedError(
        "RoseTTAFold2 stub — fill in using the rf2aa.run_inference API."
    )


def call_openfold(sequence: str, seed: int = 0) -> StructurePrediction:
    """
    Run OpenFold locally and return a StructurePrediction.

    DEPRECATED: Boltz-2 supersedes OpenFold in accuracy and ease of install.
    Prefer BOLTZ_ENABLED=True. This stub is kept for future completion if needed.

    Installation:
        pip install 'openfold @ git+https://github.com/aqlaboratory/openfold'
        # Requires CUDA for full performance; CPU mode works for short sequences.
    Then set OPENFOLD_ENABLED=True.

    Raises RuntimeError if the package is not installed.
    Raises NotImplementedError — complete the body once OpenFold is installed.
    """
    try:
        import openfold  # type: ignore  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "OpenFold is not installed. "
            "See: https://github.com/aqlaboratory/openfold"
        )

    # TODO: implement using the OpenFold data pipeline + model runner.
    # Example sketch:
    #   from openfold.data import data_pipeline, feature_pipeline
    #   from openfold.model import model as of_model
    #   ...
    raise NotImplementedError(
        "OpenFold stub — fill in using the openfold.data and openfold.model APIs."
    )


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

    # Validate SMILES before doing any work — surface the error early.
    for lig in ligands:
        lig_name = lig.get("name", "unknown") if isinstance(lig, dict) else lig.name
        lig_smiles = lig.get("smiles") if isinstance(lig, dict) else lig.smiles
        if not lig_smiles:
            raise ValueError(
                f"Ligand '{lig_name}' has no SMILES string. "
                "Boltz-2 requires SMILES for all ligands. "
                "Add smiles to the LigandContext or remove the ligand from context."
            )

    # Build YAML input — protein on chain A, ligands on B, C, ...
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
            affinity_binder = chain_id  # predict affinity for first ligand

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

        # Boltz-2 output layout can vary by version; find the CIF by glob.
        cif_hits = sorted(glob.glob(os.path.join(out_dir, "**", "*model_0.cif"), recursive=True))
        logger.info(f"Boltz-2 output tree: {glob.glob(os.path.join(out_dir, '**', '*'), recursive=True)}")
        if not cif_hits:
            raise FileNotFoundError(
                f"Boltz-2 produced no *model_0.cif under {out_dir}. "
                f"stderr: {proc.stderr[-1000:]}"
            )
        cif_path = cif_hits[0]
        results_dir = os.path.dirname(cif_path)
        # Derive the stem prefix used by Boltz for related files.
        # cif_path looks like ".../stem_model_0.cif"; strip "_model_0.cif" → stem
        cif_stem = os.path.basename(cif_path).replace("_model_0.cif", "")
        print(f"[boltz] cif_path={cif_path}  cif_stem={cif_stem}")
        pdb_string = _cif_to_pdb(cif_path)

        # pLDDT from confidence JSON
        plddt_scores: List[float] = []
        # Search for the confidence JSON next to the CIF and one level up (Boltz-2 layout varies).
        conf_candidates = [
            os.path.join(results_dir, f"{cif_stem}_confidence_model_0.json"),
            os.path.join(os.path.dirname(results_dir), f"{cif_stem}_confidence_model_0.json"),
        ]
        conf_candidates += glob.glob(os.path.join(out_dir, "**", f"*confidence*model_0.json"), recursive=True)
        conf_path = next((p for p in conf_candidates if os.path.exists(p)), None)
        print(f"[boltz] conf_path={conf_path}")
        if conf_path is not None:
            with open(conf_path) as fh:
                conf = json.load(fh)
            raw = conf.get("plddt", [])
            # Boltz-1 stores pLDDT on 0–1 scale; Boltz-2 uses 0–100.
            import importlib.metadata
            _boltz_major = int(importlib.metadata.version("boltz").split(".")[0])
            if _boltz_major < 2:
                plddt_scores = [v * 100.0 for v in raw]
            else:
                plddt_scores = list(raw)

        if not plddt_scores:
            # Fallback: Boltz-2 stores pLDDT in B-factor on 0–100 scale (not 0–1).
            import importlib.metadata
            _boltz_major = int(importlib.metadata.version("boltz").split(".")[0])
            # _parse_plddt_from_pdb multiplies B-factor by 100 (ESMFold uses 0–1 B-factor).
            # Boltz-2 stores B-factor on 0–100 scale, so that multiplication overshoots by 100x.
            raw_bfactor = _parse_plddt_from_pdb(pdb_string)
            plddt_scores = raw_bfactor if _boltz_major < 2 else [v / 100.0 for v in raw_bfactor]

        if not plddt_scores:
            raise ValueError("No pLDDT scores found in Boltz-2 output")

        mean_plddt = sum(plddt_scores) / len(plddt_scores)

        # Binding affinity (kcal/mol) — only present when ligands were provided
        affinity_score: Optional[float] = None
        if affinity_binder:
            aff_path = os.path.join(results_dir, f"{cif_stem}_affinity_0.json")
            if os.path.exists(aff_path):
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


def _align_and_compare_structures(pdb_strings: List[str]) -> Dict[str, Any]:
    """
    Align all structures to the first one and compute per-residue CA disagreement.

    Algorithm (star topology — all vs model 0):
      1. Parse CA atoms from each PDB; index by residue number.
      2. Find residue numbers common to ALL models.
      3. For each model i ≥ 1: BioPython Superimposer aligns to model 0,
         then record per-residue CA distance after alignment.
      4. Average distances across all (0,i) pairs → per-residue disagreement (nm).
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

    DISAGREEMENT_THRESHOLD_NM = 0.3  # flag regions with mean CA RMSD > 0.3 nm

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

    # per_pair_dists[j] accumulates distances at position j across all (0,i) pairs
    per_pair_dists: List[List[float]] = [[] for _ in common]

    for i in range(1, len(all_ca)):
        mobile = [all_ca[i][r] for r in common]
        sup.set_atoms(ref_atoms, mobile)
        sup.apply(mobile)  # rotate/translate mobile in place

        for j, (ref_a, mob_a) in enumerate(zip(ref_atoms, mobile)):
            dist_nm = (ref_a.get_vector() - mob_a.get_vector()).norm() / 10.0  # Å → nm
            per_pair_dists[j].append(dist_nm)

    mean_per_residue = [
        sum(ds) / len(ds) if ds else 0.0 for ds in per_pair_dists
    ]

    # Collect high-disagreement regions
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


# ---------------------------------------------------------------------------
# Protonation state assignment via PropKa3 (optional — pip install propka)
# ---------------------------------------------------------------------------

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

        # Prefer the 'AVR' (average) conformation; fall back to first available
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
                # Try direct attributes first (propka >= 3.4)
                res_name = group.res_name.strip()
                res_num = int(group.res_num)
                chain = group.chain_id.strip()
            except AttributeError:
                # Fall back to parsing the label string, e.g. "HIS  64 A"
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


# Model pKa values used when PropKa result is absent
_MODEL_PKA = {"HIS": 6.0, "ASP": 3.9, "GLU": 4.1, "LYS": 10.5, "TYR": 10.1, "CYS": 8.3}


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

    # HIS: pKa > pH → HIP (2, charged); otherwise default to HIE (1, neutral)
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
# GROMACS energy minimisation (optional — brew install gromacs / apt-get gromacs)
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

    Pipeline: PropKa → pdb2gmx → editconf → solvate → grompp → mdrun → energy
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

        # --- Protonation state assignment via PropKa ---
        logger.info(f"Running PropKa3 for protonation assignment at pH {pH:.1f}...")
        pka_dict = _run_propka(pdb_path)
        protonation = _determine_protonation_states(pdb_string, pH, pka_dict)

        # Build pdb2gmx command and stdin input for titratable residues
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
# GROMACS full MD pipeline (NVT → NPT → Production)
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

    Stages: PropKa → pdb2gmx → [insane.py membrane embed] →
            editconf → solvate → [ligand topology merge] → genion →
            EM → NVT (100 ps) → NPT (100 ps) → Production → Analysis

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
    production_steps = int(production_ns * 1_000_000 / 2)  # ns → 2-fs steps

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

        # Force-field choice: CHARMM36m when membrane present, amber99sb-ildn otherwise
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
                # insane.py writes its own .gro and .top — use them as the
                # starting point for box/solvation steps.
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
                        # Append #include directive to topol.top
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

    Pipeline: add H at pH → [membrane embed OR solvate] →
              [ligand parameterization + topology merge] →
              EM → NVT (50 ps) → NPT (50 ps) → Production → Analysis

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
    production_steps = int(production_ns * 1_000_000 / 2)   # ns → 2-fs steps
    report_interval = max(1, production_steps // 100)         # ~100 trajectory frames

    # --- Fix missing terminal atoms (OXT etc.) before OpenMM sees the structure ---
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

    # --- Load structure and add hydrogens at requested pH ---
    logger.info(f"OpenMM: adding hydrogens at pH {pH:.1f}...")

    # Choose force field: CHARMM36m for membrane runs, AMBER14 otherwise
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

    # --- Membrane embedding OR plain solvation ---
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
        # Solvate: TIP3P water box + 0.15 M NaCl
        modeller.addSolvent(
            ff,
            model="tip3p",
            padding=1.0 * unit.nanometers,
            ionicStrength=0.15 * unit.molar,
        )

    # --- Ligand preparation (OpenFF SMIRNOFF) ---
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
                        # Register the OpenFF XML with the ForceField so it is
                        # applied when createSystem() is called below.
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

    # --- Capture pre-simulation PDB (full solvated system including ligands) ---
    sim_system_pdb_str: Optional[str] = None
    try:
        _sim_pdb_io = io.StringIO()
        PDBFile.writeFile(modeller.topology, modeller.positions, _sim_pdb_io)
        sim_system_pdb_str = _sim_pdb_io.getvalue()
        logger.info("OpenMM: pre-simulation system PDB captured")
    except Exception as _e:
        logger.warning(f"OpenMM: could not capture pre-simulation PDB: {_e}")

    # --- Build system ---
    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * unit.nanometers,
        constraints=HBonds,
    )

    # Langevin thermostat (NVT integrator)
    integrator = omm.LangevinMiddleIntegrator(
        temperature,
        1.0 / unit.picoseconds,  # friction
        dt,
    )

    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    # --- Energy minimization ---
    logger.info("OpenMM: energy minimization...")
    simulation.minimizeEnergy()
    state = simulation.context.getState(getEnergy=True)
    potential_energy_kj = state.getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    logger.info(f"OpenMM: EM done. PE = {potential_energy_kj:.1f} kJ/mol")

    # --- NVT equilibration (50 ps = 25 000 steps) ---
    logger.info("OpenMM: NVT equilibration (50 ps)...")
    simulation.step(25_000)

    # --- NPT: add Monte Carlo barostat and reinitialise ---
    system.addForce(omm.MonteCarloBarostat(1.0 * unit.bar, temperature))
    simulation.context.reinitialize(preserveState=True)

    # --- NPT equilibration (50 ps) ---
    logger.info("OpenMM: NPT equilibration (50 ps)...")
    simulation.step(25_000)

    # --- Production MD with frame collection ---
    logger.info(f"OpenMM: production ({production_ns} ns, {production_steps} steps)...")
    frames: List[Any] = []
    n_chunks = production_steps // report_interval
    for _ in range(n_chunks):
        simulation.step(report_interval)
        state = simulation.context.getState(getPositions=True)
        frames.append(
            np.array(state.getPositions(asNumpy=True).value_in_unit(unit.nanometers))
        )

    # --- Trajectory analysis ---
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


# ---------------------------------------------------------------------------
# Post-processing and agent decision logic
# ---------------------------------------------------------------------------

def _count_clashes(pdb_string: str) -> int:
    """
    Count steric clashes between non-adjacent CA atoms closer than 3.8 Å.

    Uses BioPython NeighborSearch. Adjacent residues (|i-j| <= 1) are excluded
    since peptide bond CA-CA distances are naturally ~3.8 Å.
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

    num_clashes = _count_clashes(prediction.structure_pdb)
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


# ---------------------------------------------------------------------------
# Agentic refinement loop (Stage D — requires pip install anthropic + ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------

_AGENT_TOOLS = [
    {
        "name": "analyze_structure",
        "description": (
            "Identify low-confidence residue regions from the per-residue pLDDT profile. "
            "Call this first to understand where the structure needs attention."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "description": "pLDDT threshold; residues below this are flagged (default 70)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "run_rosetta_relax",
        "description": (
            "Run PyRosetta FastRelax to improve sidechain geometry. "
            "Use when pLDDT is borderline (60–75) or when clashes are detected. "
            "Requires ROSETTA_ENABLED=True."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run_simulation",
        "description": (
            "Run a full MD simulation (OpenMM preferred, GROMACS fallback). "
            "Use when membrane/ligand context requires dynamics or thermodynamic validation. "
            "Requires OPENMM_ENABLED=True or GROMACS_ENABLED=True."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "production_ns": {
                    "type": "number",
                    "description": "Production run length in nanoseconds (default: MD_PRODUCTION_NS config value)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "run_boltz_prediction",
        "description": (
            "Re-predict the structure using Boltz-2, which gives AlphaFold3-class accuracy and "
            "supports ligand co-folding with binding affinity estimation. Use when ESMFold pLDDT "
            "is low, inter-model disagreement is high, or ligands are present. "
            "Requires BOLTZ_ENABLED=True."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "accept_structure",
        "description": "Accept the current structure. Final decision — call when quality is sufficient.",
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Why this structure is accepted"}
            },
            "required": ["reasoning"],
        },
    },
    {
        "name": "escalate_structure",
        "description": (
            "Flag the structure for human review. Final decision — call when quality is too low "
            "or when required backends are unavailable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Why human review is needed"}
            },
            "required": ["reasoning"],
        },
    },
]

_AGENT_SYSTEM = """\
You are a protein structure quality-control agent. You assess a predicted protein structure, \
optionally refine it with available tools, then make a final decision.

Available prediction backends:
- ESMFold (fast, CPU-friendly, deterministic) — always available
- Boltz-2 (AlphaFold3-class accuracy, GPU, supports ligand co-folding and binding affinity) — when BOLTZ_ENABLED=True
If Boltz-2 produced the current prediction, prefer its structure for downstream refinement.
If ESMFold produced the current prediction and quality is poor, consider run_boltz_prediction before escalating.
When ligands are present and Boltz-2 predicted an affinity score, include it in your reasoning.

Guidelines:
- mean_pLDDT ≥ 75 AND ≤ 2 clashes → accept unless context requires simulation
- mean_pLDDT 60–74 → run Rosetta relax if enabled, then reassess
- mean_pLDDT < 60 → try run_boltz_prediction if BOLTZ_ENABLED, else escalate
- Membrane or ligand context present → run simulation before accepting
- If a required backend is disabled → escalate and explain

Be concise. Make a terminal decision as soon as you have enough information."""


def _execute_agent_tool(
    tool_name: str,
    tool_input: Dict[str, Any],
    state: Dict[str, Any],
) -> str:
    """Execute one agent tool call and return a JSON string result."""

    if tool_name == "analyze_structure":
        threshold = float(tool_input.get("threshold", 70.0))
        plddt = state["plddt_scores"]

        # Group residues below threshold into contiguous regions
        low = [(i + 1, s) for i, s in enumerate(plddt) if s < threshold]
        regions: List[Dict[str, Any]] = []
        if low:
            start, prev = low[0][0], low[0][0]
            for res_num, _ in low[1:]:
                if res_num > prev + 3:
                    regions.append({"start": start, "end": prev})
                    start = res_num
                prev = res_num
            regions.append({"start": start, "end": prev})

        return json.dumps({
            "total_residues": len(plddt),
            "low_confidence_count": len(low),
            "low_confidence_fraction": round(len(low) / len(plddt), 3) if plddt else 0,
            "regions_below_threshold": regions,
            "worst_residue": int(plddt.index(min(plddt))) + 1 if plddt else None,
            "worst_score": round(min(plddt), 1) if plddt else None,
        })

    if tool_name == "run_rosetta_relax":
        if not ROSETTA_ENABLED:
            return json.dumps({"error": "ROSETTA_ENABLED=False — cannot run relax"})
        try:
            relaxed_pdb, score = run_rosetta_relax(state["current_pdb"])
            state["current_pdb"] = relaxed_pdb
            state["rosetta_energy"] = score
            return json.dumps({"status": "completed", "rosetta_energy": round(score, 3)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    if tool_name == "run_simulation":
        if not (OPENMM_ENABLED or GROMACS_ENABLED):
            return json.dumps({"error": "No simulation backend enabled (OPENMM_ENABLED / GROMACS_ENABLED)"})
        production_ns = float(tool_input.get("production_ns", MD_PRODUCTION_NS))
        pH = float(state["context"].get("pH", 7.4))
        temperature_c = float(state["context"].get("temperature_c", 25.0))
        membrane_ctx = state["context"].get("membrane")
        ligand_ctx = state["context"].get("ligands") or []
        try:
            if OPENMM_ENABLED:
                sim = run_openmm_simulation(
                    state["current_pdb"], pH=pH,
                    temperature_c=temperature_c, production_ns=production_ns,
                    membrane_context=membrane_ctx,
                    ligand_contexts=ligand_ctx if ligand_ctx else None,
                )
            else:
                sim = run_gromacs_md(
                    state["current_pdb"], pH=pH,
                    temperature_c=temperature_c, production_ns=production_ns,
                    membrane_context=membrane_ctx,
                    ligand_contexts=ligand_ctx if ligand_ctx else None,
                )
            state["sim_result"] = sim
            # Return compact summary — omit full RMSD/Rg arrays to save context
            summary: Dict[str, Any] = {
                k: v for k, v in sim.items()
                if k not in ("rmsd_nm", "rg_nm", "protonation")
            }
            if sim.get("rmsd_nm"):
                summary["rmsd_final_nm"] = round(sim["rmsd_nm"][-1], 4)
                summary["rmsd_mean_nm"] = round(sum(sim["rmsd_nm"]) / len(sim["rmsd_nm"]), 4)
            if sim.get("rg_nm"):
                summary["rg_mean_nm"] = round(sum(sim["rg_nm"]) / len(sim["rg_nm"]), 4)
            return json.dumps({"status": "completed", **summary})
        except Exception as e:
            return json.dumps({"error": str(e)})

    if tool_name == "run_boltz_prediction":
        if not BOLTZ_ENABLED:
            return json.dumps({"error": "BOLTZ_ENABLED=False — cannot run Boltz-2"})
        try:
            boltz_pred = call_boltz(
                state["sequence"], context=state["context"], seed=0
            )
            state["current_pdb"] = boltz_pred.structure_pdb
            state["plddt_scores"] = boltz_pred.plddt_scores
            state["mean_plddt"] = boltz_pred.mean_plddt
            result: Dict[str, Any] = {
                "status": "completed",
                "model_name": "boltz2",
                "mean_plddt": round(boltz_pred.mean_plddt, 2),
            }
            if boltz_pred.affinity_score is not None:
                result["affinity_kcal_mol"] = round(boltz_pred.affinity_score, 3)
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": str(e)})

    if tool_name in ("accept_structure", "escalate_structure"):
        state["terminal_tool"] = tool_name
        state["agent_reasoning"] = tool_input.get("reasoning", "")
        return json.dumps({"status": "decision_recorded"})

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def run_agent_refinement(
    prediction: StructurePrediction,
    context: Dict[str, Any],
    sequence: str,
    inter_model_data: Optional[Dict[str, Any]] = None,
) -> Tuple[PostProcessingResult, Optional[str]]:
    """
    Use Claude as an adaptive agent to assess and refine a protein structure.

    The agent iterates: analyze → (optionally) refine/simulate → decide.
    Falls back to threshold logic if the anthropic package is not installed
    or ANTHROPIC_API_KEY is not set.

    inter_model_data: optional output of _align_and_compare_structures(); when
    provided the agent receives disagreement regions in its initial prompt.

    Returns:
        (PostProcessingResult, updated_pdb_or_None)
        updated_pdb is non-None when the agent ran Rosetta relax.
    """
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic not installed — falling back to threshold logic (pip install anthropic)")
        return compute_post_processing(prediction), None

    if not ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set — falling back to threshold logic")
        return compute_post_processing(prediction), None

    num_clashes = _count_clashes(prediction.structure_pdb)

    state: Dict[str, Any] = {
        "current_pdb":    prediction.structure_pdb,
        "plddt_scores":   prediction.plddt_scores,
        "mean_plddt":     prediction.mean_plddt,
        "num_clashes":    num_clashes,
        "context":        context,
        "sequence":       sequence,
        "rosetta_energy": None,
        "sim_result":     None,
        "terminal_tool":  None,
        "agent_reasoning": "",
    }

    disagreement_lines = ""
    if inter_model_data and inter_model_data.get("mean_disagreement_nm") is not None:
        disagreement_lines = (
            f"\nInter-model disagreement ({inter_model_data.get('n_models_compared', '?')} models): "
            f"mean {inter_model_data['mean_disagreement_nm']:.3f} nm CA RMSD"
        )
        if inter_model_data.get("disagreement_regions"):
            disagreement_lines += f"\n  High-disagreement regions: {inter_model_data['disagreement_regions']}"

    affinity_line = (
        f"\nBinding affinity (Boltz-2): {prediction.affinity_score:.3f} kcal/mol"
        if prediction.affinity_score is not None else ""
    )

    user_msg = (
        f"Assess this predicted protein structure:\n\n"
        f"Sequence length: {len(sequence)} residues\n"
        f"Prediction model: {prediction.model_name}\n"
        f"Mean pLDDT: {prediction.mean_plddt:.1f}/100\n"
        f"Steric clashes: {num_clashes}"
        f"{affinity_line}\n"
        f"Per-residue pLDDT (first 20): "
        f"{[round(s, 1) for s in prediction.plddt_scores[:20]]}"
        f"{'...' if len(prediction.plddt_scores) > 20 else ''}"
        f"{disagreement_lines}\n\n"
        f"Context:\n"
        f"  pH: {context.get('pH', 7.4)}\n"
        f"  Temperature: {context.get('temperature_c', 25.0)} °C\n"
        f"  Membrane: {context.get('membrane')}\n"
        f"  Ligands: {[l.get('name') for l in context.get('ligands', [])] or None}\n"
        f"  Mutations requested: {context.get('mutations')}\n\n"
        f"Available backends: "
        f"Rosetta={'enabled' if ROSETTA_ENABLED else 'disabled'}, "
        f"OpenMM={'enabled' if OPENMM_ENABLED else 'disabled'}, "
        f"Boltz-2={'enabled' if BOLTZ_ENABLED else 'disabled'}, "
        f"GROMACS={'enabled' if GROMACS_ENABLED else 'disabled'}"
    )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    messages: List[Dict[str, Any]] = [{"role": "user", "content": user_msg}]

    for iteration in range(AGENT_MAX_ITERATIONS):
        logger.info(f"Agent iteration {iteration + 1}/{AGENT_MAX_ITERATIONS}")

        response = client.messages.create(
            model=AGENT_MODEL,
            max_tokens=2048,
            system=_AGENT_SYSTEM,
            tools=_AGENT_TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            logger.warning("Agent ended without terminal tool — defaulting to escalate")
            state["terminal_tool"] = "escalate_structure"
            state["agent_reasoning"] = "Agent completed without an explicit terminal decision"
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result_str = _execute_agent_tool(block.name, block.input, state)
                logger.info(f"Tool '{block.name}' → {result_str[:120]}…")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })
            messages.append({"role": "user", "content": tool_results})

            if state["terminal_tool"] is not None:
                logger.info(f"Agent terminal decision: {state['terminal_tool']}")
                break
    else:
        logger.warning(f"Agent hit max iterations ({AGENT_MAX_ITERATIONS}) — escalating")
        state["terminal_tool"] = "escalate_structure"
        state["agent_reasoning"] = f"Max iterations ({AGENT_MAX_ITERATIONS}) reached without decision"

    # Build PostProcessingResult
    score = state["mean_plddt"] - (state["num_clashes"] * 5.0)
    decision = "accept" if state["terminal_tool"] == "accept_structure" else "escalate"

    post_proc = PostProcessingResult(
        num_clashes=num_clashes,
        rosetta_energy=state["rosetta_energy"],
        score=score,
        decision=decision,
        agent_reasoning=state["agent_reasoning"] or None,
    )

    if state["sim_result"]:
        post_proc.gromacs_potential_energy = state["sim_result"].get("potential_energy")
        post_proc.simulation_metrics = state["sim_result"]

    updated_pdb = state["current_pdb"] if state["current_pdb"] != prediction.structure_pdb else None
    return post_proc, updated_pdb


# ---------------------------------------------------------------------------
# Main Celery task
# ---------------------------------------------------------------------------

def _run_prediction_core(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Core prediction logic: ESMFold → (optional) Rosetta relax → (optional) GROMACS/OpenMM.
    Called by both the Celery task (local dev) and the Modal function (production).
    """
    run_id = request_data.get("run_id", str(uuid.uuid4()))
    sequence = request_data["sequence"]
    context = request_data.get("context", {})
    priority = request_data.get("priority", "fast")

    logger.info(f"Starting prediction task {run_id} (priority={priority})")

    try:
        cache_key = generate_cache_key(sequence, context, pipeline=priority)

        # Cache check — return immediately on hit
        try:
            cached = _get_redis().get(f"{REDIS_CACHE_PREFIX}{cache_key}")
            if cached:
                logger.info(f"Cache hit for {run_id} (key {cache_key[:12]}…)")
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")

        # Step 1: ESMFold — run ENSEMBLE_NUM_SEEDS times
        predictions: List[StructurePrediction] = []
        for seed in range(ENSEMBLE_NUM_SEEDS):
            p = call_esmfold_api(sequence, seed=seed)
            predictions.append(p)
            logger.info(f"[esmfold] seed={seed}: mean pLDDT={p.mean_plddt:.2f}")

        # Step 1b: Additional model backends (Stage E)
        if ROSETTAFOLD_ENABLED:
            try:
                rf2_pred = call_rosettafold2(sequence)
                predictions.append(rf2_pred)
                logger.info(f"[rosettafold2] mean pLDDT={rf2_pred.mean_plddt:.2f}")
            except (RuntimeError, NotImplementedError) as e:
                logger.warning(f"RoseTTAFold2 skipped: {e}")

        if OPENFOLD_ENABLED:
            try:
                of_pred = call_openfold(sequence)
                predictions.append(of_pred)
                logger.info(f"[openfold] mean pLDDT={of_pred.mean_plddt:.2f}")
            except (RuntimeError, NotImplementedError) as e:
                logger.warning(f"OpenFold skipped: {e}")

        if BOLTZ_ENABLED:
            try:
                boltz_pred = call_boltz(sequence, context=context, seed=0)
                predictions.append(boltz_pred)
                logger.info(
                    f"[boltz2] mean pLDDT={boltz_pred.mean_plddt:.2f}"
                    + (f", affinity={boltz_pred.affinity_score:.3f} kcal/mol"
                       if boltz_pred.affinity_score is not None else "")
                )
            except (RuntimeError, FileNotFoundError, ValueError) as e:
                logger.warning(f"Boltz-2 skipped: {e}")

        # Best prediction across all models
        best_prediction = max(predictions, key=lambda p: p.mean_plddt)
        models_used = {p.model_name for p in predictions}
        logger.info(
            f"Best: [{best_prediction.model_name}] seed={best_prediction.seed} "
            f"(pLDDT={best_prediction.mean_plddt:.2f}, "
            f"models run: {sorted(models_used)})"
        )

        # Inter-model structural comparison (only when 2+ distinct models succeeded)
        inter_model_data: Dict[str, Any] = {}
        if len(models_used) >= 2:
            best_per_model: Dict[str, StructurePrediction] = {}
            for pred in predictions:
                name = pred.model_name
                if name not in best_per_model or pred.mean_plddt > best_per_model[name].mean_plddt:
                    best_per_model[name] = pred

            logger.info(f"Computing inter-model disagreement across {len(best_per_model)} models...")
            try:
                inter_model_data = _align_and_compare_structures(
                    [p.structure_pdb for p in best_per_model.values()]
                )
            except Exception as e:
                logger.warning(f"Inter-model comparison failed: {e}")

        # Steps 2–4: Post-processing, refinement, and simulation
        # When AGENT_ENABLED the Claude agent decides which tools to invoke.
        # Otherwise, the original threshold + step-by-step pipeline runs.
        if AGENT_ENABLED:
            logger.info("Running Claude agent refinement loop...")
            post_proc, updated_pdb = run_agent_refinement(
                best_prediction, context, sequence,
                inter_model_data=inter_model_data or None,
            )
            if updated_pdb:
                best_prediction = StructurePrediction(
                    structure_pdb=updated_pdb,
                    plddt_scores=best_prediction.plddt_scores,
                    mean_plddt=best_prediction.mean_plddt,
                    seed=best_prediction.seed,
                    model_name=best_prediction.model_name,
                )
            logger.info(
                f"Agent decision: {post_proc.decision} — {post_proc.agent_reasoning or '(no reasoning)'}"
            )
        else:
            # Step 2: threshold-based post-processing
            post_proc = compute_post_processing(best_prediction)
            logger.info(
                f"Decision: {post_proc.decision} "
                f"(pLDDT={best_prediction.mean_plddt:.2f}, clashes={post_proc.num_clashes})"
            )

            # Step 3: Rosetta relax on refine/escalate (if enabled)
            if post_proc.decision in ("refine", "escalate") and ROSETTA_ENABLED:
                logger.info("Running Rosetta FastRelax...")
                try:
                    relaxed_pdb, rosetta_score = run_rosetta_relax(best_prediction.structure_pdb)
                    best_prediction = StructurePrediction(
                        structure_pdb=relaxed_pdb,
                        plddt_scores=best_prediction.plddt_scores,
                        mean_plddt=best_prediction.mean_plddt,
                        seed=best_prediction.seed,
                        model_name=best_prediction.model_name,
                    )
                    post_proc.rosetta_energy = rosetta_score
                except RuntimeError as e:
                    logger.warning(f"Rosetta relax skipped: {e}")

            # Step 4: MD simulation when membrane or ligand context is present
            needs_md = bool(context.get("membrane") or context.get("ligands"))
            if needs_md and (OPENMM_ENABLED or GROMACS_ENABLED):
                pH = float(context.get("pH", 7.4))
                temperature_c = float(context.get("temperature_c", 25.0))
                membrane_ctx = context.get("membrane")  # dict or None
                ligand_ctx = context.get("ligands") or []  # list of dicts
                try:
                    if OPENMM_ENABLED:
                        logger.info(
                            f"Running OpenMM simulation at pH {pH:.1f}, "
                            f"{temperature_c:.1f} °C, {MD_PRODUCTION_NS} ns"
                            + (f", membrane={membrane_ctx.get('type')}" if membrane_ctx else "")
                            + (f", ligands={[l.get('name') for l in ligand_ctx]}" if ligand_ctx else "")
                        )
                        sim_result = run_openmm_simulation(
                            best_prediction.structure_pdb,
                            pH=pH, temperature_c=temperature_c, production_ns=MD_PRODUCTION_NS,
                            membrane_context=membrane_ctx,
                            ligand_contexts=ligand_ctx if ligand_ctx else None,
                        )
                    else:
                        logger.info(
                            f"Running GROMACS MD at pH {pH:.1f}, "
                            f"{temperature_c:.1f} °C, {MD_PRODUCTION_NS} ns"
                            + (f", membrane={membrane_ctx.get('type')}" if membrane_ctx else "")
                            + (f", ligands={[l.get('name') for l in ligand_ctx]}" if ligand_ctx else "")
                        )
                        sim_result = run_gromacs_md(
                            best_prediction.structure_pdb,
                            pH=pH, temperature_c=temperature_c, production_ns=MD_PRODUCTION_NS,
                            membrane_context=membrane_ctx,
                            ligand_contexts=ligand_ctx if ligand_ctx else None,
                        )
                    post_proc.gromacs_potential_energy = sim_result["potential_energy"]
                    post_proc.simulation_metrics = sim_result
                except RuntimeError as e:
                    logger.warning(f"MD simulation skipped: {e}")

        # Extract simulation_pdb from sim metrics (present when OpenMM ran);
        # keep it out of simulation_metrics so the metrics dict stays compact.
        sim_metrics = post_proc.simulation_metrics or {}
        simulation_pdb_out: Optional[str] = sim_metrics.pop("simulation_pdb", None)
        if not sim_metrics:
            post_proc.simulation_metrics = None

        result = {
            "run_id": run_id,
            "sequence": sequence,
            "status": "completed",
            "predictions": [p.model_dump() for p in predictions],
            "ensemble_result": best_prediction.model_dump(),
            "post_processing": post_proc.model_dump(),
            "context": context,
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "error_message": None,
            "simulation_pdb": simulation_pdb_out,
            # Multi-model ensemble (Stage E)
            "n_models_used": len(models_used),
            "inter_model_disagreement": inter_model_data.get("per_residue_disagreement_nm"),
            "disagreement_regions": inter_model_data.get("disagreement_regions"),
        }

        # Store result in cache
        try:
            _get_redis().setex(
                f"{REDIS_CACHE_PREFIX}{cache_key}",
                CACHE_TTL,
                json.dumps(result),
            )
            logger.info(f"Result cached (key {cache_key[:12]}…, TTL {CACHE_TTL}s)")
        except Exception as e:
            logger.warning(f"Cache store failed: {e}")

        logger.info(f"Task {run_id} completed. Decision: {post_proc.decision}")
        return result

    except Exception as e:
        logger.error(f"Prediction task {run_id} failed: {e}", exc_info=True)
        return {
            "run_id": run_id,
            "sequence": sequence,
            "status": "failed",
            "error_message": str(e),
            "created_at": datetime.utcnow().isoformat(),
        }


@app.task(bind=True, base=CallbackTask)
def predict_protein_structure(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Celery task wrapper around _run_prediction_core (used for local dev)."""
    request_data.setdefault("run_id", self.request.id)
    return _run_prediction_core(request_data)
