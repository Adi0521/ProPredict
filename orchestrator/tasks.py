import io
import logging
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from typing import Optional, Dict, Any, List, Tuple
from celery import Celery, Task
from datetime import datetime
import requests

from config import (
    CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND,
    ESMFOLD_API_URL,
    ESMFOLD_TIMEOUT,
    ESMFOLD_RETRIES,
    PLDDT_ACCEPT_THRESHOLD,
    PLDDT_REFINE_THRESHOLD,
    ROSETTA_ENABLED,
    GROMACS_ENABLED,
    GROMACS_BIN,
)
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


def call_esmfold_api(sequence: str, seed: int = 0) -> StructurePrediction:
    """
    Call ESMFold REST API with retries.

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
            logger.info(f"ESMFold call succeeded. Mean pLDDT: {mean_plddt:.2f}")

            return StructurePrediction(
                structure_pdb=pdb_string,
                plddt_scores=plddt_scores,
                mean_plddt=mean_plddt,
                seed=seed,
            )

        except requests.exceptions.RequestException as e:
            logger.warning(f"ESMFold API call failed (attempt {attempt + 1}): {e}")
            if attempt < ESMFOLD_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"ESMFold API failed after {ESMFOLD_RETRIES} attempts")
                raise


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


def run_gromacs_em(pdb_string: str) -> Dict[str, Any]:
    """
    Run GROMACS energy minimisation on a structure.

    Pipeline: pdb2gmx → editconf → solvate → grompp → mdrun → energy
    Returns {"potential_energy": float}.
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

        def _gmx(*args):
            """Run a gmx sub-command, raise on failure."""
            cmd = [gmx] + list(args)
            logger.debug(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, cwd=tmpdir)

        logger.info("GROMACS: converting PDB to GROMACS format...")
        _gmx("pdb2gmx", "-f", "input.pdb", "-o", "processed.gro",
             "-p", "topol.top", "-ff", "amber99sb-ildn", "-water", "spc", "-ignh")

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
        return {"potential_energy": potential_energy}


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


# ---------------------------------------------------------------------------
# Post-processing and agent decision logic
# ---------------------------------------------------------------------------

def compute_post_processing(prediction: StructurePrediction) -> PostProcessingResult:
    """Compute post-processing scores and make an accept/refine/escalate decision."""
    mean_plddt = prediction.mean_plddt

    # Placeholder clash detection (real implementation would parse 3D coordinates)
    num_clashes = 0
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
# Main Celery task
# ---------------------------------------------------------------------------

@app.task(bind=True, base=CallbackTask)
def predict_protein_structure(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main orchestration task: ESMFold → (optional) Rosetta relax → (optional) GROMACS EM.

    Args:
        request_data: Dictionary matching PredictionRequest schema.

    Returns:
        Dictionary with prediction results matching PredictionResponse schema.
    """
    run_id = request_data.get("run_id", self.request.id)
    sequence = request_data["sequence"]
    context = request_data.get("context", {})
    priority = request_data.get("priority", "fast")

    logger.info(f"Starting prediction task {run_id} (priority={priority})")

    try:
        cache_key = generate_cache_key(sequence, context, pipeline=priority)
        logger.info(f"Cache key: {cache_key}")

        # Step 1: ESMFold structure prediction
        prediction = call_esmfold_api(sequence, seed=0)

        # Step 2: Post-processing and agent decision
        post_proc = compute_post_processing(prediction)
        logger.info(f"Decision: {post_proc.decision} (mean pLDDT={prediction.mean_plddt:.2f})")

        # Step 3: Rosetta relax on refine/escalate decisions (if enabled)
        if post_proc.decision in ("refine", "escalate") and ROSETTA_ENABLED:
            logger.info("Running Rosetta FastRelax...")
            try:
                relaxed_pdb, rosetta_score = run_rosetta_relax(prediction.structure_pdb)
                prediction = StructurePrediction(
                    structure_pdb=relaxed_pdb,
                    plddt_scores=prediction.plddt_scores,
                    mean_plddt=prediction.mean_plddt,
                    seed=prediction.seed,
                )
                post_proc.rosetta_energy = rosetta_score
            except RuntimeError as e:
                logger.warning(f"Rosetta relax skipped: {e}")

        # Step 4: GROMACS EM when membrane or ligand context is present (if enabled)
        needs_md = bool(
            context.get("membrane") or context.get("ligands")
        )
        if needs_md and GROMACS_ENABLED:
            logger.info("Running GROMACS energy minimisation...")
            try:
                gromacs_result = run_gromacs_em(prediction.structure_pdb)
                post_proc.gromacs_potential_energy = gromacs_result["potential_energy"]
            except RuntimeError as e:
                logger.warning(f"GROMACS EM skipped: {e}")

        result = {
            "run_id": run_id,
            "sequence": sequence,
            "status": "completed",
            "predictions": [prediction.model_dump()],
            "ensemble_result": prediction.model_dump(),
            "post_processing": post_proc.model_dump(),
            "context": context,
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "error_message": None,
        }

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
