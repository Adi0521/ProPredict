import hashlib
import json
import logging
import random
import time
import uuid
from typing import Optional, Dict, Any, List
from celery import Celery, Task
from datetime import datetime
import requests

from config import (
    CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND,
    ROSETTA_ENABLED,
    GROMACS_ENABLED,
    OPENMM_ENABLED,
    ROSETTAFOLD_ENABLED,
    OPENFOLD_ENABLED,
    BOLTZ_ENABLED,
    MD_PRODUCTION_NS,
    AGENT_ENABLED,
    ENSEMBLE_NUM_SEEDS,
    REFINEMENT_MAX_ITERATIONS,
    REFINEMENT_PLDDT_PLATEAU_DELTA,
    REDIS_URL,
    CACHE_TTL,
    REDIS_CACHE_PREFIX,
)
from models.schemas import StructurePrediction, PostProcessingResult

from orchestrator.backends.esmfold import (
    call_esmfold_api,
    _parse_plddt_from_pdb,
)
from orchestrator.backends.boltz import call_boltz
from orchestrator.backends.stubs import call_rosettafold2, call_openfold
from orchestrator.ensemble import align_and_compare_structures
from orchestrator.simulation import (
    run_rosetta_relax,
    run_gromacs_em,
    run_gromacs_md,
    run_openmm_simulation,
)
from orchestrator.scoring import count_clashes, compute_post_processing, validate_simulation_metrics
from orchestrator.agent import run_agent_refinement

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
# Main pipeline
# ---------------------------------------------------------------------------

def _run_prediction_core(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Core prediction logic with multi-seed sampling and iterative refinement.

    Pipeline:
      1. Initial multi-seed predictions (ESMFold + Boltz-2 with ENSEMBLE_NUM_SEEDS seeds each)
      2. Score best prediction -> if "accept", skip refinement
      3. Iterative refinement loop (up to REFINEMENT_MAX_ITERATIONS):
         a. Try new Boltz-2 seed (stochastic -> new structure each time)
         b. If Rosetta enabled, relax the best structure so far
         c. Re-score; exit early on accept or pLDDT plateau
      4. Post-refinement: MD simulation (OpenMM/GROMACS) if membrane/ligand context present

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

        # ---------------------------------------------------------------
        # Step 1: Initial multi-seed predictions
        # ---------------------------------------------------------------
        predictions: List[StructurePrediction] = []

        # ESMFold (deterministic — one call suffices regardless of ENSEMBLE_NUM_SEEDS)
        p = call_esmfold_api(sequence, seed=0)
        predictions.append(p)
        logger.info(f"[esmfold] mean pLDDT={p.mean_plddt:.2f}")

        # Additional model backends (Stage E stubs)
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

        # Boltz-2 multi-seed: random seeds -> stochastic diffusion gives different structures
        if BOLTZ_ENABLED:
            num_boltz_seeds = max(1, ENSEMBLE_NUM_SEEDS)
            boltz_seeds = [random.randint(0, 2**31 - 1) for _ in range(num_boltz_seeds)]
            for seed in boltz_seeds:
                try:
                    boltz_pred = call_boltz(sequence, context=context, seed=seed)
                    predictions.append(boltz_pred)
                    logger.info(
                        f"[boltz2] seed={seed}: mean pLDDT={boltz_pred.mean_plddt:.2f}"
                        + (f", affinity={boltz_pred.affinity_score:.3f} kcal/mol"
                           if boltz_pred.affinity_score is not None else "")
                    )
                except (RuntimeError, FileNotFoundError, ValueError) as e:
                    logger.warning(f"Boltz-2 seed={seed} skipped: {e}")

        # Best prediction across all initial seeds/models
        best_prediction = max(predictions, key=lambda p: p.mean_plddt)
        models_used = {p.model_name for p in predictions}
        logger.info(
            f"Initial best: [{best_prediction.model_name}] seed={best_prediction.seed} "
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
                inter_model_data = align_and_compare_structures(
                    [p.structure_pdb for p in best_per_model.values()]
                )
            except Exception as e:
                logger.warning(f"Inter-model comparison failed: {e}")

        # ---------------------------------------------------------------
        # Step 2: Iterative refinement loop
        # ---------------------------------------------------------------
        refinement_iterations = 0
        # When AGENT_ENABLED the Claude agent decides tools to invoke (replaces this loop).
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
            post_proc = compute_post_processing(best_prediction)
            logger.info(
                f"Initial decision: {post_proc.decision} "
                f"(pLDDT={best_prediction.mean_plddt:.2f}, clashes={post_proc.num_clashes})"
            )

            # Iterative refinement: re-predict (random seeds) + relax until accept or budget exhausted
            prev_best_plddt = best_prediction.mean_plddt
            refinement_iterations = 0

            while (
                post_proc.decision in ("refine", "escalate")
                and refinement_iterations < REFINEMENT_MAX_ITERATIONS
            ):
                refinement_iterations += 1
                logger.info(
                    f"Refinement iteration {refinement_iterations}/{REFINEMENT_MAX_ITERATIONS} "
                    f"(current pLDDT={best_prediction.mean_plddt:.2f})"
                )

                improved = False

                # Strategy A: Try a new random Boltz-2 seed (stochastic -> potentially better structure)
                if BOLTZ_ENABLED:
                    rand_seed = random.randint(0, 2**31 - 1)
                    try:
                        new_pred = call_boltz(sequence, context=context, seed=rand_seed)
                        predictions.append(new_pred)
                        logger.info(
                            f"[boltz2] refinement seed={rand_seed}: "
                            f"mean pLDDT={new_pred.mean_plddt:.2f}"
                        )
                        if new_pred.mean_plddt > best_prediction.mean_plddt:
                            best_prediction = new_pred
                            improved = True
                    except (RuntimeError, FileNotFoundError, ValueError) as e:
                        logger.warning(f"Boltz-2 refinement seed={rand_seed} failed: {e}")

                # Strategy B: Rosetta relax on the current best structure
                if ROSETTA_ENABLED:
                    try:
                        relaxed_pdb, rosetta_score = run_rosetta_relax(best_prediction.structure_pdb)
                        relaxed_plddt = _parse_plddt_from_pdb(relaxed_pdb)
                        if relaxed_plddt:
                            relaxed_mean = sum(relaxed_plddt) / len(relaxed_plddt)
                        else:
                            relaxed_mean = best_prediction.mean_plddt
                        if relaxed_mean >= best_prediction.mean_plddt:
                            best_prediction = StructurePrediction(
                                structure_pdb=relaxed_pdb,
                                plddt_scores=relaxed_plddt or best_prediction.plddt_scores,
                                mean_plddt=relaxed_mean,
                                seed=best_prediction.seed,
                                model_name=best_prediction.model_name,
                            )
                            improved = True
                        post_proc.rosetta_energy = rosetta_score
                        logger.info(f"Rosetta relax: score={rosetta_score:.1f}, pLDDT={relaxed_mean:.2f}")
                    except RuntimeError as e:
                        logger.warning(f"Rosetta relax skipped: {e}")

                # Re-score after this iteration
                post_proc = compute_post_processing(best_prediction)

                # Early exit: pLDDT plateau detection — stop if improvement < delta
                plddt_delta = best_prediction.mean_plddt - prev_best_plddt
                if not improved or plddt_delta < REFINEMENT_PLDDT_PLATEAU_DELTA:
                    logger.info(
                        f"Refinement plateau at iteration {refinement_iterations} "
                        f"(delta={plddt_delta:.3f}, threshold={REFINEMENT_PLDDT_PLATEAU_DELTA})"
                    )
                    break
                prev_best_plddt = best_prediction.mean_plddt

            if refinement_iterations > 0:
                logger.info(
                    f"Refinement complete after {refinement_iterations} iteration(s). "
                    f"Final pLDDT={best_prediction.mean_plddt:.2f}, decision={post_proc.decision}"
                )

            # Step 3: MD simulation when membrane or ligand context is present
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

                    # Stage 4.5: validate the trajectory; escalate if it looks unphysical.
                    failure_reason = validate_simulation_metrics(sim_result)
                    if failure_reason:
                        logger.warning(f"MD validation failed — escalating: {failure_reason}")
                        post_proc.validation_reason = failure_reason
                        post_proc.decision = "escalate"
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
            # Refinement metadata
            "refinement_iterations": refinement_iterations if not AGENT_ENABLED else None,
            "total_seeds_tried": len(predictions),
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
            "context": context,
            "created_at": datetime.utcnow().isoformat(),
        }


@app.task(bind=True, base=CallbackTask)
def predict_protein_structure(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Celery task wrapper around _run_prediction_core (used for local dev)."""
    request_data.setdefault("run_id", self.request.id)
    return _run_prediction_core(request_data)
