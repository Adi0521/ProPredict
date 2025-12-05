import logging
import hashlib
import json
import time
from typing import Optional, Dict, Any, List
from celery import Celery, Task
from datetime import datetime
import requests

from config import (
    CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND,
    ALPHAFOLD_API_URL,
    ALPHAFOLD_TIMEOUT,
    ALPHAFOLD_RETRIES,
    PLDDT_ACCEPT_THRESHOLD,
    PLDDT_REFINE_THRESHOLD,
)
from models.schemas import PredictionRequest, AlphaFoldPrediction, PostProcessingResult

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

class CallbackTask(Task):
    """Task that sends webhook callbacks on completion."""
    
    def on_success(self, retval, task_id, args, kwargs):
        """Success callback."""
        request_data = args[0] if args else None
        if request_data and "webhook_url" in request_data and request_data["webhook_url"]:
            send_webhook(request_data["webhook_url"], {"status": "completed", "result": retval})
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Failure callback."""
        request_data = args[0] if args else None
        if request_data and "webhook_url" in request_data and request_data["webhook_url"]:
            send_webhook(request_data["webhook_url"], {"status": "failed", "error": str(exc)})

app.Task = CallbackTask

def generate_cache_key(sequence: str, context: Dict[str, Any], pipeline: str = "af_base") -> str:
    """Generate cache key from sequence and context."""
    context_str = json.dumps(context, sort_keys=True)
    key_input = f"{sequence}:{context_str}:{pipeline}"
    return hashlib.sha256(key_input.encode()).hexdigest()

def call_alphafold_api(sequence: str, seed: int = 0) -> Optional[AlphaFoldPrediction]:
    """Call AlphaFold API with retries."""
    payload = {"sequence": sequence, "seed": seed}
    
    for attempt in range(ALPHAFOLD_RETRIES):
        try:
            logger.info(f"AlphaFold API call attempt {attempt + 1}/{ALPHAFOLD_RETRIES} for seed {seed}")
            response = requests.post(
                ALPHAFOLD_API_URL,
                json=payload,
                timeout=ALPHAFOLD_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            
            # Parse response (adjust based on your actual AF API response format)
            prediction = AlphaFoldPrediction(
                structure_pdb=data.get("pdb_structure", ""),
                plddt_scores=data.get("plddt_scores", []),
                mean_plddt=data.get("mean_plddt", 0.0),
                pae_scores=data.get("pae_scores"),
                seed=seed,
            )
            logger.info(f"AlphaFold API call successful for seed {seed}, pLDDT: {prediction.mean_plddt}")
            return prediction
        
        except requests.exceptions.RequestException as e:
            logger.warning(f"AlphaFold API call failed (attempt {attempt + 1}): {str(e)}")
            if attempt < ALPHAFOLD_RETRIES - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"AlphaFold API failed after {ALPHAFOLD_RETRIES} attempts")
                raise

def compute_post_processing(prediction: AlphaFoldPrediction) -> PostProcessingResult:
    """Compute post-processing scores and decisions."""
    mean_plddt = prediction.mean_plddt
    
    # Placeholder clash detection (would integrate actual 3D clash detection)
    num_clashes = 0
    
    # Simple scoring: weighted combination of pLDDT and clashes
    score = mean_plddt - (num_clashes * 5.0)
    
    # Decision logic
    if mean_plddt >= PLDDT_ACCEPT_THRESHOLD:
        decision = "accept"
    elif mean_plddt >= PLDDT_REFINE_THRESHOLD:
        decision = "refine"
    else:
        decision = "escalate"
    
    return PostProcessingResult(
        num_clashes=num_clashes,
        rosetta_energy=None,
        score=score,
        decision=decision,
    )

def send_webhook(webhook_url: str, payload: Dict[str, Any]) -> None:
    """Send webhook notification with retry logic."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            requests.post(webhook_url, json=payload, timeout=10)
            logger.info(f"Webhook sent successfully to {webhook_url}")
            return
        except requests.exceptions.RequestException as e:
            logger.warning(f"Webhook send failed (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Webhook failed after {max_retries} attempts")

@app.task(bind=True, base=CallbackTask)
def predict_protein_structure(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main task: predict protein structure given request.
    
    Args:
        request_data: Dictionary matching PredictionRequest schema
    
    Returns:
        Dictionary with prediction results
    """
    run_id = request_data.get("run_id", self.request.id)
    sequence = request_data["sequence"]
    context = request_data.get("context", {})
    priority = request_data.get("priority", "fast")
    
    logger.info(f"Starting prediction task {run_id} with priority {priority}")
    
    try:
        # Generate cache key
        cache_key = generate_cache_key(sequence, context, pipeline=priority)
        logger.info(f"Cache key: {cache_key}")
        
        # Call AlphaFold API (for MVP, single seed)
        seed = 0
        prediction = call_alphafold_api(sequence, seed=seed)
        
        # Post-process
        post_proc = compute_post_processing(prediction)
        
        result = {
            "run_id": run_id,
            "sequence": sequence,
            "status": "completed",
            "predictions": [prediction.dict()],
            "ensemble_result": prediction.dict(),
            "post_processing": post_proc.dict(),
            "context": context,
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": datetime.utcnow().isoformat(),
            "error_message": None,
        }
        
        logger.info(f"Prediction task {run_id} completed successfully with decision: {post_proc.decision}")
        return result
    
    except Exception as e:
        logger.error(f"Prediction task {run_id} failed: {str(e)}", exc_info=True)
        return {
            "run_id": run_id,
            "sequence": sequence,
            "status": "failed",
            "error_message": str(e),
            "created_at": datetime.utcnow().isoformat(),
        }