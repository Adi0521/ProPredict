"""
Shared progress-reporting constants and helpers (Stage 4.6).

Kept deliberately import-light (no Celery, Postgres, or heavy deps) so the
Celery worker, the Modal worker, and the API read-path can all import it, and so
the pure `celery_state_to_status` mapping is unit-testable without services.
"""
from typing import Any, Optional, Tuple

# Named Modal Dict used to relay per-stage progress from the Modal worker (which
# has no Celery result backend) back to the API status endpoint. Keyed by run_id.
PROGRESS_DICT_NAME = "propredict-progress"

# Coarse pipeline stage markers emitted by _run_prediction_core via progress_cb.
STAGE_FOLDING = "folding"
STAGE_POST_PROCESSING = "post_processing"
STAGE_SIMULATION = "simulation"
STAGE_FINALIZING = "finalizing"


def celery_state_to_status(state: Optional[str], info: Any) -> Tuple[str, int, Optional[str]]:
    """
    Map a Celery task state (+ its meta) into (status, progress_percent, stage).

    `info` is the task meta dict on PROGRESS, or the exception on FAILURE — we only
    read it when the state is PROGRESS, so the FAILURE path is never dereferenced.
    """
    state = (state or "PENDING").upper()
    if state == "SUCCESS":
        return "completed", 100, None
    if state == "FAILURE":
        return "failed", 0, None
    if state == "PROGRESS" and isinstance(info, dict):
        percent = int(info.get("progress_percent", 50))
        return "started", percent, info.get("stage")
    if state == "STARTED":
        return "started", 50, None
    if state == "PENDING":
        return "pending", 0, None
    # Unknown/custom state — surface it but don't claim progress.
    return state.lower(), 0, None
