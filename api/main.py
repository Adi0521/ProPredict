import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy.orm import Session

from config import API_DEBUG, LOG_LEVEL
from models.database import Job, SessionLocal, get_db, init_db
from models.schemas import PredictionRequest, PredictionResponse, JobStatus, StructurePrediction

MODAL_ENABLED = os.getenv("MODAL_ENABLED", "False") == "True"

if MODAL_ENABLED:
    import modal
    _modal_predict = modal.Function.from_name("propredict", "run_prediction")
else:
    from orchestrator.tasks import predict_protein_structure

from orchestrator.progress import PROGRESS_DICT_NAME, celery_state_to_status


def _read_modal_progress(run_id: str) -> Optional[dict]:
    """Read the latest {progress_percent, stage} the Modal worker wrote, if any."""
    try:
        d = modal.Dict.from_name(PROGRESS_DICT_NAME, create_if_missing=True)
        return d.get(run_id)
    except Exception:
        return None

# Configure logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB tables on startup; log shutdown."""
    init_db()
    logger.info("ProPredict API started")
    yield
    logger.info("ProPredict API shutdown")


app = FastAPI(
    title="ProPredict",
    description="Agentic AI Service for Protein Structure Prediction",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/predict", response_model=JobStatus)
async def predict(request: PredictionRequest, db: Session = Depends(get_db)):
    """
    Submit a protein structure prediction job.

    Returns a job ID that can be polled for status.
    """
    try:
        run_id = request.run_id or str(uuid.uuid4())
        logger.info(f"Received prediction request {run_id} for sequence: {request.sequence[:30]}...")

        # Persist job metadata to Postgres
        now = datetime.utcnow()
        job = Job(
            run_id=run_id,
            status="pending",
            progress_percent=0,
            sequence=request.sequence,
            created_at=now,
            updated_at=now,
        )
        db.add(job)
        db.commit()

        # Dispatch the prediction task
        request_data = request.model_dump()
        request_data["run_id"] = run_id
        if MODAL_ENABLED:
            fc = _modal_predict.spawn(request_data)
            job.modal_call_id = fc.object_id
            db.commit()
            logger.info(f"Task {run_id} submitted as Modal call {fc.object_id}")
        else:
            task = predict_protein_structure.apply_async(
                args=(request_data,),
                task_id=run_id,
                expires=request.job_timeout_seconds,
            )
            logger.info(f"Task {run_id} submitted with Celery task ID: {task.id}")

        return JobStatus(
            run_id=run_id,
            status="pending",
            progress_percent=0,
            created_at=now,
            updated_at=now,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting prediction request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to submit prediction")


@app.get("/predict/{run_id}", response_model=PredictionResponse)
async def get_prediction(run_id: str, db: Session = Depends(get_db)):
    """Retrieve full prediction results by job ID."""
    try:
        logger.info(f"Fetching results for run ID: {run_id}")
        job = db.get(Job, run_id)

        if MODAL_ENABLED:
            if not job or not job.modal_call_id:
                raise HTTPException(status_code=404, detail=f"Job {run_id} not found")
            fc = modal.functions.FunctionCall.from_id(job.modal_call_id)
            try:
                result = fc.get(timeout=0)
                if job:
                    job.status = "completed"
                    job.result_json = json.dumps(result)
                    job.updated_at = datetime.utcnow()
                    db.commit()
                return PredictionResponse(**result)
            except TimeoutError:
                return PredictionResponse(
                    run_id=run_id,
                    sequence=job.sequence if job else "",
                    status="started",
                    context={},
                    created_at=job.created_at if job else datetime.utcnow(),
                )
            except Exception as exc:
                if job:
                    job.status = "failed"
                    job.error_message = str(exc)
                    job.updated_at = datetime.utcnow()
                    db.commit()
                return PredictionResponse(
                    run_id=run_id,
                    sequence=job.sequence if job else "",
                    status="failed",
                    error_message=str(exc),
                    context={},
                    created_at=job.created_at if job else datetime.utcnow(),
                )
        else:
            task = predict_protein_structure.AsyncResult(run_id)

            if task.state == "SUCCESS" and task.result:
                result = task.result
                if job:
                    job.status = "completed"
                    job.result_json = json.dumps(result)
                    job.updated_at = datetime.utcnow()
                    db.commit()
                return PredictionResponse(**result)

            if task.state == "FAILURE":
                if job:
                    job.status = "failed"
                    job.error_message = str(task.info)
                    job.updated_at = datetime.utcnow()
                    db.commit()
                return PredictionResponse(
                    run_id=run_id,
                    sequence=job.sequence if job else "",
                    status="failed",
                    error_message=str(task.info),
                    context={},
                    created_at=job.created_at if job else datetime.utcnow(),
                )

            return PredictionResponse(
                run_id=run_id,
                sequence=job.sequence if job else "",
                status=task.state.lower(),
                context={},
                created_at=job.created_at if job else datetime.utcnow(),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving results for {run_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve results")


@app.get("/predict/{run_id}/status", response_model=JobStatus)
async def get_job_status(run_id: str, db: Session = Depends(get_db)):
    """Get job status without full results."""
    try:
        job = db.get(Job, run_id)
        now = datetime.utcnow()

        stage: Optional[str] = None
        if MODAL_ENABLED:
            if not job or not job.modal_call_id:
                raise HTTPException(status_code=404, detail=f"Job {run_id} not found")
            fc = modal.functions.FunctionCall.from_id(job.modal_call_id)
            try:
                fc.get(timeout=0)
                state, progress = "completed", 100
            except TimeoutError:
                # In flight — refine the coarse 50% with the worker's Dict progress.
                state, progress = "started", 50
                entry = _read_modal_progress(run_id)
                if entry:
                    progress = int(entry.get("progress_percent", progress))
                    stage = entry.get("stage")
            except Exception:
                state, progress = "failed", 0
        else:
            task = predict_protein_structure.AsyncResult(run_id)
            state, progress, stage = celery_state_to_status(task.state, task.info)

        return JobStatus(
            run_id=run_id,
            status=state,
            progress_percent=progress,
            stage=stage,
            error_message=job.error_message if job else None,
            created_at=job.created_at if job else now,
            updated_at=job.updated_at if job else now,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status for {run_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get job status")


def _get_completed_result(run_id: str, db: Session) -> dict:
    """Fetch the completed result dict for a job from Modal or Celery."""
    if MODAL_ENABLED:
        job = db.get(Job, run_id)
        if not job or not job.modal_call_id:
            raise HTTPException(status_code=404, detail=f"Job {run_id} not found")
        if job.result_json:
            return json.loads(job.result_json)
        fc = modal.functions.FunctionCall.from_id(job.modal_call_id)
        try:
            return fc.get(timeout=0)
        except TimeoutError:
            raise HTTPException(status_code=404, detail=f"PDB not available — job is still running")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"PDB not available — job failed: {exc}")
    else:
        task = predict_protein_structure.AsyncResult(run_id)
        if task.state != "SUCCESS" or not task.result:
            status = task.state.lower()
            raise HTTPException(
                status_code=404 if status == "pending" else 400,
                detail=f"PDB not available — job status is '{status}'",
            )
        return task.result


@app.get("/predict/{run_id}/pdb")
async def get_pdb(run_id: str, db: Session = Depends(get_db)):
    """
    Download the predicted PDB structure file directly.
    Returns a .pdb file attachment ready to open in PyMOL, ChimeraX, etc.
    """
    result = _get_completed_result(run_id, db)

    pdb_string = result.get("ensemble_result", {}).get("structure_pdb", "")
    if not pdb_string:
        raise HTTPException(status_code=404, detail="No PDB structure found in result")

    filename = f"propredict_{run_id[:8]}.pdb"
    return PlainTextResponse(
        content=pdb_string,
        media_type="chemical/x-pdb",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/predict/{run_id}/simulation-pdb")
async def get_simulation_pdb(run_id: str, db: Session = Depends(get_db)):
    """
    Download the full simulation-ready PDB (post-solvation / post-docking).

    Unlike /pdb (which returns the raw ESMFold output), this endpoint returns
    the prepared OpenMM system: protein + explicit water + ions, and any docked
    ligands or membrane included in the context.  Only available when OpenMM
    ran successfully (OPENMM_ENABLED=True and a membrane/ligand context was
    provided, or MD was triggered).
    """
    result = _get_completed_result(run_id, db)

    pdb_string = result.get("simulation_pdb")
    if not pdb_string:
        raise HTTPException(
            status_code=404,
            detail=(
                "No simulation PDB found. Either OpenMM did not run for this job "
                "(requires OPENMM_ENABLED=True and a membrane/ligand context), "
                "or the PDB capture failed."
            ),
        )

    filename = f"propredict_{run_id[:8]}_simulation.pdb"
    return PlainTextResponse(
        content=pdb_string,
        media_type="chemical/x-pdb",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.warning(f"HTTP exception: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=API_DEBUG)
