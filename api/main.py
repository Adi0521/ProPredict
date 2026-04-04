import json
import logging
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
from orchestrator.tasks import predict_protein_structure

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

        if len(request.sequence) > 2000:
            raise HTTPException(status_code=400, detail="Sequence too long (max 2000 residues)")

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

        # Prepare and submit async Celery task
        request_data = request.model_dump()
        request_data["run_id"] = run_id
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
        raise HTTPException(status_code=500, detail=f"Failed to submit prediction: {str(e)}")


@app.get("/predict/{run_id}", response_model=PredictionResponse)
async def get_prediction(run_id: str, db: Session = Depends(get_db)):
    """Retrieve full prediction results by job ID."""
    try:
        logger.info(f"Fetching results for run ID: {run_id}")
        job = db.get(Job, run_id)

        task = predict_protein_structure.AsyncResult(run_id)

        if task.state == "SUCCESS" and task.result:
            result = task.result
            # Update DB record with completed result
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

        # PENDING or STARTED
        return PredictionResponse(
            run_id=run_id,
            sequence=job.sequence if job else "",
            status=task.state.lower(),
            context={},
            created_at=job.created_at if job else datetime.utcnow(),
        )

    except Exception as e:
        logger.error(f"Error retrieving results for {run_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve results: {str(e)}")


@app.get("/predict/{run_id}/status", response_model=JobStatus)
async def get_job_status(run_id: str, db: Session = Depends(get_db)):
    """Get job status without full results."""
    try:
        job = db.get(Job, run_id)
        task = predict_protein_structure.AsyncResult(run_id)

        state = task.state.lower() if task.state != "PENDING" else "pending"
        if state == "started":
            progress = 50
        elif state in ("success", "completed"):
            progress = 100
        else:
            progress = 0

        now = datetime.utcnow()
        return JobStatus(
            run_id=run_id,
            status=state,
            progress_percent=progress,
            error_message=job.error_message if job else None,
            created_at=job.created_at if job else now,
            updated_at=job.updated_at if job else now,
        )

    except Exception as e:
        logger.error(f"Error getting status for {run_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")


@app.get("/predict/{run_id}/pdb")
async def get_pdb(run_id: str):
    """
    Download the predicted PDB structure file directly.
    Returns a .pdb file attachment ready to open in PyMOL, ChimeraX, etc.
    """
    task = predict_protein_structure.AsyncResult(run_id)

    if task.state != "SUCCESS" or not task.result:
        status = task.state.lower()
        raise HTTPException(
            status_code=404 if status == "pending" else 400,
            detail=f"PDB not available — job status is '{status}'",
        )

    pdb_string = task.result.get("ensemble_result", {}).get("structure_pdb", "")
    if not pdb_string:
        raise HTTPException(status_code=404, detail="No PDB structure found in result")

    filename = f"propredict_{run_id[:8]}.pdb"
    return PlainTextResponse(
        content=pdb_string,
        media_type="chemical/x-pdb",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/predict/{run_id}/simulation-pdb")
async def get_simulation_pdb(run_id: str):
    """
    Download the full simulation-ready PDB (post-solvation / post-docking).

    Unlike /pdb (which returns the raw ESMFold output), this endpoint returns
    the prepared OpenMM system: protein + explicit water + ions, and any docked
    ligands or membrane included in the context.  Only available when OpenMM
    ran successfully (OPENMM_ENABLED=True and a membrane/ligand context was
    provided, or MD was triggered).
    """
    task = predict_protein_structure.AsyncResult(run_id)

    if task.state != "SUCCESS" or not task.result:
        status = task.state.lower()
        raise HTTPException(
            status_code=404 if status == "pending" else 400,
            detail=f"Simulation PDB not available — job status is '{status}'",
        )

    pdb_string = task.result.get("simulation_pdb")
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
