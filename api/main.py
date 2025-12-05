import logging
import uuid
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from datetime import datetime

from models.schemas import PredictionRequest, PredictionResponse, JobStatus
from orchestrator.tasks import predict_protein_structure
from config import API_DEBUG, LOG_LEVEL

# Configure logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ProPredict",
    description="Agentic AI Service for Protein Structure Prediction",
    version="0.1.0",
)

# In-memory job store (in production, use Redis or database)
jobs_store = {}

@app.on_event("startup")
async def startup_event():
    logger.info("ProPredict API started")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("ProPredict API shutdown")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/predict", response_model=JobStatus)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Submit a protein structure prediction job.
    
    Returns a job ID that can be polled for status.
    """
    try:
        # Generate run ID if not provided
        run_id = request.run_id or str(uuid.uuid4())
        
        logger.info(f"Received prediction request {run_id} for sequence: {request.sequence[:30]}...")
        
        # Validate sequence length
        if len(request.sequence) > 2000:
            raise HTTPException(status_code=400, detail="Sequence too long (max 2000 residues)")
        
        # Prepare request data
        request_data = request.dict()
        request_data["run_id"] = run_id
        
        # Store job metadata
        jobs_store[run_id] = {
            "status": "pending",
            "progress_percent": 0,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "error_message": None,
        }
        
        # Submit async task
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
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
    
    except Exception as e:
        logger.error(f"Error submitting prediction request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to submit prediction: {str(e)}")

@app.get("/predict/{run_id}", response_model=PredictionResponse)
async def get_prediction(run_id: str):
    """
    Retrieve prediction results by job ID.
    """
    try:
        logger.info(f"Fetching results for run ID: {run_id}")
        
        # Get task result from Celery
        task = predict_protein_structure.AsyncResult(run_id)
        
        if task.state == "PENDING":
            return PredictionResponse(
                run_id=run_id,
                sequence="",
                status="pending",
                context={},
                created_at=jobs_store.get(run_id, {}).get("created_at", datetime.utcnow()),
            )
        
        elif task.state == "SUCCESS":
            result = task.result
            return PredictionResponse(**result)
        
        elif task.state == "FAILURE":
            return PredictionResponse(
                run_id=run_id,
                sequence="",
                status="failed",
                error_message=str(task.info),
                context={},
                created_at=jobs_store.get(run_id, {}).get("created_at", datetime.utcnow()),
            )
        
        else:
            return PredictionResponse(
                run_id=run_id,
                sequence="",
                status=task.state.lower(),
                context={},
                created_at=jobs_store.get(run_id, {}).get("created_at", datetime.utcnow()),
            )
    
    except Exception as e:
        logger.error(f"Error retrieving results for {run_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve results: {str(e)}")

@app.get("/predict/{run_id}/status", response_model=JobStatus)
async def get_job_status(run_id: str):
    """Get job status without full results."""
    try:
        task = predict_protein_structure.AsyncResult(run_id)
        job_meta = jobs_store.get(run_id, {})
        
        return JobStatus(
            run_id=run_id,
            status=task.state.lower() if task.state != "PENDING" else "pending",
            progress_percent=0 if task.state == "PENDING" else (50 if task.state == "STARTED" else 100),
            error_message=job_meta.get("error_message"),
            created_at=job_meta.get("created_at", datetime.utcnow()),
            updated_at=job_meta.get("updated_at", datetime.utcnow()),
        )
    
    except Exception as e:
        logger.error(f"Error getting status for {run_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.warning(f"HTTP exception: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=API_DEBUG)