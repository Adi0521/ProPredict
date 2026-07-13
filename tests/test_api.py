import pytest
from fastapi.testclient import TestClient
from api.main import app
from models.database import Job, SessionLocal, init_db

client = TestClient(app)


@pytest.fixture(scope="session", autouse=True)
def initialize_database():
    """Create the schema, apply pending column migrations, and clean up the
    jobs table before any test runs.

    The app normally calls init_db() in its lifespan startup, but instantiating
    TestClient(app) directly (without `with`) does not fire lifespan events, so
    the auto-migration (e.g. ADD COLUMN modal_call_id) would never run. Invoking
    it here keeps the test DB schema in sync without manual psql patching.

    We also clear the jobs table so tests that use a deterministic run_id
    (e.g. "test-run-123") don't collide with rows left by a previous run against
    the persistent Postgres volume.
    """
    init_db()
    db = SessionLocal()
    try:
        db.query(Job).delete()
        db.commit()
    finally:
        db.close()

def test_health_check():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_valid_sequence():
    """Test prediction endpoint with valid sequence."""
    payload = {
        "sequence": "MKTAYIAKQRQISFVKSHFSRQDILDLWQYVQG",
        "context": {
            "pH": 7.4,
            "temperature_c": 25,
        },
        "priority": "fast",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pending"
    assert "run_id" in data
    assert data["progress_percent"] == 0

def test_predict_invalid_sequence():
    """Test prediction with invalid amino acid codes."""
    payload = {
        "sequence": "MKXYZABC",
        "priority": "fast",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error

def test_predict_sequence_too_long():
    """Test prediction with sequence exceeding max length."""
    payload = {
        "sequence": "A" * 2001,
        "priority": "fast",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic max_length validation

def test_get_job_status():
    """Test retrieving job status."""
    # Submit a job
    payload = {
        "sequence": "MKTAYIAKQRQISFVKSHFSRQDILDLWQYVQG",
        "run_id": "test-run-123",
    }
    response = client.post("/predict", json=payload)
    run_id = response.json()["run_id"]
    
    # Get status
    status_response = client.get(f"/predict/{run_id}/status")
    assert status_response.status_code == 200
    status_data = status_response.json()
    assert status_data["run_id"] == run_id
    assert status_data["status"] in ["pending", "started", "completed", "failed"]