import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

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
    assert response.status_code == 400

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