# QuickStart Guide

## Setup

1. **Clone and navigate**:
   ```bash
   cd /Users/adi-kewalram/ProPredict
   ```

2. **Create `.env` from template**:
   ```bash
   cp .env.example .env
   ```

3. **Start services** (requires Docker & Docker Compose):
   ```bash
   docker-compose up -d
   ```
   
   This starts:
   - PostgreSQL (port 5432)
   - Redis (port 6379)
   - FastAPI (port 8000)
   - Celery Worker
   - Flower (Celery monitoring, port 5555)

## Test the API

### Health Check
```bash
curl -X GET http://localhost:8000/health
```

### Submit a Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "MKTAYIAKQRQISFVKSHFSRQDILDLWQYVQG",
    "context": {
      "pH": 7.4,
      "temperature_c": 25
    },
    "priority": "fast"
  }'
```

Response:
```json
{
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "progress_percent": 0,
  "created_at": "2025-11-27T10:00:00Z",
  "updated_at": "2025-11-27T10:00:00Z"
}
```

### Poll Results
```bash
curl -X GET http://localhost:8000/predict/{run_id}
```

### Monitor Celery Tasks
Open browser: `http://localhost:5555`

## Run Tests Locally (without Docker)

Install dev dependencies:
```bash
pip install -r requirements.txt
```

Run tests:
```bash
pytest tests/ -v
```

## Project Structure
```
ProPredict/
├── api/
│   └── main.py              # FastAPI routes
├── orchestrator/
│   └── tasks.py             # Celery tasks
├── models/
│   └── schemas.py           # Pydantic models
├── tests/
│   └── test_api.py          # API tests
├── config.py                # Configuration
├── docker-compose.yml       # Docker setup
├── Dockerfile.*             # Container definitions
├── requirements.txt         # Python dependencies
├── .env.example             # Environment template
└── README.md                # Full documentation
```

## Next Steps

1. **Customize AlphaFold API**:
   Edit `ALPHAFOLD_API_URL` in `.env` to point to your actual AlphaFold service.

2. **Add database persistence**:
   Implement a `database.py` module with SQLAlchemy ORM to store job results.

3. **Implement Rosetta integration**:
   Add a new Celery task in `orchestrator/tasks.py` called `refine_with_rosetta()`.

4. **Build UI**:
   Add React frontend in `frontend/` to visualize structures (using Mol* or NGL).
