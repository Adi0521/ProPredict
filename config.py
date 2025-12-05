import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_DEBUG = os.getenv("API_DEBUG", "False") == "True"

# AlphaFold API Configuration
ALPHAFOLD_API_URL = os.getenv("ALPHAFOLD_API_URL", "https://alphafold.ebi.ac.uk/api/prediction")
ALPHAFOLD_TIMEOUT = int(os.getenv("ALPHAFOLD_TIMEOUT", 300))
ALPHAFOLD_RETRIES = int(os.getenv("ALPHAFOLD_RETRIES", 3))

# Database Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_NAME = os.getenv("DB_NAME", "propredict")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# Celery Configuration
CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TIMEZONE = "UTC"
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = 30 * 60  # 30 minutes

# Agent Policy Thresholds
PLDDT_ACCEPT_THRESHOLD = float(os.getenv("PLDDT_ACCEPT_THRESHOLD", 75.0))
PLDDT_REFINE_THRESHOLD = float(os.getenv("PLDDT_REFINE_THRESHOLD", 60.0))
PLDDT_VARIANCE_THRESHOLD = float(os.getenv("PLDDT_VARIANCE_THRESHOLD", 30.0))
VARIANCE_REGION_PERCENT = float(os.getenv("VARIANCE_REGION_PERCENT", 10.0))

# Job Configuration
JOB_TIMEOUT_SECONDS = int(os.getenv("JOB_TIMEOUT_SECONDS", 600))
ENSEMBLE_NUM_SEEDS = int(os.getenv("ENSEMBLE_NUM_SEEDS", 1))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")