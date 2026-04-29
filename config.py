import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_DEBUG = os.getenv("API_DEBUG", "False") == "True"

# ESMFold Configuration
ESMFOLD_LOCAL = os.getenv("ESMFOLD_LOCAL", "True") == "True"
ESMFOLD_MODEL_NAME = os.getenv("ESMFOLD_MODEL_NAME", "facebook/esmfold_v1")
# Remote API fallback (used when ESMFOLD_LOCAL=False)
ESMFOLD_API_URL = os.getenv("ESMFOLD_API_URL", "https://api.esmatlas.com/foldSequence/v1/pdb/")
ESMFOLD_TIMEOUT = int(os.getenv("ESMFOLD_TIMEOUT", 300))
ESMFOLD_RETRIES = int(os.getenv("ESMFOLD_RETRIES", 3))

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

# Structure result cache (Redis)
CACHE_TTL = int(os.getenv("CACHE_TTL", 86400))           # seconds; default 24 h
REDIS_CACHE_PREFIX = os.getenv("REDIS_CACHE_PREFIX", "propredict:structure:")

# Tool Feature Flags (set to True only after installing the tool)
ROSETTA_ENABLED = os.getenv("ROSETTA_ENABLED", "False") == "True"
GROMACS_ENABLED = os.getenv("GROMACS_ENABLED", "False") == "True"
GROMACS_BIN = os.getenv("GROMACS_BIN", "gmx")
OPENMM_ENABLED = os.getenv("OPENMM_ENABLED", "False") == "True"         # conda install -c conda-forge openmm
ROSETTAFOLD_ENABLED = os.getenv("ROSETTAFOLD_ENABLED", "False") == "True"  # see github.com/baker-lab/RoseTTAFold2
OPENFOLD_ENABLED = os.getenv("OPENFOLD_ENABLED", "False") == "True"        # see github.com/aqlaboratory/openfold

# Boltz-2: high-accuracy GPU backend (install from source: pip install git+https://github.com/jwohlwend/boltz)
BOLTZ_ENABLED = os.getenv("BOLTZ_ENABLED", "False") == "True"
BOLTZ_SAMPLES = int(os.getenv("BOLTZ_SAMPLES", 1))
BOLTZ_STEPS = int(os.getenv("BOLTZ_STEPS", 200))
# False = fully local (msa: empty); True = use ColabFold MSA server (better accuracy, requires network)
BOLTZ_USE_MSA = os.getenv("BOLTZ_USE_MSA", "False") == "True"

# MD Simulation parameters
MD_PRODUCTION_NS = float(os.getenv("MD_PRODUCTION_NS", 0.1))   # production run length (nanoseconds)

# Stage F — Membrane & Ligand environments
# insane.py: download from http://cgmartini.nl — set path here or place on PATH
INSANE_PATH = os.getenv("INSANE_PATH", "")
# GROMACS force field for membrane runs (must be installed in GROMACS data dir)
MEMBRANE_FF = os.getenv("MEMBRANE_FF", "charmm36-feb2026_cgenff-5.0")
# GNINA docking binary: https://github.com/gnina/gnina/releases
GNINA_BIN = os.getenv("GNINA_BIN", "gnina")

# Agentic refinement loop (Stage D — requires ANTHROPIC_API_KEY)
AGENT_ENABLED = os.getenv("AGENT_ENABLED", "False") == "True"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-opus-4-6")
AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", 5))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
