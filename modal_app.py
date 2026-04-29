import modal
from modal import App, Image, Secret

# x86_64 Linux — ambertools, openff-*, and all other conda packages build cleanly here
image = (
    Image.micromamba()
    .micromamba_install(
        "python=3.11",
        "openmm",
        "pdbfixer",
        "openmmforcefields",
        "rdkit",
        "openff-toolkit",
        "openff-forcefields",
        "openff-interchange",
        "vina",
        channels=["conda-forge"],
    )
    .apt_install("gromacs", "curl")
    .pip_install_from_requirements("requirements.txt")
)

app = App("propredict", image=image)

# Create this secret in the Modal dashboard:
#   modal secret create propredict-secrets \
#     DATABASE_URL=postgresql://user:pass@host/db \
#     ANTHROPIC_API_KEY=sk-ant-... \
#     ESMFOLD_API_URL=https://api.esmatlas.com/foldSequence/v1/pdb/ \
#     ROSETTA_ENABLED=False \
#     GROMACS_ENABLED=True \
#     OPENMM_ENABLED=True \
#     MODAL_ENABLED=True \
#     LOG_LEVEL=INFO
secrets = [Secret.from_name("propredict-secrets")]


@app.function(timeout=1800, secrets=secrets)
def run_prediction(request_data: dict) -> dict:
    """Worker function — replaces the Celery worker in production."""
    from orchestrator.tasks import _run_prediction_core
    return _run_prediction_core(request_data)


@app.function(secrets=secrets)
@modal.asgi_app()
def fastapi_endpoint():
    """Serves the FastAPI app. MODAL_ENABLED must be set in propredict-secrets."""
    from api.main import app as fastapi_app
    return fastapi_app
