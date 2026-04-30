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
    .apt_install("gromacs", "curl", "git")
    .pip_install_from_requirements("requirements.txt")
    # Boltz-2 from source — not yet on PyPI as boltz-2, install from GitHub
    .pip_install("git+https://github.com/jwohlwend/boltz.git")
)

app = App("propredict", image=image)

# Create this secret in the Modal dashboard:
#   modal secret create propredict-secrets \
#     DATABASE_URL=postgresql://user:pass@host/db \
#     ANTHROPIC_API_KEY=sk-ant-... \
#     ROSETTA_ENABLED=False \
#     GROMACS_ENABLED=True \
#     OPENMM_ENABLED=True \
#     BOLTZ_ENABLED=True \
#     BOLTZ_SAMPLES=1 \
#     BOLTZ_STEPS=200 \
#     BOLTZ_USE_MSA=False \
#     ESMFOLD_LOCAL=True \
#     MODAL_ENABLED=True \
#     LOG_LEVEL=INFO
secrets = [Secret.from_name("propredict-secrets")]


@app.function(
    timeout=1800,
    secrets=secrets,
    gpu=modal.gpu.A10G(),
)
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
