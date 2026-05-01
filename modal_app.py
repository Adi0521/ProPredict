import modal
from modal import App, Image, Secret

# x86_64 Linux — ambertools, openff-*, and all other conda packages build cleanly here
image = (
    Image.micromamba()
    .micromamba_install(
        "python=3.11",
        "numpy=1.26",
        "scipy=1.13",
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
    .apt_install("gromacs", "curl", "git", "build-essential")
    .pip_install_from_requirements("requirements.txt")

    .pip_install(
    "torch==2.6.0",
    "torchvision==0.21.0",
    "torchaudio==2.6.0",
    extra_index_url="https://download.pytorch.org/whl/cu126",
    )

    .pip_install("cuequivariance-ops-torch-cu12")

    .pip_install("cuequivariance-torch")

    # Boltz-2 from source — not yet on PyPI as boltz-2, install from GitHub
    .pip_install("git+https://github.com/jwohlwend/boltz.git")

    #.run_commands("boltz download", timeout=1200)
    # Ship local source packages into the image
    .add_local_dir("orchestrator", remote_path="/root/orchestrator")
    .add_local_dir("models", remote_path="/root/models")
    .add_local_dir("api", remote_path="/root/api")
    .add_local_file("config.py", remote_path="/root/config.py")
    .add_local_file("modal_app.py", remote_path="/root/modal_app.py")
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
    gpu="A10G",
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


@app.function(
    timeout=600,
    gpu="A10G",
)
def test_boltz_gpu(sequence: str = "MKTAYIAKQRQISFVKSHFSRQDILDLWQYVQG") -> dict:
    """
    Standalone GPU smoke-test for Boltz-2. Does not require Postgres or Redis.

    Run with:
        modal run modal_app.py::test_boltz_gpu
        modal run modal_app.py::test_boltz_gpu --sequence MKTAYIAK
    """
    import os
    os.environ["BOLTZ_ENABLED"] = "True"
    os.environ["BOLTZ_SAMPLES"] = "1"
    os.environ["BOLTZ_STEPS"] = "200"
    os.environ["BOLTZ_USE_MSA"] = "False"

    # Import after env vars are set so config.py picks them up
    from orchestrator.tasks import call_boltz

    print(f"Running Boltz-2 on sequence of length {len(sequence)}...")
    result = call_boltz(sequence, seed=0)

    summary = {
        "model_name": result.model_name,
        "sequence_length": len(sequence),
        "mean_plddt": round(result.mean_plddt, 2),
        "num_residues_scored": len(result.plddt_scores),
        "plddt_scores_first10": [round(s, 1) for s in result.plddt_scores[:10]],
        "affinity_score": result.affinity_score,
        "pdb_lines": result.structure_pdb.count("\n"),
    }
    print(summary)
    return summary
