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
#     AGENT_API_KEY=sk-ant-... \
#     ROSETTA_ENABLED=False \
#     GROMACS_ENABLED=True \
#     OPENMM_ENABLED=True \
#     BOLTZ_ENABLED=True \
#     BOLTZ_DIFFUSION_SAMPLES=1 \
#     BOLTZ_SAMPLING_STEPS=200 \
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
    from orchestrator.progress import PROGRESS_DICT_NAME

    run_id = request_data.get("run_id")

    # Modal has no Celery result backend, so relay per-stage progress through a
    # named Modal Dict keyed by run_id; the API status endpoint reads it back.
    progress = modal.Dict.from_name(PROGRESS_DICT_NAME, create_if_missing=True)

    def progress_cb(percent: int, stage: str) -> None:
        if run_id:
            try:
                progress[run_id] = {"progress_percent": percent, "stage": stage}
            except Exception:
                pass

    try:
        return _run_prediction_core(request_data, progress_cb=progress_cb)
    finally:
        # The terminal state is reported by the FunctionCall result itself, so the
        # interim progress entry is no longer needed — drop it to bound Dict growth.
        if run_id:
            try:
                del progress[run_id]
            except Exception:
                pass


@app.function(secrets=secrets)
@modal.asgi_app()
def fastapi_endpoint():
    """Serves the FastAPI app. MODAL_ENABLED must be set in propredict-secrets."""
    from api.main import app as fastapi_app
    return fastapi_app


@app.function(timeout=600)
def test_ligands_modal() -> dict:
    """
    Real-binary smoke test for the Stage-F ligand pipeline (CPU — no GPU needed).

    Exercises the actual RDKit -> Vina -> OpenFF path in the image. GNINA and ACPYPE
    are NOT installed in this image, so prepare_ligands here genuinely walks the
    GNINA-absent -> Vina fallback and the use_openff=True branch. The mocked local
    counterpart is tests/test_ligands.py.

    Run with:
        modal run modal_app.py::test_ligands_modal
    """
    import os
    import tempfile

    from orchestrator.ligands import (
        smiles_to_3d,
        dock_vina,
        parameterize_ligand_openff,
        prepare_ligands,
    )

    # A minimal but valid multi-residue receptor (real CA coords) — enough for Vina's
    # blind-docking box and receptor prep. Ethanol is the ligand (clean neutral
    # molecule OpenFF handles cleanly).
    receptor_pdb = (
        "ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1      11.639   6.071  -5.147  1.00  0.00           C\n"
        "ATOM      3  C   ALA A   1      13.140   6.341  -5.184  1.00  0.00           C\n"
        "ATOM      4  O   ALA A   1      13.629   7.147  -5.980  1.00  0.00           O\n"
        "ATOM      5  N   GLY A   2      13.865   5.677  -4.283  1.00  0.00           N\n"
        "ATOM      6  CA  GLY A   2      15.311   5.846  -4.215  1.00  0.00           C\n"
        "ATOM      7  C   GLY A   2      15.998   4.630  -3.617  1.00  0.00           C\n"
        "ATOM      8  O   GLY A   2      15.379   3.815  -2.934  1.00  0.00           O\n"
        "ATOM      9  N   SER A   3      17.296   4.502  -3.881  1.00  0.00           N\n"
        "ATOM     10  CA  SER A   3      18.079   3.375  -3.383  1.00  0.00           C\n"
        "ATOM     11  C   SER A   3      19.529   3.759  -3.103  1.00  0.00           C\n"
        "ATOM     12  O   SER A   3      20.207   4.353  -3.944  1.00  0.00           O\n"
    )
    ethanol = "CCO"
    results: dict = {"gnina_in_image": False, "acpype_in_image": False}

    with tempfile.TemporaryDirectory() as td:
        # 1. RDKit ETKDG conformer
        try:
            sdf = smiles_to_3d(ethanol, "ETH", td)
            results["smiles_to_3d_ok"] = os.path.isfile(sdf)
        except Exception as e:  # noqa: BLE001
            results["smiles_to_3d_error"] = repr(e)
            sdf = None

        rec_path = os.path.join(td, "receptor_input.pdb")
        with open(rec_path, "w") as fh:
            fh.write(receptor_pdb)

        # 2. Real Vina blind docking (meeko is in the image)
        if sdf:
            try:
                docked = dock_vina(sdf, rec_path, None, td)
                results["dock_vina_ok"] = os.path.isfile(docked)
            except Exception as e:  # noqa: BLE001
                results["dock_vina_error"] = repr(e)

        # 3. Real OpenFF SMIRNOFF parameterization
        if sdf:
            try:
                params = parameterize_ligand_openff(sdf, "ETH", td)
                results["openff_ok"] = os.path.isfile(params.get("xml", ""))
            except Exception as e:  # noqa: BLE001
                results["openff_error"] = repr(e)

        # 4. Full pipeline: GNINA absent -> Vina fallback, use_openff=True
        entries = prepare_ligands(
            [{"name": "ETH", "smiles": ethanol, "binding_site": None}],
            receptor_pdb,
            td,
            use_openff=True,
        )
        results["prepare_ligands_n"] = len(entries)
        if entries:
            results["prepare_ligands_parameterizer"] = entries[0]["parameterizer"]
            results["prepare_ligands_docked_sdf_set"] = entries[0]["docked_sdf"] is not None

    print(results)
    return results


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
    os.environ["BOLTZ_DIFFUSION_SAMPLES"] = "1"
    os.environ["BOLTZ_SAMPLING_STEPS"] = "200"
    os.environ["BOLTZ_USE_MSA"] = "False"

    # Import after env vars are set so config.py picks them up
    from orchestrator.backends.boltz import call_boltz

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
