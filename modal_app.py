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
        # Ligand GAFF2 parameterization (real coverage for parameterize_ligand_acpype).
        # conda-forge ships a working AmberTools 21.11 + OpenBabel; the pip acpype wheel
        # bundles fragile prebuilt binaries, so prefer conda here.
        "ambertools",
        "openbabel",
        "acpype",
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

    # ProteinMPNN clone (MIT, ~26MB incl. weights) for the structural mutation scorer
    # (orchestrator/mutation_scan.py, exposed to the agent as scan_mutations). torch is
    # already in this image, so the subprocess scorer runs in-place. No .env is copied
    # into the image, so this image env var is the authoritative PROTEINMPNN_PATH here.
    .run_commands("git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git /opt/ProteinMPNN")
    .env({"PROTEINMPNN_PATH": "/opt/ProteinMPNN"})

    #.run_commands("boltz download", timeout=1200)
    # Ship local source packages into the image
    .add_local_dir("orchestrator", remote_path="/root/orchestrator")
    .add_local_dir("models", remote_path="/root/models")
    .add_local_dir("api", remote_path="/root/api")
    .add_local_file("config.py", remote_path="/root/config.py")
    .add_local_file("modal_app.py", remote_path="/root/modal_app.py")
)

# ---------------------------------------------------------------------------
# Dedicated image for the GNINA real-binary test (ROADMAP follow-up).
# GNINA has no conda package; its release binary is CUDA-linked, and the official
# gnina Docker image is Py3.6/Ubuntu18.04 (incompatible with our 3.11 stack). So we
# layer the prebuilt binary onto an NVIDIA CUDA runtime base with just RDKit +
# OpenBabel — dock_gnina() only needs the binary on PATH and smiles_to_3d() needs
# RDKit. orchestrator/__init__.py is empty, so importing orchestrator.ligands here
# does NOT pull in torch/boltz/openff.
GNINA_RELEASE = "https://github.com/gnina/gnina/releases/download/v1.3/gnina"
gnina_image = (
    Image.from_registry(
        "nvidia/cuda:12.2.2-runtime-ubuntu22.04", add_python="3.11"
    )
    .apt_install("wget", "openbabel", "libopenbabel-dev", "libgomp1")
    .run_commands(
        f"wget -q {GNINA_RELEASE} -O /usr/local/bin/gnina",
        "chmod +x /usr/local/bin/gnina",
        # Fail the build loudly if the binary can't even print its version
        # (missing runtime lib) rather than discovering it at test time.
        "/usr/local/bin/gnina --version || true",
    )
    .pip_install("rdkit==2024.3.5", "numpy<2")
    .add_local_dir("orchestrator", remote_path="/root/orchestrator")
    .add_local_file("config.py", remote_path="/root/config.py")
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
def test_membrane_modal() -> dict:
    """
    Real-binary smoke test for the OpenMM membrane builder (CPU — no GPU needed).

    Mirrors the real setup in simulation.py: PDBFixer -> ForceField(charmm36 lipids)
    -> Modeller.addHydrogens -> embed_in_membrane_openmm (Modeller.addMembrane). insane.py
    is NOT in this image, so embed_in_membrane_gromacs is covered by the mocked
    tests/test_membrane.py only. The mocked local counterpart is tests/test_membrane.py.

    Run with:
        modal run modal_app.py::test_membrane_modal
    """
    import io

    from orchestrator.membrane import embed_in_membrane_openmm

    # A small soluble peptide (villin headpiece fragment) — enough to embed in a POPC
    # patch. Membrane type POPC exercises the CHARMM36m lipid path.
    protein_pdb = (
        "ATOM      1  N   MET A   1      -8.901   4.127  -0.555  1.00  0.00           N\n"
        "ATOM      2  CA  MET A   1      -8.608   3.135  -1.618  1.00  0.00           C\n"
        "ATOM      3  C   MET A   1      -7.117   2.964  -1.897  1.00  0.00           C\n"
        "ATOM      4  O   MET A   1      -6.634   1.849  -1.758  1.00  0.00           O\n"
        "ATOM      5  N   LYS A   2      -6.379   4.031  -2.228  1.00  0.00           N\n"
        "ATOM      6  CA  LYS A   2      -4.923   4.002  -2.452  1.00  0.00           C\n"
        "ATOM      7  C   LYS A   2      -4.136   3.383  -1.301  1.00  0.00           C\n"
        "ATOM      8  O   LYS A   2      -3.391   2.416  -1.517  1.00  0.00           O\n"
        "ATOM      9  N   THR A   3      -4.354   3.961  -0.129  1.00  0.00           N\n"
        "ATOM     10  CA  THR A   3      -3.646   3.474   1.049  1.00  0.00           C\n"
        "ATOM     11  C   THR A   3      -4.297   2.229   1.652  1.00  0.00           C\n"
        "ATOM     12  O   THR A   3      -3.605   1.318   2.106  1.00  0.00           O\n"
    )
    results: dict = {"insane_in_image": False}

    from openmm.app import PDBFile, ForceField, Modeller

    # 1. PDBFixer -> clean structure
    try:
        from pdbfixer import PDBFixer
        fixer = PDBFixer(pdbfile=io.StringIO(protein_pdb))
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)
        fixed_io = io.StringIO()
        PDBFile.writeFile(fixer.topology, fixer.positions, fixed_io)
        fixed_io.seek(0)
        pdb = PDBFile(fixed_io)
        results["pdbfixer_ok"] = True
    except Exception as e:  # noqa: BLE001
        results["pdbfixer_error"] = repr(e)
        pdb = PDBFile(io.StringIO(protein_pdb))

    # 2. CHARMM36m force field with lipids
    try:
        ff = ForceField("charmm36.xml", "charmm36/water.xml", "charmm36/lipids.xml")
        results["charmm36_ff_ok"] = True
    except Exception as e:  # noqa: BLE001
        results["charmm36_ff_error"] = repr(e)
        return results

    # 3. Modeller + hydrogens + real addMembrane
    try:
        modeller = Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(ff, pH=7.0)
        n_before = modeller.topology.getNumAtoms()
        modeller = embed_in_membrane_openmm(modeller, ff, {"type": "POPC"})
        n_after = modeller.topology.getNumAtoms()
        results["addmembrane_ok"] = n_after > n_before
        results["n_atoms_before"] = n_before
        results["n_atoms_after"] = n_after
    except Exception as e:  # noqa: BLE001
        results["addmembrane_error"] = repr(e)

    print(results)
    return results


@app.function(timeout=600)
def test_ligands_modal() -> dict:
    """
    Real-binary smoke test for the Stage-F ligand pipeline (CPU — no GPU needed).

    Exercises the actual RDKit -> Vina -> OpenFF and ACPYPE(GAFF2) paths in the image.
    GNINA is NOT installed (CUDA-compiled binary — deferred; see ROADMAP), so
    prepare_ligands here genuinely walks the GNINA-absent -> Vina fallback. ACPYPE and
    AmberTools ARE in the image, so parameterize_ligand_acpype is covered for real. The
    mocked local counterpart is tests/test_ligands.py.

    Run with:
        modal run modal_app.py::test_ligands_modal
    """
    import os
    import tempfile

    from orchestrator.ligands import (
        smiles_to_3d,
        dock_vina,
        parameterize_ligand_openff,
        parameterize_ligand_acpype,
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
    results: dict = {"gnina_in_image": False, "acpype_in_image": True}

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

        # 3b. Real ACPYPE GAFF2 parameterization (AmberTools + acpype are in the image)
        if sdf:
            try:
                acp = parameterize_ligand_acpype(sdf, "ETH", td)
                results["acpype_ok"] = bool(acp.get("itp") and os.path.isfile(acp["itp"]))
            except Exception as e:  # noqa: BLE001
                results["acpype_error"] = repr(e)

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


@app.function(image=gnina_image, gpu="T4", timeout=900)
def test_gnina_modal() -> dict:
    """
    Real-binary smoke test for dock_gnina() — the one ligand path still mocked-only
    in tests/test_ligands.py (GNINA is absent from the main image; see ROADMAP
    "Real-binary GNINA coverage on Modal"). Runs the actual RDKit ETKDG -> gnina
    docking pipeline on a real GPU, covering both the binding-site box branch and the
    blind --autobox_ligand branch.

    Run with:
        modal run modal_app.py::test_gnina_modal
    """
    import os
    import shutil
    import tempfile

    from orchestrator.ligands import smiles_to_3d, dock_gnina

    # Same minimal multi-residue receptor as test_ligands_modal (real CA coords).
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
    results: dict = {"gnina_on_path": shutil.which("gnina") is not None}

    with tempfile.TemporaryDirectory() as td:
        rec_path = os.path.join(td, "receptor.pdb")
        with open(rec_path, "w") as fh:
            fh.write(receptor_pdb)

        # 1. RDKit ETKDG conformer
        try:
            sdf = smiles_to_3d(ethanol, "ETH", td)
            results["smiles_to_3d_ok"] = os.path.isfile(sdf)
        except Exception as e:  # noqa: BLE001
            results["smiles_to_3d_error"] = repr(e)
            sdf = None

        # 2. Blind docking (--autobox_ligand branch), own out_dir → its own docked.sdf
        if sdf:
            blind_dir = os.path.join(td, "blind")
            os.makedirs(blind_dir, exist_ok=True)
            try:
                blind = dock_gnina(sdf, rec_path, None, blind_dir)
                results["dock_gnina_blind_ok"] = os.path.isfile(blind)
            except Exception as e:  # noqa: BLE001
                results["dock_gnina_blind_error"] = repr(e)

        # 3. Binding-site docking (--center_x/--size_x branch, CA centroid of res 1-3)
        if sdf:
            site_dir = os.path.join(td, "site")
            os.makedirs(site_dir, exist_ok=True)
            try:
                site = dock_gnina(sdf, rec_path, [1, 2, 3], site_dir)
                results["dock_gnina_site_ok"] = os.path.isfile(site)
            except Exception as e:  # noqa: BLE001
                results["dock_gnina_site_error"] = repr(e)

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
        # Both stay None here — this smoke test folds a bare sequence with no ligand, so
        # Boltz-2 never runs the affinity head. Confirming the affinity keys parse needs a
        # ligand-bearing run (see test_boltz_affinity_gpu).
        "affinity_score": result.affinity_score,
        "affinity_probability": result.affinity_probability,
        "pdb_lines": result.structure_pdb.count("\n"),
    }
    print(summary)
    return summary


@app.function(
    timeout=1800,
    gpu="A10G",
)
def test_boltz_affinity_gpu(
    sequence: str = "MKTAYIAKQRQISFVKSHFSRQDILDLWQYVQG",
    smiles: str = "CCO",
) -> dict:
    """
    Ligand-bearing GPU test that closes the verification gap on the affinity fix
    (Process/boltz-affinity-key-fix.md).

    test_boltz_gpu folds a bare sequence, so Boltz-2 never runs the affinity head and
    affinity_score stays None there — it cannot catch a wrong key. The local unit tests
    are mocked, and a mock encoding the WRONG key is precisely what hid the original bug
    for months. So this function deliberately does not trust our own parser: it first
    reads the raw Boltz-2 output and reports the actual filenames and JSON keys as
    ground truth, then checks that call_boltz's parse agrees.

    Two GPU runs (the ground-truth one uses fewer sampling steps, since it only needs
    the file layout, not a good structure).

    Run with:
        modal run modal_app.py::test_boltz_affinity_gpu
        modal run modal_app.py::test_boltz_affinity_gpu --smiles "CC(=O)Oc1ccccc1C(=O)O"
    """
    import glob
    import json
    import os
    import subprocess
    import tempfile

    import yaml

    os.environ["BOLTZ_ENABLED"] = "True"
    os.environ["BOLTZ_DIFFUSION_SAMPLES"] = "1"
    os.environ["BOLTZ_SAMPLING_STEPS"] = "200"
    os.environ["BOLTZ_USE_MSA"] = "False"

    results: dict = {"smiles": smiles, "sequence_length": len(sequence)}

    # ------------------------------------------------------------------
    # Part 1 — ground truth: what does Boltz-2 actually write?
    # Runs the CLI directly (not via call_boltz) so nothing our code believes about
    # filenames or key names can influence the answer.
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        boltz_input = {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": sequence, "msa": "empty"}},
                {"ligand": {"id": "B", "smiles": smiles}},
            ],
            "properties": [{"affinity": {"binder": "B"}}],
        }
        yaml_path = os.path.join(tmpdir, "input.yaml")
        out_dir = os.path.join(tmpdir, "output")
        os.makedirs(out_dir)
        with open(yaml_path, "w") as fh:
            yaml.dump(boltz_input, fh, default_flow_style=False)

        proc = subprocess.run(
            [
                "boltz", "predict", yaml_path,
                "--out_dir", out_dir,
                "--diffusion_samples", "1",
                "--sampling_steps", "50",   # layout only — no need for a good structure
                "--seed", "0",
            ],
            capture_output=True, text=True, timeout=1500,
        )
        results["groundtruth_returncode"] = proc.returncode
        if proc.returncode != 0:
            results["groundtruth_stderr"] = proc.stderr[-2000:]
        else:
            all_json = sorted(glob.glob(os.path.join(out_dir, "**", "*.json"), recursive=True))
            results["all_json_basenames"] = [os.path.basename(p) for p in all_json]

            aff_files = [p for p in all_json if "affinity" in os.path.basename(p).lower()]
            results["affinity_file_basenames"] = [os.path.basename(p) for p in aff_files]
            # THE question this whole function exists to answer.
            results["affinity_json_keys"] = {
                os.path.basename(p): sorted(json.load(open(p)).keys()) for p in aff_files
            }
            results["keys_match_our_parser"] = any(
                "affinity_pred_value" in keys
                for keys in results["affinity_json_keys"].values()
            )

    # ------------------------------------------------------------------
    # Part 2 — does call_boltz's parse actually pick those values up?
    # ------------------------------------------------------------------
    from orchestrator.backends.boltz import call_boltz

    ctx = {"ligands": [{"name": "ligand", "smiles": smiles}]}
    try:
        pred = call_boltz(sequence, context=ctx, seed=0)
        results["call_boltz_mean_plddt"] = round(pred.mean_plddt, 2)
        results["affinity_score"] = pred.affinity_score
        results["affinity_probability"] = pred.affinity_probability
        # The assertions that matter: both must be populated, or the fix is not working.
        results["affinity_score_populated"] = pred.affinity_score is not None
        results["affinity_probability_populated"] = pred.affinity_probability is not None
    except Exception as e:  # noqa: BLE001
        results["call_boltz_error"] = repr(e)

    results["PASS"] = bool(
        results.get("keys_match_our_parser")
        and results.get("affinity_score_populated")
        and results.get("affinity_probability_populated")
    )

    print(json.dumps(results, indent=2, default=str))
    return results
