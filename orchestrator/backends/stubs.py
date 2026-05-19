import logging

from models.schemas import StructurePrediction

logger = logging.getLogger(__name__)


def call_rosettafold2(sequence: str, seed: int = 0) -> StructurePrediction:
    """
    Run RoseTTAFold2 locally and return a StructurePrediction.

    DEPRECATED: Boltz-2 supersedes RoseTTAFold2 in accuracy and ease of install.
    Prefer BOLTZ_ENABLED=True. This stub is kept for future completion if needed.

    Installation:
        git clone https://github.com/baker-lab/RoseTTAFold2
        cd RoseTTAFold2 && conda env create -f environment.yaml
        conda activate RF2 && pip install -e .
    Then set ROSETTAFOLD_ENABLED=True.

    Raises RuntimeError if the package is not installed.
    Raises NotImplementedError — complete the body once RF2 is installed.
    """
    try:
        import rf2aa  # type: ignore  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "RoseTTAFold2 (rf2aa) is not installed. "
            "See: https://github.com/baker-lab/RoseTTAFold2"
        )

    # TODO: implement using the RF2AA runner once the conda env is active.
    # Example sketch (API may differ by version):
    #   from rf2aa.run_inference import run_inference
    #   pdb_string, plddt = run_inference(sequence)
    raise NotImplementedError(
        "RoseTTAFold2 stub — fill in using the rf2aa.run_inference API."
    )


def call_openfold(sequence: str, seed: int = 0) -> StructurePrediction:
    """
    Run OpenFold locally and return a StructurePrediction.

    DEPRECATED: Boltz-2 supersedes OpenFold in accuracy and ease of install.
    Prefer BOLTZ_ENABLED=True. This stub is kept for future completion if needed.

    Installation:
        pip install 'openfold @ git+https://github.com/aqlaboratory/openfold'
        # Requires CUDA for full performance; CPU mode works for short sequences.
    Then set OPENFOLD_ENABLED=True.

    Raises RuntimeError if the package is not installed.
    Raises NotImplementedError — complete the body once OpenFold is installed.
    """
    try:
        import openfold  # type: ignore  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "OpenFold is not installed. "
            "See: https://github.com/aqlaboratory/openfold"
        )

    # TODO: implement using the OpenFold data pipeline + model runner.
    # Example sketch:
    #   from openfold.data import data_pipeline, feature_pipeline
    #   from openfold.model import model as of_model
    #   ...
    raise NotImplementedError(
        "OpenFold stub — fill in using the openfold.data and openfold.model APIs."
    )
