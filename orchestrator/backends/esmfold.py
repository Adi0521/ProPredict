import io
import logging
import time
from typing import List

import requests

from config import (
    ESMFOLD_LOCAL,
    ESMFOLD_MODEL_NAME,
    ESMFOLD_API_URL,
    ESMFOLD_TIMEOUT,
    ESMFOLD_RETRIES,
)
from models.schemas import StructurePrediction

logger = logging.getLogger(__name__)

_esmfold_model = None
_esmfold_tokenizer = None


def _get_esmfold_local():
    """Lazy-load the ESMFold model and tokenizer (heavy — only once per worker)."""
    global _esmfold_model, _esmfold_tokenizer
    if _esmfold_model is None:
        from transformers import EsmForProteinFolding, AutoTokenizer  # type: ignore
        import torch
        logger.info(f"Loading ESMFold model: {ESMFOLD_MODEL_NAME}")
        _esmfold_tokenizer = AutoTokenizer.from_pretrained(ESMFOLD_MODEL_NAME)
        _esmfold_model = EsmForProteinFolding.from_pretrained(ESMFOLD_MODEL_NAME)
        _esmfold_model.eval()
        if torch.cuda.is_available():
            _esmfold_model = _esmfold_model.cuda()
        elif torch.backends.mps.is_available():
            _esmfold_model = _esmfold_model.to("mps")
        logger.info("ESMFold model loaded.")
    return _esmfold_model, _esmfold_tokenizer


def _parse_plddt_from_pdb(pdb_string: str) -> List[float]:
    """
    Extract per-residue pLDDT from ESMFold PDB output.
    ESMFold stores pLDDT in the B-factor column of CA atoms as a 0–1 fraction.
    Multiply by 100 to normalise to the standard 0–100 scale used by thresholds.
    """
    scores: List[float] = []
    for line in pdb_string.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                scores.append(float(line[60:66].strip()) * 100)
            except ValueError:
                pass
    return scores


def _call_esmfold_remote(sequence: str, seed: int = 0) -> StructurePrediction:
    """
    Call the ESMFold REST API with retries (fallback when ESMFOLD_LOCAL=False).

    ESMFold endpoint accepts a raw amino acid sequence as the POST body
    (application/x-www-form-urlencoded) and returns a PDB string.
    pLDDT scores are embedded in the B-factor column of CA atoms.
    """
    for attempt in range(ESMFOLD_RETRIES):
        try:
            logger.info(f"ESMFold API call attempt {attempt + 1}/{ESMFOLD_RETRIES}")
            response = requests.post(
                ESMFOLD_API_URL,
                data=sequence,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=ESMFOLD_TIMEOUT,
            )
            response.raise_for_status()

            pdb_string = response.text
            plddt_scores = _parse_plddt_from_pdb(pdb_string)

            if not plddt_scores:
                raise ValueError("No CA atoms found in ESMFold PDB output — response may be malformed")

            mean_plddt = sum(plddt_scores) / len(plddt_scores)
            logger.info(f"ESMFold remote call succeeded. Mean pLDDT: {mean_plddt:.2f}")

            return StructurePrediction(
                structure_pdb=pdb_string,
                plddt_scores=plddt_scores,
                mean_plddt=mean_plddt,
                seed=seed,
                model_name="esmfold",
            )

        except requests.exceptions.RequestException as e:
            logger.warning(f"ESMFold API call failed (attempt {attempt + 1}): {e}")
            if attempt < ESMFOLD_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error(f"ESMFold API failed after {ESMFOLD_RETRIES} attempts")
                raise


def call_esmfold_local(sequence: str, seed: int = 0) -> StructurePrediction:
    """
    Run ESMFold locally via HuggingFace Transformers.

    ESMFold is deterministic — the seed parameter does not affect output. When
    ENSEMBLE_NUM_SEEDS > 1, the ensemble loop will call this N times but receive
    identical structures. This is a known limitation of ESMFold, not a ProPredict bug.
    """
    import torch
    model, tokenizer = _get_esmfold_local()

    logger.info(f"ESMFold local inference on sequence of length {len(sequence)}")
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)

    pdb_string = model.output_to_pdb(output)[0]
    plddt_scores = _parse_plddt_from_pdb(pdb_string)

    if not plddt_scores:
        raise ValueError("No CA atoms found in ESMFold local output — model output may be malformed")

    mean_plddt = sum(plddt_scores) / len(plddt_scores)
    logger.info(f"ESMFold local inference succeeded. Mean pLDDT: {mean_plddt:.2f}")

    return StructurePrediction(
        structure_pdb=pdb_string,
        plddt_scores=plddt_scores,
        mean_plddt=mean_plddt,
        seed=seed,
        model_name="esmfold_local",
    )


def call_esmfold_api(sequence: str, seed: int = 0) -> StructurePrediction:
    """Dispatch to local model or remote API based on ESMFOLD_LOCAL flag."""
    if ESMFOLD_LOCAL:
        return call_esmfold_local(sequence, seed)
    return _call_esmfold_remote(sequence, seed)
