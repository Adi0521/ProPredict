"""Shared ProteinGym cache preparation for the benchmark harnesses.

Third-party data is auto-downloaded once into the gitignored `benchmarks/proteingym_cache/`
rather than committed (the same convention as `build_hiv_pr_dataset.py`), so a fresh checkout
can reproduce any of these runs without a manual data step.

Lifted out of `benchmark_proteinmpnn_checkpoints.py` so the epistasis harness can share one
implementation instead of hardcoding the cache path and crashing on a fresh checkout. That
module keeps `_ensure_parquet` as a thin alias.
"""

import hashlib
import os
import subprocess

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proteingym_cache")
PARQUET_S3 = "s3://proteingym/DMS_substitutions.parquet"
PARQUET_NAME = "DMS_substitutions.parquet"


def ensure_parquet() -> str:
    """Download the combined ProteinGym substitutions parquet once (~89 MB), then cache.

    Needs the AWS CLI on PATH; the bucket is public, hence `--no-sign-request`.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, PARQUET_NAME)
    if not os.path.isfile(path):
        print(f"Downloading ProteinGym parquet (~89 MB) -> {path}")
        subprocess.run(["aws", "s3", "cp", "--no-sign-request", PARQUET_S3, path], check=True)
    return path


def file_digest(path: str, chunk: int = 1 << 20) -> str:
    """sha256 of a file, for recording dataset provenance in a results artifact."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
