"""Results-artifact path helpers for the benchmark harnesses.

Both `benchmark_score_only_epistasis.py` and `benchmark_proteinmpnn_checkpoints.py` used to
write mode `"w"` to a fixed default path, so a plain re-run at default flags silently
destroyed a documented result (ISSUES I-1/I-2/I-3: a 300-pair epistasis run was replaced by a
150-pair one, and a three-seed checkpoint run by a single-seed one, both unnoticed).

Two guards, used together:
  * `stamped_path` puts the run parameters in the filename, so a run at different parameters
    cannot land on an existing artifact's name and the provenance is self-describing.
  * `guard_overwrite` refuses to clobber an existing file without an explicit `--force`.
"""

import os


def stamped_path(directory: str, stem: str, ext: str, **params) -> str:
    """`<directory>/<stem>_<k><v>..._.<ext>` — e.g. stamped_path(d, "eps", "jsonl", n=300, k=8)
    -> "<d>/eps_n300_k8.jsonl". Params are stamped in the order given."""
    tags = "".join(f"_{k}{v}" for k, v in params.items())
    return os.path.join(directory, f"{stem}{tags}.{ext}")


def guard_overwrite(path: str, force: bool) -> str:
    """Return `path`, or exit non-zero if it already exists and `force` is False."""
    if os.path.exists(path) and not force:
        raise SystemExit(
            f"refusing to overwrite existing results artifact:\n  {path}\n"
            "Re-run with --force to replace it, or pass --out to write elsewhere. "
            "(Committed artifacts are cited in research_plan/ and Process/ write-ups.)"
        )
    return path
