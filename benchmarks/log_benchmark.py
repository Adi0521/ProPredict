"""
Append a benchmark run to results.jsonl and optionally log to Weights & Biases.

Each JSONL entry is self-contained and paper-ready: full per-target results table,
aggregate statistics with std dev / IQR / confidence intervals, config snapshot,
git state, and environment metadata.

Usage from benchmark_modal.py:
    from benchmarks.log_benchmark import log_run
    log_run(results, config, source, notes)

Standalone (re-log an existing benchmark_results.json):
    python -m benchmarks.log_benchmark benchmark_results.json --notes "baseline run"
"""

import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RESULTS_FILE = Path(__file__).parent / "results.jsonl"


def _git_info() -> dict:
    def _run(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(
                cmd, stderr=subprocess.DEVNULL
            ).decode().strip()
        except Exception:
            return ""

    sha_short = _run(["git", "rev-parse", "--short", "HEAD"])
    sha_full = _run(["git", "rev-parse", "HEAD"])
    msg = _run(["git", "log", "-1", "--pretty=%s"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"]) != ""

    return {
        "commit_sha": sha_full or "unknown",
        "commit_short": sha_short or "unknown",
        "commit_message": msg,
        "branch": branch,
        "dirty": dirty,
    }


def _environment_info() -> dict:
    env = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_count"] = torch.cuda.device_count()
            env["cuda_version"] = torch.version.cuda or ""
    except ImportError:
        pass
    # NOTE: no `import boltz` probe here. log_run() executes on the dispatching machine
    # (a laptop), never in the GPU container, so that import always failed and silently
    # wrote nothing — every historical row lacks it. Worse, boltz.__version__ would report
    # "2.2.1" for builds that are commits apart. The build is captured properly via the
    # `backend_build` argument, reported by the workers that actually ran it.
    return env


# Matches the pinned commit in modal_app.py / requirements-gpu.txt. Deliberately duplicated
# from scripts/check_boltz_updates.py rather than imported: benchmarks/ importing from
# scripts/ would couple two unrelated entry points for one regex.
_BOLTZ_PIN_RE = re.compile(r"jwohlwend/boltz(?:\.git)?@([0-9a-f]{40})\b")


def _declared_boltz_pin() -> str | None:
    """
    Fall back to the Boltz commit *declared* in the repo when no worker reported one
    (e.g. every target failed, or a re-log of an old results file).

    Marked `declared:` in the record because it is weaker evidence than a worker report:
    it says what the source pins, not what actually ran.
    """
    root = Path(__file__).parent.parent
    for name in ("modal_app.py", "requirements-gpu.txt"):
        try:
            found = set(_BOLTZ_PIN_RE.findall((root / name).read_text()))
        except OSError:
            continue
        if len(found) == 1:
            return f"declared:{found.pop()[:12]}"
    return None


def _next_run_id() -> str:
    n = 1
    if RESULTS_FILE.exists():
        for line in RESULTS_FILE.read_text().splitlines():
            if line.strip():
                n += 1
    return f"run-{n:03d}"


def _ci95(values: list[float]) -> float:
    """95% confidence interval half-width (t-distribution for small n)."""
    n = len(values)
    if n < 2:
        return 0.0
    se = statistics.stdev(values) / math.sqrt(n)
    # t critical value approximation for 95% CI
    t_crit = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
    }.get(n, 1.96)
    return round(se * t_crit, 4)


def _percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] * (c - k) + s[c] * (k - f)


def _compute_summary(results: list[dict]) -> dict:
    good = [r for r in results if isinstance(r, dict) and "error" not in r]
    bad = len(results) - len(good)

    if not good:
        return {"total_targets": len(results), "succeeded": 0, "failed": bad}

    tms = [r["tm_score"] for r in good]
    rmsds = [r["rmsd"] for r in good]
    plddts = [r["mean_plddt"] for r in good]
    lengths = [r["length"] for r in good]

    def _stats(values: list[float], label: str) -> dict:
        return {
            f"{label}_mean": round(statistics.mean(values), 4),
            f"{label}_median": round(statistics.median(values), 4),
            f"{label}_std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            f"{label}_ci95": _ci95(values),
            f"{label}_q25": round(_percentile(values, 25), 4),
            f"{label}_q75": round(_percentile(values, 75), 4),
            f"{label}_min": round(min(values), 4),
            f"{label}_max": round(max(values), 4),
        }

    return {
        "total_targets": len(results),
        "succeeded": len(good),
        "failed": bad,
        **_stats(tms, "tm_score"),
        "tm_gte_0_5": sum(t >= 0.5 for t in tms),
        "tm_gte_0_7": sum(t >= 0.7 for t in tms),
        **_stats(rmsds, "rmsd"),
        **_stats(plddts, "plddt"),
        "length_mean": round(statistics.mean(lengths), 1),
        "length_median": round(statistics.median(lengths), 1),
        "length_min": min(lengths),
        "length_max": max(lengths),
    }


def _build_per_target_table(results: list[dict]) -> list[dict]:
    table = []
    for r in results:
        if not isinstance(r, dict):
            continue
        if "error" in r:
            table.append({
                "pdb_id": r.get("pdb_id", "?"),
                "status": "failed",
                "error": r["error"],
            })
        else:
            table.append({
                "pdb_id": r["pdb_id"],
                "name": r.get("name", r["pdb_id"]),
                "length": r["length"],
                "mean_plddt": r["mean_plddt"],
                "tm_score": r["tm_score"],
                "rmsd": r["rmsd"],
                "n_aligned": r.get("n_aligned"),
                "status": "ok",
            })
    return table


def log_run(
    results: list[dict],
    config: dict,
    source: str = "casp15",
    backend: str = "boltz-2",
    notes: str = "",
    duration_seconds: float | None = None,
    backend_build: str | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
) -> dict:
    """
    Append a paper-ready benchmark entry to results.jsonl.

    `backend_build` identifies the exact backend that produced these numbers, e.g.
    "2.2.1@b1ebfc46ecf5" (version + resolved git commit), as reported by the workers that
    ran it. Falls back to the commit declared in the repo, prefixed `declared:`. Without it
    a row is not reproducible: Boltz's version string does not uniquely identify a build,
    and rows 001-011 were recorded against builds that can no longer be identified
    (Process/boltz-version-pin.md).

    If wandb_project is set (or WANDB_PROJECT env var), also logs to W&B.
    Returns the logged entry dict.
    """
    git = _git_info()
    run_id = _next_run_id()
    summary = _compute_summary(results)
    per_target = _build_per_target_table(results)

    entry = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "source": source,
        "backend": backend,
        "backend_build": backend_build or _declared_boltz_pin() or "unknown",
        "config": config,
        "environment": _environment_info(),
        "duration_seconds": round(duration_seconds, 1) if duration_seconds else None,
        "summary": summary,
        "per_target": per_target,
        "notes": notes,
    }

    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"\n  Logged {run_id} to {RESULTS_FILE}")
    if duration_seconds:
        print(f"  Run duration: {duration_seconds:.0f}s ({duration_seconds/60:.1f}m)")

    wb_project = wandb_project or os.getenv("WANDB_PROJECT")
    if wb_project:
        wb_entity = wandb_entity or os.getenv("WANDB_ENTITY") or None
        _log_wandb(entry, wb_project, entity=wb_entity)

    return entry


def _log_wandb(entry: dict, project: str, entity: str | None = None):
    """
    Mirror an entry to Weights & Biases. Best-effort by design: results.jsonl is already
    written by the time this runs, so nothing here may raise — a telemetry hiccup must not
    destroy a benchmark run that just spent GPU-minutes producing numbers.
    """
    try:
        import wandb
    except ImportError:
        print("  [warn] wandb not installed, skipping W&B logging. pip install wandb")
        return

    # `import wandb` can SUCCEED and still be useless: this repo has a local wandb/ run
    # artifact directory, and when the real package is absent (e.g. running from the wrong
    # conda env) Python resolves that directory as a NAMESPACE PACKAGE. The ImportError
    # guard above passes and the failure surfaces much later as
    # `AttributeError: module 'wandb' has no attribute 'init'` — after the GPU work is done.
    if not hasattr(wandb, "init"):
        where = getattr(wandb, "__path__", None) or getattr(wandb, "__file__", "?")
        print(f"  [warn] 'wandb' resolved to {where}, which is the local run-artifact "
              "directory rather than the installed package — skipping W&B logging.")
        print("         Activate the env that has wandb installed (the ProPredict env), "
              "or run from outside the repo root.")
        return

    try:
        _log_wandb_inner(entry, wandb, project, entity)
    except Exception as e:  # noqa: BLE001 — telemetry must never fail the run
        print(f"  [warn] W&B logging failed: {type(e).__name__}: {e}")
        print(f"         The results.jsonl entry ({entry['run_id']}) was already written "
              "and is unaffected.")


def _log_wandb_inner(entry: dict, wandb, project: str, entity: str | None = None):
    good = [r for r in entry["per_target"] if r["status"] == "ok"]

    run = wandb.init(
        project=project,
        entity=entity or None,
        name=entry["run_id"],
        config={
            **entry["git"],
            "source": entry["source"],
            "backend": entry["backend"],
            "backend_build": entry["backend_build"],
            **entry["config"],
            **entry["environment"],
        },
        notes=entry["notes"],
        tags=[entry["backend"], entry["source"]],
    )

    run.summary.update(entry["summary"])
    if entry["duration_seconds"]:
        run.summary["duration_seconds"] = entry["duration_seconds"]

    if good:
        table = wandb.Table(
            columns=["pdb_id", "name", "length", "mean_plddt", "tm_score", "rmsd", "n_aligned"]
        )
        for r in good:
            table.add_data(
                r["pdb_id"], r.get("name", ""), r["length"],
                r["mean_plddt"], r["tm_score"], r["rmsd"], r.get("n_aligned"),
            )
        run.log({"per_target": table})

        tms = [r["tm_score"] for r in good]
        rmsds = [r["rmsd"] for r in good]
        plddts = [r["mean_plddt"] for r in good]
        run.log({
            "tm_score_hist": wandb.Histogram(tms),
            "rmsd_hist": wandb.Histogram(rmsds),
            "plddt_hist": wandb.Histogram(plddts),
        })

    run.finish()
    print(f"  Logged to W&B project '{project}': {run.url}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Log a benchmark result file")
    parser.add_argument("file", help="Path to benchmark_results.json")
    parser.add_argument("--notes", default="", help="Run notes")
    parser.add_argument("--source", default="casp15", help="Target source")
    parser.add_argument("--backend", default="boltz-2", help="Backend used")
    parser.add_argument("--wandb-project", default=None, help="W&B project name")
    parser.add_argument("--wandb-entity", default=None, help="W&B entity (team or username)")
    args = parser.parse_args()

    with open(args.file) as f:
        data = json.load(f)

    targets = data.get("targets", data.get("results", []))
    config_snapshot = {
        "BOLTZ_DIFFUSION_SAMPLES": int(os.getenv("BOLTZ_DIFFUSION_SAMPLES", 1)),
        "BOLTZ_SAMPLING_STEPS": int(os.getenv("BOLTZ_SAMPLING_STEPS", 200)),
        "BOLTZ_USE_MSA": os.getenv("BOLTZ_USE_MSA", "False") == "True",
    }

    log_run(
        targets, config_snapshot,
        source=args.source, backend=args.backend,
        notes=args.notes,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )
