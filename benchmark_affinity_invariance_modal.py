"""
Modal entrypoint for the Row-A affinity-invariance experiment.

This is the GPU wrapper around `benchmarks/benchmark_affinity_invariance.py`. That script
is standalone and CPU-launchable but shells out to `boltz predict`, which needs a GPU and
the baked Boltz-2 weights — both of which live in the Modal image (see `modal_app.image`,
BOLTZ_CACHE=/opt/boltz-cache). So the pattern mirrors `benchmark_pipeline_modal.py`:

  * reuse `app` and `image` from modal_app (weights already baked),
  * mount the invariance script + dataset into the image,
  * run one Boltz job per (isolate, MSA-arm) on an A10G via `.starmap`,
  * write results to a LOCAL jsonl with the same resume/dedup contract as the CLI,
  * print an immediate H_invariant / H_responsive read so you don't have to run the
    analyzer separately to know which way it broke.

Run (MSA-off arm — the important one; see note in the local entrypoint):
    modal run benchmark_affinity_invariance_modal.py

Add the MSA-on arm as an ablation:
    modal run benchmark_affinity_invariance_modal.py --both-msa-arms

Smoke test on a handful of isolates first:
    modal run benchmark_affinity_invariance_modal.py --limit 6
"""
import json
import os

from modal_app import app, image

# Mount the existing standalone harness (its build_yaml/run_boltz are reused verbatim) and
# the dataset, so the GPU function imports the SAME code the CLI runs — no logic fork.
image = (
    image
    .add_local_file(
        "benchmarks/benchmark_affinity_invariance.py",
        remote_path="/root/benchmark_affinity_invariance.py",
    )
    .add_local_file(
        "benchmarks/hiv_pr_resistance_dataset.json",
        remote_path="/root/hiv_pr_resistance_dataset.json",
    )
)

# Columns copied straight from each dataset job onto its result row (same set the CLI uses).
_PASSTHROUGH = (
    "seq_id", "drug", "drug_name", "mutations", "n_mutations",
    "fold_change", "log10_fold_change", "censored",
)


@app.function(timeout=3600, gpu="A10G", image=image)
def affinity_one(
    job: dict,
    use_msa: bool,
    seed: int,
    diffusion_samples: int,
    sampling_steps: int,
    msa_server_url: str,
) -> dict:
    """Run Boltz-2 affinity on a single WT/isolate job and return its result row.

    Imports build_yaml/run_boltz from the mounted CLI script so this stays a thin GPU
    shim: identical YAML, identical affinity-key parsing, identical timing.
    """
    import sys
    sys.path.insert(0, "/root")
    from benchmark_affinity_invariance import run_boltz

    rec = {k: job.get(k) for k in _PASSTHROUGH}
    rec.update({"msa": use_msa, "seed": seed})
    try:
        rec.update(run_boltz(
            job["sequence"], job["smiles"], seed, use_msa,
            diffusion_samples, sampling_steps, msa_server_url,
        ))
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001 — record the failure, don't kill the sweep
        rec["ok"] = False
        rec["error"] = str(e)[:500]
    return rec


def _spearman(x, y):
    """Tie-averaged Spearman; returns None if <3 points or zero variance."""
    if len(x) < 3:
        return None

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return None if dx == 0 or dy == 0 else num / (dx * dy)


def _slope(x, y):
    """OLS slope of y on x — the invariance discriminator: ~0 slope == practical blindness
    even if a weak correlation squeaks past significance."""
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    den = sum((a - mx) ** 2 for a in x)
    return None if den == 0 else sum((a - mx) * (b - my) for a, b in zip(x, y)) / den


def _summarize(rows):
    """Print the headline read per MSA arm: predicted Δaffinity vs experimental log10 FC.

    predicted Δ = affinity_pred_value(isolate) - affinity_pred_value(WT_same_drug)
    experiment  = log10_fold_change
    Spearman ~ 0 AND slope ~ 0  -> H_invariant (affinity is mutation-blind; physics tier
    has a real gap to fill). Spearman > 0 with real slope -> H_responsive.
    """
    for arm in sorted({r["msa"] for r in rows}):
        arm_rows = [r for r in rows if r["msa"] == arm and r.get("ok")]
        # WT baseline per drug (seq_id starting WT_), then Δ for each isolate of that drug.
        wt = {r["drug"]: r["affinity_pred_value"]
              for r in arm_rows if str(r["seq_id"]).startswith("WT")}
        pred_delta, exp_delta = [], []
        for r in arm_rows:
            if str(r["seq_id"]).startswith("WT") or r["drug"] not in wt:
                continue
            if r.get("log10_fold_change") is None:
                continue
            pred_delta.append(r["affinity_pred_value"] - wt[r["drug"]])
            exp_delta.append(r["log10_fold_change"])

        print(f"\n=== MSA={'on' if arm else 'off'} : {len(pred_delta)} isolate Δ pairs ===")
        if len(pred_delta) < 3:
            print("  too few completed pairs to call it yet")
            continue
        rho = _spearman(exp_delta, pred_delta)
        slope = _slope(exp_delta, pred_delta)
        spread = max(pred_delta) - min(pred_delta)
        print(f"  Spearman(exp, pred) = {rho:+.3f}" if rho is not None else "  Spearman = n/a")
        print(f"  slope(pred~exp)     = {slope:+.4f}" if slope is not None else "  slope = n/a")
        print(f"  pred Δ spread       = {spread:.3f}  (log10 units)")
        verdict = ("H_invariant (affinity ~blind to mutations)"
                   if (rho is None or abs(rho) < 0.2 or (slope is not None and abs(slope) < 0.05))
                   else "H_responsive (affinity tracks resistance)")
        print(f"  --> {verdict}")
    print("\n(Definitive call: run benchmarks/analyze_affinity_invariance.py on the jsonl — "
          "it handles PhenoSense censoring at 100 properly.)")


@app.local_entrypoint()
def affinity_invariance(
    dataset: str = "benchmarks/hiv_pr_resistance_dataset.json",
    out: str = "benchmarks/affinity_invariance_results.jsonl",
    limit: int = 0,
    seed: int = 37,
    diffusion_samples: int = 1,
    sampling_steps: int = 200,
    msa_server_url: str = "https://api.colabfold.com",
    both_msa_arms: bool = False,
):
    """Sweep WT + isolates on Modal, append to a local jsonl (resumable), print the verdict.

    Default arm is MSA-OFF. Rationale (from analyze_affinity_invariance.py): an HIV-protease
    MSA contains both WT and resistant variants, so an MSA-on model can copy resistance from
    homologs — a plausible *mechanism* for invariance that would confound the result. The
    clean test of whether the model reasons about the mutation is MSA-off; --both-msa-arms
    adds the MSA-on arm as an ablation to see if the MSA changes the picture.
    """
    data = json.load(open(dataset))
    jobs = data["reference_runs"] + data["records"]  # WT rows first — needed as Δ baselines
    if limit:
        jobs = jobs[:limit]

    arms = [True, False] if both_msa_arms else [False]

    done = set()
    if os.path.exists(out):
        for line in open(out):
            try:
                r = json.loads(line)
                done.add((r["seq_id"], r["drug"], r["msa"]))
            except Exception:
                pass

    tasks = [
        (job, use_msa, seed, diffusion_samples, sampling_steps, msa_server_url)
        for use_msa in arms
        for job in jobs
        if (job["seq_id"], job["drug"], use_msa) not in done
    ]
    print(f"{len(tasks)} jobs to run "
          f"({len(jobs)} isolates x {len(arms)} MSA arm(s)), {len(done)} already done.")

    all_rows = []
    if os.path.exists(out):
        all_rows = [json.loads(l) for l in open(out) if l.strip()]

    with open(out, "a") as fh:
        for rec in affinity_one.starmap(tasks, return_exceptions=True):
            if isinstance(rec, Exception):
                print(f"  worker raised: {rec}")
                continue
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            all_rows.append(rec)
            tag = (f"aff={rec['affinity_pred_value']:+.3f}"
                   if rec.get("ok") else f"FAILED: {rec.get('error', '')[:60]}")
            print(f"  {rec['seq_id']:>16} {rec['drug']} msa={rec['msa']} {tag}")

    _summarize(all_rows)
