"""
Row A experiment: is Boltz-2's affinity head blind to HIV-1 protease resistance mutations?

Runs Boltz-2 on WT (subtype-B consensus) and on clinical isolates with measured
PhenoSense fold-change, then compares:

    predicted  Delta = affinity_pred_value(mutant) - affinity_pred_value(WT)
    experiment Delta = log10(experimental fold-change)

These are DIRECTLY comparable with no unit conversion and no fitted parameters:
Boltz-2's README states affinity_pred_value is log10(IC50) with IC50 in micromolar,
so the WT->mutant difference IS a predicted log10 fold-change.

Deliberately standalone (does NOT import orchestrator.backends.boltz) because that
backend currently has two blocking bugs for this experiment:
  1. it reads aff_data.get("affinity") -- Boltz-2 writes "affinity_pred_value";
     the key "affinity" does not exist, so affinity_score is always None.
  2. it builds a single protein chain {"id": "A"}. HIV-1 protease is an obligate
     homodimer whose active site forms at the dimer interface; a monomer has no
     binding pocket. Boltz supports id: [A, B] for homomers.
Fix those in the backend (see the plan doc), then this script can be folded back in.

Usage:
    python benchmark_affinity_invariance.py hiv_pr_resistance_dataset.json results.jsonl [--limit N]
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

import yaml


def build_yaml(sequence: str, smiles: str, use_msa: bool) -> dict:
    """Homodimeric protease (chains A+B, identical sequence) + ligand C, affinity on C."""
    protein = {"id": ["A", "B"], "sequence": sequence}
    if not use_msa:
        protein["msa"] = "empty"
    return {
        "version": 1,
        "sequences": [
            {"protein": protein},
            {"ligand": {"id": "C", "smiles": smiles}},
        ],
        "properties": [{"affinity": {"binder": "C"}}],
    }


def run_boltz(sequence, smiles, seed, use_msa, diffusion_samples, sampling_steps,
              msa_server_url, timeout=3600):
    with tempfile.TemporaryDirectory() as tmp:
        ypath = os.path.join(tmp, "input.yaml")
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir)
        with open(ypath, "w") as fh:
            yaml.dump(build_yaml(sequence, smiles, use_msa), fh, default_flow_style=False)

        cmd = ["boltz", "predict", ypath, "--out_dir", out_dir,
               "--diffusion_samples", str(diffusion_samples),
               "--sampling_steps", str(sampling_steps),
               "--seed", str(seed)]
        if use_msa:
            cmd += ["--use_msa_server", "--msa_server_url", msa_server_url]

        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"boltz failed ({proc.returncode}): {proc.stderr[-1500:]}")

        hits = glob.glob(os.path.join(out_dir, "**", "affinity_*.json"), recursive=True)
        if not hits:
            raise RuntimeError("no affinity_*.json produced -- check the properties block")
        with open(hits[0]) as fh:
            aff = json.load(fh)

        # Verified against boltz-2.2.1 src/boltz/data/write/writer.py: the JSON keys are
        # affinity_pred_value / affinity_probability_binary (+ per-model 1/2 variants).
        return {
            "affinity_pred_value": aff["affinity_pred_value"],
            "affinity_probability_binary": aff["affinity_probability_binary"],
            "affinity_pred_value1": aff.get("affinity_pred_value1"),
            "affinity_pred_value2": aff.get("affinity_pred_value2"),
            "wall_seconds": round(time.time() - t0, 1),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out", help="results .jsonl (appended; reruns skip completed rows)")
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--no-msa", action="store_true",
                    help="MSA-off arm. Worth running as an ablation: AF-class models lean "
                         "hard on the MSA, and an HIV-protease MSA contains both WT and "
                         "resistant variants, which is a plausible mechanism for invariance.")
    ap.add_argument("--diffusion-samples", type=int, default=1)
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--msa-server-url", default="https://api.colabfold.com")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    data = json.load(open(args.dataset))
    jobs = data["reference_runs"] + data["records"]   # WT first: needed as the Delta baseline
    if args.limit:
        jobs = jobs[: args.limit]

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                r = json.loads(line)
                done.add((r["seq_id"], r["drug"], r["msa"]))
            except Exception:
                pass

    use_msa = not args.no_msa
    with open(args.out, "a") as fh:
        for i, job in enumerate(jobs, 1):
            key = (job["seq_id"], job["drug"], use_msa)
            if key in done:
                print(f"[{i}/{len(jobs)}] skip {key}")
                continue
            rec = {k: job[k] for k in
                   ("seq_id", "drug", "drug_name", "mutations", "n_mutations",
                    "fold_change", "log10_fold_change", "censored")}
            rec.update({"msa": use_msa, "seed": args.seed})
            try:
                rec.update(run_boltz(job["sequence"], job["smiles"], args.seed, use_msa,
                                     args.diffusion_samples, args.sampling_steps,
                                     args.msa_server_url))
                rec["ok"] = True
                print(f"[{i}/{len(jobs)}] {job['seq_id']:>16} {job['drug']} "
                      f"aff={rec['affinity_pred_value']:+.3f} ({rec['wall_seconds']}s)")
            except Exception as e:
                rec["ok"] = False
                rec["error"] = str(e)[:500]
                print(f"[{i}/{len(jobs)}] {job['seq_id']} {job['drug']} FAILED: {e}", file=sys.stderr)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
