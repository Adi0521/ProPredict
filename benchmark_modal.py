"""
Boltz-2 structure prediction benchmark.

Fetches targets from CASP15 (default) or a custom RCSB query, runs Boltz-2
in parallel on Modal, and scores each prediction with TM-score and CA-RMSD.

Usage:
    modal run benchmark_modal.py                          # CASP15 targets
    modal run benchmark_modal.py --source rcsb            # RCSB quality query
    modal run benchmark_modal.py --pdb-ids 1UBQ,1VII      # specific PDB IDs
    modal run benchmark_modal.py --max-targets 20         # limit for quick tests
"""

import json
import numpy as np
from typing import Optional

import requests

from modal_app import app, image


# ---------------------------------------------------------------------------
# Target fetching
# ---------------------------------------------------------------------------

def fetch_casp15_targets() -> list[dict]:
    """
    Fetch CASP15 single-chain targets from RCSB by searching for entries
    that are tagged with the CASP15 experiment.  Falls back to a hardcoded
    core list if the API call fails.
    """
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "struct_keywords.pdbx_keywords",
                "operator": "contains_phrase",
                "value": "CASP15",
            },
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": 0, "rows": 200},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
    }
    try:
        resp = requests.post(
            "https://search.rcsb.org/rcsbsearch/v2/query",
            json=query,
            timeout=20,
        )
        resp.raise_for_status()
        hits = resp.json().get("result_set", [])
        targets = []
        for hit in hits:
            eid = hit["identifier"]          # e.g. "7TY4_1"
            pdb_id, entity_num = eid.split("_")
            targets.append({"pdb_id": pdb_id, "chain": "A", "name": f"CASP15/{pdb_id}"})
        if targets:
            return targets
    except Exception as e:
        print(f"[warn] RCSB CASP15 query failed ({e}), using hardcoded fallback.")

    # Hardcoded fallback — a representative cross-section of CASP15 regular targets
    return [
        {"pdb_id": p, "chain": "A", "name": f"CASP15/{p}"}
        for p in [
            "7TY4","7TY5","7UL4","7UL5","7UXB","7UXC","7V3E","7V3F",
            "7VDL","7VDM","7VQ6","7VQ7","7WGP","7WGQ","7WV6","7WV7",
            "7X26","7X27","7XPT","7XPU","7YEK","7YEL","7YG3","7Z6M",
            "7Z6N","7ZPD","7ZPE","8A0N","8A0O","8A2O","8A2P","8ACS",
            "8APC","8APD","8APE","8APG","8APH","8B1J","8B8T","8BCY",
            "8BCZ","8BD0","8BDE","8BDF","8BDG","8BEP","8BG1","8BOZ",
            "8BP0","8BPP","8BPQ","8C7B","8C7C","8CBN","8CBO","8CGU",
            "8CGV","8CIQ","8CIR","8CMJ","8CMK","8D40","8D41","8D8C",
            "8D8D","8DGS","8DGT","8DPN","8DPO","8DPP","8DPQ","8DPR",
            "8DPS","8DPT","8DPU","8FLY","8FLZ","8FM0","8FM1","8GEO",
            "8GEP","8GEQ","8H8J","8H8K","8HAK","8HAL","8HAM","8HAN",
        ]
    ]


def fetch_rcsb_quality_targets(
    max_resolution: float = 2.0,
    min_length: int = 50,
    max_length: int = 400,
    deposited_after: str = "2023-01-01",
    max_results: int = 300,
) -> list[dict]:
    """
    Fetch a diverse set of high-quality, recently deposited single-chain
    protein structures from RCSB.  Use deposited_after to approximate a
    held-out set (structures after Boltz-2's training cutoff).
    """
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_entity_polymer_type",
                        "operator": "exact_match",
                        "value": "Protein",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": max_resolution,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_sample_sequence_length",
                        "operator": "greater_or_equal",
                        "value": min_length,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_sample_sequence_length",
                        "operator": "less_or_equal",
                        "value": max_length,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_accession_info.deposit_date",
                        "operator": "greater_or_equal",
                        "value": deposited_after,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
                        "operator": "equals",
                        "value": 1,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_results},
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
    }
    resp = requests.post(
        "https://search.rcsb.org/rcsbsearch/v2/query",
        json=query,
        timeout=30,
    )
    resp.raise_for_status()
    hits = resp.json().get("result_set", [])
    return [
        {"pdb_id": h["identifier"], "chain": "A", "name": h["identifier"]}
        for h in hits
    ]


# ---------------------------------------------------------------------------
# Structural scoring — pure numpy, no extra deps needed in the container
# ---------------------------------------------------------------------------

def _parse_ca_coords(pdb_str: str, chain_id: Optional[str] = None) -> np.ndarray:
    coords = []
    for line in pdb_str.splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        if chain_id is not None and line[21].strip() != chain_id:
            continue
        coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(coords, dtype=float)


def _kabsch_rmsd(P: np.ndarray, Q: np.ndarray):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Pc.T @ Qc)
    R = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T
    diff = Pc @ R.T - Qc
    return float(np.sqrt((diff ** 2).sum(1).mean())), diff


def score_structures(pred_pdb: str, true_pdb: str, chain_id: str = "A") -> dict:
    """
    TM-score and CA-RMSD using RMSD-optimal superposition (Kabsch).
    Note: this slightly underestimates TM-score vs TM-align's TM-optimal
    rotation, but avoids any external binary dependency.
    """
    pred_ca = _parse_ca_coords(pred_pdb)
    true_ca = _parse_ca_coords(true_pdb, chain_id)

    n = min(len(pred_ca), len(true_ca))
    if n < 5:
        return {"tm_score": 0.0, "rmsd": 999.0, "n_aligned": n}

    pred_ca, true_ca = pred_ca[:n], true_ca[:n]
    rmsd, diff = _kabsch_rmsd(pred_ca, true_ca)

    L = len(true_ca)
    d0 = max(1.24 * (L - 15) ** (1 / 3) - 1.8, 0.5)
    tm = float((1.0 / L) * (1.0 / (1.0 + (diff ** 2).sum(1) / d0 ** 2)).sum())

    return {"tm_score": round(tm, 4), "rmsd": round(rmsd, 3), "n_aligned": n}


# ---------------------------------------------------------------------------
# Per-target Modal function
# ---------------------------------------------------------------------------

@app.function(timeout=600, gpu="A10G", image=image)
def benchmark_one(target: dict) -> dict:
    import io
    import os
    import requests
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa, protein_letters_3to1

    os.environ.update({
        "BOLTZ_ENABLED": "True",
        "BOLTZ_SAMPLES": "1",
        "BOLTZ_STEPS": "200",
        "BOLTZ_USE_MSA": "False",
    })
    from orchestrator.tasks import call_boltz

    pdb_id, chain = target["pdb_id"], target["chain"]

    # Fetch experimental structure
    try:
        resp = requests.get(
            f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=30
        )
        resp.raise_for_status()
    except Exception as e:
        return {"pdb_id": pdb_id, "error": f"PDB fetch failed: {e}"}

    true_pdb = resp.text

    # Extract canonical sequence from ATOM records
    parser = PDBParser(QUIET=True)
    try:
        struct = parser.get_structure(pdb_id, io.StringIO(true_pdb))
        chain_obj = struct[0][chain]
    except KeyError:
        # Try the first chain if "A" is not present
        chains = list(struct[0].get_chains())
        if not chains:
            return {"pdb_id": pdb_id, "error": "No chains found in PDB"}
        chain_obj = chains[0]
        chain = chain_obj.id

    seq = "".join(
        protein_letters_3to1.get(res.get_resname(), "X")
        for res in chain_obj
        if is_aa(res, standard=True)
    )

    if len(seq) < 10:
        return {"pdb_id": pdb_id, "error": f"Sequence too short ({len(seq)} aa)"}

    # Predict
    try:
        result = call_boltz(seq, seed=0)
    except Exception as e:
        return {"pdb_id": pdb_id, "length": len(seq), "error": f"Boltz failed: {e}"}

    # Score
    scores = score_structures(result.structure_pdb, true_pdb, chain)

    return {
        "pdb_id": pdb_id,
        "name": target.get("name", pdb_id),
        "length": len(seq),
        "mean_plddt": round(result.mean_plddt, 2),
        **scores,
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def run_benchmark(
    source: str = "casp15",
    pdb_ids: str = "",
    max_targets: int = 0,
    out: str = "benchmark_results.json",
):
    """
    source:      casp15 | rcsb | custom  (default: casp15)
    pdb-ids:     comma-separated PDB IDs, implies source=custom
    max-targets: cap the number of targets (0 = unlimited, useful for quick tests)
    out:         output JSON file

    Examples:
        modal run benchmark_modal.py
        modal run benchmark_modal.py --source rcsb --max-targets 50
        modal run benchmark_modal.py --pdb-ids 1UBQ,1VII,1GB1
    """
    if pdb_ids:
        targets = [
            {"pdb_id": p.strip().upper(), "chain": "A", "name": p.strip().upper()}
            for p in pdb_ids.split(",")
        ]
    elif source == "rcsb":
        print("Querying RCSB for high-quality single-chain proteins deposited ≥ 2023...")
        targets = fetch_rcsb_quality_targets()
    else:
        print("Fetching CASP15 targets from RCSB...")
        targets = fetch_casp15_targets()

    if max_targets and max_targets < len(targets):
        targets = targets[:max_targets]

    print(f"Running Boltz-2 on {len(targets)} targets in parallel...\n")

    results = list(benchmark_one.map(targets, return_exceptions=True))

    # Print table
    header = (
        f"{'PDB':>6}  {'Name':<24}  {'Len':>4}  "
        f"{'pLDDT':>6}  {'TM-score':>8}  {'RMSD (Å)':>9}"
    )
    print(header)
    print("-" * len(header))

    good, errors = [], []
    for r in results:
        if isinstance(r, Exception):
            errors.append(str(r))
            print(f"  EXCEPTION: {r}")
            continue
        if "error" in r:
            errors.append(r["error"])
            print(f"{r['pdb_id']:>6}  {'':24}  ERROR: {r['error']}")
            continue
        good.append(r)
        print(
            f"{r['pdb_id']:>6}  {r.get('name', r['pdb_id']):<24}  {r['length']:>4}  "
            f"{r['mean_plddt']:>6.1f}  {r['tm_score']:>8.4f}  {r['rmsd']:>9.3f}"
        )

    # Aggregate stats
    if good:
        tms = [r["tm_score"] for r in good]
        plddts = [r["mean_plddt"] for r in good]
        rmsds = [r["rmsd"] for r in good]
        print(
            f"\n  Targets: {len(good)} succeeded, {len(errors)} failed\n"
            f"  TM-score  mean={np.mean(tms):.4f}  median={np.median(tms):.4f}"
            f"  ≥0.5: {sum(t >= 0.5 for t in tms)}/{len(tms)}"
            f"  ≥0.7: {sum(t >= 0.7 for t in tms)}/{len(tms)}\n"
            f"  RMSD (Å)  mean={np.mean(rmsds):.2f}  median={np.median(rmsds):.2f}\n"
            f"  pLDDT     mean={np.mean(plddts):.1f}\n"
        )

    with open(out, "w") as f:
        json.dump({"targets": results, "errors": errors}, f, indent=2, default=str)
    print(f"  Full results saved to {out}")
