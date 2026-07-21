"""Build the WT-vs-resistant-isolate dataset for the Boltz-2 affinity-invariance test.

Source: Stanford HIVDB PhenoSense genotype-phenotype dataset (PI_DataSet.txt).
Output: dataset.json — a manifest of (isolate, drug, sequence, experimental fold-change).
"""
import csv, json, math, os, random, sys, urllib.request

HIVDB_URL = "https://hivdb.stanford.edu/download/GenoPhenoDatasets/PI_DataSet.txt"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hivdb_cache")
CACHE_PATH = os.path.join(CACHE_DIR, "PI_DataSet.txt")


def ensure_dataset(path: str = CACHE_PATH) -> str:
    """Download the Stanford HIVDB PI genotype-phenotype dataset if not cached.

    Mirrors the benchmarks/proteingym_cache/ convention: third-party data is
    auto-downloaded into a gitignored cache rather than committed.
    """
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {HIVDB_URL} -> {path}")
    urllib.request.urlretrieve(HIVDB_URL, path)
    return path

# HIV-1 protease subtype-B consensus, 99 aa. Verified against the dataset's own
# mutation annotations (e.g. row 1 lists "D30N, M46I, R57G, L63P, N88D" and
# P30=N,P46=I,P57=G,P63=P,P88=D against consensus D30,M46,R57,L63,N88).
CONSENSUS = ("PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYD"
             "QILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF")
assert len(CONSENSUS) == 99

# Isomeric SMILES from PubChem PUG-REST (stereochemistry included — required;
# canonical SMILES drop it and darunavir/PI stereo is load-bearing).
DRUGS = {
    "DRV": ("darunavir",  "CC(C)CN(C[C@H]([C@H](CC1=CC=CC=C1)NC(=O)O[C@H]2CO[C@@H]3[C@H]2CCO3)O)S(=O)(=O)C4=CC=C(C=C4)N"),
    "NFV": ("nelfinavir", "CC1=C(C=CC=C1O)C(=O)N[C@@H](CSC2=CC=CC=C2)[C@@H](CN3C[C@H]4CCCC[C@H]4C[C@H]3C(=O)NC(C)(C)C)O"),
    "SQV": ("saquinavir", "CC(C)(C)NC(=O)[C@@H]1C[C@@H]2CCCC[C@@H]2CN1C[C@H]([C@H](CC3=CC=CC=C3)NC(=O)[C@H](CC(=O)N)NC(=O)C4=NC5=CC=CC=C5C=C4)O"),
    "IDV": ("indinavir",  "CC(C)(C)NC(=O)[C@@H]1CN(CCN1C[C@H](C[C@@H](CC2=CC=CC=C2)C(=O)N[C@@H]3[C@@H](CC4=CC=CC=C34)O)O)CC5=CN=CC=C5"),
}

POS = [f"P{i}" for i in range(1, 100)]

def parse(path):
    return list(csv.DictReader(open(path), delimiter="\t"))

def mutations(row):
    """Return [(pos, aa)] or None if the isolate has a mixture/indel (excluded)."""
    out = []
    for i, c in enumerate(POS, 1):
        v = (row.get(c) or "").strip()
        if v in ("-", ""):
            continue
        if len(v) != 1 or not v.isalpha():
            return None          # mixture ("IV") or indel ("#", "~") -> exclude
        out.append((i, v.upper()))
    return out

def build_seq(muts):
    s = list(CONSENSUS)
    for p, a in muts:
        s[p - 1] = a
    return "".join(s)

def main(path=None, n_per_drug=40, seed=0):
    rows = parse(ensure_dataset(path or CACHE_PATH))
    rng = random.Random(seed)
    records, skipped = [], 0

    for drug, (drug_name, smiles) in DRUGS.items():
        cands = []
        for r in rows:
            fc = (r.get(drug) or "NA").strip()
            if fc in ("NA", ""):
                continue
            try:
                fc = float(fc)
            except ValueError:
                continue
            if fc <= 0:
                continue
            muts = mutations(r)
            if muts is None or not muts:
                skipped += 1
                continue
            cands.append({
                "seq_id": r["SeqID"], "drug": drug, "drug_name": drug_name,
                "smiles": smiles,
                "mutations": [f"{CONSENSUS[p-1]}{p}{a}" for p, a in muts],
                "n_mutations": len(muts),
                "sequence": build_seq(muts),
                "fold_change": fc,
                "log10_fold_change": round(math.log10(fc), 4),
                "censored": fc >= 100.0,   # PhenoSense caps at 100 -> rank stats only
            })

        # Stratify across the log10 fold-change range so the sample spans
        # susceptible -> highly resistant rather than clustering at the median.
        cands.sort(key=lambda c: c["log10_fold_change"])
        if len(cands) > n_per_drug:
            idx = [round(i * (len(cands) - 1) / (n_per_drug - 1)) for i in range(n_per_drug)]
            cands = [cands[i] for i in sorted(set(idx))]
        records.extend(cands)

    wt = [{"seq_id": "WT_CONSENSUS_B", "drug": d, "drug_name": n, "smiles": s,
           "mutations": [], "n_mutations": 0, "sequence": CONSENSUS,
           "fold_change": 1.0, "log10_fold_change": 0.0, "censored": False}
          for d, (n, s) in DRUGS.items()]

    out = {
        "description": "WT-vs-isolate Boltz-2 affinity invariance test (HIV-1 protease + PIs)",
        "source": "Stanford HIVDB PhenoSense PI_DataSet.txt",
        "consensus": CONSENSUS,
        "note": ("HIV-1 protease is an obligate homodimer: the active site forms at the "
                 "dimer interface. Both chains carry the same mutations. Boltz-2 YAML must "
                 "use id: [A, B] for the protein entry."),
        "reference_runs": wt,
        "records": records,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "hiv_pr_resistance_dataset.json")
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"Saved -> {out_path}")
    print(f"skipped (mixture/indel/no-mut): {skipped}")
    print(f"WT reference runs: {len(wt)}   isolate runs: {len(records)}   total: {len(wt)+len(records)}")
    for d in DRUGS:
        sub = [r for r in records if r["drug"] == d]
        if sub:
            lo, hi = sub[0]["log10_fold_change"], sub[-1]["log10_fold_change"]
            nm = sorted(r["n_mutations"] for r in sub)
            print(f"  {d}: n={len(sub):3d}  log10FC {lo:+.2f}..{hi:+.2f}  "
                  f"mutation load median={nm[len(nm)//2]}  censored={sum(r['censored'] for r in sub)}")

if __name__ == "__main__":
    main(*sys.argv[1:])
