"""
Real-data epistasis study (sub-question A): does ProteinMPNN `score_only` exhibit
NON-ADDITIVITY on a real protein — i.e., is there epistasis for the combinatorial search
(orchestrator.mutation_search) to exploit, or is score_only effectively additive (which would
make AdaLead over it near-vacuous on real proteins)?

This is the mutation-search analogue of the Row A affinity-invariance study, but it runs on
CPU (score_only needs no GPU). It is a METHODOLOGY probe on real structure + real variants,
NOT a ProteinGym-style accuracy gate.

Epistasis for a double mutant (fitness convention, higher = better; f = -global_score):
    eps(m1, m2) = f(WT+m1+m2) - f(WT+m1) - f(WT+m2) + f(WT)
Zero = additive; |eps| large relative to the single-mutation effect scale = real coupling
the search can exploit.

Data: one Tsuboyama-2023 megascale domain from the cached ProteinGym parquet (AMFR_HUMAN,
PDB 4G3O, 47 aa, X-ray, gap-free). Sequences come straight from the parquet's
`mutated_sequence`; the WT backbone is the real PDB trimmed to the DMS residue slice and
renumbered 1..L so score_only threads each variant onto the correct positions.

Run: python -m benchmarks.benchmark_score_only_epistasis [--n-doubles N] [--decoding-orders K]
"""
import argparse
import io
import json
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROTEINMPNN_PATH, PROTEINMPNN_MODEL_NAME, PROTEINMPNN_SEED
from orchestrator.mutation_search import score_only_oracle

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARQUET = os.path.join(_HERE, "proteingym_cache", "DMS_substitutions.parquet")
_STRUCT_DIR = os.path.join(_HERE, "epistasis_structures")

DMS_ID = "AMFR_HUMAN_Tsuboyama_2023_4G3O"
PDB_CODE = "4G3O"


def _fetch_pdb(pdb_code: str) -> str:
    raw = os.path.join(_STRUCT_DIR, f"{pdb_code}.raw.pdb")
    if not os.path.isfile(raw):
        import urllib.request
        os.makedirs(_STRUCT_DIR, exist_ok=True)
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_code}.pdb", raw)
    with open(raw) as fh:
        return fh.read()


def trim_pdb_to_dms(raw_pdb: str, target_seq: str) -> str:
    """Extract the model-0, first-chain residue slice whose sequence == target_seq, renumber
    it 1..L, and return an ATOM-only PDB. Raises if the slice isn't found gap-free."""
    from Bio.PDB import PDBParser, PPBuilder

    struct = PDBParser(QUIET=True).get_structure(PDB_CODE, io.StringIO(raw_pdb))
    chain = list(list(struct)[0])[0]
    peptides = PPBuilder().build_peptides(chain)
    if len(peptides) != 1:
        raise RuntimeError(f"{PDB_CODE}: chain is not a single gap-free fragment")
    residues = list(peptides[0])
    chain_seq = str(peptides[0].get_sequence())
    off = chain_seq.find(target_seq)
    if off < 0:
        raise RuntimeError(f"{PDB_CODE}: DMS target_seq not found in chain sequence")
    slice_res = residues[off:off + len(target_seq)]

    lines = []
    for new_num, res in enumerate(slice_res, start=1):
        for atom in res:
            e = atom.element or atom.get_name()[0]
            x, y, z = atom.coord
            lines.append(
                f"ATOM  {atom.serial_number:>5} {atom.get_name():<4} "
                f"{res.get_resname():>3} A{new_num:>4}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{atom.get_occupancy() or 1.0:6.2f}"
                f"{atom.get_bfactor() or 0.0:6.2f}          {e:>2}"
            )
    lines.append("END")
    return "\n".join(lines) + "\n"


def load_variants(dms_id: str):
    """Return (wt_seq, singles, doubles) from the cached parquet.
    singles: {mutant_str: (sequence, dms_score)};
    doubles: [{"mutant","sequence","dms_score","components":[m1,m2]}] with BOTH singles present."""
    import pandas as pd
    df = pd.read_parquet(
        _PARQUET, columns=["mutant", "mutated_sequence", "target_seq", "DMS_score", "DMS_id"]
    )
    df = df[df.DMS_id == dms_id]
    wt_seq = df.target_seq.iloc[0]

    singles, doubles = {}, []
    for _, row in df.iterrows():
        parts = str(row.mutant).split(":")
        if len(parts) == 1:
            singles[parts[0]] = (row.mutated_sequence, float(row.DMS_score))
    for _, row in df.iterrows():
        parts = str(row.mutant).split(":")
        if len(parts) == 2 and parts[0] in singles and parts[1] in singles:
            doubles.append({
                "mutant": row.mutant, "sequence": row.mutated_sequence,
                "dms_score": float(row.DMS_score), "components": parts,
            })
    return wt_seq, singles, doubles


def _ca_distance(pdb_string: str, pos_i: int, pos_j: int) -> float:
    """CA-CA distance (A) between two 1-indexed positions in the trimmed PDB."""
    from Bio.PDB import PDBParser
    struct = PDBParser(QUIET=True).get_structure("t", io.StringIO(pdb_string))
    ca = {r.get_id()[1]: r["CA"].coord for r in list(struct.get_residues()) if "CA" in r}
    return float(np.linalg.norm(ca[pos_i] - ca[pos_j]))


def _pos(mut: str) -> int:
    return int(mut[1:-1])


def _eps_for_sample(pdb, wt_seq, singles, sample, decoding_orders, mpnn_seed):
    """Model epistasis for each double in `sample`, as a numpy array (fitness units)."""
    need = sorted({m for d in sample for m in d["components"]})
    seqs = [wt_seq] + [singles[m][0] for m in need] + [d["sequence"] for d in sample]
    f = score_only_oracle(pdb, seqs, proteinmpnn_dir=PROTEINMPNN_PATH,
                          model_name=PROTEINMPNN_MODEL_NAME, seed=mpnn_seed,
                          num_decoding_orders=decoding_orders)
    f_wt = f[0]
    f_single = {m: f[1 + i] for i, m in enumerate(need)}
    f_double = {d["mutant"]: f[1 + len(need) + i] for i, d in enumerate(sample)}
    return np.array([
        (f_double[d["mutant"]] - f_wt)
        - ((f_single[d["components"][0]] - f_wt) + (f_single[d["components"][1]] - f_wt))
        for d in sample
    ])


def noise_check(pdb, wt_seq, singles, sample, decoding_orders):
    """Score the same doubles at two ProteinMPNN decoding seeds and report signal-vs-noise:
    if eps is stable across seeds it is real coupling, not decoding-order noise."""
    e_a = _eps_for_sample(pdb, wt_seq, singles, sample, decoding_orders, mpnn_seed=37)
    e_b = _eps_for_sample(pdb, wt_seq, singles, sample, decoding_orders, mpnn_seed=38)
    noise = float((e_a - e_b).std() / np.sqrt(2))
    print(f"\n=== noise floor (K={decoding_orders} decoding orders, n={len(sample)} pairs) ===")
    print(f"eps std (signal) : {e_a.std():.4f}")
    print(f"noise std        : {noise:.4f}")
    print(f"corr(seed37,38)  : {np.corrcoef(e_a, e_b)[0, 1]:.3f}")
    print(f"SNR              : {e_a.std() / noise:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-doubles", type=int, default=150,
                    help="how many doubles to sample (deterministic)")
    ap.add_argument("--decoding-orders", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-check", action="store_true",
                    help="also score at two decoding seeds and report signal-vs-noise")
    ap.add_argument("--out", default=os.path.join(_HERE, "score_only_epistasis.jsonl"))
    args = ap.parse_args()

    if not PROTEINMPNN_PATH:
        sys.exit("PROTEINMPNN_PATH not set (see .env)")

    wt_seq, singles, doubles = load_variants(DMS_ID)
    pdb = trim_pdb_to_dms(_fetch_pdb(PDB_CODE), wt_seq)
    print(f"# {DMS_ID}: WT {len(wt_seq)} aa | {len(singles)} singles | "
          f"{len(doubles)} doubles with both singles present")

    rng = np.random.default_rng(args.seed)
    if len(doubles) > args.n_doubles:
        idx = rng.choice(len(doubles), size=args.n_doubles, replace=False)
        doubles = [doubles[i] for i in sorted(idx)]

    # One batched score_only over WT + every involved single + the sampled doubles.
    needed_singles = sorted({m for d in doubles for m in d["components"]})
    seqs = [wt_seq] + [singles[m][0] for m in needed_singles] + [d["sequence"] for d in doubles]
    print(f"# scoring {len(seqs)} sequences (1 WT + {len(needed_singles)} singles + "
          f"{len(doubles)} doubles) at {args.decoding_orders} decoding orders...")
    fits = score_only_oracle(pdb, seqs, proteinmpnn_dir=PROTEINMPNN_PATH,
                             model_name=PROTEINMPNN_MODEL_NAME, seed=PROTEINMPNN_SEED,
                             num_decoding_orders=args.decoding_orders)
    f_wt = fits[0]
    f_single = {m: fits[1 + i] for i, m in enumerate(needed_singles)}
    f_double = {d["mutant"]: fits[1 + len(needed_singles) + i] for i, d in enumerate(doubles)}

    rows = []
    for d in doubles:
        m1, m2 = d["components"]
        d1, d2 = f_single[m1] - f_wt, f_single[m2] - f_wt      # single-effect deltas (model)
        eps_model = (f_double[d["mutant"]] - f_wt) - (d1 + d2)  # non-additivity
        dist = _ca_distance(pdb, _pos(m1), _pos(m2))
        rows.append({
            "mutant": d["mutant"], "eps_model": eps_model,
            "delta1": d1, "delta2": d2, "ca_distance": dist,
        })

    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    eps = [r["eps_model"] for r in rows]
    main_effects = [abs(r["delta1"]) for r in rows] + [abs(r["delta2"]) for r in rows]
    abs_eps = [abs(e) for e in eps]
    near = [r["eps_model"] for r in rows if r["ca_distance"] <= 8.0]
    far = [r["eps_model"] for r in rows if r["ca_distance"] > 8.0]

    print("\n=== score_only internal non-additivity (fitness units) ===")
    print(f"pairs analyzed         : {len(eps)}")
    print(f"eps  mean / median     : {statistics.mean(eps):+.4f} / {statistics.median(eps):+.4f}")
    print(f"|eps| median / max     : {statistics.median(abs_eps):.4f} / {max(abs_eps):.4f}")
    print(f"single |delta| median  : {statistics.median(main_effects):.4f}")
    print(f"|eps| / |delta| (median ratio): {statistics.median(abs_eps)/statistics.median(main_effects):.3f}")
    print(f"pairs with |eps| > 0.5*|delta_median|: "
          f"{sum(e > 0.5*statistics.median(main_effects) for e in abs_eps)}/{len(eps)}")
    if near and far:
        print(f"|eps| median  contacting (CA<=8A, n={len(near)}): {statistics.median([abs(x) for x in near]):.4f}")
        print(f"|eps| median  distal    (CA>8A,  n={len(far)}): {statistics.median([abs(x) for x in far]):.4f}")
    print(f"\nwrote {args.out}")

    if args.noise_check:
        noise_check(pdb, wt_seq, singles, doubles[:min(60, len(doubles))], args.decoding_orders)


if __name__ == "__main__":
    main()
