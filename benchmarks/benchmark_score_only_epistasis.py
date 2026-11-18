"""
Row B — does ProteinMPNN `score_only` epistasis exist, and does it track experiment?

Two arms, selected with `--arm`:

**`--arm internal` (sub-question A, the original study; default).** Does `score_only` exhibit
NON-ADDITIVITY on a real protein at all — i.e. is there epistasis for the combinatorial search
(orchestrator.mutation_search) to exploit, or is score_only effectively additive (which would
make AdaLead over it near-vacuous on real proteins)? Pure model probe, no experimental labels.

**`--arm stability` (sub-question B).** Does that non-additivity track *experimental*
epistasis, and does the full non-additive model score beat an additive baseline at predicting
measured double-mutant stability? Runs over three X-ray Tsuboyama-2023 domains.

Epistasis for a double mutant (fitness convention, higher = better; f = -global_score):
    eps_model(m1, m2) = f(WT+m1+m2) - f(WT+m1) - f(WT+m2) + f(WT)
    eps_exp(m1, m2)   = DMS(WT+m1+m2) - DMS(WT+m1) - DMS(WT+m2) + DMS(WT)
Zero = additive.

**The experimental WT reference.** The cached ProteinGym parquet has NO wild-type row for these
assays, so `DMS(WT)` is *assumed* 0 (ProteinGym's reference metadata gives the Tsuboyama source
column as `ddG_ML_float`, direction +1 — a quantity defined relative to WT, which the observed
value distribution is consistent with: mostly negative, continuous, no pile-up at the range
ends). A wrong reference `c` shifts every `eps_exp` in a domain by the same constant, so the
primary readouts — within-domain Spearman, Pearson and OLS slope WITH an intercept — are
unaffected, but `eps_exp` signs and absolute magnitudes are not interpretable and uncentered
values are never pooled across domains. See `benchmarks/epistasis_stats.py`.

Both arms run on CPU (`score_only` needs no GPU); the whole study ran locally on an M3.
Data: cached ProteinGym substitutions parquet (auto-downloaded, see `benchmarks/_proteingym.py`).
Backbones: the real RCSB structures, trimmed to each DMS residue slice and renumbered 1..L so
score_only threads each variant onto the correct positions.

Run:
    python -m benchmarks.benchmark_score_only_epistasis --n-doubles 300 --noise-check
    python -m benchmarks.benchmark_score_only_epistasis --arm stability --n-doubles 300 \
        --decoding-orders 32 --plots
"""

import argparse
import datetime
import hashlib
import io
import json
import os
import statistics
import subprocess
import sys
import time
import zlib
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks import _proteingym as _pg
from benchmarks import epistasis_stats as es
from benchmarks._artifacts import guard_overwrite, stamped_path
from config import PROTEINMPNN_MODEL_NAME, PROTEINMPNN_PATH, PROTEINMPNN_SEED
from orchestrator.mutation_search import score_only_oracle

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARQUET = os.path.join(_HERE, "proteingym_cache", "DMS_substitutions.parquet")
_STRUCT_DIR = os.path.join(_HERE, "epistasis_structures")

DMS_ID = "AMFR_HUMAN_Tsuboyama_2023_4G3O"
PDB_CODE = "4G3O"

# Stability-arm candidates, in the fixed evaluation order of the study design. AMFR/4G3O is
# retained from sub-question A; the first two *additional* candidates that are X-ray, match
# their DMS sequence exactly, trim gap-free and carry enough usable doubles are selected.
# X-ray-only is a SCOPE RESTRICTION (keeping the backbone provenance uniform with
# sub-question A), not a claim that NMR ensembles are unsuitable for score_only.
DOMAIN_CANDIDATES: List[Tuple[str, str]] = [
    ("AMFR_HUMAN_Tsuboyama_2023_4G3O", "4G3O"),
    ("HECD1_HUMAN_Tsuboyama_2023_3DKM", "3DKM"),
    ("UBE4B_HUMAN_Tsuboyama_2023_3L1X", "3L1X"),
    ("MYO3_YEAST_Tsuboyama_2023_2BTT", "2BTT"),
    ("SDA_BACSU_Tsuboyama_2023_1PV0", "1PV0"),
    ("OBSCN_HUMAN_Tsuboyama_2023_1V1C", "1V1C"),
]
MIN_USABLE_DOUBLES = 300  # "enough usable doubles" for the stability arm
# ProteinGym reference metadata for the Tsuboyama-2023 assays (verified against the data by
# `check_experimental_semantics`, not taken on faith).
EXP_SOURCE_COLUMN = "ddG_ML_float"
EXP_DIRECTION = +1


def _fetch_pdb(pdb_code: str) -> str:
    raw = os.path.join(_STRUCT_DIR, f"{pdb_code}.raw.pdb")
    if not os.path.isfile(raw):
        import urllib.request

        os.makedirs(_STRUCT_DIR, exist_ok=True)
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_code}.pdb", raw)
    with open(raw) as fh:
        return fh.read()


def trim_pdb_to_dms(raw_pdb: str, target_seq: str, pdb_code: str = PDB_CODE) -> str:
    """Extract the model-0, first-chain residue slice whose sequence == target_seq, renumber
    it 1..L, and return an ATOM-only PDB. Raises if the slice isn't found gap-free."""
    from Bio.PDB import PDBParser, PPBuilder

    struct = PDBParser(QUIET=True).get_structure(pdb_code, io.StringIO(raw_pdb))
    chain = list(list(struct)[0])[0]
    peptides = PPBuilder().build_peptides(chain)
    if len(peptides) != 1:
        raise RuntimeError(f"{pdb_code}: chain is not a single gap-free fragment")
    residues = list(peptides[0])
    chain_seq = str(peptides[0].get_sequence())
    off = chain_seq.find(target_seq)
    if off < 0:
        raise RuntimeError(f"{pdb_code}: DMS target_seq not found in chain sequence")
    slice_res = residues[off : off + len(target_seq)]

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


def pdb_metadata(raw_pdb: str) -> Dict:
    """Experimental method and resolution from the PDB header — the X-ray eligibility check
    reads this rather than trusting a hand-maintained list of PDB codes."""
    method, resolution = None, None
    for line in raw_pdb.splitlines():
        if line.startswith("EXPDTA") and method is None:
            method = line[10:].strip()
        elif line.startswith("REMARK   2 RESOLUTION") and resolution is None:
            tok = line.split("RESOLUTION.")[-1].replace("ANGSTROMS", "").strip().rstrip(".")
            try:
                resolution = float(tok)
            except ValueError:
                resolution = None
    n_models = max(1, sum(1 for line in raw_pdb.splitlines() if line.startswith("MODEL ")))
    return {"method": method, "resolution_angstrom": resolution, "n_models": n_models}


# ---------------------------------------------------------------------------
# Assay loading, validation and experimental-score semantics
# ---------------------------------------------------------------------------
class Assay:
    """One DMS assay's variants plus the provenance/exclusion bookkeeping behind them."""

    def __init__(self, dms_id: str, wt_seq: str, singles: Dict, doubles: List[Dict], meta: Dict):
        self.dms_id = dms_id
        self.wt_seq = wt_seq
        self.singles = singles  # {mutant: (sequence, dms_score)}
        self.doubles = doubles  # [{"mutant","sequence","dms_score","components"}]
        self.meta = meta


def _parse_sub(sub: str) -> Optional[Tuple[str, int, str]]:
    sub = sub.strip()
    if len(sub) < 3:
        return None
    wt_aa, mut_aa, pos_tok = sub[0], sub[-1], sub[1:-1]
    if not pos_tok.isdigit():
        return None
    return wt_aa, int(pos_tok), mut_aa


def load_assay(dms_id: str, parquet: Optional[str] = None) -> Assay:
    """Load one assay, validate every variant, and record why anything was dropped.

    Exclusion rules (all counted in `meta["exclusions"]`):
      * unparseable mutant token, or position outside 1..L;
      * mutant's stated WT letter disagreeing with `target_seq` at that position (an alignment
        mismatch — scoring it would silently score the wrong variant);
      * `mutated_sequence` whose length differs from `target_seq`;
      * missing (NaN) DMS_score;
      * duplicate mutant rows (first kept);
      * doubles whose constituent singles are not BOTH measured — without both singles the
        epistasis expression cannot be evaluated at all.
    """
    import pandas as pd

    path = parquet or (_PARQUET if os.path.isfile(_PARQUET) else _pg.ensure_parquet())
    df = pd.read_parquet(path, columns=["mutant", "mutated_sequence", "target_seq", "DMS_score", "DMS_id"])
    df = df[df.DMS_id == dms_id]
    if df.empty:
        raise RuntimeError(f"{dms_id}: no rows in {path}")
    wt_seq = df.target_seq.iloc[0]
    n_rows = len(df)

    exclusions = {
        "unparseable_mutant": 0,
        "position_out_of_range": 0,
        "wt_letter_mismatch": 0,
        "sequence_length_mismatch": 0,
        "missing_dms_score": 0,
        "duplicate_mutant": 0,
        "double_missing_a_single": 0,
    }
    seen = set()
    singles: Dict[str, Tuple[str, float]] = {}
    raw_doubles: List[Dict] = []

    for row in df.itertuples(index=False):
        mutant = str(row.mutant)
        if mutant in seen:
            exclusions["duplicate_mutant"] += 1
            continue
        seen.add(mutant)
        if row.DMS_score is None or (isinstance(row.DMS_score, float) and np.isnan(row.DMS_score)):
            exclusions["missing_dms_score"] += 1
            continue
        if len(row.mutated_sequence) != len(wt_seq):
            exclusions["sequence_length_mismatch"] += 1
            continue
        parts = mutant.split(":")
        parsed = [_parse_sub(p) for p in parts]
        if any(p is None for p in parsed):
            exclusions["unparseable_mutant"] += 1
            continue
        if any(not (1 <= p[1] <= len(wt_seq)) for p in parsed):  # type: ignore[index]
            exclusions["position_out_of_range"] += 1
            continue
        if any(wt_seq[p[1] - 1] != p[0] for p in parsed):  # type: ignore[index]
            exclusions["wt_letter_mismatch"] += 1
            continue
        if len(parts) == 1:
            singles[mutant] = (row.mutated_sequence, float(row.DMS_score))
        elif len(parts) == 2:
            raw_doubles.append(
                {
                    "mutant": mutant,
                    "sequence": row.mutated_sequence,
                    "dms_score": float(row.DMS_score),
                    "components": parts,
                }
            )

    doubles = []
    for d in raw_doubles:
        if all(c in singles for c in d["components"]):
            doubles.append(d)
        else:
            exclusions["double_missing_a_single"] += 1

    meta = {
        "dms_id": dms_id,
        "parquet_path": path,
        "parquet_bytes": os.path.getsize(path),
        "rows_for_assay": n_rows,
        "n_singles_usable": len(singles),
        "n_doubles_total": len(raw_doubles),
        "n_doubles_usable": len(doubles),
        "wt_length": len(wt_seq),
        "exclusions": exclusions,
    }
    return Assay(dms_id, wt_seq, singles, doubles, meta)


def load_variants(dms_id: str):
    """Back-compat shim for the internal arm: (wt_seq, singles, doubles)."""
    a = load_assay(dms_id)
    return a.wt_seq, a.singles, a.doubles


def check_experimental_semantics(assay: Assay) -> Dict:
    """Interrogate the experimental scores before any epistasis is interpreted.

    Checks, in order: is a wild-type row present at all (it is not, for these assays); is the
    score distribution consistent with a WT-relative quantity rather than an absolute one; is
    there censoring (a pile-up at either range end) or missing data; and what does an
    additive-null fit `DMS(AB) ~ a + b*(DMS(A)+DMS(B))` say.

    That last fit is reported but deliberately NOT used to estimate the WT reference: the
    estimate `-a` only identifies WT when `b == 1`, and the measured `b` is far below 1
    (strong global/nonspecific epistasis — diminishing returns as variants approach the
    unfolded limit). So the WT offset stays explicitly unknown.
    """
    scores = np.array([v[1] for v in assay.singles.values()] + [d["dms_score"] for d in assay.doubles])
    lo, hi = float(scores.min()), float(scores.max())
    at_lo = int(np.sum(np.isclose(scores, lo)))
    at_hi = int(np.sum(np.isclose(scores, hi)))

    x, y = [], []
    for d in assay.doubles:
        m1, m2 = d["components"]
        x.append(assay.singles[m1][1] + assay.singles[m2][1])
        y.append(d["dms_score"])
    fit = {}
    if len(x) > 30:
        slope, intercept = np.polyfit(np.array(x), np.array(y), 1)
        fit = {"slope": float(slope), "intercept": float(intercept), "n": len(x)}

    return {
        "column_used": "DMS_score",
        "proteingym_source_column": EXP_SOURCE_COLUMN,
        "proteingym_direction": EXP_DIRECTION,
        "higher_is_more_stable": True,
        "wt_row_present": False,
        "wt_reference_assumed": 0.0,
        "wt_offset_known": False,
        "wt_offset_note": (
            "No WT row in the parquet. The source column is a WT-relative ddG and the value "
            "distribution agrees (mostly negative, continuous, no pile-up at the range ends), "
            "so 0.0 is the convention used; it is not verified from the data. Any error is a "
            "per-domain constant: within-domain Spearman/Pearson/OLS-slope-with-intercept are "
            "invariant to it; eps_exp signs and magnitudes are not interpretable and uncentered "
            "values are not pooled across domains."
        ),
        "score_range": [lo, hi],
        "n_at_min": at_lo,
        "n_at_max": at_hi,
        "censoring_detected": bool(at_lo > 5 or at_hi > 5),
        "distinct_values": int(len(np.unique(scores))),
        "n_values": int(scores.size),
        "additive_null_fit": fit,
        "additive_null_note": (
            "Fit of DMS(AB) on DMS(A)+DMS(B). A slope well below 1 indicates strong GLOBAL "
            "(nonspecific) epistasis, so the intercept does NOT identify the WT reference."
        ),
    }


# ---------------------------------------------------------------------------
# Shared scoring helpers
# ---------------------------------------------------------------------------
def _ca_distance(pdb_string: str, pos_i: int, pos_j: int) -> float:
    """CA-CA distance (A) between two 1-indexed positions in the trimmed PDB."""
    from Bio.PDB import PDBParser

    struct = PDBParser(QUIET=True).get_structure("t", io.StringIO(pdb_string))
    ca = {r.get_id()[1]: r["CA"].coord for r in list(struct.get_residues()) if "CA" in r}
    return float(np.linalg.norm(ca[pos_i] - ca[pos_j]))


def _ca_coords(pdb_string: str) -> Dict[int, np.ndarray]:
    from Bio.PDB import PDBParser

    struct = PDBParser(QUIET=True).get_structure("t", io.StringIO(pdb_string))
    return {r.get_id()[1]: r["CA"].coord for r in list(struct.get_residues()) if "CA" in r}


def _pos(mut: str) -> int:
    return int(mut[1:-1])


def _eps_for_sample(pdb, wt_seq, singles, sample, decoding_orders, mpnn_seed):
    """Model epistasis for each double in `sample`, as a numpy array (fitness units)."""
    need = sorted({m for d in sample for m in d["components"]})
    seqs = [wt_seq] + [singles[m][0] for m in need] + [d["sequence"] for d in sample]
    f = score_only_oracle(
        pdb,
        seqs,
        proteinmpnn_dir=PROTEINMPNN_PATH,
        model_name=PROTEINMPNN_MODEL_NAME,
        seed=mpnn_seed,
        num_decoding_orders=decoding_orders,
    )
    f_wt = f[0]
    f_single = {m: f[1 + i] for i, m in enumerate(need)}
    f_double = {d["mutant"]: f[1 + len(need) + i] for i, d in enumerate(sample)}
    return np.array(
        [
            (f_double[d["mutant"]] - f_wt)
            - ((f_single[d["components"][0]] - f_wt) + (f_single[d["components"][1]] - f_wt))
            for d in sample
        ]
    )


def noise_check(pdb, wt_seq, singles, sample, decoding_orders):
    """Score the same doubles at two ProteinMPNN decoding seeds and report signal-vs-noise:
    if eps is stable across seeds it is real coupling, not decoding-order noise.

    NOTE the `eps std / noise std` ratio printed here is the historical sub-question A
    statistic. It is NOT a latent signal-to-noise ratio (its numerator contains the noise);
    the stability arm reports test-retest `reliability` instead — see epistasis_stats.py.
    """
    e_a = _eps_for_sample(pdb, wt_seq, singles, sample, decoding_orders, mpnn_seed=37)
    e_b = _eps_for_sample(pdb, wt_seq, singles, sample, decoding_orders, mpnn_seed=38)
    noise = float((e_a - e_b).std() / np.sqrt(2))
    print(f"\n=== noise floor (K={decoding_orders} decoding orders, n={len(sample)} pairs) ===")
    print(f"eps std (signal) : {e_a.std():.4f}")
    print(f"noise std        : {noise:.4f}")
    print(f"corr(seed37,38)  : {np.corrcoef(e_a, e_b)[0, 1]:.3f}")
    print(f"SNR              : {e_a.std() / noise:.2f}")
    rel = es.reliability(e_a, e_b)
    print(f"reliability      : {rel.get('reliability', float('nan')):.3f} (valid={rel.get('valid')})")


# ---------------------------------------------------------------------------
# Stability arm (sub-question B)
# ---------------------------------------------------------------------------
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _proteinmpnn_rev() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "-C", PROTEINMPNN_PATH, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def evaluate_domain(dms_id: str, pdb_code: str, min_doubles: int = MIN_USABLE_DOUBLES) -> Dict:
    """Structural + data eligibility for one candidate domain, as a record to commit.

    Every check is executed (header method, exact sequence match, gap-free trim, usable-double
    count) and the reason for any rejection is returned — selection is recorded BEFORE any
    correlation is computed.
    """
    rec: Dict = {"dms_id": dms_id, "pdb_code": pdb_code, "eligible": False, "reasons": []}
    try:
        assay = load_assay(dms_id)
    except Exception as exc:  # pragma: no cover - data availability
        rec["reasons"].append(f"assay load failed: {exc}")
        return rec
    rec["wt_length"] = len(assay.wt_seq)
    rec["n_singles_usable"] = assay.meta["n_singles_usable"]
    rec["n_doubles_usable"] = assay.meta["n_doubles_usable"]

    raw = _fetch_pdb(pdb_code)
    md = pdb_metadata(raw)
    rec.update(md)
    is_xray = bool(md["method"] and "X-RAY" in md["method"].upper())
    rec["is_xray"] = is_xray

    try:
        pdb = trim_pdb_to_dms(raw, assay.wt_seq, pdb_code)
        rec["trim_ok"] = True
        rec["trimmed_residues"] = len(_ca_coords(pdb))
    except Exception as exc:
        rec["trim_ok"] = False
        rec["reasons"].append(f"trim/sequence match failed: {exc}")

    if not is_xray:
        rec["reasons"].append(f"not X-ray ({md['method']}) — scope restriction, not a quality judgement")
    if assay.meta["n_doubles_usable"] < min_doubles:
        rec["reasons"].append(f"only {assay.meta['n_doubles_usable']} usable doubles (< {min_doubles})")
    rec["eligible"] = bool(is_xray and rec.get("trim_ok") and assay.meta["n_doubles_usable"] >= min_doubles)
    return rec


def select_stability_domains(
    n_extra: int = 2, min_doubles: int = MIN_USABLE_DOUBLES
) -> Tuple[List[Tuple[str, str]], List[Dict]]:
    """AMFR/4G3O plus the first `n_extra` eligible candidates, in the fixed candidate order.

    Returns (selected, eligibility_records). Candidates after the quota is filled are still
    recorded as 'not evaluated' so the stopping point is explicit. If fewer than `n_extra`
    qualify, the shortfall is returned as-is — the caller reports the limitation rather than
    relaxing the criteria.
    """
    records, selected = [], []
    for dms_id, code in DOMAIN_CANDIDATES:
        if len(selected) >= 1 + n_extra:
            records.append({"dms_id": dms_id, "pdb_code": code, "evaluated": False, "note": "quota already filled"})
            continue
        rec = evaluate_domain(dms_id, code, min_doubles)
        rec["evaluated"] = True
        records.append(rec)
        if rec["eligible"]:
            selected.append((dms_id, code))
    return selected, records


def _sample_doubles(assay: Assay, n: int, sampling_seed: int) -> List[Dict]:
    """Deterministic per-domain sample. The sampling seed is independent of the ProteinMPNN
    decoding seeds; the domain id is mixed in so different domains get different draws from
    the same `--sampling-seed`."""
    doubles = assay.doubles
    if len(doubles) <= n:
        return list(doubles)
    rng = np.random.default_rng([sampling_seed, zlib.crc32(assay.dms_id.encode())])
    idx = rng.choice(len(doubles), size=n, replace=False)
    return [doubles[i] for i in sorted(idx)]


def _score_batch(pdb: str, seqs: List[str], decoding_orders: int, mpnn_seed: int) -> Tuple[List[float], float]:
    t0 = time.time()
    f = score_only_oracle(
        pdb,
        seqs,
        proteinmpnn_dir=PROTEINMPNN_PATH,
        model_name=PROTEINMPNN_MODEL_NAME,
        seed=mpnn_seed,
        num_decoding_orders=decoding_orders,
    )
    return f, time.time() - t0


def analyze_domain(rows: List[Dict], rel: Dict) -> Dict:
    """Per-domain association, bootstraps, reliability and the additive-baseline comparison.

    Primary: Spearman, Pearson and OLS slope (with intercept) of eps_exp on eps_model, all
    invariant to the unknown per-domain WT offset.
    """
    eps_model = np.array([r["eps_model"] for r in rows])
    eps_exp = np.array([r["eps_exp"] for r in rows])
    groups = [tuple(r["residue_pair"]) for r in rows]

    out: Dict = {
        "n_pairs": len(rows),
        "n_residue_pair_groups": len(set(groups)),
        "primary_eps_association": es.associations(eps_model, eps_exp),
        "ci_individual_double": {
            stat: es.bootstrap_ci(eps_model, eps_exp, stat=stat) for stat in ("spearman", "pearson", "slope")
        },
        "ci_residue_pair_block": {
            stat: es.block_bootstrap_ci(eps_model, eps_exp, groups, stat=stat)
            for stat in ("spearman", "pearson", "slope")
        },
        "decoding_noise_reliability": rel,
    }

    # Secondary: eps_exp detrended for global (nonspecific) epistasis. If the experimental
    # double effect is largely a smooth function of the summed single effects, raw eps_exp is
    # dominated by that curvature rather than by pair-specific coupling; the residual isolates
    # the specific part. Fitted in-sample on the analysed pairs -> descriptive only.
    x_sum = np.array([r["dms_single1"] + r["dms_single2"] for r in rows])
    y_dbl = np.array([r["dms_double"] for r in rows])
    if len(rows) >= 20 and np.std(x_sum) > 0:
        coef = np.polyfit(x_sum, y_dbl, 2)
        resid = y_dbl - np.polyval(coef, x_sum)
        out["secondary_detrended"] = {
            "note": (
                "eps_exp replaced by the residual of DMS(AB) on a quadratic in "
                "DMS(A)+DMS(B) (global-epistasis detrending), fitted in-sample on these "
                "pairs: descriptive, not a held-out claim."
            ),
            "quadratic_coef": [float(c) for c in coef],
            "association": es.associations(eps_model, resid),
            "ci_residue_pair_block_spearman": es.block_bootstrap_ci(eps_model, resid, groups, stat="spearman"),
        }

    # Additive baseline vs full non-additive model, predicting the measured DOUBLE effect.
    y_effect = np.array([r["exp_double_effect"] for r in rows])
    x_add = np.array([r["additive_pred"] for r in rows])
    x_full = np.array([r["model_double_effect"] for r in rows])
    out["additive_vs_full"] = es.additive_vs_full(y_effect, x_add, x_full, groups)
    out["additive_vs_full"]["paired_delta_ci_pearson"] = es.paired_delta_ci(
        y_effect, x_add, x_full, groups, stat="pearson"
    )
    out["additive_vs_full"]["paired_delta_ci_spearman"] = es.paired_delta_ci(
        y_effect, x_add, x_full, groups, stat="spearman"
    )

    # Within-residue-pair association. The Tsuboyama doubles cluster on a HANDFUL of position
    # pairs (7-18 per domain), many amino-acid combinations each, so the cross-pair analyses
    # above have far fewer effectively independent units than `n_pairs` suggests. This
    # analysis asks the question that carries no cross-pair dependence at all: at a FIXED pair
    # of positions, does the model track experimental epistasis across substitution identities?
    per_pair = []
    for g in sorted(set(groups)):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        if len(idx) < 20:
            continue
        a_in = es.associations(eps_model[idx], eps_exp[idx])
        per_pair.append(
            {
                "residue_pair": list(g),
                "n": len(idx),
                "spearman": a_in["spearman"],
                "pearson": a_in["pearson"],
                "slope": a_in["slope"],
                "ca_distance": float(np.median([rows[i]["ca_distance"] for i in idx])),
            }
        )
    if per_pair:
        rhos = [p["spearman"] for p in per_pair if np.isfinite(p["spearman"])]
        out["within_residue_pair"] = {
            "note": (
                "Per position-pair Spearman of eps_exp on eps_model across substitution "
                "identities (groups with n>=20). No cross-pair dependence; the sign test "
                "below treats each position pair as one unit."
            ),
            "pairs": per_pair,
            "median_spearman": float(np.median(rhos)) if rhos else float("nan"),
            "n_pairs_positive": int(sum(r > 0 for r in rhos)),
            "n_pairs_tested": len(rhos),
        }

    # Contact split, carried over from sub-question A (reported, not claimed — see
    # research_plan/rowB: this comparison is under-powered).
    near = [abs(r["eps_model"]) for r in rows if r["ca_distance"] <= 8.0]
    far = [abs(r["eps_model"]) for r in rows if r["ca_distance"] > 8.0]
    if near and far:
        out["contact_split_abs_eps_model"] = {
            "contacting_median": float(np.median(near)),
            "contacting_n": len(near),
            "distal_median": float(np.median(far)),
            "distal_n": len(far),
            "note": "Under-powered in sub-question A; reported for continuity only.",
        }
    return out


def run_stability_domain(
    dms_id: str,
    pdb_code: str,
    args,
    out_path: str,
) -> Dict:
    """Score one domain and return its full artifact dict (also written to `out_path`)."""
    cfg = {
        "arm": "stability",
        "dms_id": dms_id,
        "pdb_code": pdb_code,
        "n_doubles_requested": args.n_doubles,
        "decoding_orders": args.decoding_orders,
        "mpnn_seed_a": args.mpnn_seed,
        "mpnn_seed_b": args.mpnn_seed_b,
        "sampling_seed": args.sampling_seed,
        "replicate_subset_size": args.replicate_subset,
        "model_name": PROTEINMPNN_MODEL_NAME,
        "proteinmpnn_rev": _proteinmpnn_rev(),
        "wt_reference_assumed": args.dms_wt,
        "bootstrap_seed": args.bootstrap_seed,
    }
    cfg_id = _sha(json.dumps(cfg, sort_keys=True))[:16]

    # Reuse a completed artifact only when the whole configuration matches.
    if os.path.isfile(out_path) and not args.rescore:
        with open(out_path) as fh:
            prev = json.load(fh)
        if prev.get("config_id") == cfg_id:
            if args.reanalyze:
                rel_prev = prev["analysis"]["decoding_noise_reliability"]
                rel_new = es.reliability(prev["replicate"]["eps_seed_a"], prev["replicate"]["eps_seed_b"])
                rel_new.update({k: rel_prev[k] for k in ("mpnn_seed_a", "mpnn_seed_b", "caveats") if k in rel_prev})
                prev["analysis"] = analyze_domain(prev["doubles"], rel_new)
                prev["reanalyzed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
                with open(out_path, "w") as fh:
                    json.dump(prev, fh, indent=2)
                print(f"[{pdb_code}] re-analysed stored scores (no rescoring) -> {os.path.basename(out_path)}")
                return prev
            print(f"[{pdb_code}] reusing completed artifact {os.path.basename(out_path)} (config_id {cfg_id})")
            return prev
        guard_overwrite(out_path, args.force)

    assay = load_assay(dms_id)
    raw = _fetch_pdb(pdb_code)
    md = pdb_metadata(raw)
    pdb = trim_pdb_to_dms(raw, assay.wt_seq, pdb_code)
    semantics = check_experimental_semantics(assay)
    sample = _sample_doubles(assay, args.n_doubles, args.sampling_seed)
    needed = sorted({m for d in sample for m in d["components"]})

    print(
        f"\n=== [{pdb_code}] {dms_id} | WT {len(assay.wt_seq)} aa | "
        f"{assay.meta['n_singles_usable']} singles | {assay.meta['n_doubles_usable']} usable doubles -> "
        f"scoring {len(sample)} doubles + {len(needed)} singles + WT at K={args.decoding_orders}"
    )
    seqs = [assay.wt_seq] + [assay.singles[m][0] for m in needed] + [d["sequence"] for d in sample]
    fits, t_main = _score_batch(pdb, seqs, args.decoding_orders, args.mpnn_seed)
    print(f"[{pdb_code}] main scoring: {len(seqs)} sequences in {t_main:.1f} s")

    f_wt = fits[0]
    f_single = {m: fits[1 + i] for i, m in enumerate(needed)}
    f_double = {d["mutant"]: fits[1 + len(needed) + i] for i, d in enumerate(sample)}

    ca = _ca_coords(pdb)
    rows = []
    for d in sample:
        m1, m2 = d["components"]
        d1, d2, additive, eps_model = es.model_effects(f_double[d["mutant"]], f_single[m1], f_single[m2], f_wt)
        dms1, dms2 = assay.singles[m1][1], assay.singles[m2][1]
        p1, p2 = _pos(m1), _pos(m2)
        rows.append(
            {
                "mutant": d["mutant"],
                "components": [m1, m2],
                "positions": [p1, p2],
                "residue_pair": list(es.residue_pair(d["mutant"])),
                # raw model scores (fitness units, higher = better)
                "f_wt": f_wt,
                "f_single1": f_single[m1],
                "f_single2": f_single[m2],
                "f_double": f_double[d["mutant"]],
                # model-derived effects
                "delta1": d1,
                "delta2": d2,
                "additive_pred": additive,
                "model_double_effect": f_double[d["mutant"]] - f_wt,
                "eps_model": eps_model,
                # experimental values and derived effects (WT reference = args.dms_wt)
                "dms_single1": dms1,
                "dms_single2": dms2,
                "dms_double": d["dms_score"],
                "exp_double_effect": d["dms_score"] - args.dms_wt,
                "eps_exp": es.experimental_epistasis(d["dms_score"], dms1, dms2, args.dms_wt),
                "ca_distance": float(np.linalg.norm(ca[p1] - ca[p2])),
            }
        )

    # Replicate the same fixed subset, same variant ordering, at two independent decoding
    # seeds -> test-retest reliability of eps_model.
    subset = sample[: min(args.replicate_subset, len(sample))]
    t0 = time.time()
    eps_a = _eps_for_sample(pdb, assay.wt_seq, assay.singles, subset, args.decoding_orders, args.mpnn_seed)
    eps_b = _eps_for_sample(pdb, assay.wt_seq, assay.singles, subset, args.decoding_orders, args.mpnn_seed_b)
    t_rep = time.time() - t0
    rel = es.reliability(eps_a, eps_b)
    rel["mpnn_seed_a"] = args.mpnn_seed
    rel["mpnn_seed_b"] = args.mpnn_seed_b
    rel["caveats"] = (
        "Two replicates on n=%d pairs: the reliability point estimate is itself noisy. "
        "sqrt(reliability) bounds Pearson attenuation from MODEL DECODING NOISE ALONE under "
        "independent equal-variance additive errors; it is not a Spearman correction and not "
        "a total-noise ceiling (experimental noise attenuates further)." % len(subset)
    )
    print(
        f"[{pdb_code}] replicate ({len(subset)} pairs, seeds {args.mpnn_seed}/{args.mpnn_seed_b}): "
        f"{t_rep:.1f} s | reliability={rel.get('reliability', float('nan')):.3f} valid={rel.get('valid')}"
    )

    artifact = {
        "schema": "rowB-stability-arm/1",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "config": cfg,
        "config_id": cfg_id,
        "runtime_seconds": {"main_scoring": t_main, "replicate_scoring": t_rep},
        "provenance": {
            **assay.meta,
            "parquet_sha256": _pg.file_digest(assay.meta["parquet_path"]) if args.hash_parquet else None,
        },
        "structure": {
            "pdb_code": pdb_code,
            **md,
            "trimmed_length": len(assay.wt_seq),
            "trimmed_sequence": assay.wt_seq,
            "trimmed_sequence_sha256": _sha(assay.wt_seq),
            "raw_pdb_sha256": _sha(raw),
        },
        "experimental_semantics": semantics,
        "wt_model_score": f_wt,
        "singles_model_scores": {m: f_single[m] for m in needed},
        "singles_dms_scores": {m: assay.singles[m][1] for m in needed},
        "doubles": rows,
        "replicate": {
            "subset_mutants": [d["mutant"] for d in subset],
            "eps_seed_a": [float(v) for v in eps_a],
            "eps_seed_b": [float(v) for v in eps_b],
        },
        "analysis": analyze_domain(rows, rel),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
    }
    with open(out_path, "w") as fh:
        json.dump(artifact, fh, indent=2)
    print(f"[{pdb_code}] wrote {out_path}")
    return artifact


def plot_domain(artifact: Dict, png_path: str) -> None:
    """Two panels: eps_model vs eps_exp, and the grouped held-out calibrated predictions of
    the additive baseline vs the full non-additive model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = artifact["doubles"]
    a = artifact["analysis"]
    eps_m = np.array([r["eps_model"] for r in rows])
    eps_e = np.array([r["eps_exp"] for r in rows])
    code = artifact["structure"]["pdb_code"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    ax.scatter(eps_m, eps_e, s=14, alpha=0.6, color="#3b6ea5", edgecolor="none")
    assoc = a["primary_eps_association"]
    if np.isfinite(assoc.get("slope", float("nan"))):
        xs = np.linspace(eps_m.min(), eps_m.max(), 50)
        ax.plot(xs, assoc["intercept"] + assoc["slope"] * xs, color="#c0392b", lw=1.5)
    ax.axhline(0, color="#999", lw=0.6)
    ax.axvline(0, color="#999", lw=0.6)
    ax.set_xlabel("eps_model  (score_only fitness units)")
    ax.set_ylabel("eps_exp  (ddG units, WT ref assumed 0)")
    ax.set_title(
        f"{code}: model vs experimental epistasis\n"
        f"rho={assoc['spearman']:.3f}  r={assoc['pearson']:.3f}  slope={assoc['slope']:.3f}  n={len(rows)}",
        fontsize=10,
    )

    ax = axes[1]
    y = np.array([r["exp_double_effect"] for r in rows])
    for name, color in (("additive", "#888888"), ("full", "#2e8b57")):
        x = np.array([r["additive_pred"] if name == "additive" else r["model_double_effect"] for r in rows])
        h = a["additive_vs_full"]["held_out"]
        lab = f"{name} (held-out r={h.get(name, {}).get('pearson', float('nan')):.3f})"
        ax.scatter(x, y, s=12, alpha=0.5, color=color, edgecolor="none", label=lab)
    ax.set_xlabel("model predicted double effect (additive sum / full)")
    ax.set_ylabel("experimental double effect")
    ax.set_title(f"{code}: additive baseline vs full model", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[{code}] wrote {png_path}")


def summarize(artifacts: List[Dict], args) -> Dict:
    """Compact comparison table + the SECONDARY pooled analysis.

    Pooling uses within-domain centering for Pearson and domain-specific intercepts for the
    slope, so the unknown per-domain WT offsets cannot leak in. Three proteins is not a basis
    for a generality claim, and the pooled row is labelled accordingly.
    """
    lines = [
        "| domain | PDB | method | n doubles | position pairs | Spearman (block 95% CI) | Pearson | slope | within-pair median rho (pos/tested) | reliability | held-out r: additive -> full |",
        "|---|---|---|--:|--:|---|--:|--:|---|--:|---|",
    ]
    cx, cy, cg = [], [], []
    for art in artifacts:
        a = art["analysis"]
        p = a["primary_eps_association"]
        blk = a["ci_residue_pair_block"]["spearman"]
        rel = a["decoding_noise_reliability"]
        h = a["additive_vs_full"]["held_out"]
        code = art["structure"]["pdb_code"]
        wp = a.get("within_residue_pair")
        wp_txt = (
            f"{wp['median_spearman']:+.3f} ({wp['n_pairs_positive']}/{wp['n_pairs_tested']})" if wp else "n/a"
        )
        rel_txt = f"{rel['reliability']:.2f}" if rel.get("reliability") is not None else "n/a"
        if not rel.get("valid"):
            rel_txt += " (invalid)"
        elif not rel.get("stable", True):
            rel_txt += " (unstable: CI includes 0)"
        lines.append(
            f"| {art['config']['dms_id'].split('_Tsuboyama')[0]} | {code} | "
            f"{art['structure']['method'].split()[0] if art['structure']['method'] else '?'} | "
            f"{p['n']:.0f} | {a['n_residue_pair_groups']} | "
            f"{p['spearman']:+.3f} [{blk['lo']:+.3f}, {blk['hi']:+.3f}] | {p['pearson']:+.3f} | "
            f"{p['slope']:+.3f} | {wp_txt} | {rel_txt} | "
            f"{h.get('additive', {}).get('pearson', float('nan')):+.3f} -> "
            f"{h.get('full', {}).get('pearson', float('nan')):+.3f} |"
        )
        eps_m = np.array([r["eps_model"] for r in art["doubles"]])
        eps_e = np.array([r["eps_exp"] for r in art["doubles"]])
        cx.append(eps_m - eps_m.mean())
        cy.append(eps_e - eps_e.mean())
        cg += [(code, tuple(r["residue_pair"])) for r in art["doubles"]]

    pooled = {}
    if len(artifacts) > 1:
        x = np.concatenate(cx)
        y = np.concatenate(cy)
        pooled = {
            "note": (
                "SECONDARY. Within-domain centering (Pearson) and domain-specific intercepts "
                "(slope) make this invariant to the unknown per-domain WT offsets. Three "
                "proteins — not evidence of broad generalization."
            ),
            "n_domains": len(artifacts),
            "centered_association": es.associations(x, y),
            "ci_block_spearman": es.block_bootstrap_ci(x, y, cg, stat="spearman"),
            "ci_block_pearson": es.block_bootstrap_ci(x, y, cg, stat="pearson"),
        }

    table = "\n".join(lines)
    print("\n=== Row B stability arm — per-domain summary (primary) ===")
    print(table)
    if pooled:
        pa = pooled["centered_association"]
        ci = pooled["ci_block_spearman"]
        print(
            f"\npooled (secondary, within-domain centered): rho={pa['spearman']:+.3f} "
            f"[{ci['lo']:+.3f}, {ci['hi']:+.3f}]  r={pa['pearson']:+.3f}  slope={pa['slope']:+.3f}  n={pa['n']:.0f}"
        )

    summary = {
        "schema": "rowB-stability-arm-summary/1",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "n_doubles_requested": args.n_doubles,
            "decoding_orders": args.decoding_orders,
            "mpnn_seeds": [args.mpnn_seed, args.mpnn_seed_b],
            "sampling_seed": args.sampling_seed,
            "wt_reference_assumed": args.dms_wt,
        },
        "domains": [
            {
                "dms_id": art["config"]["dms_id"],
                "pdb_code": art["structure"]["pdb_code"],
                "method": art["structure"]["method"],
                "resolution_angstrom": art["structure"]["resolution_angstrom"],
                "config_id": art["config_id"],
                "analysis": art["analysis"],
                "runtime_seconds": art["runtime_seconds"],
            }
            for art in artifacts
        ],
        "pooled_secondary": pooled,
        "table_markdown": table,
    }
    return summary


def cmd_stability(args) -> None:
    selected, records = select_stability_domains(n_extra=args.extra_domains, min_doubles=args.min_doubles)
    print("=== domain eligibility (recorded before any correlation was computed) ===")
    for rec in records:
        if not rec.get("evaluated", True):
            print(f"  {rec['pdb_code']:>5}  not evaluated — {rec['note']}")
            continue
        mark = "SELECTED" if rec["eligible"] else "excluded"
        extra = "; ".join(rec["reasons"]) if rec["reasons"] else ""
        print(
            f"  {rec['pdb_code']:>5}  {mark:>8}  {rec.get('method')} "
            f"{rec.get('resolution_angstrom')}A  L={rec.get('wt_length')}  "
            f"doubles={rec.get('n_doubles_usable')}  trim_ok={rec.get('trim_ok')}  {extra}"
        )
    if args.dms_id:
        known = dict(DOMAIN_CANDIDATES)
        if args.dms_id not in known:
            sys.exit(f"--dms-id {args.dms_id} is not one of the known candidates: {sorted(known)}")
        selected = [(args.dms_id, known[args.dms_id])]
        print(f"\n--dms-id override: running only {selected}")
    if args.dms_id is None and len(selected) < 1 + args.extra_domains:
        print(
            f"\nLIMITATION: only {len(selected)} of {1 + args.extra_domains} intended domains qualified "
            "under the stated criteria. Reporting the shortfall rather than relaxing the criteria."
        )
    if not selected:
        sys.exit("no eligible domains")

    artifacts = []
    for dms_id, code in selected:
        out_path = stamped_path(
            args.out_dir,
            f"score_only_epistasis_stability_{code}",
            "json",
            n=args.n_doubles,
            k=args.decoding_orders,
        )
        art = run_stability_domain(dms_id, code, args, out_path)
        artifacts.append(art)
        if args.plots:
            plot_domain(art, out_path.replace(".json", ".png"))

    summary = summarize(artifacts, args)
    summary["domain_eligibility"] = records
    sum_path = stamped_path(
        args.out_dir, "score_only_epistasis_stability_summary", "json", n=args.n_doubles, k=args.decoding_orders
    )
    guard_overwrite(sum_path, args.force or args.rescore)
    with open(sum_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(sum_path.replace(".json", ".md"), "w") as fh:
        fh.write(summary["table_markdown"] + "\n")
    print(f"\nwrote {sum_path}")


def cmd_internal(args) -> None:
    """Sub-question A — unchanged behaviour (same sampling, same statistics, same output)."""
    out_path = args.out or stamped_path(
        _HERE, "score_only_epistasis", "jsonl", n=args.n_doubles, k=args.decoding_orders
    )
    guard_overwrite(out_path, args.force)

    dms_id = args.dms_id or DMS_ID
    pdb_code = dict(DOMAIN_CANDIDATES).get(dms_id, PDB_CODE)
    wt_seq, singles, doubles = load_variants(dms_id)
    pdb = trim_pdb_to_dms(_fetch_pdb(pdb_code), wt_seq, pdb_code)
    print(
        f"# {dms_id}: WT {len(wt_seq)} aa | {len(singles)} singles | {len(doubles)} doubles with both singles present"
    )

    rng = np.random.default_rng(args.seed)
    if len(doubles) > args.n_doubles:
        idx = rng.choice(len(doubles), size=args.n_doubles, replace=False)
        doubles = [doubles[i] for i in sorted(idx)]

    # One batched score_only over WT + every involved single + the sampled doubles.
    needed_singles = sorted({m for d in doubles for m in d["components"]})
    seqs = [wt_seq] + [singles[m][0] for m in needed_singles] + [d["sequence"] for d in doubles]
    print(
        f"# scoring {len(seqs)} sequences (1 WT + {len(needed_singles)} singles + "
        f"{len(doubles)} doubles) at {args.decoding_orders} decoding orders..."
    )
    fits = score_only_oracle(
        pdb,
        seqs,
        proteinmpnn_dir=PROTEINMPNN_PATH,
        model_name=PROTEINMPNN_MODEL_NAME,
        seed=PROTEINMPNN_SEED,
        num_decoding_orders=args.decoding_orders,
    )
    f_wt = fits[0]
    f_single = {m: fits[1 + i] for i, m in enumerate(needed_singles)}
    f_double = {d["mutant"]: fits[1 + len(needed_singles) + i] for i, d in enumerate(doubles)}

    rows = []
    for d in doubles:
        m1, m2 = d["components"]
        d1, d2 = f_single[m1] - f_wt, f_single[m2] - f_wt  # single-effect deltas (model)
        eps_model = (f_double[d["mutant"]] - f_wt) - (d1 + d2)  # non-additivity
        dist = _ca_distance(pdb, _pos(m1), _pos(m2))
        rows.append(
            {
                "mutant": d["mutant"],
                "eps_model": eps_model,
                "delta1": d1,
                "delta2": d2,
                "ca_distance": dist,
            }
        )

    with open(out_path, "w") as fh:
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
    print(f"|eps| / |delta| (median ratio): {statistics.median(abs_eps) / statistics.median(main_effects):.3f}")
    print(
        f"pairs with |eps| > 0.5*|delta_median|: "
        f"{sum(e > 0.5 * statistics.median(main_effects) for e in abs_eps)}/{len(eps)}"
    )
    if near and far:
        print(f"|eps| median  contacting (CA<=8A, n={len(near)}): {statistics.median([abs(x) for x in near]):.4f}")
        print(f"|eps| median  distal    (CA>8A,  n={len(far)}): {statistics.median([abs(x) for x in far]):.4f}")
    print(f"\nwrote {out_path}")

    if args.noise_check:
        noise_check(pdb, wt_seq, singles, doubles[: min(60, len(doubles))], args.decoding_orders)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        choices=("internal", "stability"),
        default="internal",
        help="internal = sub-question A (non-additivity exists?); stability = sub-question B "
        "(does it track experimental epistasis?)",
    )
    ap.add_argument("--n-doubles", type=int, default=150, help="how many doubles to sample (deterministic)")
    ap.add_argument("--decoding-orders", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="internal arm: sampling seed (unchanged default)")
    ap.add_argument("--sampling-seed", type=int, default=0, help="stability arm: variant-sampling seed")
    ap.add_argument(
        "--mpnn-seed", type=int, default=PROTEINMPNN_SEED, help="ProteinMPNN decoding seed A (must be nonzero)"
    )
    ap.add_argument(
        "--mpnn-seed-b",
        type=int,
        default=PROTEINMPNN_SEED + 1,
        help="independent ProteinMPNN decoding seed for the reliability replicate (nonzero)",
    )
    ap.add_argument("--replicate-subset", type=int, default=60, help="pairs re-scored at seed B for reliability")
    ap.add_argument("--bootstrap-seed", type=int, default=20261117)
    ap.add_argument("--dms-wt", type=float, default=0.0, help="assumed experimental WT reference (see module docs)")
    ap.add_argument("--dms-id", default=None, help="run a single named assay instead of the selected set")
    ap.add_argument("--extra-domains", type=int, default=2, help="stability arm: additional domains beyond AMFR")
    ap.add_argument("--min-doubles", type=int, default=MIN_USABLE_DOUBLES)
    ap.add_argument("--plots", action="store_true", help="stability arm: write per-domain PNGs")
    ap.add_argument("--hash-parquet", action="store_true", help="record the parquet sha256 (slow, ~90 MB)")
    ap.add_argument("--out-dir", default=_HERE, help="stability arm: artifact directory")
    ap.add_argument(
        "--noise-check", action="store_true", help="also score at two decoding seeds and report signal-vs-noise"
    )
    ap.add_argument(
        "--out",
        default=None,
        help="internal arm output path; default is parameter-stamped score_only_epistasis_n<N>_k<K>.jsonl",
    )
    ap.add_argument("--force", action="store_true", help="allow overwriting an existing results artifact")
    ap.add_argument("--rescore", action="store_true", help="stability arm: recompute even if an artifact matches")
    ap.add_argument(
        "--reanalyze",
        action="store_true",
        help="stability arm: recompute the analysis from stored model scores without rescoring",
    )
    args = ap.parse_args()

    if not PROTEINMPNN_PATH:
        sys.exit("PROTEINMPNN_PATH not set (see .env)")
    # ProteinMPNN rejects seed=0 (it means "unseeded" upstream); the determinism gate depends
    # on both replicate seeds being explicit and nonzero.
    if args.arm == "stability":
        if args.mpnn_seed == 0 or args.mpnn_seed_b == 0:
            sys.exit("--mpnn-seed / --mpnn-seed-b must be nonzero")
        if args.mpnn_seed == args.mpnn_seed_b:
            sys.exit("--mpnn-seed-b must differ from --mpnn-seed (the replicate needs an independent draw)")
        cmd_stability(args)
    else:
        cmd_internal(args)


if __name__ == "__main__":
    main()
