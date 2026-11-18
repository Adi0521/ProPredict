"""Diagnose ProteinMPNN `score_only` sensitivity to BATCH COMPOSITION at a fixed seed.

Why this exists
---------------
`Process/rowB-step2-stability-arm.md` recorded, in passing, that `score_only` scores are
"not invariant to batch composition" and quoted correlations of 0.28/0.55/0.92. Those
numbers are **not reproducible from any committed artifact**: what the stability-arm
artifacts store is `replicate.eps_seed_a` vs `eps_seed_b` at mpnn seeds 37 vs 38 (r =
0.931/0.063/0.379), which is cross-*seed* reliability of an epistasis CONTRAST, not
fixed-seed cross-invocation variation of a raw score. The mechanism is real (traced below)
but its magnitude was never measured. This module measures it.

The mechanism, traced through the pinned clone (8907e66)
--------------------------------------------------------
`protein_mpnn_run.py:29` calls `torch.manual_seed(seed)` ONCE per process. The score_only
branch then loops `for fc in range(1+loop_c)` over the fasta sequences (`:244`) and, inside
each, `for j in range(NUM_BATCHES)` draws `randn_1 = torch.randn(chain_M.shape)` (`:256`)
from that one global stream. `randn_1` is the decoding-order draw. So with `--batch_size 1`,
fasta sequence k (0 = the native-PDB pass) consumes stream draws [K*k .. K*k+K-1]:

  * identical fasta CONTENT AND ORDER  -> identical draws  -> bit-identical scores;
  * a sequence moved to another index  -> different draws  -> different score.

Decoding order is the ONLY stochastic input here: `--backbone_noise` defaults to 0.00, so
`augment_eps=0` and no coordinate jitter is applied. `--batch_size 1` means no batch-dim
reduction nondeterminism either.

Two properties that must not be conflated, and that this module reports separately:
  * EXACT REPRODUCIBILITY under identical inputs (expected to hold);
  * INVARIANCE TO BATCH COMPOSITION (expected to fail).

Conditions
----------
All hold model, backbone, K (decoding orders), seed and device fixed; only the composition
of the scored batch varies.

  repeat        the identical panel, identical order, scored twice
  reorder       the panel deterministically shuffled, scores mapped back by sequence
  insert        unrelated sequences (not in the panel) spliced in at fixed positions
  chunked       the panel split into N equal chunks, one invocation each
  individual    one invocation per sequence (every sequence lands at index 1 -> all share
                the SAME decoding orders; this is the shared-draw condition)
  stream_offset one fixed probe sequence scored at index 1..M of otherwise-identical
                batches, isolating stream-offset dependence from everything else

Metrics
-------
Correlation alone is insufficient (a tight rank correlation can still hide score shifts that
swamp an epistasis contrast), so each condition reports absolute score deltas, absolute
epistasis deltas, Spearman on both, and top-candidate overlap.

Run
---
    python -m benchmarks.diagnose_score_only_batching --k 8 --n-doubles 30
    python -m benchmarks.diagnose_score_only_batching --k 32 --out-dir benchmarks
"""

import argparse
import datetime
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks._artifacts import guard_overwrite, stamped_path
from benchmarks.benchmark_score_only_epistasis import (
    DOMAIN_CANDIDATES,
    _fetch_pdb,
    _proteinmpnn_rev,
    _score_batch,
    load_assay,
    trim_pdb_to_dms,
)
from config import PROTEINMPNN_MODEL_NAME, PROTEINMPNN_SEED

_HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------


class Panel:
    """WT + singles + doubles for one domain, with the index bookkeeping to turn a
    sequence->score mapping into per-double epistasis.

    `names` are stable labels ("WT", "A12V", "A12V:L30P"); `seqs[i]` is the sequence for
    `names[i]`. Sequences are unique by construction (duplicates would make a
    sequence-keyed score mapping ambiguous)."""

    def __init__(self, dms_id: str, pdb_code: str, n_doubles: int, seed: int):
        assay = load_assay(dms_id)
        raw = _fetch_pdb(pdb_code)
        self.pdb = trim_pdb_to_dms(raw, assay.wt_seq, pdb_code)
        self.dms_id = dms_id
        self.pdb_code = pdb_code
        self.wt_seq = assay.wt_seq

        rng = np.random.default_rng(seed)
        # Doubles drive the panel; the singles are then exactly the constituents those
        # doubles need. (Capping singles independently starves the panel: the cap drops
        # doubles whose partner fell outside it, so a 12-single cap left only 3 doubles.)
        doubles = list(assay.doubles)
        if len(doubles) > n_doubles:
            idx = rng.choice(len(doubles), size=n_doubles, replace=False)
            doubles = [doubles[i] for i in sorted(idx)]
        needed = sorted({m for d in doubles for m in d["components"]})
        if not doubles:
            raise SystemExit(f"{dms_id} has no usable doubles to build a panel from")

        self.names: List[str] = ["WT"] + needed + [d["mutant"] for d in doubles]
        self.seqs: List[str] = [assay.wt_seq] + [assay.singles[m][0] for m in needed] + [d["sequence"] for d in doubles]
        if len(set(self.seqs)) != len(self.seqs):
            raise SystemExit("panel contains duplicate sequences — cannot key scores by sequence")
        self.doubles = [{"mutant": d["mutant"], "components": list(d["components"])} for d in doubles]
        self.n_singles = len(needed)

        # Decoys: sequences NOT in the panel, used by the `insert` condition. Drawn from
        # unsampled doubles so they are the same length and the same kind of input.
        pool = [d for d in assay.doubles if d["mutant"] not in {x["mutant"] for x in self.doubles}]
        n_decoy = min(3, len(pool))
        didx = rng.choice(len(pool), size=n_decoy, replace=False) if pool else []
        self.decoys: List[str] = [pool[int(i)]["sequence"] for i in sorted(didx)]

    def epistasis(self, by_seq: Dict[str, float]) -> np.ndarray:
        """eps = f(AB) - f(A) - f(B) + f(WT), one per double, in `self.doubles` order."""
        f_wt = by_seq[self.wt_seq]
        out = []
        for d in self.doubles:
            i = self.names.index(d["mutant"])
            a = self.names.index(d["components"][0])
            b = self.names.index(d["components"][1])
            out.append(by_seq[self.seqs[i]] - by_seq[self.seqs[a]] - by_seq[self.seqs[b]] + f_wt)
        return np.array(out)


# ---------------------------------------------------------------------------
# Scoring conditions — each returns {sequence: fitness} for the whole panel
# ---------------------------------------------------------------------------


def _score_map(pdb: str, seqs: List[str], k: int, seed: int) -> Tuple[Dict[str, float], float, int]:
    """Score `seqs` in ONE invocation; return (seq->fitness, seconds, n_sequences_scored)."""
    fits, dt = _score_batch(pdb, seqs, k, seed)
    return dict(zip(seqs, fits, strict=True)), dt, len(seqs)


def cond_same(panel: Panel, k: int, seed: int) -> Tuple[Dict[str, float], float, int]:
    return _score_map(panel.pdb, panel.seqs, k, seed)


def cond_reorder(panel: Panel, k: int, seed: int, shuffle_seed: int) -> Tuple[Dict[str, float], float, int]:
    order = np.random.default_rng(shuffle_seed).permutation(len(panel.seqs))
    shuffled = [panel.seqs[int(i)] for i in order]
    return _score_map(panel.pdb, shuffled, k, seed)


def cond_insert(panel: Panel, k: int, seed: int) -> Tuple[Dict[str, float], float, int]:
    """Splice decoys in at fixed positions; return scores for the PANEL members only."""
    if not panel.decoys:
        raise SystemExit("no decoy sequences available for the `insert` condition")
    seqs = list(panel.seqs)
    for offset, decoy in enumerate(panel.decoys):
        # Spread the insertions so the shift accumulates across the panel rather than
        # displacing only its tail.
        at = min(len(seqs), 1 + offset * (1 + len(seqs) // (len(panel.decoys) + 1)))
        seqs.insert(at, decoy)
    by_seq, dt, n = _score_map(panel.pdb, seqs, k, seed)
    return {s: by_seq[s] for s in panel.seqs}, dt, n


def cond_chunked(panel: Panel, k: int, seed: int, n_chunks: int) -> Tuple[Dict[str, float], float, int]:
    """Split the panel across `n_chunks` invocations (a batch-SIZE change, in the sense the
    search actually varies: how many candidates ride in one invocation)."""
    bounds = np.array_split(np.arange(len(panel.seqs)), n_chunks)
    out: Dict[str, float] = {}
    total_dt = 0.0
    total_n = 0
    for chunk in bounds:
        seqs = [panel.seqs[int(i)] for i in chunk]
        if not seqs:
            continue
        by_seq, dt, n = _score_map(panel.pdb, seqs, k, seed)
        out.update(by_seq)
        total_dt += dt
        total_n += n
    return out, total_dt, total_n


def cond_individual(panel: Panel, k: int, seed: int) -> Tuple[Dict[str, float], float, int]:
    """One invocation per sequence. Every sequence therefore lands at fasta index 1 and
    consumes the SAME stream draws — i.e. all sequences share one set of decoding orders."""
    return cond_chunked(panel, k, seed, n_chunks=len(panel.seqs))


def cond_stream_offset(panel: Panel, k: int, seed: int, max_offset: int) -> Dict:
    """Score one fixed probe sequence at fasta index 1..max_offset, padding before it with
    distinct filler sequences, and report the spread of ITS OWN score.

    This isolates stream-offset dependence: the probe, the backbone, K and the seed are all
    identical across invocations; only how many draws were consumed before it differs."""
    probe = panel.seqs[-1]  # a double — the quantity the search actually ranks
    filler = [s for s in panel.seqs if s != probe]
    scores = []
    total_dt = 0.0
    total_n = 0
    for offset in range(1, max_offset + 1):
        if offset - 1 > len(filler):
            break
        seqs = filler[: offset - 1] + [probe]
        by_seq, dt, n = _score_map(panel.pdb, seqs, k, seed)
        scores.append(by_seq[probe])
        total_dt += dt
        total_n += n
    arr = np.array(scores)
    return {
        "probe_is_double": panel.names[-1],
        "offsets": list(range(1, len(scores) + 1)),
        "scores": [float(x) for x in arr],
        "min": float(arr.min()),
        "max": float(arr.max()),
        "range": float(arr.max() - arr.min()),
        "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "seconds": round(total_dt, 1),
        "sequences_scored": total_n,
    }


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------


def shared_draw_benefit(panel: Panel, k: int, seed_a: int, seed_b: int) -> Dict:
    """Does sharing decoding orders across the cycle reduce epistasis variance?

    `eps = f(AB) - f(A) - f(B) + f(WT)` is a four-score contrast. Under the upstream
    protocol each of those four scores sits at a different RNG stream offset, so each
    carries an INDEPENDENT decoding-order perturbation and the contrast accumulates all
    four. If instead every sequence in the cycle is scored against the SAME K decoding
    orders, the order-dependent component is common to all four terms and largely cancels
    in the difference — classic common random numbers.

    The `individual` condition realises shared draws exactly (every sequence lands at fasta
    index 1, so all consume stream draws [K..2K-1]), which is why it is used here as the
    shared-draw stand-in. It is diagnostic, not the proposed implementation: it costs one
    model load per sequence.

    Measured as cross-SEED reliability of eps — re-draw the orders (new seed) and see how
    much of the per-double eps ordering survives. Higher = the contrast is measuring
    coupling rather than draw noise. Both protocols estimate the SAME per-sequence quantity
    (an unbiased mean over K orders drawn from the same distribution); only the correlation
    structure of their errors differs."""
    from scipy.stats import pearsonr, spearmanr

    out: Dict[str, Dict] = {}
    for label, fn in (
        ("batched_v1", lambda s: cond_same(panel, k, s)),
        ("shared_draws", lambda s: cond_individual(panel, k, s)),
    ):
        a, dt_a, n_a = fn(seed_a)
        b, dt_b, n_b = fn(seed_b)
        ea, eb = panel.epistasis(a), panel.epistasis(b)
        diff = ea - eb
        noise_var = float(diff.var(ddof=1) / 2.0)
        observed_var = float((ea.var(ddof=1) + eb.var(ddof=1)) / 2.0)
        rel = 1.0 - noise_var / observed_var if observed_var > 0 else float("nan")
        out[label] = {
            "mpnn_seed_a": seed_a,
            "mpnn_seed_b": seed_b,
            "n_doubles": len(ea),
            "eps_pearson_across_seeds": float(pearsonr(ea, eb).statistic),
            "eps_spearman_across_seeds": float(spearmanr(ea, eb).statistic),
            "eps_noise_sd": float(np.sqrt(noise_var)),
            "eps_observed_sd": float(np.sqrt(observed_var)),
            "reliability": float(rel),
            "seconds": round(dt_a + dt_b, 1),
            "sequences_scored": n_a + n_b,
        }
        print(
            f"[shared_draw_benefit/{label}] reliability={rel:.3f} "
            f"r={out[label]['eps_pearson_across_seeds']:.3f} "
            f"noise_sd={out[label]['eps_noise_sd']:.4f} "
            f"observed_sd={out[label]['eps_observed_sd']:.4f} ({dt_a + dt_b:.1f}s)"
        )
    return out


def _spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 3:
        return None
    from scipy.stats import spearmanr

    r = spearmanr(x, y).statistic
    return None if np.isnan(r) else float(r)


def compare(panel: Panel, ref: Dict[str, float], alt: Dict[str, float], top_k: int = 5) -> Dict:
    """Reference-vs-condition metrics on raw scores, on epistasis, and on ranking."""
    seqs = panel.seqs
    a = np.array([ref[s] for s in seqs])
    b = np.array([alt[s] for s in seqs])
    d = b - a
    exact = bool(np.array_equal(a, b))

    ea, eb = panel.epistasis(ref), panel.epistasis(alt)
    ed = eb - ea

    # Top-candidate overlap over the DOUBLES only — what a search would actually select.
    dbl = [panel.seqs[panel.names.index(x["mutant"])] for x in panel.doubles]
    kk = min(top_k, len(dbl))
    top_a = {s for s in sorted(dbl, key=lambda s: -ref[s])[:kk]}
    top_b = {s for s in sorted(dbl, key=lambda s: -alt[s])[:kk]}

    return {
        "exact_match": exact,
        "n_sequences": len(seqs),
        "score_abs_delta_mean": float(np.abs(d).mean()),
        "score_abs_delta_max": float(np.abs(d).max()),
        "score_delta_sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
        "score_ref_sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "score_spearman": _spearman(a, b),
        "n_doubles": len(ea),
        "eps_abs_delta_mean": float(np.abs(ed).mean()),
        "eps_abs_delta_max": float(np.abs(ed).max()),
        "eps_ref_sd": float(ea.std(ddof=1)) if len(ea) > 1 else 0.0,
        "eps_spearman": _spearman(ea, eb),
        "eps_pearson": (float(np.corrcoef(ea, eb)[0, 1]) if len(ea) > 2 else None),
        f"top{kk}_overlap": len(top_a & top_b),
        f"top{kk}_of": kk,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(args) -> Dict:
    code_by_id = dict(DOMAIN_CANDIDATES)
    if args.dms_id not in code_by_id:
        raise SystemExit(f"unknown --dms-id {args.dms_id!r}; known: {sorted(code_by_id)}")
    pdb_code = code_by_id[args.dms_id]

    panel = Panel(args.dms_id, pdb_code, args.n_doubles, args.panel_seed)
    print(
        f"=== panel: {pdb_code} {args.dms_id} | WT {len(panel.wt_seq)} aa | "
        f"1 WT + {panel.n_singles} singles + {len(panel.doubles)} doubles = {len(panel.seqs)} seqs | "
        f"K={args.k} seed={args.seed} ==="
    )

    t0 = time.time()
    results: Dict[str, Dict] = {}
    cost: Dict[str, Dict] = {}

    ref, dt, n = cond_same(panel, args.k, args.seed)
    cost["reference"] = {"invocations": 1, "sequences_scored": n, "seconds": round(dt, 1)}
    print(f"[reference] {n} seqs in {dt:.1f}s")

    conditions = [
        ("repeat", lambda: cond_same(panel, args.k, args.seed), 1),
        ("reorder", lambda: cond_reorder(panel, args.k, args.seed, args.shuffle_seed), 1),
        ("insert", lambda: cond_insert(panel, args.k, args.seed), 1),
        ("chunked", lambda: cond_chunked(panel, args.k, args.seed, args.chunks), args.chunks),
        ("individual", lambda: cond_individual(panel, args.k, args.seed), len(panel.seqs)),
    ]
    for name, fn, n_inv in conditions:
        alt, dt, n = fn()
        results[name] = compare(panel, ref, alt, args.top_k)
        cost[name] = {"invocations": n_inv, "sequences_scored": n, "seconds": round(dt, 1)}
        r = results[name]
        print(
            f"[{name}] exact={r['exact_match']} "
            f"|dscore| mean={r['score_abs_delta_mean']:.4f} max={r['score_abs_delta_max']:.4f} "
            f"(ref sd {r['score_ref_sd']:.4f}) | |deps| mean={r['eps_abs_delta_mean']:.4f} "
            f"(ref sd {r['eps_ref_sd']:.4f}) | rho_score={r['score_spearman']} "
            f"rho_eps={r['eps_spearman']} | top{args.top_k} overlap "
            f"{r[f'top{min(args.top_k, len(panel.doubles))}_overlap']}"
            f"/{r[f'top{min(args.top_k, len(panel.doubles))}_of']} | {dt:.1f}s"
        )

    so = cond_stream_offset(panel, args.k, args.seed, args.max_offset)
    cost["stream_offset"] = {
        "invocations": len(so["offsets"]),
        "sequences_scored": so["sequences_scored"],
        "seconds": so["seconds"],
    }
    print(
        f"[stream_offset] probe {so['probe_is_double']} scored at fasta index "
        f"{so['offsets'][0]}..{so['offsets'][-1]}: range={so['range']:.4f} sd={so['sd']:.4f}"
    )

    sdb = shared_draw_benefit(panel, args.k, args.seed, args.seed_b)
    cost["shared_draw_benefit"] = {
        "invocations": 2 + 2 * len(panel.seqs),
        "sequences_scored": sum(v["sequences_scored"] for v in sdb.values()),
        "seconds": round(sum(v["seconds"] for v in sdb.values()), 1),
    }

    return {
        "schema": "score_only_batching_diagnostic/1",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "runtime_seconds": round(time.time() - t0, 1),
        "config": {
            "dms_id": args.dms_id,
            "pdb_code": pdb_code,
            "decoding_orders": args.k,
            "mpnn_seed": args.seed,
            "mpnn_seed_b": args.seed_b,
            "panel_seed": args.panel_seed,
            "shuffle_seed": args.shuffle_seed,
            "chunks": args.chunks,
            "max_offset": args.max_offset,
            "top_k": args.top_k,
            "model_name": PROTEINMPNN_MODEL_NAME,
            "proteinmpnn_rev": _proteinmpnn_rev(),
            "scoring_protocol": "score_only/v1 (upstream CLI, one global RNG stream per process)",
        },
        "panel": {
            "wt_length": len(panel.wt_seq),
            "names": panel.names,
            "doubles": panel.doubles,
            "n_decoys": len(panel.decoys),
        },
        "reference_scores": {panel.names[i]: ref[panel.seqs[i]] for i in range(len(panel.seqs))},
        "reference_epistasis": [float(x) for x in panel.epistasis(ref)],
        "conditions": results,
        "stream_offset": so,
        "shared_draw_benefit": sdb,
        "scoring_cost": cost,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dms-id", default="AMFR_HUMAN_Tsuboyama_2023_4G3O")
    ap.add_argument("--k", type=int, default=8, help="decoding orders (num_seq_per_target)")
    ap.add_argument("--seed", type=int, default=PROTEINMPNN_SEED, help="ProteinMPNN decoding seed (non-zero)")
    ap.add_argument(
        "--n-doubles",
        type=int,
        default=30,
        help="doubles in the panel; the singles are then exactly their constituents",
    )
    ap.add_argument("--panel-seed", type=int, default=20261118)
    ap.add_argument("--seed-b", type=int, default=38, help="second decoding seed for the shared-draw comparison")
    ap.add_argument("--shuffle-seed", type=int, default=7)
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--max-offset", type=int, default=6)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out-dir", default=_HERE)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.seed == 0:
        raise SystemExit("--seed 0 is rejected: ProteinMPNN's `if args.seed:` treats 0 as unset and randomizes.")

    out = args.out or stamped_path(
        args.out_dir,
        "score_only_batching_diagnostic",
        "json",
        **{"": dict(DOMAIN_CANDIDATES)[args.dms_id], "k": args.k, "d": args.n_doubles},
    )
    guard_overwrite(out, args.force)
    artifact = run(args)
    with open(out, "w") as fh:
        json.dump(artifact, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
