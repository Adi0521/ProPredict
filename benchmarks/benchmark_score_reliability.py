"""Row C Study 1 — does the K=8 decoding-order operating point generalize across proteins?

Pre-registered in `research_plan/rowC-reliability-generalization.md` (Study 1, including
amendment A1). This harness runs it and nothing else.

**What is measured.** For one backbone, M sampled single mutants plus WT are scored D
independent times at each K, each draw being a fresh draw of the K shared decoding orders
(protocol v2). The quantity analysed is `f(mut) - f(WT)` -- the per-sequence score the
mutation search actually ranks by, NOT the four-term epistasis contrast that every earlier K
figure measured and that the pipeline never computes. Reliability is the one-way
random-effects ICC(1,1) over the resulting (M, D) matrix.

**Why no experimental data.** Test-retest reliability compares the model against itself, so
DMS labels are needed only for the *association* claim (Study 2). Dropping them removes every
data dependency -- no doubles requirement, no assay family, no ProteinGym parquet -- which is
what lets the backbone set be chosen for structural diversity instead of for what happens to
carry labels.

**What this cannot establish.** A backbone reaching ICC 0.99 buys no experimental validity.
This is Monte-Carlo stability of a score against re-drawing its decoding orders, nothing more.
A K that makes the score self-consistent does not make it biologically right.

**Why D >= 3.** Every reliability figure in Row B came from exactly two draws, and
`es.reliability`'s bootstrap resamples variants while conditioning on those two order-sets, so
it cannot see draw-level luck (ISSUES I-23(b)). D=5 plus `es.icc_reliability` fixes that; the
old two-draw interval is recomputed from draws 1-2 of the same panel and reported beside the
new one, so the understatement is a measured number rather than an argument.

Run:
    python -m benchmarks.benchmark_score_reliability --dry-run
    python -m benchmarks.benchmark_score_reliability
    python -m benchmarks.benchmark_score_reliability --backbones 4G3O,1R69 --k-values 8
    python -m benchmarks.benchmark_score_reliability --resume
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks import epistasis_stats as es  # noqa: E402
from benchmarks._artifacts import guard_overwrite, stamped_path  # noqa: E402
from benchmarks.benchmark_score_only_epistasis import _fetch_pdb, trim_pdb_to_dms  # noqa: E402
from benchmarks.screen_backbones import ROWC, screen  # noqa: E402
from config import PROTEINMPNN_MODEL_NAME, PROTEINMPNN_PATH  # noqa: E402
from orchestrator.mpnn_score_only import SCORE_PROTOCOL_V2  # noqa: E402
from orchestrator.mutation_search import score_only_oracle  # noqa: E402

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA = "rowC-study1-score-reliability/1"

# Pre-registered design. Changing any of these changes the study, so they are defaults with
# explicit flags rather than constants edited in place.
DEFAULT_K_VALUES = (2, 4, 8, 16)
DEFAULT_SEEDS = (37, 38, 39, 40, 41)  # D = 5, distinct and nonzero
DEFAULT_N_VARIANTS = 100
DEFAULT_VARIANT_SEED = 20261121

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Decision rule, fixed in advance (Study 1). `stable` -- the F-interval lower bound -- is the
# gate, never `valid`: a pure-noise panel's ICC scatters around zero and lands slightly
# positive about half the time, so `valid` alone would accept noise.
ICC_TARGET = 0.95
ICC_LOWER_BOUND = 0.90
RULE_BACKBONE_FRACTION = 0.90


# Measured on five of the chosen backbones with the model cache warm, amortised over 41
# sequences (Row C, "Cost -- measured, not extrapolated"). Used ONLY for --dry-run estimates.
def estimated_sec_per_seq_per_order(length: int) -> float:
    return 0.0004 * length + 0.004


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------
def sample_single_mutants(wt_seq: str, n: int, seed_material: Sequence[int]) -> List[Tuple[str, str]]:
    """`n` (mutant_label, sequence) pairs sampled uniformly without replacement over the
    L x 19 (position, non-WT amino acid) grid.

    Uniform over the grid, not over positions: every (position, substitution) cell is equally
    likely, so no position is over-weighted and the panel is not biased toward or away from
    any residue type. Labels are 1-based WT-position-mutant, matching DMS notation.
    """
    length = len(wt_seq)
    candidates = [(i, aa) for i in range(length) for aa in AMINO_ACIDS if aa != wt_seq[i]]
    if n > len(candidates):
        raise SystemExit(f"asked for {n} mutants but only {len(candidates)} (position, aa) cells exist")
    rng = np.random.default_rng(np.random.SeedSequence(list(seed_material)))
    picked = sorted(rng.choice(len(candidates), size=n, replace=False).tolist())
    out = []
    for c in picked:
        pos, aa = candidates[c]
        out.append((f"{wt_seq[pos]}{pos + 1}{aa}", wt_seq[:pos] + aa + wt_seq[pos + 1 :]))
    return out


def backbone_descriptors(pdb_string: str, pdb_code: str) -> Dict:
    """Cheap structural descriptors for the EXPLORATORY question of whether the K a protein
    needs is predictable from its backbone. Declared exploratory in the design: 16 points
    against three descriptors supports a hypothesis to test later, not a conclusion.
    """
    import io

    from Bio.PDB import NeighborSearch, PDBParser

    struct = PDBParser(QUIET=True).get_structure(pdb_code, io.StringIO(pdb_string))
    residues = [r for r in list(struct)[0]["A"] if "CA" in r]
    cas = [r["CA"] for r in residues]
    search = NeighborSearch(cas)
    # CA-CA contacts within 8 A, self excluded: a standard coarse packing measure.
    contacts = [len(search.search(ca.coord, 8.0)) - 1 for ca in cas]
    out: Dict = {
        "n_residues_with_ca": len(residues),
        "mean_contact_number_ca_8a": float(np.mean(contacts)) if contacts else float("nan"),
    }
    try:
        from Bio.PDB.SASA import ShrakeRupley

        ShrakeRupley().compute(list(struct)[0], level="R")
        sasa = [float(r.sasa) for r in residues if getattr(r, "sasa", None) is not None]
        if sasa:
            out["mean_sasa_per_residue"] = float(np.mean(sasa))
            # Absolute threshold, NOT relative accessibility: a relative measure needs a
            # per-residue-type max-ASA reference table this harness deliberately does not carry.
            out["fraction_buried_sasa_lt_20a2"] = float(np.mean([s < 20.0 for s in sasa]))
    except Exception as exc:  # noqa: BLE001 - descriptors are exploratory, never load-bearing
        out["sasa_error"] = f"{type(exc).__name__}: {exc}"
    return out


class Backbone:
    """One eligible backbone and the exact panel of sequences Study 1 scores on it."""

    def __init__(self, pdb_code: str, n_variants: int, variant_seed: int, with_descriptors: bool = True):
        self.pdb_code = pdb_code.upper()
        rec = screen(self.pdb_code)
        if "error" in rec:
            raise SystemExit(f"{self.pdb_code}: {rec['error']}")
        if not rec["eligible"]:
            # The eligibility rule is part of the pre-registration, so it is enforced here
            # rather than trusted to have been checked once by hand.
            raise SystemExit(
                f"{self.pdb_code} fails Study 1's eligibility rule ({'; '.join(rec['reasons'])}). "
                "Run `python -m benchmarks.screen_backbones <code>` for the full facts."
            )
        self.method = rec["method"]
        self.resolution = rec["resolution"]
        self.wt_seq = rec["sequence"]
        self.length = len(self.wt_seq)
        raw = _fetch_pdb(self.pdb_code)
        # Reused from the epistasis harness: model-0, first chain, ATOM-only, renumbered 1..L.
        # Passing the whole chain sequence as the target makes it a chain extraction rather
        # than a slice, and it raises loudly if the screener and this parse disagree on which
        # chain is first.
        self.pdb = trim_pdb_to_dms(raw, self.wt_seq, self.pdb_code)
        self.raw_sha256 = hashlib.sha256(raw.encode()).hexdigest()
        stable = int.from_bytes(hashlib.sha256(self.pdb_code.encode()).digest()[:8], "big")
        self.seed_material = [int(variant_seed), stable]
        self.mutants = sample_single_mutants(self.wt_seq, n_variants, self.seed_material)
        self.descriptors = backbone_descriptors(self.pdb, self.pdb_code) if with_descriptors else {}

    @property
    def sequences(self) -> List[str]:
        """WT first, then the sampled mutants in panel order."""
        return [self.wt_seq] + [seq for _, seq in self.mutants]

    @property
    def n_sequences(self) -> int:
        return 1 + len(self.mutants)

    def deltas(self, f: Sequence[float]) -> np.ndarray:
        """`f(mut) - f(WT)` for one scored batch. WT is re-scored in every draw, so its own
        decoding noise is inside each delta -- which is correct: that is the quantity the
        search ranks, noise and all."""
        return np.asarray(f[1:], dtype=float) - float(f[0])

    def provenance(self) -> Dict:
        return {
            "pdb_code": self.pdb_code,
            "method": self.method,
            "resolution_angstrom": self.resolution,
            "length": self.length,
            "raw_pdb_sha256": self.raw_sha256,
            "n_variants": len(self.mutants),
            "variant_seed_material": self.seed_material,
            "mutants": [m for m, _ in self.mutants],
            "descriptors": self.descriptors,
        }


# ---------------------------------------------------------------------------
# One (backbone, K) point
# ---------------------------------------------------------------------------
def score_draw(backbone: Backbone, k: int, seed: int) -> Tuple[List[float], float]:
    t0 = time.time()
    f = score_only_oracle(
        backbone.pdb,
        backbone.sequences,
        proteinmpnn_dir=PROTEINMPNN_PATH,
        model_name=PROTEINMPNN_MODEL_NAME,
        seed=seed,
        num_decoding_orders=k,
        protocol=SCORE_PROTOCOL_V2,
    )
    return f, time.time() - t0


def measure_point(backbone: Backbone, k: int, seeds: Sequence[int], verbose: bool = True) -> Dict:
    """Score the panel once per seed at this K, then the full D-draw analysis."""
    cols, runtimes = [], []
    for seed in seeds:
        f, dt = score_draw(backbone, k, seed)
        cols.append(backbone.deltas(f))
        runtimes.append(dt)
        if verbose:
            print(f"    [{backbone.pdb_code}] K={k} seed={seed}: {backbone.n_sequences} seqs in {dt:.1f} s")
    x = np.column_stack(cols)

    icc = es.icc_reliability(x)
    # The same panel read the old way: draws 1 and 2 only, variant-resampling bootstrap,
    # conditioned on those two order-sets. Reported so the understatement is measurable.
    two_draw = es.reliability(x[:, 0], x[:, 1])

    jack = icc.get("draw_jackknife", {})
    f_ci = icc.get("icc_ci95_f")
    point = {
        "k": k,
        "seeds": list(seeds),
        "n_variants": int(x.shape[0]),
        "n_draws": int(x.shape[1]),
        "n_sequences_per_draw": backbone.n_sequences,
        "icc": icc,
        "two_draw_reliability_seeds_1_2": {
            kk: two_draw[kk]
            for kk in ("n", "reliability", "reliability_ci95", "replicate_corr", "valid", "stable", "reason")
            if kk in two_draw
        },
        "runtime_seconds": {"per_draw": runtimes, "total": float(sum(runtimes))},
        "seconds_per_sequence_per_order": float(sum(runtimes)) / (len(seeds) * backbone.n_sequences * k),
    }
    # Declared in advance as a diagnostic to REPORT, not an automatic override: a jackknife
    # spread wider than the F interval means the F interval is understating draw sensitivity
    # and its normality assumption is doing work the data does not support.
    if f_ci and isinstance(jack.get("spread"), float) and math.isfinite(jack["spread"]):
        width = f_ci[1] - f_ci[0]
        point["draw_sensitivity_check"] = {
            "f_interval_width": width,
            "jackknife_spread": jack["spread"],
            "jackknife_exceeds_f_width": bool(jack["spread"] > width),
        }
    point["meets_rule"] = bool(
        icc.get("valid")
        and icc.get("stable")
        and icc.get("icc", 0.0) >= ICC_TARGET
        and f_ci is not None
        and f_ci[0] >= ICC_LOWER_BOUND
    )
    return point


# ---------------------------------------------------------------------------
# Cross-point and cross-backbone analysis
# ---------------------------------------------------------------------------
def noise_scaling(points: List[Dict]) -> Dict:
    """Log-log slope of noise SD against K. Pre-registered expectation: -0.5 (plain
    Monte-Carlo averaging). The K-sweep measured [-0.68, -0.44] over six curves, so a clear
    departure is informative and is declared here rather than discovered later."""
    ks, sds = [], []
    for p in points:
        sd = p["icc"].get("noise_sd")
        if sd and sd > 0 and p["k"] > 0:
            ks.append(p["k"])
            sds.append(sd)
    if len(ks) < 2:
        return {"valid": False, "reason": f"need >= 2 usable K points, have {len(ks)}"}
    slope, intercept = np.polyfit(np.log(ks), np.log(sds), 1)
    return {
        "valid": True,
        "k_values": ks,
        "noise_sd": sds,
        "log_log_slope": float(slope),
        "log_log_intercept": float(intercept),
        "expected_slope": -0.5,
    }


def min_qualifying_k(points: List[Dict]) -> Optional[int]:
    qualifying = [p["k"] for p in points if p["meets_rule"]]
    return min(qualifying) if qualifying else None


def descriptor_associations(rows: List[Dict], k: int) -> Dict:
    """EXPLORATORY: does ICC at a given K track a cheap backbone property? 16 points."""
    pairs = []
    for r in rows:
        point = next((p for p in r["points"] if p["k"] == k), None)
        if point and point["icc"].get("valid"):
            pairs.append((r["provenance"], point["icc"]["icc"]))
    if len(pairs) < 5:
        return {"valid": False, "reason": f"only {len(pairs)} backbones with a valid ICC at K={k}"}
    y = [icc for _, icc in pairs]
    out: Dict = {
        "valid": True,
        "k": k,
        "n_backbones": len(pairs),
        "status": "EXPLORATORY — declared in advance as hypothesis-generating, not a finding",
    }
    for name in ("length", "mean_contact_number_ca_8a", "mean_sasa_per_residue", "fraction_buried_sasa_lt_20a2"):
        xs = [
            prov.get(name, prov.get("descriptors", {}).get(name)) if name == "length" else prov["descriptors"].get(name)
            for prov, _ in pairs
        ]
        if any(v is None or not isinstance(v, (int, float)) for v in xs):
            continue
        out[name] = es.associations(xs, y)
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def _fmt_ci(ci: Optional[Sequence[float]]) -> str:
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "n/a"


def markdown_table(rows: List[Dict]) -> str:
    lines = [
        "| backbone | L | K | ICC(1,1) | F 95% CI | variant boot CI | jackknife spread "
        "| old 2-draw rel [CI] | noise SD | signal SD | s/seq/order | rule |",
        "|---|--:|--:|--:|---|---|--:|---|--:|--:|--:|---|",
    ]
    for r in rows:
        for p in sorted(r["points"], key=lambda q: q["k"]):
            icc = p["icc"]
            old = p["two_draw_reliability_seeds_1_2"]
            jack = icc.get("draw_jackknife", {})
            spread = jack.get("spread")
            old_txt = (
                f"{old['reliability']:.3f} {_fmt_ci(old.get('reliability_ci95'))}" if "reliability" in old else "n/a"
            )
            lines.append(
                f"| {r['provenance']['pdb_code']} | {r['provenance']['length']} | {p['k']} "
                f"| {icc.get('icc', float('nan')):.4f} "
                f"| {_fmt_ci(icc.get('icc_ci95_f'))} "
                f"| {_fmt_ci(icc.get('icc_ci95_boot_variants'))} "
                f"| {'n/a' if spread is None else f'{spread:.4f}'} "
                f"| {old_txt} "
                f"| {icc.get('noise_sd', float('nan')):.4f} "
                f"| {icc.get('signal_sd', float('nan')):.4f} "
                f"| {p['seconds_per_sequence_per_order']:.5f} "
                f"| {'pass' if p['meets_rule'] else 'FAIL'} |"
            )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backbones", default=",".join(ROWC), help="comma-separated PDB codes")
    ap.add_argument("--k-values", default=",".join(str(k) for k in DEFAULT_K_VALUES))
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS), help="D decoding seeds")
    ap.add_argument("--n-variants", type=int, default=DEFAULT_N_VARIANTS)
    ap.add_argument("--variant-seed", type=int, default=DEFAULT_VARIANT_SEED)
    ap.add_argument("--out-dir", default=BENCH_DIR)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true", help="allow overwriting an existing artifact")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="reuse the (backbone, K) points already in the artifact and score only what is missing",
    )
    ap.add_argument("--dry-run", action="store_true", help="screen, build panels, print the plan and cost; no scoring")
    args = ap.parse_args(argv)

    k_values = [int(t) for t in args.k_values.split(",") if t.strip()]
    seeds = [int(t) for t in args.seeds.split(",") if t.strip()]
    codes = [t.strip().upper() for t in args.backbones.split(",") if t.strip()]
    if not k_values or any(k < 1 for k in k_values):
        raise SystemExit("--k-values must be one or more positive integers")
    if len(seeds) < 2:
        raise SystemExit("need at least 2 draws; the study is pre-registered at D=5")
    if len(set(seeds)) != len(seeds):
        raise SystemExit("--seeds must be distinct: re-scoring the same orders measures nothing")
    if 0 in seeds:
        raise SystemExit("decoding seeds must be nonzero (seed 0 is rejected by the scorer)")
    if len(seeds) < 3:
        print(
            f"WARNING: D={len(seeds)} reproduces the two-draw flaw this study exists to fix "
            "(ISSUES I-23(b)); the draw jackknife needs D >= 3.",
            file=sys.stderr,
        )

    config = {
        "k_values": k_values,
        "seeds": seeds,
        "n_draws": len(seeds),
        "n_variants": args.n_variants,
        "variant_seed": args.variant_seed,
        "scoring_protocol": SCORE_PROTOCOL_V2,
        "model_name": PROTEINMPNN_MODEL_NAME,
        "icc_target": ICC_TARGET,
        "icc_lower_bound": ICC_LOWER_BOUND,
        "rule_backbone_fraction": RULE_BACKBONE_FRACTION,
    }
    out_path = args.out or stamped_path(
        args.out_dir,
        "score_reliability",
        "json",
        n=args.n_variants,
        k="-".join(str(k) for k in k_values),
        d=len(seeds),
        v="2",
    )

    # Resume before the overwrite guard: a 4-5 hour run that dies to a closed laptop should not
    # have to choose between losing its work and --force'ing over it.
    done: Dict[Tuple[str, int], Dict] = {}
    prior_runtime = 0.0
    if args.resume and os.path.exists(out_path):
        with open(out_path) as fh:
            prev = json.load(fh)
        differing = {k: (v, config.get(k)) for k, v in prev.get("config", {}).items() if config.get(k) != v}
        if differing:
            raise SystemExit(
                f"--resume refuses to mix runs: {out_path} was written at different parameters "
                f"{differing!r}. Use --out for a separate artifact."
            )
        for row in prev.get("backbones", []):
            for p in row.get("points", []):
                done[(row["provenance"]["pdb_code"], p["k"])] = p
        prior_runtime = float(prev.get("total_runtime_seconds") or 0.0)
        print(f"resuming from {out_path}: {len(done)} (backbone, K) points already scored")
    elif not args.dry_run:
        guard_overwrite(out_path, args.force)

    backbones = [Backbone(c, args.n_variants, args.variant_seed, with_descriptors=not args.dry_run) for c in codes]

    if args.dry_run:
        total = 0.0
        print(f"{'pdb':6} {'L':>5} {'seqs':>5} {'seq*orders':>11} {'est s':>9}")
        for b in backbones:
            work = b.n_sequences * len(seeds) * sum(k_values)
            est = work * estimated_sec_per_seq_per_order(b.length)
            total += est
            print(f"{b.pdb_code:6} {b.length:>5} {b.n_sequences:>5} {work:>11} {est:>9.0f}")
        print(f"\n{len(backbones)} backbones, K={k_values}, D={len(seeds)}, M={args.n_variants}")
        print(f"estimated total {total / 3600:.1f} h (throughput model 0.0004*L+0.004 s/seq/order)")
        print(f"would write {out_path}")
        print("\nEstimates only. Measured throughput on this host varies with CPU contention;")
        print("the K-sweep saw a 2.5x swing on one domain when the host was busy.")
        return 0

    rows: List[Dict] = []
    t_start = time.time()
    for b in backbones:
        print(
            f"\n=== [{b.pdb_code}] L={b.length} {b.method} {b.resolution} A | "
            f"{b.n_sequences} sequences x {len(seeds)} draws x K in {k_values}"
        )
        points: List[Dict] = []
        for k in k_values:
            cached = done.get((b.pdb_code, k))
            if cached is not None:
                points.append(cached)
                print(
                    f"  [{b.pdb_code}] K={k}: reused from artifact (ICC {cached['icc'].get('icc', float('nan')):.4f})"
                )
                continue
            p = measure_point(b, k, seeds)
            points.append(p)
            print(
                f"  [{b.pdb_code}] K={k}: ICC={p['icc'].get('icc', float('nan')):.4f} "
                f"F{_fmt_ci(p['icc'].get('icc_ci95_f'))} "
                f"noise_sd={p['icc'].get('noise_sd', float('nan')):.4f} "
                f"{'pass' if p['meets_rule'] else 'FAIL'} in {p['runtime_seconds']['total']:.1f} s"
            )
        row = {
            "provenance": b.provenance(),
            "points": sorted(points, key=lambda q: q["k"]),
        }
        row["noise_scaling"] = noise_scaling(row["points"])
        row["min_qualifying_k"] = min_qualifying_k(row["points"])
        rows.append(row)

        # Written after every backbone, so a run that dies keeps everything finished so far.
        _write(out_path, SCHEMA, config, rows, seeds, k_values, time.time() - t_start + prior_runtime, partial=True)

    out = _write(out_path, SCHEMA, config, rows, seeds, k_values, time.time() - t_start + prior_runtime)
    print("\n=== Row C Study 1 — reliability of f(mut)-f(WT) across backbones ===")
    print(out["table_markdown"])
    rule = out["analysis"]["decision_rule"]
    print(
        f"\ndecision rule at K={rule['k']}: {rule['n_pass']}/{rule['n_backbones']} backbones pass "
        f"(ICC >= {ICC_TARGET} with F lower bound >= {ICC_LOWER_BOUND}) -> "
        f"{'RETAIN' if rule['retain_default'] else 'DOES NOT RETAIN'} K={rule['k']} as the default"
    )
    print(f"\nwrote {out_path}")
    print(f"wrote {os.path.splitext(out_path)[0] + '.md'}")
    return 0


def _decision_rule(rows: List[Dict], k: int) -> Dict:
    judged = [(r["provenance"]["pdb_code"], next((p for p in r["points"] if p["k"] == k), None)) for r in rows]
    judged = [(code, p) for code, p in judged if p is not None]
    n_pass = sum(1 for _, p in judged if p["meets_rule"])
    return {
        "k": k,
        "n_backbones": len(judged),
        "n_pass": n_pass,
        "failing": [code for code, p in judged if not p["meets_rule"]],
        "fraction_pass": (n_pass / len(judged)) if judged else None,
        "retain_default": bool(judged and n_pass / len(judged) >= RULE_BACKBONE_FRACTION),
        "rule": (
            f"retain K={k} iff ICC >= {ICC_TARGET} with F-interval lower bound >= {ICC_LOWER_BOUND} "
            f"on >= {RULE_BACKBONE_FRACTION:.0%} of backbones"
        ),
    }


def _write(
    out_path: str,
    schema: str,
    config: Dict,
    rows: List[Dict],
    seeds: Sequence[int],
    k_values: Sequence[int],
    elapsed: float,
    partial: bool = False,
) -> Dict:
    table = markdown_table(rows)
    rule_k = 8 if 8 in k_values else sorted(k_values)[-1]
    out = {
        "schema": schema,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "partial": partial,
        "config": config,
        "what_is_measured": (
            "Test-retest reliability of f(mut)-f(WT) -- the per-sequence score the mutation "
            "search ranks by -- across D independent draws of the K shared decoding orders "
            "under protocol score_only/v2, as one-way random-effects ICC(1,1). No experimental "
            "data is involved. This is Monte-Carlo stability of the score against re-drawing "
            "its decoding orders and says NOTHING about biological validity: a backbone "
            "reaching ICC 0.99 buys no experimental validity."
        ),
        "backbones": rows,
        "analysis": {
            "decision_rule": _decision_rule(rows, rule_k),
            "min_qualifying_k_by_backbone": {r["provenance"]["pdb_code"]: r["min_qualifying_k"] for r in rows},
            "descriptor_associations": descriptor_associations(rows, rule_k),
        },
        "total_runtime_seconds": elapsed,
        "table_markdown": table,
    }
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    with open(os.path.splitext(out_path)[0] + ".md", "w") as fh:
        fh.write(table + "\n")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
