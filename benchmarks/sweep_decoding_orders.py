"""
How many decoding orders does `score_only` actually need? (ISSUES I-23)

The `PROTEINMPNN_NUM_DECODING_ORDERS >= 32` recommendation was calibrated against scoring
protocol v1's noise, and most of that noise turned out to be a batch-composition artifact
rather than decoding-order variance (`Process/mutation-scoring-consistency.md`). Under v2 the
same 300-double panel reaches per-double epistasis reliability 0.99 on all three domains at
K=32, and the diagnostic reached 0.977 at K=8 on 4G3O — so K=32 is over-provisioned, but only
one domain had ever been measured below 32. This sweeps K and reports where the knee is.

WHAT IS MEASURED. Two reliability curves per domain, because they answer different questions
and only one of them is the config value's question:

  * `eps` — the four-term epistasis contrast f(AB) - f(A) - f(B) + f(WT). This is what every
    K figure on record so far has measured, so it keeps the sweep continuous with I-23's
    history. It sums four independent perturbations, so it is roughly 4x noisier than...
  * `single_delta` — f(mut) - f(WT) over the panel's constituent single mutants. THIS is the
    quantity `orchestrator/mutation_search.py` actually ranks candidates by. Reading a config
    default off the `eps` curve is conservative by about that 4x.

Reliability here is test-retest across two independent draws of the shared decoding orders
(seeds 37 and 38) at the same K. It measures Monte-Carlo stability of the score against
re-drawing the orders — which is exactly the "how many orders do I need" question — and
NOTHING about whether the score is biologically right. A K that reaches reliability 0.99 buys
no experimental validity whatsoever.

THE PANEL. Identical to the committed v2 stability-arm artifacts, not re-sampled: the stored
mutant list is read out of each `*_n300_k32_v2.json` and re-derived through `load_assay`, the
same path `_replicate_full_sample` uses. Nothing in this script writes to those artifacts.

K=32 comes free. The committed artifacts already hold exactly the two draws this needs for the
`eps` curve (main pass = seed 37, full-sample replicate = seed 38), so a K=32 `eps` row is
reused rather than re-scored, tagged `source="reused_artifact"`. They do NOT store raw
per-sequence scores, so there is no reused K=32 `single_delta` point; pass `--k-values
2,4,8,16,32` to score K=32 fresh and get one (roughly doubles the runtime, and the fresh
K=32 `eps` row then also serves as an independent cross-check of the committed numbers).

Run (~1 h on an M3, CPU):
    python -m benchmarks.sweep_decoding_orders
    python -m benchmarks.sweep_decoding_orders --k-values 2,4,8,16,32
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
from benchmarks import epistasis_stats as es
from benchmarks._artifacts import guard_overwrite, stamped_path
from benchmarks.benchmark_score_only_epistasis import (
    _fetch_pdb,
    load_assay,
    trim_pdb_to_dms,
)
from config import PROTEINMPNN_MODEL_NAME, PROTEINMPNN_PATH
from orchestrator.mpnn_score_only import SCORE_PROTOCOL_V2
from orchestrator.mutation_search import score_only_oracle

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
# Order matches the stability arm's own domain order.
DOMAINS = ["4G3O", "3DKM", "3L1X"]
DEFAULT_K_VALUES = (2, 4, 8, 16)


def artifact_path(pdb_code: str) -> str:
    return os.path.join(BENCH_DIR, f"score_only_epistasis_stability_{pdb_code}_n300_k32_v2.json")


class Panel:
    """The exact scored panel of one committed v2 artifact: WT, constituent singles, doubles."""

    def __init__(self, pdb_code: str):
        with open(artifact_path(pdb_code)) as fh:
            self.prev = json.load(fh)
        cfg = self.prev["config"]
        protocol = cfg.get("scoring_protocol")
        if protocol != SCORE_PROTOCOL_V2:
            raise SystemExit(
                f"{pdb_code}: artifact was scored under {protocol!r}, not {SCORE_PROTOCOL_V2!r}. "
                "This sweep characterises v2; re-run the stability arm under v2 first."
            )
        self.pdb_code = pdb_code
        self.dms_id = cfg["dms_id"]
        self.k_stored = int(cfg["decoding_orders"])

        assay = load_assay(self.dms_id)
        by_mutant = {d["mutant"]: d for d in assay.doubles}
        self.mutants = [r["mutant"] for r in self.prev["doubles"]]
        missing = [m for m in self.mutants if m not in by_mutant]
        if missing:
            raise SystemExit(
                f"{pdb_code}: {len(missing)} stored mutants are not in the re-derived assay "
                f"(first: {missing[0]}). The panel is not reproducible; refusing to sweep."
            )
        self.sample = [by_mutant[m] for m in self.mutants]
        self.singles = assay.singles
        self.wt_seq = assay.wt_seq
        # Sorted, exactly as _eps_for_sample orders them, so the batch is the same batch.
        self.need = sorted({m for d in self.sample for m in d["components"]})
        self.pdb = trim_pdb_to_dms(_fetch_pdb(pdb_code), assay.wt_seq, pdb_code)
        self.groups = [tuple(es.residue_pair(m)) for m in self.mutants]
        self.eps_exp = np.array([r["eps_exp"] for r in self.prev["doubles"]])

    @property
    def sequences(self) -> List[str]:
        return [self.wt_seq] + [self.singles[m][0] for m in self.need] + [d["sequence"] for d in self.sample]

    @property
    def n_sequences(self) -> int:
        return 1 + len(self.need) + len(self.sample)

    def split(self, f: List[float]) -> Tuple[float, np.ndarray, np.ndarray]:
        """(f_wt, single deltas over `need`, epsilon over `sample`) from one scored batch."""
        f_wt = f[0]
        f_single = {m: f[1 + i] for i, m in enumerate(self.need)}
        f_double = {d["mutant"]: f[1 + len(self.need) + i] for i, d in enumerate(self.sample)}
        single_delta = np.array([f_single[m] - f_wt for m in self.need])
        eps = np.array(
            [
                (f_double[d["mutant"]] - f_wt)
                - ((f_single[d["components"][0]] - f_wt) + (f_single[d["components"][1]] - f_wt))
                for d in self.sample
            ]
        )
        return f_wt, single_delta, eps


def score_panel(panel: Panel, k: int, seed: int) -> Tuple[List[float], float]:
    t0 = time.time()
    f = score_only_oracle(
        panel.pdb,
        panel.sequences,
        proteinmpnn_dir=PROTEINMPNN_PATH,
        model_name=PROTEINMPNN_MODEL_NAME,
        seed=seed,
        num_decoding_orders=k,
        protocol=SCORE_PROTOCOL_V2,
    )
    return f, time.time() - t0


def _rel_summary(rel: Dict) -> Dict:
    """The fields worth carrying per point, without the full caveat payload on every row."""
    keep = ("n", "reliability", "reliability_ci95", "replicate_corr", "noise_var", "observed_var", "valid", "reason")
    out = {k: rel[k] for k in keep if k in rel}
    if "noise_var" in rel and "observed_var" in rel:
        out["noise_sd"] = float(np.sqrt(rel["noise_var"]))
        out["signal_sd"] = float(np.sqrt(max(rel["observed_var"] - rel["noise_var"], 0.0)))
    return out


def sweep_point(panel: Panel, k: int, seed_a: int, seed_b: int) -> Dict:
    """Score the panel twice at K and report both reliability curves plus the association."""
    f_a, dt_a = score_panel(panel, k, seed_a)
    f_b, dt_b = score_panel(panel, k, seed_b)
    _, sd_a, eps_a = panel.split(f_a)
    _, sd_b, eps_b = panel.split(f_b)

    rel_eps = es.reliability(eps_a, eps_b)
    rel_single = es.reliability(sd_a, sd_b)
    decomp = es.reliability_decomposition(eps_a, eps_b, panel.groups)
    return {
        "k": k,
        "source": "sweep",
        "seed_a": seed_a,
        "seed_b": seed_b,
        "n_sequences_per_draw": panel.n_sequences,
        "eps_reliability": _rel_summary(rel_eps),
        "single_delta_reliability": _rel_summary(rel_single),
        "eps_reliability_decomposition": {
            which: _rel_summary(decomp[which]) for which in ("between_group", "within_group") if which in decomp
        },
        # Association at seed A alone, so it is directly comparable to the stability-arm
        # artifacts (whose primary numbers are the seed-37 main pass), not an averaged draw.
        "eps_association_seed_a": es.associations(eps_a, panel.eps_exp),
        "runtime_seconds": {"draw_a": dt_a, "draw_b": dt_b, "total": dt_a + dt_b},
        "seconds_per_sequence_per_order": (dt_a + dt_b) / (2 * panel.n_sequences * k),
    }


def reused_point(panel: Panel) -> Optional[Dict]:
    """The free K=32 `eps` row from the committed artifact's two existing draws."""
    rep = panel.prev.get("replicate") or {}
    eps_b = rep.get("full_sample_eps_seed_b")
    if not eps_b:
        return None
    stored_mutants = rep.get("full_sample_mutants")
    if stored_mutants is not None and stored_mutants != panel.mutants:
        raise SystemExit(
            f"{panel.pdb_code}: the artifact's full-sample replicate covers a different "
            "mutant list than its own doubles; refusing to reuse it."
        )
    eps_a = np.array([r["eps_model"] for r in panel.prev["doubles"]])
    eps_b = np.array(eps_b, dtype=float)
    rel_eps = es.reliability(eps_a, eps_b)
    decomp = es.reliability_decomposition(eps_a, eps_b, panel.groups)
    return {
        "k": panel.k_stored,
        "source": "reused_artifact",
        "seed_a": panel.prev["config"]["mpnn_seed_a"],
        "seed_b": panel.prev["config"]["mpnn_seed_b"],
        "n_sequences_per_draw": panel.n_sequences,
        "eps_reliability": _rel_summary(rel_eps),
        # Raw per-sequence scores are not stored in the artifact, only eps.
        "single_delta_reliability": {
            "valid": False,
            "reason": "the committed artifact stores eps only, not raw per-sequence scores; "
            "pass --k-values ...,32 to score K=32 fresh and measure this",
        },
        "eps_reliability_decomposition": {
            which: _rel_summary(decomp[which]) for which in ("between_group", "within_group") if which in decomp
        },
        "eps_association_seed_a": panel.prev["analysis"]["primary_eps_association"],
        "runtime_seconds": dict(panel.prev.get("runtime_seconds", {})),
        "note": "reused from the committed v2 artifact; not re-scored by this sweep",
    }


def _fmt_rel(point: Dict, key: str) -> str:
    rel = point.get(key, {})
    if not rel.get("valid", True) or "reliability" not in rel:
        return "n/a"
    ci = rel.get("reliability_ci95")
    base = f"{rel['reliability']:.3f}"
    return f"{base} [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else base


def markdown_table(rows: List[Dict]) -> str:
    lines = [
        "| domain | K | source | single-delta reliability [95% CI] | eps reliability [95% CI] "
        "| eps between / within pair | eps rho vs experiment | noise SD (eps) | s/seq/order |",
        "|---|--:|---|---|---|---|--:|--:|--:|",
    ]
    for r in rows:
        for p in r["points"]:
            decomp = p.get("eps_reliability_decomposition", {})

            def d(which, decomp=decomp):
                x = decomp.get(which, {})
                return f"{x['reliability']:.3f}" if x.get("valid") and "reliability" in x else "n/a"

            noise = p["eps_reliability"].get("noise_sd")
            spp = p.get("seconds_per_sequence_per_order")
            rho = p["eps_association_seed_a"].get("spearman")
            lines.append(
                f"| {r['pdb_code']} | {p['k']} | {p['source']} "
                f"| {_fmt_rel(p, 'single_delta_reliability')} "
                f"| {_fmt_rel(p, 'eps_reliability')} "
                f"| {d('between_group')} / {d('within_group')} "
                f"| {'n/a' if rho is None else f'{rho:+.3f}'} "
                f"| {'n/a' if noise is None else f'{noise:.4f}'} "
                f"| {'n/a' if spp is None else f'{spp:.5f}'} |"
            )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--k-values",
        default=",".join(str(k) for k in DEFAULT_K_VALUES),
        help="comma-separated decoding-order counts to score fresh (default 2,4,8,16; K=32 is "
        "reused from the committed artifacts for the eps curve)",
    )
    ap.add_argument("--domains", default=",".join(DOMAINS), help="comma-separated PDB codes")
    ap.add_argument("--seed-a", type=int, default=37)
    ap.add_argument("--seed-b", type=int, default=38)
    ap.add_argument("--no-reuse", action="store_true", help="skip the free K=32 reused rows")
    ap.add_argument("--out-dir", default=BENCH_DIR)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    k_values = [int(tok) for tok in args.k_values.split(",") if tok.strip()]
    if not k_values or any(k < 1 for k in k_values):
        raise SystemExit("--k-values must be one or more positive integers")
    if args.seed_a == args.seed_b:
        raise SystemExit("--seed-a and --seed-b must differ: two draws of the SAME orders measure nothing")
    if 0 in (args.seed_a, args.seed_b):
        raise SystemExit("decoding seeds must be nonzero (seed 0 is rejected by the scorer)")
    domains = [tok.strip().upper() for tok in args.domains.split(",") if tok.strip()]

    out_path = args.out or stamped_path(
        args.out_dir,
        "decoding_order_sweep",
        "json",
        n=300,
        k="-".join(str(k) for k in k_values),
        v="2",
    )
    guard_overwrite(out_path, args.force)

    rows: List[Dict] = []
    t_start = time.time()
    for pdb_code in domains:
        panel = Panel(pdb_code)
        print(
            f"\n=== [{pdb_code}] {panel.dms_id} | {len(panel.sample)} doubles "
            f"+ {len(panel.need)} singles + WT = {panel.n_sequences} sequences per draw"
        )
        points: List[Dict] = []
        if not args.no_reuse:
            reused = reused_point(panel)
            if reused is None:
                print(f"[{pdb_code}] no full-sample replicate stored; nothing to reuse")
            else:
                points.append(reused)
                print(f"[{pdb_code}] K={reused['k']} (reused) eps reliability={_fmt_rel(reused, 'eps_reliability')}")
        for k in k_values:
            print(f"[{pdb_code}] K={k}: scoring {panel.n_sequences} sequences x 2 draws...")
            p = sweep_point(panel, k, args.seed_a, args.seed_b)
            points.append(p)
            print(
                f"[{pdb_code}] K={k}: single-delta={_fmt_rel(p, 'single_delta_reliability')} "
                f"eps={_fmt_rel(p, 'eps_reliability')} "
                f"in {p['runtime_seconds']['total']:.1f} s"
            )
        points.sort(key=lambda p: (p["k"], p["source"]))
        rows.append(
            {
                "pdb_code": pdb_code,
                "dms_id": panel.dms_id,
                "n_doubles": len(panel.sample),
                "n_singles": len(panel.need),
                "n_sequences_per_draw": panel.n_sequences,
                "source_artifact": os.path.basename(artifact_path(pdb_code)),
                "source_config_id": panel.prev.get("config_id"),
                "points": points,
            }
        )

    table = markdown_table(rows)
    out = {
        "schema": "rowB-decoding-order-sweep/1",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "k_values_scored": k_values,
            "k_reused": None if args.no_reuse else 32,
            "seed_a": args.seed_a,
            "seed_b": args.seed_b,
            "scoring_protocol": SCORE_PROTOCOL_V2,
            "model_name": PROTEINMPNN_MODEL_NAME,
            "domains": domains,
        },
        "what_is_measured": (
            "Test-retest reliability across two independent draws of the shared decoding orders "
            "at a fixed K. `single_delta` is f(mut)-f(WT), the quantity the mutation search "
            "ranks by; `eps` is the four-term epistasis contrast, ~4x noisier, and is what "
            "earlier K figures measured. This is Monte-Carlo stability of the score against "
            "re-drawing the orders. It says NOTHING about whether the score is biologically "
            "right: a K reaching reliability 0.99 buys no experimental validity."
        ),
        "domains": rows,
        "total_runtime_seconds": time.time() - t_start,
        "table_markdown": table,
    }
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    md_path = os.path.splitext(out_path)[0] + ".md"
    with open(md_path, "w") as fh:
        fh.write(table + "\n")

    print("\n=== decoding-order sweep (protocol v2) ===")
    print(table)
    print(f"\nwrote {out_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
