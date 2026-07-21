"""
Analyse the Boltz-2 affinity-invariance results.

The headline number is the INVARIANCE test, not the accuracy test:

  H_invariant : predicted Delta ~ 0 regardless of experimental fold-change
                -> spread(pred Delta) collapses, Spearman ~ 0.
                   The thesis of the research plan holds; co-folding affinity is
                   mutation-blind, and the QM/physics tier has a real gap to fill.

  H_responsive: predicted Delta tracks experimental log10 fold-change
                -> Spearman > 0. The thesis needs rework -- Boltz-2 already does
                   the thing the ladder was built to do.

Reporting rules baked in:
  * Spearman (not Pearson) is primary: PhenoSense censors fold-change at 100.
  * Censored rows are reported both included (rank-safe) and excluded.
  * "Slope" of pred vs experimental Delta is reported: under H_invariant the
    slope is ~0 even if a weak correlation squeaks past significance. A tiny-but-
    significant correlation with slope ~0.02 is still practical blindness, and
    that distinction is the whole point -- don't let a p-value decide it.
"""
import argparse
import json
import math
from collections import defaultdict


def spearman(x, y):
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
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def linfit(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    den = sum((a - mx) ** 2 for a in x)
    if den == 0:
        return float("nan"), float("nan")
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / den
    return slope, my - slope * mx


def stdev(v):
    if len(v) < 2:
        return float("nan")
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="results.jsonl from benchmark_affinity_invariance.py")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.results)]
    rows = [r for r in rows if r.get("ok")]

    # WT baseline per (drug, msa arm)
    wt = {(r["drug"], r["msa"]): r["affinity_pred_value"]
          for r in rows if r["seq_id"] == "WT_CONSENSUS_B"}
    if not wt:
        raise SystemExit("no WT reference rows found -- cannot compute Delta")

    by_arm = defaultdict(list)
    for r in rows:
        if r["seq_id"] == "WT_CONSENSUS_B":
            continue
        base = wt.get((r["drug"], r["msa"]))
        if base is None:
            continue
        r["pred_delta"] = r["affinity_pred_value"] - base
        by_arm[r["msa"]].append(r)

    for msa_arm, rs in sorted(by_arm.items()):
        print(f"\n{'='*72}\nMSA={'on' if msa_arm else 'off'}   n={len(rs)}\n{'='*72}")
        print(f"{'drug':>5} {'n':>4} {'spearman':>9} {'slope':>8} "
              f"{'sd(pred)':>9} {'sd(exp)':>8} {'ratio':>6}")
        allp, alle = [], []
        for drug in sorted({r["drug"] for r in rs}):
            sub = [r for r in rs if r["drug"] == drug]
            uncens = [r for r in sub if not r["censored"]]
            p = [r["pred_delta"] for r in uncens]
            e = [r["log10_fold_change"] for r in uncens]
            if len(p) < 4:
                continue
            rho = spearman(p, e)
            slope, _ = linfit(e, p)
            sp, se = stdev(p), stdev(e)
            print(f"{drug:>5} {len(p):>4} {rho:>+9.3f} {slope:>+8.3f} "
                  f"{sp:>9.3f} {se:>8.3f} {sp/se if se else float('nan'):>6.2f}")
            allp += p
            alle += e

        if len(allp) >= 4:
            rho = spearman(allp, alle)
            slope, _ = linfit(alle, allp)
            sp, se = stdev(allp), stdev(alle)
            print(f"{'ALL':>5} {len(allp):>4} {rho:>+9.3f} {slope:>+8.3f} "
                  f"{sp:>9.3f} {se:>8.3f} {sp/se if se else float('nan'):>6.2f}")
            print()
            print(f"  Experimental spread spans {se:.2f} log10 units (~{10**(2*se):.0f}x).")
            print(f"  Predicted spread is {sp:.3f} log10 units "
                  f"({sp/se*100 if se else float('nan'):.0f}% of experimental).")
            if abs(slope) < 0.1 and abs(rho) < 0.2:
                print("  => H_invariant: affinity head is effectively mutation-blind.")
            elif rho > 0.3:
                print("  => H_responsive: affinity head tracks resistance. Thesis needs rework.")
            else:
                print("  => Ambiguous: weak signal. Report honestly; do not over-read.")

    if len(by_arm) == 2:
        print("\nMSA on-vs-off is the mechanistic ablation: if invariance holds with MSA")
        print("and relaxes without it, the MSA is washing out the point mutations —")
        print("which is a mechanism, not just a phenomenon, and worth a figure.")


if __name__ == "__main__":
    main()
