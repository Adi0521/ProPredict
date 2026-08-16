"""
Ablation: does the tier-2 AdaLead-lite search (orchestrator.mutation_search) actually beat
naive top-N single-site stacking on a case with KNOWN epistasis?

This is a METHODOLOGY / ablation check on a synthetic landscape, NOT a ProteinGym-style gate:
multi-mutant experimental ground truth is scarce, so there is no real-data label to score
against here. The point is narrow and honest — demonstrate the qualitative claim that a
combinatorial search fuses epistatic combinations that a single-site ranker structurally
cannot, and show WHERE that advantage turns on (as a function of synergy strength).

Landscape (hidden synergy, 8-residue WT "AAAAAAAA", fitness 0):
  pos1->C (+1.0), pos8->C (+1.0)   decoys: the best SINGLES, purely additive, no synergy
  pos3->D (+0.6), pos6->K (+0.6)   decent singles (elite-eligible) but not top-ranked
  pos3->D AND pos6->K              + SYNERGY bonus (swept below)
  any other substitution           -0.5

Naive top-N stacker = the strawman `scan_mutations`-style strategy: score every single
substitution, greedily stack the best distinct-position ones up to the k-cap, evaluate the
combination. It ranks the two decoys first and never fuses D3+K6.

Run: python -m benchmarks.ablation_mutation_search
"""
from typing import Callable, Dict, List, Tuple

from orchestrator.mutation_search import adalead_search

_WT = "AAAAAAAA"
_AA = "ACDEFGHIKLMNPQRSTVWY"


def make_landscape(synergy: float) -> Callable[[str], float]:
    def landscape(seq: str) -> float:
        f = 0.0
        for i, (w, c) in enumerate(zip(_WT, seq)):
            if w == c:
                continue
            if i == 0 and c == "C":
                f += 1.0
            elif i == 7 and c == "C":
                f += 1.0
            elif i == 2 and c == "D":
                f += 0.6
            elif i == 5 and c == "K":
                f += 0.6
            else:
                f += -0.5
        if seq[2] == "D" and seq[5] == "K":
            f += synergy
        return f
    return landscape


def naive_stack(landscape: Callable[[str], float], k: int) -> Tuple[float, str]:
    """Score all single substitutions, greedily stack the best distinct-position ones up to
    k, and return (true_fitness, sequence). Deterministic — this is the baseline."""
    singles = []
    for i in range(len(_WT)):
        for aa in _AA:
            if aa == _WT[i]:
                continue
            m = list(_WT)
            m[i] = aa
            singles.append((i, aa, landscape("".join(m))))
    singles.sort(key=lambda t: -t[2])
    chars, used = list(_WT), set()
    for i, aa, _ in singles:
        if len(used) >= k:
            break
        if i in used:
            continue
        chars[i] = aa
        used.add(i)
    seq = "".join(chars)
    return landscape(seq), seq


def _has_pair(seq: str) -> bool:
    return seq[2] == "D" and seq[5] == "K"


def run_ablation(
    synergies: List[float],
    n_seeds: int = 30,
    rounds: int = 25,
    candidates_per_round: int = 30,
    max_sites: int = 3,
) -> List[Dict]:
    rows = []
    for synergy in synergies:
        landscape = make_landscape(synergy)
        oracle = lambda seqs: [landscape(s) for s in seqs]  # noqa: E731
        naive_fit, _ = naive_stack(landscape, max_sites)

        best_fits, found, beat = [], 0, 0
        for seed in range(n_seeds):
            res = adalead_search(_WT, oracle, rounds=rounds,
                                 candidates_per_round=candidates_per_round,
                                 max_sites=max_sites, seed=seed, oracle_name="ablation")
            top = res.candidates[0]
            best_fits.append(top.score)
            found += _has_pair(top.sequence)
            beat += top.score > naive_fit + 1e-9
        rows.append({
            "synergy": synergy,
            "naive_fit": naive_fit,
            "adalead_mean": sum(best_fits) / len(best_fits),
            "adalead_max": max(best_fits),
            "found_pair_frac": found / n_seeds,
            "beat_naive_frac": beat / n_seeds,
            "n_seeds": n_seeds,
        })
    return rows


def main() -> None:
    synergies = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    rows = run_ablation(synergies)
    hdr = (f"{'synergy':>7} | {'naive':>6} | {'ada_mean':>8} | {'ada_max':>7} | "
           f"{'found_pair':>10} | {'beat_naive':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['synergy']:>7.1f} | {r['naive_fit']:>6.2f} | {r['adalead_mean']:>8.2f} | "
              f"{r['adalead_max']:>7.2f} | {r['found_pair_frac']:>9.0%} | "
              f"{r['beat_naive_frac']:>9.0%}")
    print(f"\n(n_seeds={rows[0]['n_seeds']}, rounds=25, lambda=30, k=3. Naive baseline is "
          "deterministic.\nAdaLead oracle calls = 1+rounds = 26 vs naive = 1 single-site "
          "pass — the search\ncosts ~26x more oracle calls to find combos the single pass "
          "cannot represent.)")


if __name__ == "__main__":
    main()
