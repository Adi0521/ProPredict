"""Statistics for the Row B stability arm (sub-question B): does ProteinMPNN `score_only`
non-additivity track *experimental* epistasis?

Pure functions only — no I/O, no ProteinMPNN, no parquet — so every piece of arithmetic that
a conclusion rests on is unit-testable without weights or data (`tests/test_epistasis_stats.py`).

Three things here are easy to get wrong and are therefore spelled out:

1. **The unknown WT offset.** ProteinGym's Tsuboyama parquet has no wild-type row, so
   `eps_exp` is computed with `dms_wt` as an explicit argument (0.0 by convention). If the
   true WT reference is `c`, every pair's `eps_exp` in that domain is off by the same
   constant `+c` (see `experimental_epistasis`). A constant per-domain
   shift leaves Pearson, Spearman and the OLS *slope* (with an intercept) unchanged, and
   changes only the intercept — hence the primary readouts are within-domain and
   offset-invariant, while signs and absolute magnitudes of `eps_exp` are NOT interpretable
   and uncentered values must not be pooled across domains.

2. **Reliability, not "SNR".** The old internal-arm harness reported
   `observed_SD / noise_SD` and called it SNR. That ratio is not a latent-signal-to-noise
   ratio: the numerator contains the noise. `reliability()` instead returns
   `1 - noise_var/observed_var`, the standard test-retest reliability, whose square root is
   the Pearson attenuation factor under independent additive errors with equal variance.
   It bounds attenuation *from model decoding noise alone* — experimental noise attenuates
   further and is not estimated here.

3. **Dependence between doubles.** Doubles share constituent singles and several variants can
   share the same residue pair, so an individual-double bootstrap understates uncertainty.
   `block_bootstrap_ci` resamples residue-pair groups instead; shared singles across groups
   leave residual dependence even then.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats


def _group_index(groups: Sequence) -> Tuple[List, Dict]:
    """(unique group keys in first-appearance order, {key: row indices}).

    Keys are normalised to tuples so a residue pair passed as a list is hashable; building the
    index by hand avoids `np.asarray` turning a list of 2-tuples into a 2-D array.
    """
    keys = [tuple(g) if isinstance(g, (list, tuple, np.ndarray)) else g for g in groups]
    index: Dict = {}
    for i, k in enumerate(keys):
        index.setdefault(k, []).append(i)
    return list(index), {k: np.asarray(v, dtype=int) for k, v in index.items()}


# ---------------------------------------------------------------------------
# Epistasis arithmetic
# ---------------------------------------------------------------------------
def experimental_epistasis(dms_double: float, dms_m1: float, dms_m2: float, dms_wt: float = 0.0) -> float:
    """eps_exp = DMS(AB) - DMS(A) - DMS(B) + DMS(WT).

    With single/double effects taken relative to WT this is the usual double-mutant-cycle
    term. If the true WT reference is `c` rather than the `dms_wt` passed in, every pair in
    the domain shifts by exactly `c - dms_wt` — a constant, hence invisible to Pearson,
    Spearman and the OLS slope-with-intercept.
    """
    return (dms_double - dms_m1 - dms_m2) + dms_wt


def model_effects(f_double: float, f_m1: float, f_m2: float, f_wt: float) -> Tuple[float, float, float, float]:
    """Return (delta1, delta2, additive_pred, eps_model) in fitness units (higher = better).

    additive_pred = delta1 + delta2 is the additive baseline's predicted double effect;
    the full model effect is `f_double - f_wt = additive_pred + eps_model`.
    """
    d1 = f_m1 - f_wt
    d2 = f_m2 - f_wt
    additive = d1 + d2
    eps = (f_double - f_wt) - additive
    return d1, d2, additive, eps


def residue_pair(mutant: str) -> Tuple[int, int]:
    """Sorted residue positions of a 'A10C:N41G'-style double, for block grouping."""
    parts = str(mutant).split(":")
    if len(parts) != 2:
        raise ValueError(f"not a double mutant: {mutant!r}")
    return tuple(sorted(int(p[1:-1]) for p in parts))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Association
# ---------------------------------------------------------------------------
def associations(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Spearman, Pearson and OLS slope/intercept of y on x. Offset-invariant in y except for
    `intercept`, which absorbs any WT-reference error."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": float(n), "spearman": float("nan"), "pearson": float("nan"), "slope": float("nan")}
    sp = stats.spearmanr(x, y)
    pe = stats.pearsonr(x, y)
    slope, intercept, r_value, p_value, stderr = stats.linregress(x, y)
    return {
        "n": float(n),
        "spearman": float(sp.statistic),
        "spearman_p": float(sp.pvalue),
        "pearson": float(pe.statistic),
        "pearson_p": float(pe.pvalue),
        "slope": float(slope),
        "slope_stderr": float(stderr),
        "intercept": float(intercept),
        "r_squared": float(r_value**2),
    }


def _stat_or_nan(fn: Callable[[np.ndarray, np.ndarray], float], x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    try:
        return float(fn(x, y))
    except Exception:
        return float("nan")


STAT_FNS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "spearman": lambda x, y: stats.spearmanr(x, y).statistic,
    "pearson": lambda x, y: stats.pearsonr(x, y).statistic,
    "slope": lambda x, y: stats.linregress(x, y).slope,
}


def bootstrap_ci(
    x: Sequence[float],
    y: Sequence[float],
    stat: str = "spearman",
    n_boot: int = 2000,
    seed: int = 20261117,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Percentile CI by resampling INDIVIDUAL doubles.

    Conditional/descriptive only: doubles are not independent (they share constituent
    singles and residue pairs), so this interval is narrower than the truth. Use
    `block_bootstrap_ci` for the sensitivity analysis.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    fn = STAT_FNS[stat]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), size=len(x))
        v = _stat_or_nan(fn, x[idx], y[idx])
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return {"lo": float("nan"), "hi": float("nan"), "n_boot_ok": 0}
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"lo": float(lo), "hi": float(hi), "n_boot_ok": len(vals), "kind": "individual-double (conditional)"}


def block_bootstrap_ci(
    x: Sequence[float],
    y: Sequence[float],
    groups: Sequence,
    stat: str = "spearman",
    n_boot: int = 2000,
    seed: int = 20261117,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Percentile CI resampling residue-pair GROUPS with replacement (all doubles in a drawn
    group travel together). Residual dependence remains: distinct residue pairs still share
    individual single mutations, which this does not break."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    uniq, index_of = _group_index(groups)
    rng = np.random.default_rng(seed)
    fn = STAT_FNS[stat]
    vals = []
    for _ in range(n_boot):
        drawn = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([index_of[uniq[d]] for d in drawn])
        v = _stat_or_nan(fn, x[idx], y[idx])
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return {"lo": float("nan"), "hi": float("nan"), "n_boot_ok": 0}
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "lo": float(lo),
        "hi": float(hi),
        "n_boot_ok": len(vals),
        "n_groups": len(uniq),
        "kind": "residue-pair block (residual dependence via shared singles)",
    }


# ---------------------------------------------------------------------------
# Decoding-noise reliability
# ---------------------------------------------------------------------------
def reliability(x_a: Sequence[float], x_b: Sequence[float], n_boot: int = 1000, seed: int = 20261117) -> Dict:
    """Test-retest reliability of model epistasis from two independent decoding-seed replicates.

        noise_var    = Var(x_a - x_b) / 2
        observed_var = (Var(x_a) + Var(x_b)) / 2
        reliability  = 1 - noise_var / observed_var

    `sqrt(reliability)` estimates the Pearson *attenuation factor* due to model decoding noise
    alone, under independent additive errors of equal variance. It is NOT a Spearman
    correction, NOT a total-noise ceiling (experimental noise attenuates further), and is a
    noisy estimate itself at these sample sizes. A non-positive reliability means the
    replicate difference is as large as the spread — reported as invalid rather than clipped.
    """
    a = np.asarray(x_a, dtype=float)
    b = np.asarray(x_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"replicate length mismatch: {a.shape} vs {b.shape}")
    n = len(a)
    out: Dict = {"n": n}
    if n < 3:
        out.update(valid=False, reason=f"n={n} too small for a variance estimate")
        return out
    noise_var = float(np.var(a - b, ddof=1) / 2.0)
    observed_var = float((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    out.update(noise_var=noise_var, observed_var=observed_var)
    if observed_var <= 0:
        out.update(valid=False, reason="observed variance is zero")
        return out
    rel = 1.0 - noise_var / observed_var
    out["reliability"] = rel
    out["replicate_corr"] = _stat_or_nan(STAT_FNS["pearson"], a, b)
    if rel <= 0:
        out.update(
            valid=False,
            reason=(
                f"reliability={rel:.3f} <= 0: replicate difference is as large as the spread, "
                "so no attenuation factor can be estimated"
            ),
        )
        return out
    out.update(valid=True, attenuation=math.sqrt(rel))

    # The point estimate is itself noisy at these sample sizes, so bootstrap it over pairs and
    # say so: a lower bound at or below zero means the reliability (and any attenuation factor
    # derived from it) is not a trustworthy number, even though the point estimate is positive.
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        aa, bb = a[idx], b[idx]
        ov = (np.var(aa, ddof=1) + np.var(bb, ddof=1)) / 2.0
        if ov > 0:
            draws.append(1.0 - (np.var(aa - bb, ddof=1) / 2.0) / ov)
    if draws:
        lo, hi = np.percentile(draws, [2.5, 97.5])
        out["reliability_ci95"] = [float(lo), float(hi)]
        out["stable"] = bool(lo > 0)
        if not out["stable"]:
            out["reason"] = (
                f"reliability={rel:.3f} but its 95% CI [{lo:.3f}, {hi:.3f}] includes 0 — "
                "treat the attenuation factor as unusable, not as a correction"
            )
    return out


# ---------------------------------------------------------------------------
# Additive baseline vs full (non-additive) model, with grouped held-out evaluation
# ---------------------------------------------------------------------------
def grouped_folds(groups: Sequence, n_folds: int = 5, seed: int = 20261117) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Deterministic group-disjoint folds: every double sharing a residue pair lands in the
    same fold, so a calibration fitted on train groups is evaluated on unseen pairs."""
    uniq, index_of = _group_index(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    fold_of = np.empty(sum(len(v) for v in index_of.values()), dtype=int)
    for i, g in enumerate(order):
        fold_of[index_of[uniq[g]]] = i % n_folds
    folds = []
    for k in range(n_folds):
        test = np.flatnonzero(fold_of == k)
        train = np.flatnonzero(fold_of != k)
        if len(test) and len(train) >= 3:
            folds.append((train, test))
    return folds


def additive_vs_full(
    y_exp: Sequence[float],
    x_additive: Sequence[float],
    x_full: Sequence[float],
    groups: Sequence,
    n_folds: int = 5,
    seed: int = 20261117,
) -> Dict:
    """Does the model's NON-additive double score beat the additive baseline at predicting the
    measured double-mutant effect?

    `y_exp`   — experimental double effect (relative to the assumed WT reference).
    `x_additive` — sum of model single effects, delta1 + delta2.
    `x_full`  — model double effect, (f_double - f_wt) = x_additive + eps_model.

    In-sample rank/linear association for both, plus a grouped held-out comparison: a linear
    calibration `y = a + b*x` is fitted on train groups only and scored on held-out groups
    (model and experiment are in different units, so a calibration is required and slope=1 is
    NOT the ideal). Held-out numbers are the predictive claim; in-sample ones are not.
    """
    y = np.asarray(y_exp, dtype=float)
    xa = np.asarray(x_additive, dtype=float)
    xf = np.asarray(x_full, dtype=float)
    out: Dict = {
        "in_sample": {"additive": associations(xa, y), "full": associations(xf, y)},
        "n": len(y),
    }
    folds = grouped_folds(groups, n_folds=n_folds, seed=seed)
    out["n_folds_used"] = len(folds)
    out["n_groups"] = len(_group_index(groups)[0])
    if not folds:
        out["held_out"] = {"valid": False, "reason": "no usable group-disjoint folds"}
        return out

    preds = {"additive": np.full(len(y), np.nan), "full": np.full(len(y), np.nan)}
    for train, test in folds:
        for name, x in (("additive", xa), ("full", xf)):
            if np.std(x[train]) == 0:
                continue
            lr = stats.linregress(x[train], y[train])
            preds[name][test] = lr.intercept + lr.slope * x[test]

    held: Dict = {"valid": True, "n_scored": int(np.sum(~np.isnan(preds["full"])))}
    for name in ("additive", "full"):
        p = preds[name]
        ok = ~np.isnan(p)
        held[name] = {
            "pearson": _stat_or_nan(STAT_FNS["pearson"], p[ok], y[ok]),
            "spearman": _stat_or_nan(STAT_FNS["spearman"], p[ok], y[ok]),
            "rmse": float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))) if ok.any() else float("nan"),
        }
    held["delta_pearson_full_minus_additive"] = held["full"]["pearson"] - held["additive"]["pearson"]
    held["delta_rmse_full_minus_additive"] = held["full"]["rmse"] - held["additive"]["rmse"]
    held["note"] = (
        "Group-disjoint by residue pair; distinct pairs still share individual single "
        "mutations, so train/test are not fully independent."
    )
    out["held_out"] = held
    return out


def paired_delta_ci(
    y_exp: Sequence[float],
    x_additive: Sequence[float],
    x_full: Sequence[float],
    groups: Sequence,
    stat: str = "pearson",
    n_boot: int = 2000,
    seed: int = 20261117,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Block-bootstrap CI for the PAIRED difference (full - additive) in association with the
    experimental effect, resampled over residue-pair groups on the same draws for both."""
    y = np.asarray(y_exp, dtype=float)
    xa = np.asarray(x_additive, dtype=float)
    xf = np.asarray(x_full, dtype=float)
    uniq, index_of = _group_index(groups)
    fn = STAT_FNS[stat]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        drawn = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([index_of[uniq[d]] for d in drawn])
        v_full = _stat_or_nan(fn, xf[idx], y[idx])
        v_add = _stat_or_nan(fn, xa[idx], y[idx])
        if not (math.isnan(v_full) or math.isnan(v_add)):
            vals.append(v_full - v_add)
    if not vals:
        return {"lo": float("nan"), "hi": float("nan"), "n_boot_ok": 0}
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "lo": float(lo),
        "hi": float(hi),
        "median": float(np.median(vals)),
        "n_boot_ok": len(vals),
        "kind": f"paired block bootstrap of delta {stat} (full - additive)",
    }
