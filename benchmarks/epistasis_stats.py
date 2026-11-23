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
    held["predictions"] = {k: [None if np.isnan(v) else float(v) for v in preds[k]] for k in preds}
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


def reliability_decomposition(x_a: Sequence[float], x_b: Sequence[float], groups: Sequence) -> Dict:
    """Split test-retest reliability into BETWEEN-group and WITHIN-group parts.

    Motivation: these DMS doubles cluster on a handful of position pairs, so model epistasis
    can be reliable at the position-pair level (which pairs couple) while being noise at the
    substitution level (which amino acids couple at that pair) — or the reverse. A single
    per-double reliability cannot tell those apart, and the two support very different claims.

    `between` is the reliability of the per-group means; `within` is the reliability of the
    deviations from those means. Both are estimated on few units here, so both carry the same
    stability caveats as `reliability`.
    """
    a = np.asarray(x_a, dtype=float)
    b = np.asarray(x_b, dtype=float)
    uniq, index_of = _group_index(groups)
    if len(uniq) < 3:
        return {"valid": False, "reason": f"only {len(uniq)} groups"}
    means_a, means_b, dev_a, dev_b = [], [], [], []
    for g in uniq:
        idx = index_of[g]
        means_a.append(a[idx].mean())
        means_b.append(b[idx].mean())
        if len(idx) > 1:
            dev_a.extend(a[idx] - a[idx].mean())
            dev_b.extend(b[idx] - b[idx].mean())
    out = {
        "n_groups": len(uniq),
        "between_group": reliability(means_a, means_b),
        "note": (
            "between_group = reliability of per-position-pair means (does the model agree with "
            "itself about WHICH pairs couple); within_group = reliability of substitution-level "
            "deviations at a fixed pair."
        ),
    }
    if len(dev_a) >= 3:
        out["within_group"] = reliability(dev_a, dev_b)
    return out


# ---------------------------------------------------------------------------
# Reliability from D >= 3 decoding draws (one-way random-effects ICC)
# ---------------------------------------------------------------------------
def _icc_point(x: np.ndarray, alpha: float = 0.05) -> Dict:
    """ICC(1,1) and its variance components from a balanced (n_variants, n_draws) matrix."""
    n, d = x.shape
    row_means = x.mean(axis=1)
    grand = float(row_means.mean())
    ms_between = float(d * np.sum((row_means - grand) ** 2) / (n - 1))
    ms_within = float(np.sum((x - row_means[:, None]) ** 2) / (n * (d - 1)))
    out: Dict = {
        "n_variants": n,
        "n_draws": d,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "noise_var": ms_within,
        "signal_var": (ms_between - ms_within) / d,
    }
    out["observed_var"] = out["signal_var"] + out["noise_var"]
    out["noise_sd"] = math.sqrt(max(out["noise_var"], 0.0))
    out["signal_sd"] = math.sqrt(max(out["signal_var"], 0.0))
    denom = ms_between + (d - 1) * ms_within
    if denom <= 0:
        out.update(icc=float("nan"), valid=False, reason="both mean squares are zero (constant input)")
        return out
    out["icc"] = (ms_between - ms_within) / denom

    # Exact interval under normality (Searle; Shrout & Fleiss ICC(1,1)). MSB carries n-1 df and
    # MSW carries n(d-1) df, so this propagates sampling error in BOTH variance components --
    # which is the thing the two-draw bootstrap could not do.
    if ms_within > 0:
        f_obs = ms_between / ms_within
        df1, df2 = n - 1, n * (d - 1)
        f_lo = f_obs / stats.f.ppf(1 - alpha / 2, df1, df2)
        f_hi = f_obs * stats.f.ppf(1 - alpha / 2, df2, df1)
        out["icc_ci95_f"] = [
            float((f_lo - 1.0) / (f_lo + d - 1)),
            float((f_hi - 1.0) / (f_hi + d - 1)),
        ]
    return out


def icc_reliability(
    x: Sequence[Sequence[float]],
    n_boot: int = 1000,
    seed: int = 20261117,
    alpha: float = 0.05,
) -> Dict:
    """Test-retest reliability from D independent decoding draws, as one-way random-effects
    ICC(1,1) over an `(n_variants, n_draws)` matrix.

        MSB = d * sum_i (mean_i - grand)^2 / (n - 1)
        MSW = sum_ij (x_ij - mean_i)^2 / (n * (d - 1))
        ICC = (MSB - MSW) / (MSB + (d - 1) * MSW)

    **Why this exists.** `reliability` takes exactly two replicates and bootstraps over
    variants only, so its interval conditions on the two decoding-order sets that happened to
    be drawn: it cannot see draw-level luck (ISSUES I-23(b)). At D = 2 this function is the same
    estimator -- the algebra reduces to `1 - Var(a-b)/2 / ((Var(a)+Var(b))/2)`, up to MSW using
    n(d-1) rather than n-1 degrees of freedom -- so it generalises `reliability` rather than
    replacing it. `reliability` is kept untouched: the committed two-draw artifacts must keep
    reproducing their published numbers.

    **Three intervals, because no single one covers both axes distribution-free.**

    * `icc_ci95_f` (primary) is exact under normality and propagates error in both mean
      squares, i.e. both the variant and the draw axis.
    * `icc_ci95_boot_variants` resamples VARIANT rows with replacement, keeping all d columns.
      Distribution-free on the axis with ~300 units; says nothing about the draw axis.
    * `draw_jackknife` recomputes the ICC on each leave-one-draw-out subset. This is the
      direct, distribution-free answer to "how much did the answer depend on which order-sets
      we drew", and needs d >= 3.

    The design note called for a nonparametric bootstrap over variants AND draws. That is not
    implemented, deliberately: resampling d columns with replacement attenuates the bootstrap
    MSW by a factor (d-1)/d in expectation (0.8 at d = 5, since duplicated columns contribute
    no within-variant spread), which inflates the bootstrap ICC distribution and pushes its
    LOWER bound up -- anti-conservative in exactly the direction that matters for a rule of
    the form "retain iff the lower bound clears a threshold". The jackknife has no such bias.

    **What ICC(1,1) assumes.** Draws are exchangeable, with a common error variance. A draw
    that is systematically offset from the others is absorbed into MSW rather than modelled as
    a draw main effect, so the ICC is biased DOWN by such an effect, not up -- conservative for
    a retention rule. Detecting one needs a two-way model and more than a handful of draws.

    A non-positive ICC is reported as invalid rather than clipped, matching `reliability`.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D (n_variants, n_draws) matrix, got shape {arr.shape}")
    n, d = arr.shape
    if d < 2:
        raise ValueError(f"need at least 2 draws, got {d}")
    if n < 3:
        return {"n_variants": n, "n_draws": d, "valid": False, "reason": f"n={n} too small for a variance estimate"}
    if not np.isfinite(arr).all():
        raise ValueError("input contains non-finite values")

    out = _icc_point(arr, alpha=alpha)
    if out.get("valid") is False:
        return out
    icc = out["icc"]
    if icc <= 0:
        out.update(
            valid=False,
            reason=(
                f"icc={icc:.3f} <= 0: spread across draws is as large as spread across variants, "
                "so no attenuation factor can be estimated"
            ),
        )
        return out
    out.update(valid=True, attenuation=math.sqrt(icc))

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        p = _icc_point(arr[idx], alpha=alpha)
        if p.get("icc") is not None and math.isfinite(p["icc"]):
            draws.append(p["icc"])
    if draws:
        lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out["icc_ci95_boot_variants"] = [float(lo), float(hi)]
        out["n_boot_ok"] = len(draws)

    if d >= 3:
        loo = []
        for j in range(d):
            sub = _icc_point(np.delete(arr, j, axis=1), alpha=alpha)
            loo.append(float(sub["icc"]) if math.isfinite(sub.get("icc", float("nan"))) else float("nan"))
        finite = [v for v in loo if math.isfinite(v)]
        out["draw_jackknife"] = {
            "icc_leave_one_out": loo,
            "min": min(finite) if finite else float("nan"),
            "max": max(finite) if finite else float("nan"),
            "spread": (max(finite) - min(finite)) if finite else float("nan"),
            "kind": "leave-one-draw-out ICC (distribution-free draw-axis sensitivity)",
        }
    else:
        out["draw_jackknife"] = {"valid": False, "reason": "needs d >= 3 draws"}

    lo_f = out.get("icc_ci95_f", [float("nan")])[0]
    if math.isfinite(lo_f) and lo_f <= 0:
        out["reason"] = (
            f"icc={icc:.3f} but its 95% F interval lower bound is {lo_f:.3f} <= 0 — "
            "treat the attenuation factor as unusable, not as a correction"
        )
    out["stable"] = bool(math.isfinite(lo_f) and lo_f > 0)
    return out
