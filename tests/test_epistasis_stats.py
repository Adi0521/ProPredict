"""
Tests for the Row B stability arm (sub-question B): does `score_only` non-additivity track
experimental epistasis?

Everything a conclusion rests on is tested without ProteinMPNN weights, without a GPU and
without the 90 MB ProteinGym parquet:

  * `benchmarks/epistasis_stats.py` is pure, so the epistasis arithmetic, the WT-offset
    invariance that licenses the primary readouts, the reliability formula and the grouped
    held-out comparison are all exercised directly on synthetic arrays;
  * the harness's data handling (`load_assay` exclusion rules, experimental-score semantics,
    PDB header parsing, domain selection) runs against a tiny parquet written in a tmpdir and
    hand-written PDB headers;
  * the INTERNAL arm (sub-question A) is run end-to-end against a fake oracle to assert its
    behaviour and output format did not change when the stability arm was added.
"""

import json
import os

import numpy as np
import pytest

from benchmarks import benchmark_score_only_epistasis as bse
from benchmarks import epistasis_stats as es


# ---------------------------------------------------------------------------
# Epistasis arithmetic
# ---------------------------------------------------------------------------
def test_experimental_epistasis_matches_double_mutant_cycle():
    # DMS(AB) = -1.0, DMS(A) = -0.4, DMS(B) = -0.3, WT = 0 -> eps = -0.3
    assert es.experimental_epistasis(-1.0, -0.4, -0.3, 0.0) == pytest.approx(-0.3)
    # Perfectly additive pair -> exactly zero.
    assert es.experimental_epistasis(-0.7, -0.4, -0.3, 0.0) == pytest.approx(0.0)
    # A nonzero WT reference enters with a + sign.
    assert es.experimental_epistasis(-1.0, -0.4, -0.3, 0.5) == pytest.approx(0.2)


def test_model_effects_decomposition_is_exact():
    d1, d2, additive, eps = es.model_effects(f_double=-2.0, f_m1=-0.5, f_m2=-0.8, f_wt=0.0)
    assert (d1, d2) == (-0.5, -0.8)
    assert additive == pytest.approx(-1.3)
    assert eps == pytest.approx(-0.7)
    # The identity the additive-baseline comparison relies on.
    assert additive + eps == pytest.approx(-2.0 - 0.0)


def test_residue_pair_is_sorted_and_rejects_non_doubles():
    assert es.residue_pair("N41G:D28A") == (28, 41)
    assert es.residue_pair("D28A:N41G") == (28, 41)
    with pytest.raises(ValueError):
        es.residue_pair("D28A")


# ---------------------------------------------------------------------------
# The WT-offset invariance that licenses the primary readouts
# ---------------------------------------------------------------------------
def test_unknown_wt_offset_shifts_eps_exp_by_a_constant_only():
    rng = np.random.default_rng(7)
    dms_d = rng.normal(size=40)
    dms_1 = rng.normal(size=40)
    dms_2 = rng.normal(size=40)
    eps_0 = np.array([es.experimental_epistasis(d, a, b, 0.0) for d, a, b in zip(dms_d, dms_1, dms_2, strict=True)])
    eps_c = np.array([es.experimental_epistasis(d, a, b, 0.75) for d, a, b in zip(dms_d, dms_1, dms_2, strict=True)])
    assert np.allclose(eps_c - eps_0, 0.75)


def test_associations_are_invariant_to_the_wt_offset_except_the_intercept():
    rng = np.random.default_rng(11)
    x = rng.normal(size=60)  # eps_model
    y = 0.4 * x + rng.normal(scale=0.3, size=60)  # eps_exp at WT ref 0
    a0 = es.associations(x, y)
    a1 = es.associations(x, y + 1.3)  # a wrong WT reference
    for key in ("spearman", "pearson", "slope", "r_squared"):
        assert a1[key] == pytest.approx(a0[key], rel=1e-9, abs=1e-9)
    assert a1["intercept"] == pytest.approx(a0["intercept"] + 1.3)


def test_associations_degrade_gracefully_on_constant_input():
    out = es.associations([1.0, 1.0, 1.0, 1.0], [2.0, 3.0, 4.0, 5.0])
    assert np.isnan(out["spearman"]) and np.isnan(out["pearson"])


# ---------------------------------------------------------------------------
# Decoding-noise reliability
# ---------------------------------------------------------------------------
def test_reliability_is_one_for_identical_replicates():
    x = np.linspace(-1, 1, 30)
    out = es.reliability(x, x)
    assert out["valid"] is True
    assert out["reliability"] == pytest.approx(1.0)
    assert out["attenuation"] == pytest.approx(1.0)


def test_reliability_matches_the_formula_on_known_variances():
    rng = np.random.default_rng(3)
    signal = rng.normal(scale=1.0, size=4000)
    a = signal + rng.normal(scale=0.5, size=4000)
    b = signal + rng.normal(scale=0.5, size=4000)
    out = es.reliability(a, b)
    expected = 1.0 - (np.var(a - b, ddof=1) / 2) / ((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    assert out["reliability"] == pytest.approx(expected)
    # signal var 1, noise var 0.25 -> reliability ~ 1/1.25 = 0.8
    assert out["reliability"] == pytest.approx(0.8, abs=0.05)
    assert out["attenuation"] == pytest.approx(np.sqrt(out["reliability"]))


def test_reliability_flags_pure_noise_as_unstable_rather_than_trustworthy():
    rng = np.random.default_rng(5)
    a = rng.normal(size=200)
    b = rng.normal(size=200)  # independent: no shared signal
    out = es.reliability(a, b)
    # The point estimate scatters around 0 and can land just above it, so the guard that
    # matters is the CI: it must not be sold as a usable attenuation factor.
    assert out["reliability"] < 0.3
    assert out["stable"] is False
    assert out["reliability_ci95"][0] <= 0
    assert out["reason"]


def test_reliability_is_stable_when_the_shared_signal_dominates():
    rng = np.random.default_rng(41)
    signal = rng.normal(size=300)
    out = es.reliability(signal + rng.normal(scale=0.3, size=300), signal + rng.normal(scale=0.3, size=300))
    assert out["valid"] is True and out["stable"] is True
    assert out["reliability_ci95"][0] > 0


def test_reliability_reports_negative_estimates_as_invalid():
    # Anti-correlated replicates: the difference variance exceeds the spread.
    x = np.linspace(-1, 1, 50)
    out = es.reliability(x, -x)
    assert out["valid"] is False
    assert "attenuation" not in out
    assert out["reason"]


def test_reliability_flags_tiny_samples_and_rejects_mismatched_lengths():
    assert es.reliability([1.0, 2.0], [1.0, 2.0])["valid"] is False
    with pytest.raises(ValueError):
        es.reliability([1.0, 2.0, 3.0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# D >= 3 draws: one-way random-effects ICC (Row C's estimator)
# ---------------------------------------------------------------------------
def _draw_matrix(n=300, d=5, signal_sd=1.0, noise_sd=0.2, seed=0):
    """(n, d) matrix of per-variant truth plus independent per-draw decoding noise."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, signal_sd, n)
    return mu[:, None] + rng.normal(0.0, noise_sd, (n, d))


def test_icc_recovers_known_variance_components():
    # ICC's definition on this generator: signal_var / (signal_var + noise_var).
    x = _draw_matrix(n=400, d=5, signal_sd=1.0, noise_sd=0.25, seed=7)
    true = 1.0**2 / (1.0**2 + 0.25**2)
    out = es.icc_reliability(x, n_boot=200)
    assert out["valid"] is True
    assert out["icc"] == pytest.approx(true, abs=0.02)
    assert out["noise_sd"] == pytest.approx(0.25, rel=0.1)
    assert out["signal_sd"] == pytest.approx(1.0, rel=0.1)
    assert out["n_variants"] == 400 and out["n_draws"] == 5


def test_icc_matches_the_two_draw_reliability_at_d_equals_two():
    """The new estimator GENERALISES the committed one rather than replacing it.

    At d=2 the ICC(1,1) algebra reduces to `1 - Var(a-b)/2 / ((Var(a)+Var(b))/2)`; the only
    difference is MSW's n(d-1) degrees of freedom, an O(1/n) effect. If this ever drifts, the
    two-draw numbers already published in Process/mutation-scoring-consistency.md and the new
    D-draw numbers would no longer be on the same scale.
    """
    for seed in (0, 1, 2, 3):
        x = _draw_matrix(n=300, d=2, signal_sd=1.0, noise_sd=0.4, seed=seed)
        old = es.reliability(x[:, 0], x[:, 1], n_boot=50)
        new = es.icc_reliability(x, n_boot=50)
        assert new["icc"] == pytest.approx(old["reliability"], abs=2e-3)


def test_icc_is_one_for_identical_draws_and_never_calls_pure_noise_stable():
    same = np.tile(np.linspace(0.0, 1.0, 50)[:, None], (1, 4))
    assert es.icc_reliability(same, n_boot=50)["icc"] == pytest.approx(1.0)

    # No shared per-variant signal at all: every draw is independent noise. The point estimate
    # scatters around zero and can land slightly positive, so what must hold is not `valid is
    # False` but that the interval refuses it: a near-zero ICC is never reported as usable.
    for seed in (5, 6, 7):
        out = es.icc_reliability(np.random.default_rng(seed).normal(size=(200, 4)), n_boot=100)
        assert abs(out["icc"]) < 0.06
        assert out["stable"] is False
        assert "reason" in out


def test_icc_f_interval_brackets_the_estimate_and_narrows_with_more_draws():
    narrow = es.icc_reliability(_draw_matrix(n=300, d=8, noise_sd=0.5, seed=11), n_boot=100)
    wide = es.icc_reliability(_draw_matrix(n=300, d=2, noise_sd=0.5, seed=11), n_boot=100)
    for out in (narrow, wide):
        lo, hi = out["icc_ci95_f"]
        assert lo < out["icc"] < hi
    assert (narrow["icc_ci95_f"][1] - narrow["icc_ci95_f"][0]) < (wide["icc_ci95_f"][1] - wide["icc_ci95_f"][0])


def test_draw_jackknife_needs_three_draws_and_finds_the_odd_draw_out():
    two = es.icc_reliability(_draw_matrix(n=100, d=2, seed=3), n_boot=50)
    assert two["draw_jackknife"]["valid"] is False

    # Draw 2 is noisier than the rest; dropping it should give the HIGHEST leave-one-out ICC.
    x = _draw_matrix(n=300, d=5, noise_sd=0.2, seed=4)
    x[:, 2] += np.random.default_rng(99).normal(0.0, 0.8, x.shape[0])
    loo = es.icc_reliability(x, n_boot=100)["draw_jackknife"]["icc_leave_one_out"]
    assert np.argmax(loo) == 2
    assert loo[2] > max(v for j, v in enumerate(loo) if j != 2)


def test_icc_treats_a_systematically_offset_draw_as_noise_not_as_signal():
    """One order-set sitting at a different level is absorbed into MSW, so the ICC goes DOWN.

    ICC(1,1) has no draw main effect, which makes it conservative for a retention rule — the
    direction a decision rule can tolerate. A two-way model would be needed to separate this
    from per-variant noise, and D=5 cannot support one.
    """
    x = _draw_matrix(n=300, d=5, noise_sd=0.2, seed=8)
    before = es.icc_reliability(x, n_boot=100)["icc"]
    x[:, 0] += 0.5
    after = es.icc_reliability(x, n_boot=100)["icc"]
    assert after < before


def test_naive_variant_and_draw_bootstrap_would_be_anticonservative():
    """Why `icc_reliability` does NOT resample draw columns with replacement.

    Duplicated columns contribute no within-variant spread, so the bootstrap MSW is attenuated
    by ~(d-1)/d and the resulting ICC distribution sits ABOVE the point estimate. Its lower
    bound would then clear a threshold the data does not actually support. Locked in as a test
    because the Row C design note originally called for exactly this resampling scheme.
    """
    x = _draw_matrix(n=300, d=5, noise_sd=0.3, seed=12)
    point = es.icc_reliability(x, n_boot=100)["icc"]
    rng = np.random.default_rng(1)
    naive = [es._icc_point(x[:, rng.integers(0, 5, size=5)])["icc"] for _ in range(300)]
    # Duplication can only remove within-variant spread, so the naive distribution is bounded
    # BELOW by the point estimate (hit exactly when the resample happens to be a permutation)
    # and centred above it. Any percentile lower bound taken from it is therefore >= the
    # estimate, which is not an interval at all.
    assert min(naive) >= point - 1e-12
    assert np.median(naive) > point
    assert np.percentile(naive, 2.5) >= point


def test_icc_is_deterministic_for_a_fixed_seed_and_rejects_malformed_input():
    x = _draw_matrix(n=120, d=4, seed=2)
    assert (
        es.icc_reliability(x, n_boot=100)["icc_ci95_boot_variants"]
        == (es.icc_reliability(x, n_boot=100)["icc_ci95_boot_variants"])
    )
    assert es.icc_reliability(np.zeros((2, 4)), n_boot=10)["valid"] is False  # n too small
    with pytest.raises(ValueError):
        es.icc_reliability([1.0, 2.0, 3.0])  # 1-D
    with pytest.raises(ValueError):
        es.icc_reliability(np.zeros((10, 1)))  # single draw
    with pytest.raises(ValueError):
        es.icc_reliability([[1.0, 2.0], [3.0, np.nan], [1.0, 1.0]])


# ---------------------------------------------------------------------------
# Dependence-aware uncertainty
# ---------------------------------------------------------------------------
def test_block_bootstrap_accepts_list_groups_and_counts_distinct_pairs():
    rng = np.random.default_rng(13)
    x = rng.normal(size=60)
    y = 0.5 * x + rng.normal(scale=0.5, size=60)
    groups = [[10, 20] if i % 3 == 0 else [11, 30] if i % 3 == 1 else [12, 40] for i in range(60)]
    out = es.block_bootstrap_ci(x, y, groups, stat="spearman", n_boot=200)
    assert out["n_groups"] == 3
    assert out["lo"] <= out["hi"]
    assert "block" in out["kind"]


def test_block_bootstrap_is_wider_than_the_individual_bootstrap_when_pairs_cluster():
    rng = np.random.default_rng(17)
    # 20 residue pairs, 10 near-duplicate doubles each: individual resampling sees 200
    # "independent" points, block resampling correctly sees 20.
    xs, ys, groups = [], [], []
    for g in range(20):
        base_x, base_y = rng.normal(), rng.normal()
        for _ in range(10):
            xs.append(base_x + rng.normal(scale=0.01))
            ys.append(base_y + rng.normal(scale=0.01))
            groups.append((g, g + 100))
    ind = es.bootstrap_ci(xs, ys, stat="pearson", n_boot=400)
    blk = es.block_bootstrap_ci(xs, ys, groups, stat="pearson", n_boot=400)
    assert (blk["hi"] - blk["lo"]) > (ind["hi"] - ind["lo"])
    assert ind["kind"].startswith("individual-double")


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(19)
    x = rng.normal(size=40)
    y = rng.normal(size=40)
    a = es.bootstrap_ci(x, y, n_boot=100, seed=123)
    b = es.bootstrap_ci(x, y, n_boot=100, seed=123)
    assert (a["lo"], a["hi"]) == (b["lo"], b["hi"])


# ---------------------------------------------------------------------------
# Grouped held-out evaluation and the additive baseline
# ---------------------------------------------------------------------------
def test_grouped_folds_never_split_a_residue_pair_across_train_and_test():
    groups = [(i % 10, 100 + i % 10) for i in range(80)]
    folds = es.grouped_folds(groups, n_folds=5, seed=1)
    assert folds
    g = np.array([str(x) for x in groups])
    for train, test in folds:
        assert not (set(g[train]) & set(g[test]))


def test_additive_vs_full_prefers_the_full_model_when_epistasis_is_real():
    rng = np.random.default_rng(23)
    n = 240
    groups = [(i % 40, 100 + i % 40) for i in range(n)]
    d1 = rng.normal(size=n)
    d2 = rng.normal(size=n)
    eps = rng.normal(scale=1.0, size=n)
    x_add = d1 + d2
    x_full = x_add + eps
    # Experiment is a linear function of the FULL model effect (different units) + noise.
    y = 2.0 * x_full + 0.5 + rng.normal(scale=0.2, size=n)
    out = es.additive_vs_full(y, x_add, x_full, groups)
    assert out["held_out"]["valid"] is True
    assert out["held_out"]["full"]["pearson"] > out["held_out"]["additive"]["pearson"]
    assert out["held_out"]["delta_rmse_full_minus_additive"] < 0  # full predicts better
    assert out["n_groups"] == 40


def test_additive_vs_full_shows_no_gain_when_the_model_epistasis_is_noise():
    rng = np.random.default_rng(29)
    n = 240
    groups = [(i % 40, 100 + i % 40) for i in range(n)]
    d1 = rng.normal(size=n)
    d2 = rng.normal(size=n)
    x_add = d1 + d2
    x_full = x_add + rng.normal(scale=1.0, size=n)  # the extra term is pure noise
    y = 2.0 * x_add + rng.normal(scale=0.2, size=n)
    out = es.additive_vs_full(y, x_add, x_full, groups)
    assert out["held_out"]["full"]["pearson"] < out["held_out"]["additive"]["pearson"]
    delta = es.paired_delta_ci(y, x_add, x_full, groups, stat="pearson", n_boot=300)
    assert delta["hi"] < 0  # the full model is reliably WORSE here


def test_additive_vs_full_reports_in_sample_and_held_out_separately():
    rng = np.random.default_rng(31)
    n = 100
    groups = [(i, i + 1000) for i in range(n)]  # every double its own group
    x_add = rng.normal(size=n)
    x_full = x_add + rng.normal(size=n)
    y = rng.normal(size=n)
    out = es.additive_vs_full(y, x_add, x_full, groups)
    assert set(out["in_sample"]) == {"additive", "full"}
    # In-sample association of noise is not reported as predictive performance.
    assert out["held_out"]["n_scored"] == n


# ---------------------------------------------------------------------------
# Harness: assay loading, exclusions and experimental semantics
# ---------------------------------------------------------------------------
WT = "ACDEFGHIKL"


def _variant(seq, pos, aa):
    return seq[: pos - 1] + aa + seq[pos:]


def _write_parquet(tmp_path, rows):
    import pandas as pd

    path = os.path.join(tmp_path, "mini.parquet")
    pd.DataFrame(rows).to_parquet(path)
    return path


def _row(mutant, seq, score, dms_id="TEST_ASSAY"):
    return {
        "mutant": mutant,
        "mutated_sequence": seq,
        "target_seq": WT,
        "DMS_score": score,
        "DMS_id": dms_id,
    }


def test_load_assay_keeps_valid_variants_and_counts_every_exclusion(tmp_path):
    rows = [
        _row("A1V", _variant(WT, 1, "V"), -0.5),
        _row("C2S", _variant(WT, 2, "S"), -0.3),
        _row("D3N", _variant(WT, 3, "N"), -0.2),
        # usable double (both singles present)
        _row("A1V:C2S", _variant(_variant(WT, 1, "V"), 2, "S"), -1.0),
        # double whose second single was never measured -> excluded
        _row("A1V:E4Q", _variant(_variant(WT, 1, "V"), 4, "Q"), -1.1),
        # exclusion cases
        _row("A1", _variant(WT, 1, "V"), -0.4),  # unparseable
        _row("A99V", _variant(WT, 1, "V"), -0.4),  # position out of range
        _row("G1V", _variant(WT, 1, "V"), -0.4),  # WT letter mismatch (WT[0] == 'A')
        _row("C2S", _variant(WT, 2, "S"), -0.9),  # duplicate mutant
        _row("F5L", _variant(WT, 5, "L")[:-1], -0.4),  # sequence length mismatch
        _row("G6A", _variant(WT, 6, "A"), float("nan")),  # missing score
        _row("OTHER", WT, -1.0, dms_id="SOMETHING_ELSE"),  # different assay
    ]
    assay = bse.load_assay("TEST_ASSAY", parquet=_write_parquet(tmp_path, rows))

    assert assay.wt_seq == WT
    assert sorted(assay.singles) == ["A1V", "C2S", "D3N"]
    assert [d["mutant"] for d in assay.doubles] == ["A1V:C2S"]
    exc = assay.meta["exclusions"]
    assert exc["unparseable_mutant"] == 1
    assert exc["position_out_of_range"] == 1
    assert exc["wt_letter_mismatch"] == 1
    assert exc["duplicate_mutant"] == 1
    assert exc["sequence_length_mismatch"] == 1
    assert exc["missing_dms_score"] == 1
    assert exc["double_missing_a_single"] == 1
    assert assay.meta["n_doubles_usable"] == 1
    assert assay.meta["n_doubles_total"] == 2


def test_load_variants_shim_preserves_the_internal_arm_signature(tmp_path, monkeypatch):
    rows = [
        _row("A1V", _variant(WT, 1, "V"), -0.5),
        _row("C2S", _variant(WT, 2, "S"), -0.3),
        _row("A1V:C2S", _variant(_variant(WT, 1, "V"), 2, "S"), -1.0),
    ]
    path = _write_parquet(tmp_path, rows)
    monkeypatch.setattr(bse, "_PARQUET", path)
    wt, singles, doubles = bse.load_variants("TEST_ASSAY")
    assert wt == WT
    assert singles["A1V"] == (_variant(WT, 1, "V"), pytest.approx(-0.5))
    assert doubles[0]["components"] == ["A1V", "C2S"]


def test_experimental_semantics_records_the_unknown_wt_offset_and_checks_censoring(tmp_path):
    rows = [_row("A1V", _variant(WT, 1, "V"), -0.5), _row("C2S", _variant(WT, 2, "S"), -0.3)]
    rows.append(_row("A1V:C2S", _variant(_variant(WT, 1, "V"), 2, "S"), -1.0))
    assay = bse.load_assay("TEST_ASSAY", parquet=_write_parquet(tmp_path, rows))
    sem = bse.check_experimental_semantics(assay)
    assert sem["wt_row_present"] is False
    assert sem["wt_offset_known"] is False
    assert sem["wt_reference_assumed"] == 0.0
    assert sem["proteingym_source_column"] == "ddG_ML_float"
    assert sem["censoring_detected"] is False
    assert sem["score_range"] == [pytest.approx(-1.0), pytest.approx(-0.3)]


# ---------------------------------------------------------------------------
# Harness: structure metadata and domain selection
# ---------------------------------------------------------------------------
def test_pdb_metadata_reads_method_and_resolution():
    xray = "EXPDTA    X-RAY DIFFRACTION\nREMARK   2 RESOLUTION.    1.60 ANGSTROMS.\nATOM      1  N\n"
    nmr = "EXPDTA    SOLUTION NMR\nREMARK   2 RESOLUTION. NOT APPLICABLE.\nMODEL        1\nMODEL        2\n"
    mx = bse.pdb_metadata(xray)
    mn = bse.pdb_metadata(nmr)
    assert mx["method"] == "X-RAY DIFFRACTION" and mx["resolution_angstrom"] == pytest.approx(1.60)
    assert mn["method"] == "SOLUTION NMR" and mn["resolution_angstrom"] is None
    assert mn["n_models"] == 2 and mx["n_models"] == 1


def test_domain_selection_takes_the_first_eligible_candidates_in_order(monkeypatch):
    eligible = {"4G3O", "3L1X", "1PV0"}  # 3DKM deliberately ineligible here

    def fake_eval(dms_id, pdb_code, min_doubles=0):
        return {
            "dms_id": dms_id,
            "pdb_code": pdb_code,
            "eligible": pdb_code in eligible,
            "reasons": [] if pdb_code in eligible else ["synthetic rejection"],
        }

    monkeypatch.setattr(bse, "evaluate_domain", fake_eval)
    selected, records = bse.select_stability_domains(n_extra=2)
    assert [code for _, code in selected] == ["4G3O", "3L1X", "1PV0"]
    # Evaluation stops once the quota is filled, and that is recorded rather than silent.
    not_evaluated = [r["pdb_code"] for r in records if not r.get("evaluated", True)]
    assert "1V1C" in not_evaluated
    # The ineligible candidate keeps its rejection reason in the record.
    rejected = [r for r in records if r["pdb_code"] == "3DKM"][0]
    assert rejected["eligible"] is False and rejected["reasons"]


def test_domain_selection_reports_a_shortfall_instead_of_relaxing_criteria(monkeypatch):
    monkeypatch.setattr(
        bse,
        "evaluate_domain",
        lambda dms_id, pdb_code, min_doubles=0: {
            "dms_id": dms_id,
            "pdb_code": pdb_code,
            "eligible": pdb_code == "4G3O",
            "reasons": ["synthetic rejection"],
        },
    )
    selected, records = bse.select_stability_domains(n_extra=2)
    assert [code for _, code in selected] == ["4G3O"]
    assert all(r.get("evaluated", True) for r in records)  # every candidate was tried


# ---------------------------------------------------------------------------
# Internal arm (sub-question A) — behaviour preserved
# ---------------------------------------------------------------------------
class _Args:
    """argparse-shaped stand-in for the internal arm."""

    arm = "internal"
    n_doubles = 2
    decoding_orders = 8
    seed = 0
    noise_check = False
    force = False
    dms_id = None
    # The internal arm (sub-question A) was run under the upstream CLI, so reproducing its
    # committed artifact means pinning v1 — the harness default is now v2.
    score_protocol = "score_only/v1"

    def __init__(self, out):
        self.out = out


def test_internal_arm_still_writes_stamped_jsonl_with_the_same_epistasis_formula(tmp_path, monkeypatch):
    wt = WT
    s1, s2 = _variant(wt, 1, "V"), _variant(wt, 2, "S")
    dbl = _variant(s1, 2, "S")
    singles = {"A1V": (s1, -0.5), "C2S": (s2, -0.3)}
    doubles = [{"mutant": "A1V:C2S", "sequence": dbl, "dms_score": -1.0, "components": ["A1V", "C2S"]}]

    monkeypatch.setattr(bse, "load_variants", lambda dms_id: (wt, singles, doubles))
    monkeypatch.setattr(bse, "_fetch_pdb", lambda code: "RAW")
    monkeypatch.setattr(bse, "trim_pdb_to_dms", lambda raw, seq, code=None: "PDB")
    monkeypatch.setattr(bse, "_ca_distance", lambda pdb, i, j: 7.5)
    # f(WT)=0.0, f(A1V)=-0.1, f(C2S)=-0.2, f(double)=-0.5 -> eps = -0.5 - (-0.3) = -0.2
    monkeypatch.setattr(bse, "score_only_oracle", lambda *a, **k: [0.0, -0.1, -0.2, -0.5])
    monkeypatch.setattr(bse, "PROTEINMPNN_PATH", "/fake/proteinmpnn")

    out = os.path.join(tmp_path, "internal.jsonl")
    bse.cmd_internal(_Args(out))
    rows = [json.loads(line) for line in open(out)]
    assert len(rows) == 1
    assert rows[0]["mutant"] == "A1V:C2S"
    assert rows[0]["delta1"] == pytest.approx(-0.1)
    assert rows[0]["delta2"] == pytest.approx(-0.2)
    assert rows[0]["eps_model"] == pytest.approx(-0.2)
    assert rows[0]["ca_distance"] == pytest.approx(7.5)
    # No stability-arm keys leak into the internal arm's rows.
    assert "eps_exp" not in rows[0]


def test_internal_arm_refuses_to_overwrite_an_existing_artifact(tmp_path, monkeypatch):
    out = os.path.join(tmp_path, "internal.jsonl")
    with open(out, "w") as fh:
        fh.write("{}\n")
    monkeypatch.setattr(bse, "PROTEINMPNN_PATH", "/fake/proteinmpnn")
    with pytest.raises(SystemExit):
        bse.cmd_internal(_Args(out))
    assert open(out).read() == "{}\n"  # untouched


def test_default_internal_output_path_is_parameter_stamped():
    from benchmarks._artifacts import stamped_path

    path = stamped_path("/tmp", "score_only_epistasis", "jsonl", n=300, k=32)
    assert path.endswith("score_only_epistasis_n300_k32.jsonl")
