"""Tests for Row C Study 1's harness (`benchmarks/benchmark_score_reliability.py`).

The scoring itself needs ProteinMPNN weights and the PDB, so everything here runs against a
fake oracle and stub backbones. What is tested is the part that can silently be wrong without
anything crashing: the panel sampling, the (M, D) matrix assembly, the pre-registered decision
rule, the K^-0.5 fit, and the guards that stop a run from mixing parameters.
"""

import numpy as np
import pytest

from benchmarks import benchmark_score_reliability as bsr

WT = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"


# ---------------------------------------------------------------------------
# Panel sampling
# ---------------------------------------------------------------------------
def test_sampled_mutants_are_well_formed_and_never_wild_type():
    picked = bsr.sample_single_mutants(WT, 25, [20261121, 7])
    assert len(picked) == 25
    assert len({label for label, _ in picked}) == 25
    for label, seq in picked:
        wt_aa, mut_aa, pos = label[0], label[-1], int(label[1:-1])
        assert len(seq) == len(WT)
        assert wt_aa == WT[pos - 1] and mut_aa != wt_aa  # label agrees with WT, and mutates it
        assert seq[pos - 1] == mut_aa
        assert [i for i in range(len(WT)) if seq[i] != WT[i]] == [pos - 1]  # exactly one change


def test_sampled_mutants_are_reproducible_and_backbone_specific():
    a = bsr.sample_single_mutants(WT, 20, [20261121, 7])
    assert a == bsr.sample_single_mutants(WT, 20, [20261121, 7])
    assert a != bsr.sample_single_mutants(WT, 20, [20261121, 8])  # different backbone
    assert a != bsr.sample_single_mutants(WT, 20, [1, 7])  # different --variant-seed


def test_sampling_refuses_to_ask_for_more_cells_than_exist():
    with pytest.raises(SystemExit):
        bsr.sample_single_mutants("MKT", 100, [1, 1])  # 3 x 19 = 57 cells


# ---------------------------------------------------------------------------
# One (backbone, K) point, against a fake oracle
# ---------------------------------------------------------------------------
class _StubBackbone:
    """Only the attributes `measure_point` touches."""

    def __init__(self, n_variants=40):
        self.pdb_code = "TEST"
        self.pdb = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        self.mutants = bsr.sample_single_mutants(WT, n_variants, [1, 1])

    @property
    def sequences(self):
        return [WT] + [s for _, s in self.mutants]

    @property
    def n_sequences(self):
        return 1 + len(self.mutants)

    deltas = bsr.Backbone.deltas


def _fake_oracle(noise_scale=1.0):
    """Per-sequence truth plus independent per-(seed, sequence) noise falling as 1/sqrt(K)."""

    def oracle(pdb_string, sequences, seed=37, num_decoding_orders=8, **kw):
        truth = np.random.default_rng(0).normal(0.0, 1.0, len(sequences))
        noise = np.random.default_rng([seed, num_decoding_orders]).normal(
            0.0, noise_scale / np.sqrt(num_decoding_orders), len(sequences)
        )
        return (truth + noise).tolist()

    return oracle


def test_measure_point_builds_the_variant_by_draw_matrix_and_reports_both_intervals(monkeypatch):
    monkeypatch.setattr(bsr, "score_only_oracle", _fake_oracle(noise_scale=0.05))
    b = _StubBackbone(n_variants=60)
    seeds = [37, 38, 39, 40, 41]
    p = bsr.measure_point(b, k=8, seeds=seeds, verbose=False)

    assert p["n_variants"] == 60 and p["n_draws"] == 5  # WT is the reference, not a row
    assert p["n_sequences_per_draw"] == 61
    assert p["icc"]["valid"] is True and p["meets_rule"] is True
    # The old two-draw reading of the SAME panel, for the understatement comparison.
    assert p["two_draw_reliability_seeds_1_2"]["n"] == 60
    assert len(p["icc"]["draw_jackknife"]["icc_leave_one_out"]) == 5
    assert p["seconds_per_sequence_per_order"] > 0
    assert len(p["runtime_seconds"]["per_draw"]) == 5


def test_measure_point_is_deterministic_and_rejects_a_noise_only_panel(monkeypatch):
    monkeypatch.setattr(bsr, "score_only_oracle", _fake_oracle(noise_scale=0.05))
    b = _StubBackbone(n_variants=50)
    first = bsr.measure_point(b, 8, [37, 38, 39], verbose=False)
    assert first["icc"]["icc"] == bsr.measure_point(b, 8, [37, 38, 39], verbose=False)["icc"]["icc"]

    # Noise swamping the signal must FAIL the rule rather than pass it on a positive estimate.
    monkeypatch.setattr(bsr, "score_only_oracle", _fake_oracle(noise_scale=30.0))
    assert bsr.measure_point(b, 8, [37, 38, 39], verbose=False)["meets_rule"] is False


def test_deltas_reference_wild_type_in_the_same_draw():
    b = _StubBackbone(n_variants=4)
    assert b.deltas([10.0, 11.0, 9.0, 10.0, 12.5]) == pytest.approx([1.0, -1.0, 0.0, 2.5])


# ---------------------------------------------------------------------------
# Pre-registered analysis
# ---------------------------------------------------------------------------
def _point(k, icc, lo, hi, noise_sd):
    return {
        "k": k,
        "icc": {"icc": icc, "icc_ci95_f": [lo, hi], "noise_sd": noise_sd, "valid": True, "stable": lo > 0},
        "meets_rule": bool(icc >= bsr.ICC_TARGET and lo >= bsr.ICC_LOWER_BOUND),
    }


def test_noise_scaling_recovers_the_preregistered_half_power_law():
    points = [_point(k, 0.99, 0.98, 0.999, 0.1 * k**-0.5) for k in (2, 4, 8, 16)]
    fit = bsr.noise_scaling(points)
    assert fit["valid"] is True
    assert fit["log_log_slope"] == pytest.approx(-0.5, abs=1e-6)
    assert fit["expected_slope"] == -0.5
    assert bsr.noise_scaling(points[:1])["valid"] is False  # a slope needs two points


def test_min_qualifying_k_is_the_cheapest_passing_k():
    points = [
        _point(2, 0.93, 0.89, 0.96, 0.07),  # fails on both legs
        _point(4, 0.96, 0.88, 0.98, 0.05),  # point estimate clears, lower bound does not
        _point(8, 0.98, 0.96, 0.99, 0.035),
        _point(16, 0.99, 0.98, 0.995, 0.025),
    ]
    assert bsr.min_qualifying_k(points) == 8
    assert bsr.min_qualifying_k(points[:2]) is None


def test_decision_rule_needs_ninety_percent_of_backbones_and_names_the_failures():
    def row(code, point):
        return {"provenance": {"pdb_code": code, "length": 100, "descriptors": {}}, "points": [point]}

    passing = [row(f"P{i}", _point(8, 0.98, 0.96, 0.99, 0.03)) for i in range(9)]
    failing = row("BAD", _point(8, 0.80, 0.70, 0.88, 0.2))

    ten = bsr._decision_rule(passing + [failing], 8)
    assert ten["n_pass"] == 9 and ten["fraction_pass"] == pytest.approx(0.9)
    assert ten["retain_default"] is True and ten["failing"] == ["BAD"]

    five = bsr._decision_rule(passing[:4] + [failing], 8)
    assert five["fraction_pass"] == pytest.approx(0.8) and five["retain_default"] is False


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "argv",
    [
        ["--seeds", "37,37,38"],  # re-scoring the same orders measures nothing
        ["--seeds", "37,0"],  # the scorer rejects seed 0
        ["--seeds", "37"],  # one draw cannot give a reliability
        ["--k-values", "0,8"],
        ["--k-values", ""],
    ],
)
def test_cli_rejects_designs_that_cannot_measure_anything(argv):
    with pytest.raises(SystemExit):
        bsr.main(argv + ["--dry-run", "--backbones", "4G3O"])


def test_backbone_refuses_a_structure_that_fails_the_eligibility_rule(monkeypatch):
    monkeypatch.setattr(
        bsr, "screen", lambda code: {"code": code, "eligible": False, "reasons": ["6 fragments (longest 64/225)"]}
    )
    with pytest.raises(SystemExit, match="eligibility"):
        bsr.Backbone("1EMA", 10, 1)

    monkeypatch.setattr(bsr, "screen", lambda code: {"code": code, "error": "fetch failed: HTTPError"})
    with pytest.raises(SystemExit, match="fetch failed"):
        bsr.Backbone("ZZZZ", 10, 1)


def test_default_artifact_path_is_parameter_stamped():
    from benchmarks._artifacts import stamped_path

    path = stamped_path("/tmp", "score_reliability", "json", n=100, k="2-4-8-16", d=5, v="2")
    assert path.endswith("score_reliability_n100_k2-4-8-16_d5_v2.json")
