"""Tests for the bounded finalist-evaluation stage in `orchestrator/mutation_search.py`.

These are pure: the ProteinMPNN oracle is patched with a deterministic fake, so what is
under test is the stage's own contract — independent-draw enforcement, source merging, the
uncertainty summary, ranking-consistency reporting and separate cost accounting. The
numerical claims about the scoring protocol itself are not testable this way and live in
`tests/test_mpnn_score_only.py` against the real model.
"""

from typing import Dict, List
from unittest.mock import patch

import numpy as np
import pytest

from models.schemas import MutationCandidate
from orchestrator.mutation_search import (
    additive_proposals,
    evaluate_finalists,
)

WT = "AAAAAA"


def _cand(mutations: List[str], sequence: str, score: float, oracle: str = "score_only") -> MutationCandidate:
    return MutationCandidate(mutations=mutations, sequence=sequence, score=score, oracle=oracle)


def _seq(**subs: str) -> str:
    """_seq(p1="V", p3="L") -> a WT-length sequence with those 1-indexed substitutions."""
    chars = list(WT)
    for key, aa in subs.items():
        chars[int(key[1:]) - 1] = aa
    return "".join(chars)


S1, S2, S3, S4 = _seq(p1="V"), _seq(p2="L"), _seq(p3="W"), _seq(p4="G")


def _fake_oracle(level_by_seq: Dict[str, float], seed_offsets: Dict[int, float] = None):
    """A deterministic stand-in for `score_only_oracle`.

    `level_by_seq` sets each sequence's true level; `seed_offsets` optionally adds a
    per-(seed, sequence) perturbation so ranking instability can be simulated."""

    def fake(pdb_string, sequences, *, proteinmpnn_dir, model_name, seed, num_decoding_orders, protocol):
        out = []
        for i, s in enumerate(sequences):
            jitter = 0.0 if not seed_offsets else seed_offsets.get(seed, 0.0) * ((i % 2) * 2 - 1)
            out.append(level_by_seq[s] + jitter)
        return out

    return fake


def _run(proposals, *, eval_seeds=(101, 102, 103), search_seed=37, levels=None, seed_offsets=None, **kw):
    levels = levels or {WT: 0.0, S1: 1.0, S2: 0.8, S3: 0.5, S4: 0.2}
    with patch(
        "orchestrator.mutation_search.score_only_oracle",
        side_effect=_fake_oracle(levels, seed_offsets),
    ):
        return evaluate_finalists(
            "PDBSTR",
            WT,
            proposals,
            proteinmpnn_dir="/fake",
            eval_seeds=list(eval_seeds),
            search_seed=search_seed,
            **kw,
        )


# ---------------------------------------------------------------------------
# Independent-draw enforcement — the property the stage exists for
# ---------------------------------------------------------------------------


def test_eval_seeds_must_exclude_the_search_seed():
    """Re-scoring with the search's own draw would reproduce the selected score and leave
    the winner's-curse bias in place, which is the whole point of the stage."""
    with pytest.raises(ValueError, match="includes the search seed"):
        _run({"score_only": [_cand(["A1V"], S1, 1.0)]}, eval_seeds=(37, 102), search_seed=37)


def test_eval_seeds_must_be_non_empty():
    with pytest.raises(ValueError, match="non-empty"):
        _run({"score_only": [_cand(["A1V"], S1, 1.0)]}, eval_seeds=())


def test_eval_seeds_must_be_non_zero():
    with pytest.raises(ValueError, match="non-zero"):
        _run({"score_only": [_cand(["A1V"], S1, 1.0)]}, eval_seeds=(0, 102))


def test_eval_seeds_must_be_distinct():
    """Duplicated seeds would understate the uncertainty by double-counting one draw."""
    with pytest.raises(ValueError, match="distinct"):
        _run({"score_only": [_cand(["A1V"], S1, 1.0)]}, eval_seeds=(101, 101))


def test_empty_proposals_rejected():
    with pytest.raises(ValueError, match="no finalists"):
        _run({"score_only": [], "additive": []})


def test_wild_type_proposal_is_not_a_finalist():
    """WT is the reference row, not a candidate; a proposer emitting it must not create a
    self-comparison."""
    with pytest.raises(ValueError, match="no finalists"):
        _run({"score_only": [_cand([], WT, 0.0)]})


# ---------------------------------------------------------------------------
# Source merging
# ---------------------------------------------------------------------------


def test_both_proposers_are_represented():
    """The additive baseline must actually appear in the shortlist, not be crowded out."""
    rep = _run(
        {
            "score_only": [_cand(["A1V"], S1, 1.0), _cand(["A2L"], S2, 0.8)],
            "additive": [_cand(["A3W"], S3, 0.5, "additive"), _cand(["A4G"], S4, 0.2, "additive")],
        }
    )
    assert rep["n_finalists"] == 4
    sources = {f["sequence"]: f["sources"] for f in rep["finalists"]}
    assert sources[S1] == ["score_only"]
    assert sources[S3] == ["additive"]
    assert rep["sources"] == {"score_only": 2, "additive": 2}


def test_shared_candidate_is_evaluated_once_and_records_both_sources():
    rep = _run(
        {
            "score_only": [_cand(["A1V"], S1, 1.0)],
            "additive": [_cand(["A1V"], S1, 0.9, "additive")],
        }
    )
    assert rep["n_finalists"] == 1
    f = rep["finalists"][0]
    assert sorted(f["sources"]) == ["additive", "score_only"]
    # Each source's own search score is kept, so the two can be compared after the fact.
    assert f["search_scores"] == {"score_only": 1.0, "additive": 0.9}


def test_max_finalists_cap_is_round_robin_not_first_come():
    """A long list from one proposer must not exhaust the budget before the other is seen."""
    rep = _run(
        {
            "score_only": [_cand(["A1V"], S1, 1.0), _cand(["A2L"], S2, 0.8)],
            "additive": [_cand(["A3W"], S3, 0.5, "additive")],
        },
        max_finalists=2,
    )
    assert rep["n_finalists"] == 2
    got = {f["sequence"] for f in rep["finalists"]}
    assert got == {S1, S3}, "cap should take one from each source before a second from either"


# ---------------------------------------------------------------------------
# Uncertainty and ranking consistency
# ---------------------------------------------------------------------------


def test_mean_sd_and_delta_vs_wild_type():
    rep = _run({"score_only": [_cand(["A1V"], S1, 99.0)]})
    f = rep["finalists"][0]
    assert f["mean_score"] == pytest.approx(1.0)
    assert f["sd_score"] == pytest.approx(0.0)
    assert f["n_eval_draws"] == 3
    assert f["mean_delta_vs_wt"] == pytest.approx(1.0)
    # The search score is retained but is NOT what gets reported as the finalist score.
    assert f["search_scores"]["score_only"] == 99.0
    assert f["mean_score"] != f["search_scores"]["score_only"]


def test_finalists_are_ranked_by_mean_not_by_search_score():
    """S2 has the better search score but the worse true level; the report must follow the
    independent draws."""
    rep = _run(
        {
            "score_only": [_cand(["A2L"], S2, 50.0), _cand(["A1V"], S1, 1.0)],
        }
    )
    assert [f["sequence"] for f in rep["finalists"]] == [S1, S2]


def test_stable_ranking_is_reported_as_stable():
    rep = _run(
        {
            "score_only": [
                _cand(["A1V"], S1, 1.0),
                _cand(["A2L"], S2, 0.8),
                _cand(["A3W"], S3, 0.5),
            ]
        }
    )
    rc = rep["ranking_consistency"]
    assert rc["mean_pairwise_spearman"] == pytest.approx(1.0)
    assert rc["max_rank_spread"] == 0
    assert set(rc["top3_stable_across_all_draws"]) == {S1, S2, S3}


def test_unstable_ranking_is_visible():
    """With draw-dependent jitter big enough to reorder near-ties, the report must say so
    rather than present one draw's order as the answer."""
    levels = {WT: 0.0, S1: 1.00, S2: 0.99, S3: 0.98, S4: 0.97}
    rep = _run(
        {
            "score_only": [
                _cand(["A1V"], S1, 1.0),
                _cand(["A2L"], S2, 0.99),
                _cand(["A3W"], S3, 0.98),
                _cand(["A4G"], S4, 0.97),
            ]
        },
        levels=levels,
        seed_offsets={101: 0.0, 102: 0.5, 103: -0.5},
    )
    rc = rep["ranking_consistency"]
    assert rc["max_rank_spread"] > 0
    assert rc["mean_pairwise_spearman"] is not None
    assert rc["mean_pairwise_spearman"] < 1.0
    # Every finalist carries its per-draw rank, so instability is attributable.
    for f in rep["finalists"]:
        assert set(f["ranks_by_seed"]) == {101, 102, 103}


def test_per_seed_scores_are_recorded():
    """Raw per-draw scores are preserved so an alternative analysis needs no rescoring."""
    rep = _run({"score_only": [_cand(["A1V"], S1, 1.0)]})
    assert rep["finalists"][0]["scores_by_seed"] == {101: 1.0, 102: 1.0, 103: 1.0}
    assert rep["wild_type"]["scores_by_seed"] == {101: 0.0, 102: 0.0, 103: 0.0}


def test_single_draw_reports_no_sd_rather_than_zero():
    """One draw cannot estimate a spread; reporting sd=0 would imply certainty."""
    rep = _run({"score_only": [_cand(["A1V"], S1, 1.0)]}, eval_seeds=(101,))
    f = rep["finalists"][0]
    assert f["n_eval_draws"] == 1
    assert f["sd_score"] is None
    assert f["sem_score"] is None
    assert rep["ranking_consistency"]["mean_pairwise_spearman"] is None


# ---------------------------------------------------------------------------
# Cost accounting and protocol recording
# ---------------------------------------------------------------------------


def test_scoring_cost_is_tracked_separately():
    rep = _run(
        {"score_only": [_cand(["A1V"], S1, 1.0), _cand(["A2L"], S2, 0.8)]},
        eval_seeds=(101, 102, 103),
    )
    cost = rep["scoring_cost"]
    assert cost["invocations"] == 3
    assert cost["sequences_scored"] == 3 * (1 + 2)  # WT + 2 finalists, per draw
    assert "separately" in cost["note"]


def test_protocol_block_records_what_was_run():
    rep = _run({"score_only": [_cand(["A1V"], S1, 1.0)]}, num_decoding_orders=16)
    p = rep["protocol"]
    assert p["scoring_protocol"] == "score_only/v2"
    assert p["num_decoding_orders"] == 16
    assert p["eval_seeds"] == [101, 102, 103]
    assert p["search_seed"] == 37
    # The caveat must travel with the numbers.
    assert "does not correct model bias" in p["note"]


# ---------------------------------------------------------------------------
# The additive baseline proposer
# ---------------------------------------------------------------------------


def test_additive_proposals_use_the_additive_oracle():
    """The baseline is labelled as such and costs no score_only calls."""
    rng = np.random.default_rng(0)
    log_p = rng.normal(size=(len(WT), 21))
    result = additive_proposals(
        log_p,
        WT,
        rounds=3,
        candidates_per_round=5,
        max_sites=2,
        seed=1,
        top_k=4,
    )
    assert result.oracle == "additive"
    assert all(c.oracle == "additive" for c in result.candidates)
    assert 0 < len(result.candidates) <= 4
    assert all(1 <= len(c.mutations) <= 2 for c in result.candidates)
    assert all(len(c.sequence) == len(WT) for c in result.candidates)


def test_additive_proposals_are_deterministic():
    rng = np.random.default_rng(0)
    log_p = rng.normal(size=(len(WT), 21))
    kw = dict(rounds=3, candidates_per_round=5, max_sites=2, seed=1, top_k=4)
    a = additive_proposals(log_p, WT, **kw)
    b = additive_proposals(log_p, WT, **kw)
    assert [c.sequence for c in a.candidates] == [c.sequence for c in b.candidates]
