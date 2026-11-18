"""Tests for `orchestrator/mpnn_score_only.py` — scoring protocol v2 (shared decoding orders).

Two kinds of test here, and the split matters:

  * **Guard tests** are pure and always run — argument validation, protocol dispatch.
  * **Real-model tests** load the actual ProteinMPNN weights and run forward passes on CPU.
    They skip when `PROTEINMPNN_PATH` is unset. They cannot be replaced by mocks: the whole
    claim of v2 is a NUMERICAL one (a score does not depend on batch composition), and a
    mocked model would only test the plumbing that arranges the call. The invariance tests
    use a tiny K and a short backbone so they stay inside a normal test run.

The real-model tests cover both properties that must hold, which `_run_proteinmpnn_score_only`
(v1) satisfies only the first of:

  1. exact reproducibility under identical inputs;
  2. invariance to batch composition — order, insertion of unrelated sequences, chunking,
     and one-call-per-sequence must all return the same numbers.

Plus a fidelity check: fed the exact decoding orders the upstream CLI would have used,
`_score_with_orders` must reproduce v1's `global_score`. That is what catches drift in model
construction, featurization or the scoring reduction if the pinned clone ever moves. It has
to replicate upstream's call ORDER too, because constructing `ProteinMPNN` initializes
weights from the global torch RNG and so offsets the stream before any decoding order is
drawn.
"""

import os

import pytest

from orchestrator import mpnn_score_only as mso
from orchestrator.mpnn_score_only import (
    SCORE_PROTOCOL_V1,
    SCORE_PROTOCOL_V2,
    score_only_shared_orders,
)

PROTEINMPNN_PATH = os.getenv("PROTEINMPNN_PATH", "")
needs_model = pytest.mark.skipif(
    not PROTEINMPNN_PATH or not os.path.isfile(os.path.join(PROTEINMPNN_PATH, "protein_mpnn_run.py")),
    reason="PROTEINMPNN_PATH not set to a ProteinMPNN clone — real-model invariance checks skipped",
)

# A 12-residue poly-alanine helix: enough for ProteinMPNN to featurize, small enough that a
# forward pass is milliseconds. Backbone-only (N, CA, C, O) is what score_only needs.
_HELIX_TEMPLATE = [
    (1, 0.000, 0.000, 0.000),
    (2, 1.500, 0.000, 0.000),
    (3, 2.100, 1.400, 0.000),
    (4, 1.400, 2.400, 0.200),
]


def _poly_ala_pdb(n_res: int = 12) -> str:
    """Synthesize an ideal-ish alpha helix PDB (N/CA/C/O per residue) of `n_res` residues.

    Coordinates follow the standard helical parameters (1.5 A rise, 100 deg twist, 2.3 A
    radius) — not a real structure, but geometrically sane enough that ProteinMPNN's
    k-nearest-neighbour featurization produces sensible graphs, which is all these tests need."""
    import math

    lines = []
    serial = 1
    for i in range(n_res):
        theta = math.radians(100.0 * i)
        z = 1.5 * i
        for name, dr, dtheta, dz, element in (
            ("N", 2.3, -0.4, -0.5, "N"),
            ("CA", 2.3, 0.0, 0.0, "C"),
            ("C", 2.3, 0.4, 0.5, "C"),
            ("O", 3.2, 0.5, 0.6, "O"),
        ):
            x = dr * math.cos(theta + dtheta)
            y = dr * math.sin(theta + dtheta)
            lines.append(
                f"ATOM  {serial:>5} {name:<4} ALA A{i + 1:>4}    "
                f"{x:8.3f}{y:8.3f}{z + dz:8.3f}  1.00  0.00          {element:>2}"
            )
            serial += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


def _panel(n_res: int = 12):
    """WT + two singles + their double, on the poly-alanine backbone. Deliberately a
    WT/single/single/double cycle: that is the contrast v2 exists to stabilise."""
    wt = "A" * n_res
    m1 = "V" + wt[1:]
    m2 = wt[:-1] + "L"
    dbl = "V" + wt[1:-1] + "L"
    return wt, [wt, m1, m2, dbl]


# ---------------------------------------------------------------------------
# Guard tests (pure)
# ---------------------------------------------------------------------------


def test_seed_zero_rejected():
    """seed=0 must be refused for parity with the CLI, where `if args.seed:` randomizes."""
    with pytest.raises(ValueError, match="seed=0"):
        score_only_shared_orders(_poly_ala_pdb(), ["AAAA"], "/nonexistent", seed=0)


def test_empty_sequences_short_circuits():
    """No sequences -> no work, and in particular no model load (the path is /nonexistent)."""
    assert score_only_shared_orders(_poly_ala_pdb(), [], "/nonexistent", seed=37) == []


def test_zero_decoding_orders_rejected():
    with pytest.raises(ValueError, match="num_decoding_orders"):
        score_only_shared_orders(_poly_ala_pdb(), ["AAAA"], "/nonexistent", seed=37, num_decoding_orders=0)


def test_mixed_lengths_rejected():
    with pytest.raises(ValueError, match="equal-length"):
        score_only_shared_orders(_poly_ala_pdb(), ["AAAA", "AAA"], "/nonexistent", seed=37)


def test_non_alphabet_characters_rejected():
    """A stray 'B'/'Z'/'-' would otherwise raise deep inside the index lookup."""
    with pytest.raises(ValueError, match="alphabet"):
        score_only_shared_orders(_poly_ala_pdb(), ["AABA"], "/nonexistent", seed=37)


def test_missing_clone_reports_clearly():
    with pytest.raises(RuntimeError, match="not a directory"):
        score_only_shared_orders(_poly_ala_pdb(), ["AAAA"], "/nonexistent/proteinmpnn", seed=37)


def test_unknown_protocol_rejected():
    """`score_only_oracle` dispatches on the protocol string; an unknown one must not fall
    through to a default silently."""
    from orchestrator.mutation_search import score_only_oracle

    with pytest.raises(ValueError, match="unknown scoring protocol"):
        score_only_oracle(_poly_ala_pdb(), ["AAAA"], proteinmpnn_dir="/nonexistent", protocol="v3")


def test_protocol_constants_distinct():
    assert SCORE_PROTOCOL_V1 != SCORE_PROTOCOL_V2
    assert SCORE_PROTOCOL_V2 == "score_only/v2"


def test_oracle_defaults_to_v2():
    """The whole point of the fix is that callers get the invariant protocol WITHOUT opting
    in. If this default regressed to v1, the search would quietly resume comparing scores
    drawn under different decoding orders."""
    import inspect

    from orchestrator.mutation_search import score_only_oracle

    assert inspect.signature(score_only_oracle).parameters["protocol"].default == SCORE_PROTOCOL_V2


def test_config_default_protocol_is_v2():
    """`PROTEINMPNN_SCORE_PROTOCOL` is what the agent call site passes."""
    from config import PROTEINMPNN_SCORE_PROTOCOL

    assert PROTEINMPNN_SCORE_PROTOCOL == SCORE_PROTOCOL_V2


# ---------------------------------------------------------------------------
# Real-model tests — the numerical claims
# ---------------------------------------------------------------------------


@needs_model
def test_v2_is_exactly_reproducible():
    """Identical inputs -> identical outputs, bit for bit."""
    pdb = _poly_ala_pdb()
    _, seqs = _panel()
    a = score_only_shared_orders(pdb, seqs, PROTEINMPNN_PATH, seed=37, num_decoding_orders=2)
    b = score_only_shared_orders(pdb, seqs, PROTEINMPNN_PATH, seed=37, num_decoding_orders=2)
    assert a == b


@needs_model
def test_v2_is_invariant_to_batch_composition():
    """The property v1 lacks: a score must not depend on what else was in the batch.

    Asserted as exact equality, not approximate — v2's draws are a function of the seed
    alone, so there is no float wobble to absorb. Four conditions, matching the ones
    measured in `benchmarks/diagnose_score_only_batching.py`."""
    pdb = _poly_ala_pdb()
    _, seqs = _panel()
    k = {"seed": 37, "num_decoding_orders": 2}

    together = score_only_shared_orders(pdb, seqs, PROTEINMPNN_PATH, **k)

    # (1) reordered, then mapped back
    reversed_ = score_only_shared_orders(pdb, seqs[::-1], PROTEINMPNN_PATH, **k)[::-1]
    assert reversed_ == together

    # (2) one call per sequence
    individually = [score_only_shared_orders(pdb, [s], PROTEINMPNN_PATH, **k)[0] for s in seqs]
    assert individually == together

    # (3) chunked across calls
    chunked = score_only_shared_orders(pdb, seqs[:2], PROTEINMPNN_PATH, **k) + score_only_shared_orders(
        pdb, seqs[2:], PROTEINMPNN_PATH, **k
    )
    assert chunked == together

    # (4) an unrelated sequence spliced in at the front
    decoy = "W" + ("A" * (len(seqs[0]) - 1))
    with_decoy = score_only_shared_orders(pdb, [decoy] + seqs, PROTEINMPNN_PATH, **k)[1:]
    assert with_decoy == together


@needs_model
def test_v2_epistasis_contrast_is_batch_invariant():
    """The quantity that actually broke: eps = f(AB) - f(A) - f(B) + f(WT) must be identical
    however the four terms were batched. Under v1 this contrast is where the error lands."""
    pdb = _poly_ala_pdb()
    _, (wt, m1, m2, dbl) = _panel()
    k = {"seed": 37, "num_decoding_orders": 2}

    f = score_only_shared_orders(pdb, [wt, m1, m2, dbl], PROTEINMPNN_PATH, **k)
    eps_together = f[3] - f[1] - f[2] + f[0]

    # Same four scores, gathered from four separate calls in a different order.
    g = {s: score_only_shared_orders(pdb, [s], PROTEINMPNN_PATH, **k)[0] for s in (dbl, m2, wt, m1)}
    eps_split = g[dbl] - g[m1] - g[m2] + g[wt]

    assert eps_together == eps_split


@needs_model
def test_different_seeds_give_different_scores():
    """Sanity: the seed still controls the draws. If this passed trivially, the invariance
    tests above would be vacuous (e.g. if the orders were constant)."""
    pdb = _poly_ala_pdb()
    _, seqs = _panel()
    a = score_only_shared_orders(pdb, seqs, PROTEINMPNN_PATH, seed=37, num_decoding_orders=2)
    b = score_only_shared_orders(pdb, seqs, PROTEINMPNN_PATH, seed=38, num_decoding_orders=2)
    assert a != b


@needs_model
def test_scores_are_sequence_dependent():
    """Sanity: distinct sequences get distinct scores (so invariance is not masking a
    degenerate scorer)."""
    pdb = _poly_ala_pdb()
    _, seqs = _panel()
    out = score_only_shared_orders(pdb, seqs, PROTEINMPNN_PATH, seed=37, num_decoding_orders=2)
    assert len(set(out)) == len(seqs)


@needs_model
def test_v2_reproduces_upstream_cli_given_upstream_draws():
    """Fidelity: with the CLI's own decoding orders, v2's arithmetic must match v1's.

    v2 and v1 do NOT agree on their own terms — v1 burns K draws on its native-PDB pass
    before the first fasta sequence, so the two use different orders by construction. What
    must agree is the scoring computation itself. So replicate upstream's exact call order
    (manual_seed -> parse/construct model, whose weight init consumes the stream ->
    per-fasta-entry blocks of K draws) and feed block i+1 to `_score_with_orders` for
    sequence i.

    This is the test that would fail if the pinned clone's model construction, featurization
    or `_scores` reduction ever changed under us."""
    import torch

    from orchestrator.mutation_search import score_only_oracle

    pdb = _poly_ala_pdb()
    _, seqs = _panel()
    k_orders, seed = 2, 37

    v1 = [
        -x
        for x in score_only_oracle(
            pdb,
            seqs,
            proteinmpnn_dir=PROTEINMPNN_PATH,
            seed=seed,
            num_decoding_orders=k_orders,
            protocol=SCORE_PROTOCOL_V1,
        )
    ]

    # Model construction draws from the global RNG, so it must happen AFTER the seed and
    # BEFORE the order blocks, exactly as in protein_mpnn_run.py.
    mso._MODEL_CACHE.clear()
    torch.manual_seed(seed)
    model, feats, up, device = mso.prepare(pdb, PROTEINMPNN_PATH)
    shape = feats["chain_M"].shape
    blocks = [[torch.randn(shape) for _ in range(k_orders)] for _ in range(1 + len(seqs))]

    replicated = [mso._score_with_orders(model, feats, [s], blocks[i + 1], up, device)[0] for i, s in enumerate(seqs)]

    for got, want in zip(replicated, v1, strict=True):
        assert got == pytest.approx(want, abs=1e-5), (
            "v2's scoring path no longer reproduces the upstream CLI given upstream's own "
            "decoding orders — check the pinned ProteinMPNN clone (expect 8907e66)."
        )


@needs_model
def test_oracle_v2_negates_global_score():
    """`score_only_oracle` must keep the higher-is-better convention under v2 (global_score
    is a mean NLL, so the oracle returns its negation)."""
    from orchestrator.mutation_search import score_only_oracle

    pdb = _poly_ala_pdb()
    _, seqs = _panel()
    raw = score_only_shared_orders(pdb, seqs, PROTEINMPNN_PATH, seed=37, num_decoding_orders=2)
    fit = score_only_oracle(
        pdb, seqs, proteinmpnn_dir=PROTEINMPNN_PATH, seed=37, num_decoding_orders=2, protocol=SCORE_PROTOCOL_V2
    )
    assert fit == [-x for x in raw]
    assert all(x > 0 for x in raw)  # NLL is positive; the oracle's values are negative
