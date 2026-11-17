"""
Tests for orchestrator/qmmm_oracle.py (tier-3 physics oracle for mutation search).

Mirrors tests/test_mutation_search.py's philosophy: mock at the expensive boundary, assert
the interface. The two heavy internals (_carve_qm_region, _engine_energy) are a scaffold and
would raise NotImplementedError, so every interface test patches `_interaction_energy` with a
deterministic fake and injects a recording `fold_fn`. No GPU, no Boltz, no xtb/ORCA.

What these lock (the parts of the oracle that are final, per the module's IMPLEMENTATION
STATUS):
  * build-time gating on QMMM_ENABLED (graceful degradation)
  * the budget cap that keeps it OUT of the AdaLead per-round loop
  * WT folded exactly once, cached; repeated candidates not re-folded
  * WT/mutant protocol symmetry: identical protein_copies + seed on every fold
  * the mutated-position diff handed to the energy step
  * fitness == -E_int (higher = better)
Two tripwire tests assert the QM internals STILL raise NotImplementedError, so filling them
in forces an intentional test update rather than silently changing behavior.
"""

from unittest.mock import patch

import pytest

from orchestrator import qmmm_oracle
from orchestrator.qmmm_oracle import make_qmmm_oracle

_WT = "ACDEFGHIK"  # 9-residue toy wild type
_SMILES = "CC(=O)O"


class _RecordingFold:
    """Fake fold_fn: records every (sequence, smiles, protein_copies, seed) call and returns
    a deterministic 'PDB' tag so the fake energy step can stay pure."""

    def __init__(self):
        self.calls = []

    def __call__(self, sequence, ligand_smiles, protein_copies=1, seed=0):
        self.calls.append((sequence, ligand_smiles, protein_copies, seed))
        return f"PDB::{sequence}"

    @property
    def sequences(self):
        return [c[0] for c in self.calls]


def _fake_energy(_pdb, mutated_positions, _engine, _radius):
    """Deterministic stand-in for _interaction_energy: more mutations -> lower (more
    favorable) E_int, so distinct mutants get distinct fitness and ordering is checkable."""
    return 10.0 - float(len(mutated_positions))


def _enabled():
    """Patch the module gate on (read inside make_qmmm_oracle at call time)."""
    return patch.object(qmmm_oracle, "QMMM_ENABLED", True)


def _fake_ie(sink=None):
    """Patch _interaction_energy; optionally record the mutated_positions it receives."""
    if sink is None:
        return patch.object(qmmm_oracle, "_interaction_energy", new=_fake_energy)

    def recording(pdb, mutated_positions, engine, radius):
        sink.append(list(mutated_positions))
        return _fake_energy(pdb, mutated_positions, engine, radius)

    return patch.object(qmmm_oracle, "_interaction_energy", new=recording)


# --- gating -----------------------------------------------------------------------------


def test_disabled_raises_at_build():
    fold = _RecordingFold()
    with patch.object(qmmm_oracle, "QMMM_ENABLED", False):
        with pytest.raises(RuntimeError, match="QMMM_ENABLED"):
            make_qmmm_oracle(_WT, _SMILES, fold_fn=fold)
    assert fold.calls == []  # never even folded WT


# --- WT handling: folded once, cached, exposed ------------------------------------------


def test_build_folds_wt_once_and_exposes_wt_fitness():
    fold = _RecordingFold()
    with _enabled(), _fake_ie():
        oracle = make_qmmm_oracle(_WT, _SMILES, fold_fn=fold, max_candidates=5)
    assert fold.sequences == [_WT]  # WT folded exactly once at build
    assert oracle.wt_fitness == -10.0  # fitness == -E_int, WT has 0 mutations


def test_wt_query_uses_cache_no_refold():
    fold = _RecordingFold()
    with _enabled(), _fake_ie():
        oracle = make_qmmm_oracle(_WT, _SMILES, fold_fn=fold, max_candidates=5)
        before = len(fold.calls)
        assert oracle([_WT]) == [-10.0]
    assert len(fold.calls) == before  # WT not re-folded on query


def test_repeated_candidate_folded_once():
    fold = _RecordingFold()
    mut = "ADDEFGHIK"  # differs from _WT at position 1
    with _enabled(), _fake_ie():
        oracle = make_qmmm_oracle(_WT, _SMILES, fold_fn=fold, max_candidates=5)
        oracle([mut, mut])  # same candidate twice in one batch
    assert fold.sequences.count(mut) == 1  # cached after first fold


# --- budget cap: keeps tier-3 out of the round loop -------------------------------------


def test_budget_cap_raises_over_limit():
    fold = _RecordingFold()
    with _enabled(), _fake_ie():
        oracle = make_qmmm_oracle(_WT, _SMILES, fold_fn=fold, max_candidates=2)
        with pytest.raises(ValueError, match="budget cap"):
            oracle(["ADDEFGHIK", "ACDEFGHIL", "ACDEFGHIM"])  # 3 > cap of 2


def test_budget_cap_allows_at_limit():
    fold = _RecordingFold()
    with _enabled(), _fake_ie():
        oracle = make_qmmm_oracle(_WT, _SMILES, fold_fn=fold, max_candidates=2)
        out = oracle(["ADDEFGHIK", "ACDEFGHIL"])  # exactly at cap
    assert len(out) == 2


# --- input guard ------------------------------------------------------------------------


def test_unequal_length_sequence_raises():
    fold = _RecordingFold()
    with _enabled(), _fake_ie():
        oracle = make_qmmm_oracle(_WT, _SMILES, fold_fn=fold, max_candidates=5)
        with pytest.raises(ValueError, match="equal-length"):
            oracle(["ACDEFGHI"])  # length 8 != WT length 9


# --- protocol symmetry: identical fold params on WT and every mutant --------------------


def test_fold_receives_protein_copies_and_seed_on_every_call():
    fold = _RecordingFold()
    with _enabled(), _fake_ie():
        oracle = make_qmmm_oracle(_WT, _SMILES, fold_fn=fold, max_candidates=5, protein_copies=2, seed=7)
        oracle(["ADDEFGHIK"])
    # WT (build) + one mutant — both must carry the same (protein_copies, seed).
    assert len(fold.calls) == 2
    for _seq, smiles, copies, seed in fold.calls:
        assert (smiles, copies, seed) == (_SMILES, 2, 7)


# --- the mutated-position diff handed to the energy step --------------------------------


def test_mutated_positions_are_the_diff_indices():
    fold = _RecordingFold()
    positions = []
    with _enabled(), _fake_ie(sink=positions):
        oracle = make_qmmm_oracle(_WT, _SMILES, fold_fn=fold, max_candidates=5)
        # positions[0] is the WT build call (no mutations).
        oracle(["ADDEFGHIL"])  # differs at index 1 (C->D) and 8 (K->L)
    assert positions[0] == []  # WT -> empty region-mutation set
    assert positions[1] == [1, 8]


# --- fitness convention -----------------------------------------------------------------


def test_fitness_is_negated_interaction_energy():
    fold = _RecordingFold()
    with _enabled(), patch.object(qmmm_oracle, "_interaction_energy", new=lambda *_a, **_k: 3.5):
        oracle = make_qmmm_oracle(_WT, _SMILES, fold_fn=fold, max_candidates=5)
        assert oracle(["ADDEFGHIK"]) == [-3.5]  # higher = better -> negate E_int


# --- scaffold tripwires: QM internals must still be unimplemented ------------------------


def test_carve_qm_region_still_not_implemented():
    with pytest.raises(NotImplementedError):
        qmmm_oracle._carve_qm_region("PDB", [1], 5.0)


def test_engine_energy_still_not_implemented():
    with pytest.raises(NotImplementedError):
        qmmm_oracle._engine_energy({"stub": True}, "xtb")
