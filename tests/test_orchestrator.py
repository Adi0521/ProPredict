"""
Unit tests for core orchestrator helper functions (Phase 4.4).

Fully mocked / pure-function tests — no Postgres, Redis, GPU, or network needed.
Covers:
  * _parse_plddt_from_pdb  (backends/esmfold.py)
  * count_clashes          (scoring.py)
  * generate_cache_key     (tasks.py)
  * compute_post_processing(scoring.py)
  * _determine_protonation_states (simulation.py)
"""
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _atom_line(
    serial: int,
    name: str,
    resname: str,
    chain: str,
    resnum: int,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    bfactor: float = 0.0,
    occ: float = 1.0,
) -> str:
    """
    Emit a column-correct PDB ATOM record.

    Column alignment is exact so the manual slicing in _parse_plddt_from_pdb
    (name [12:16], bfactor [60:66]) and _get_titratable_residues
    (resName [17:20], chain [21], resSeq [22:26]) all read the right fields,
    and BioPython's PDBParser parses it too.
    """
    return (
        f"ATOM  {serial:>5} {name:<4} {resname:>3} {chain}{resnum:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{occ:>6.2f}{bfactor:>6.2f}"
    )


# ---------------------------------------------------------------------------
# 1. _parse_plddt_from_pdb
# ---------------------------------------------------------------------------

class TestParsePlddtFromPdb:
    def test_scales_bfactor_to_0_100(self):
        from orchestrator.backends.esmfold import _parse_plddt_from_pdb

        pdb = "\n".join([
            _atom_line(1, "CA", "MET", "A", 1, bfactor=0.50),
            _atom_line(2, "CA", "LYS", "A", 2, bfactor=0.80),
        ])
        assert _parse_plddt_from_pdb(pdb) == pytest.approx([50.0, 80.0])

    def test_ignores_non_ca_and_malformed_lines(self):
        from orchestrator.backends.esmfold import _parse_plddt_from_pdb

        pdb = "\n".join([
            "HEADER    SOME PROTEIN",
            _atom_line(1, "N", "MET", "A", 1, bfactor=0.99),   # not CA -> skipped
            _atom_line(2, "CA", "MET", "A", 1, bfactor=0.42),  # counted
            "ATOM  garbage line that should not crash",
            _atom_line(3, "CB", "MET", "A", 1, bfactor=0.99),  # not CA -> skipped
        ])
        assert _parse_plddt_from_pdb(pdb) == pytest.approx([42.0])

    def test_empty_pdb_returns_empty_list(self):
        from orchestrator.backends.esmfold import _parse_plddt_from_pdb

        assert _parse_plddt_from_pdb("") == []


# ---------------------------------------------------------------------------
# 2. count_clashes  (requires BioPython)
# ---------------------------------------------------------------------------

def _has_biopython() -> bool:
    try:
        import Bio  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_biopython(), reason="BioPython required for clash detection")
class TestCountClashes:
    def test_detects_nonadjacent_clash(self):
        from orchestrator.scoring import count_clashes

        # Res 1 and res 5 are 2.5 A apart (< 3.8) and non-adjacent -> 1 clash.
        # Res 2 sits far away so it contributes nothing.
        pdb = "\n".join([
            _atom_line(1, "CA", "GLY", "A", 1, x=0.0, y=0.0, z=0.0),
            _atom_line(2, "CA", "GLY", "A", 2, x=0.0, y=0.0, z=10.0),
            _atom_line(3, "CA", "GLY", "A", 5, x=0.0, y=0.0, z=2.5),
        ])
        assert count_clashes(pdb) == 1

    def test_excludes_adjacent_residues(self):
        from orchestrator.scoring import count_clashes

        # Res 1 and res 2 are only 2.0 A apart but adjacent (|i-j| <= 1) -> excluded.
        pdb = "\n".join([
            _atom_line(1, "CA", "GLY", "A", 1, x=0.0, y=0.0, z=0.0),
            _atom_line(2, "CA", "GLY", "A", 2, x=0.0, y=0.0, z=2.0),
        ])
        assert count_clashes(pdb) == 0

    def test_no_clashes_when_all_far_apart(self):
        from orchestrator.scoring import count_clashes

        pdb = "\n".join([
            _atom_line(1, "CA", "GLY", "A", 1, x=0.0, y=0.0, z=0.0),
            _atom_line(2, "CA", "GLY", "A", 2, x=0.0, y=0.0, z=10.0),
            _atom_line(3, "CA", "GLY", "A", 3, x=0.0, y=0.0, z=20.0),
        ])
        assert count_clashes(pdb) == 0


# ---------------------------------------------------------------------------
# 3. generate_cache_key
# ---------------------------------------------------------------------------

class TestGenerateCacheKey:
    def test_deterministic_for_same_inputs(self):
        from orchestrator.tasks import generate_cache_key

        ctx = {"pH": 7.4, "ligands": ["ATP"]}
        assert generate_cache_key("MKTAYIAK", ctx) == generate_cache_key("MKTAYIAK", ctx)

    def test_context_key_order_invariant(self):
        from orchestrator.tasks import generate_cache_key

        a = generate_cache_key("MKTAYIAK", {"pH": 7.4, "ligands": ["ATP"]})
        b = generate_cache_key("MKTAYIAK", {"ligands": ["ATP"], "pH": 7.4})
        assert a == b  # json.dumps(sort_keys=True) normalises ordering

    def test_different_sequence_changes_key(self):
        from orchestrator.tasks import generate_cache_key

        assert generate_cache_key("MKTAYIAK", {}) != generate_cache_key("MKTAYIAL", {})

    def test_different_context_changes_key(self):
        from orchestrator.tasks import generate_cache_key

        assert generate_cache_key("MKTAYIAK", {"pH": 7.4}) != generate_cache_key("MKTAYIAK", {"pH": 5.0})

    def test_different_pipeline_changes_key(self):
        from orchestrator.tasks import generate_cache_key

        assert generate_cache_key("MKTAYIAK", {}, "esm_base") != generate_cache_key("MKTAYIAK", {}, "boltz")


# ---------------------------------------------------------------------------
# 4. compute_post_processing  (threshold decision logic)
# ---------------------------------------------------------------------------

def _make_prediction(mean_plddt: float):
    from models.schemas import StructurePrediction

    return StructurePrediction(
        structure_pdb="DUMMY",
        plddt_scores=[mean_plddt],
        mean_plddt=mean_plddt,
        seed=0,
    )


class TestComputePostProcessing:
    # count_clashes is patched to 0 so these tests isolate the pLDDT thresholds
    # (accept >= 75, refine >= 60, else escalate) from BioPython geometry.

    @pytest.mark.parametrize("mean_plddt, expected", [
        (90.0, "accept"),
        (75.0, "accept"),    # boundary: >= accept threshold
        (74.9, "refine"),
        (65.0, "refine"),
        (60.0, "refine"),    # boundary: >= refine threshold
        (59.9, "escalate"),
        (40.0, "escalate"),
    ])
    def test_decision_thresholds(self, mean_plddt, expected):
        with patch("orchestrator.scoring.count_clashes", return_value=0):
            from orchestrator.scoring import compute_post_processing

            result = compute_post_processing(_make_prediction(mean_plddt))
        assert result.decision == expected

    def test_clashes_penalise_score(self):
        # score = mean_plddt - num_clashes * 5.0
        with patch("orchestrator.scoring.count_clashes", return_value=3):
            from orchestrator.scoring import compute_post_processing

            result = compute_post_processing(_make_prediction(90.0))
        assert result.num_clashes == 3
        assert result.score == pytest.approx(90.0 - 15.0)


# ---------------------------------------------------------------------------
# 5. _determine_protonation_states
# ---------------------------------------------------------------------------

class TestDetermineProtonationStates:
    # pdb2gmx integer encoding:
    #   HIS: 1 = HIE (neutral), 2 = HIP (charged)
    #   ASP/GLU: 0 = deprotonated (neutral), 1 = protonated (charged)
    # Default model pKa (no pka_dict): HIS 6.0, ASP 3.9, GLU 4.1.

    PDB = "\n".join([
        _atom_line(1, "CA", "HIS", "A", 1),
        _atom_line(2, "CA", "ASP", "A", 2),
        _atom_line(3, "CA", "GLU", "A", 3),
    ])

    def test_neutral_at_physiological_ph(self):
        from orchestrator.simulation import _determine_protonation_states

        states = _determine_protonation_states(self.PDB, pH=7.4, pka_dict={})
        # All model pKa < 7.4 -> neutral codes
        assert states == {"his": [1], "asp": [0], "glu": [0]}

    def test_charged_at_low_ph(self):
        from orchestrator.simulation import _determine_protonation_states

        states = _determine_protonation_states(self.PDB, pH=2.0, pka_dict={})
        # All model pKa > 2.0 -> charged codes
        assert states == {"his": [2], "asp": [1], "glu": [1]}

    def test_pka_dict_overrides_model_value(self):
        from orchestrator.simulation import _determine_protonation_states

        # Force HIS pKa to 8.0 so at pH 7.4 it becomes charged (2) despite the
        # model default of 6.0 (which would give neutral 1).
        states = _determine_protonation_states(
            self.PDB, pH=7.4, pka_dict={(1, "A", "HIS"): 8.0}
        )
        assert states["his"] == [2]


# ---------------------------------------------------------------------------
# 6. validate_simulation_metrics  (Stage 4.5)
# ---------------------------------------------------------------------------

def _healthy_sim(**overrides):
    """A physically sane OpenMM-style trajectory result."""
    base = {
        "potential_energy": -45000.0,
        "rmsd_nm": [0.0, 0.12, 0.18, 0.21, 0.20],
        "rg_nm": [1.40, 1.41, 1.39, 1.42, 1.40],
        "n_frames": 5,
        "backend": "openmm",
    }
    base.update(overrides)
    return base


class TestValidateSimulationMetrics:
    def test_healthy_trajectory_passes(self):
        from orchestrator.scoring import validate_simulation_metrics

        assert validate_simulation_metrics(_healthy_sim()) is None

    def test_nan_potential_energy_fails(self):
        from orchestrator.scoring import validate_simulation_metrics

        reason = validate_simulation_metrics(_healthy_sim(potential_energy=float("nan")))
        assert reason is not None and "energy" in reason.lower()

    def test_inf_potential_energy_fails(self):
        from orchestrator.scoring import validate_simulation_metrics

        assert validate_simulation_metrics(_healthy_sim(potential_energy=float("inf"))) is not None

    def test_rmsd_blowup_fails(self):
        from orchestrator.scoring import validate_simulation_metrics

        # Final frame jumps past the 2.0 nm default threshold.
        reason = validate_simulation_metrics(_healthy_sim(rmsd_nm=[0.0, 0.3, 4.3]))
        assert reason is not None and "rmsd" in reason.lower()

    def test_rmsd_from_final_scalar_fallback(self):
        from orchestrator.scoring import validate_simulation_metrics

        # GROMACS may omit the per-frame list and provide only a final scalar.
        sim = _healthy_sim()
        del sim["rmsd_nm"]
        sim["rmsd_final_nm"] = 3.5
        assert validate_simulation_metrics(sim) is not None

    def test_rg_divergence_fails(self):
        from orchestrator.scoring import validate_simulation_metrics

        # Rg grows past 3x its initial value -> unfolding/explosion.
        reason = validate_simulation_metrics(_healthy_sim(rg_nm=[1.40, 2.0, 4.5]))
        assert reason is not None and "gyration" in reason.lower()

    def test_missing_metrics_do_not_false_fail(self):
        from orchestrator.scoring import validate_simulation_metrics

        # Empty / minimal results can't be checked -> treated as sane, not escalated.
        assert validate_simulation_metrics({}) is None
        assert validate_simulation_metrics({"backend": "openmm", "pH": 7.4}) is None

    def test_decision_flips_to_escalate_on_failure(self):
        # End-to-end: a failing sim_result should drive an escalate decision.
        from orchestrator.scoring import validate_simulation_metrics
        from models.schemas import PostProcessingResult

        post_proc = PostProcessingResult(num_clashes=0, score=90.0, decision="accept")
        reason = validate_simulation_metrics(_healthy_sim(rmsd_nm=[0.0, 5.0]))
        if reason:
            post_proc.validation_reason = reason
            post_proc.decision = "escalate"
        assert post_proc.decision == "escalate"
        assert post_proc.validation_reason is not None


# ---------------------------------------------------------------------------
# 7. celery_state_to_status  (Stage 4.6 — status endpoint progress mapping)
# ---------------------------------------------------------------------------

class TestCeleryStateToStatus:
    def test_pending(self):
        from orchestrator.progress import celery_state_to_status

        assert celery_state_to_status("PENDING", None) == ("pending", 0, None)

    def test_none_state_treated_as_pending(self):
        from orchestrator.progress import celery_state_to_status

        assert celery_state_to_status(None, None) == ("pending", 0, None)

    def test_started_before_first_progress(self):
        from orchestrator.progress import celery_state_to_status

        assert celery_state_to_status("STARTED", None) == ("started", 50, None)

    def test_progress_reads_meta(self):
        from orchestrator.progress import celery_state_to_status

        info = {"progress_percent": 60, "stage": "simulation"}
        assert celery_state_to_status("PROGRESS", info) == ("started", 60, "simulation")

    def test_progress_with_missing_meta_keys_defaults(self):
        from orchestrator.progress import celery_state_to_status

        assert celery_state_to_status("PROGRESS", {}) == ("started", 50, None)

    def test_success(self):
        from orchestrator.progress import celery_state_to_status

        assert celery_state_to_status("SUCCESS", None) == ("completed", 100, None)

    def test_failure_does_not_dereference_exception_info(self):
        from orchestrator.progress import celery_state_to_status

        # On FAILURE, Celery sets info to the exception — must not crash or read it.
        assert celery_state_to_status("FAILURE", ValueError("boom")) == ("failed", 0, None)

    def test_lowercase_state_is_normalised(self):
        from orchestrator.progress import celery_state_to_status

        assert celery_state_to_status("progress", {"progress_percent": 90, "stage": "finalizing"}) == (
            "started", 90, "finalizing",
        )


# ---------------------------------------------------------------------------
# 8. _run_prediction_core progress emission (Stage 4.6 wiring)
# ---------------------------------------------------------------------------

def _accept_prediction():
    from models.schemas import StructurePrediction

    # mean_plddt 95 >= accept threshold -> decision "accept" -> no refinement, no MD.
    return StructurePrediction(
        structure_pdb="ATOM",
        plddt_scores=[95.0],
        mean_plddt=95.0,
        seed=0,
        model_name="esmfold",
        backend_version="facebook/esmfold_v1",
    )


class TestRunPredictionCoreProgress:
    def _run(self, cb):
        from unittest.mock import patch, MagicMock
        import orchestrator.tasks as tasks

        fake_redis = MagicMock()
        fake_redis.get.return_value = None  # cache miss -> full pipeline runs

        with patch.object(tasks, "_get_redis", return_value=fake_redis), \
             patch.object(tasks, "call_esmfold_api", return_value=_accept_prediction()), \
             patch.object(tasks, "AGENT_ENABLED", False), \
             patch("orchestrator.scoring.count_clashes", return_value=0):
            return tasks._run_prediction_core(
                {"run_id": "test-progress", "sequence": "MKTAYIAK", "context": {}},
                progress_cb=cb,
            )

    def test_emits_expected_stage_sequence_on_accept_path(self):
        events = []
        result = self._run(lambda pct, stage: events.append((pct, stage)))

        assert result["status"] == "completed"
        # No MD (empty context) -> simulation stage is skipped.
        assert events == [(10, "folding"), (40, "post_processing"), (90, "finalizing")]

    def test_failing_callback_does_not_break_prediction(self):
        def boom(pct, stage):
            raise RuntimeError("status backend down")

        result = self._run(boom)  # must not raise
        assert result["status"] == "completed"

    def test_backend_version_survives_to_the_result(self):
        """
        The stamp is worthless if the pipeline drops it before storage. The backend itself
        is mocked here, so this pins propagation only — that the value the backend stamped
        reaches both `predictions[]` and `ensemble_result`. The stamping itself is covered
        in tests/test_esmfold_local.py and tests/test_boltz.py.
        """
        result = self._run(lambda pct, stage: None)
        assert result["ensemble_result"]["backend_version"] == "facebook/esmfold_v1"
        assert result["predictions"][0]["backend_version"] == "facebook/esmfold_v1"


# ---------------------------------------------------------------------------
# 9. Provenance is preserved when a prediction is rebuilt mid-pipeline
# ---------------------------------------------------------------------------

class TestPredictionFieldPreservation:
    """
    Rosetta relax and the agent branch replace `best_prediction` mid-run. They used to
    rebuild it field-by-field, which silently dropped every field not listed —
    affinity_score, affinity_probability, backend_version. The bug was invisible while
    affinity was always None; fixing the affinity key made it live. model_copy() is the fix,
    and these tests pin the behaviour so a future field cannot regress it.
    """

    def _boltz_pred(self):
        from models.schemas import StructurePrediction
        return StructurePrediction(
            structure_pdb="ATOM_ORIG",
            plddt_scores=[50.0],
            mean_plddt=50.0,        # below accept -> triggers the refinement loop
            seed=7,
            model_name="boltz2",
            affinity_score=-1.35,
            affinity_probability=0.91,
            backend_version="2.2.1@b1ebfc46ecf5",
        )

    def test_rosetta_relax_preserves_affinity_and_build(self):
        from unittest.mock import patch, MagicMock
        import orchestrator.tasks as tasks

        fake_redis = MagicMock()
        fake_redis.get.return_value = None

        # Relax returns a better structure, so the rebuild branch is taken.
        with patch.object(tasks, "_get_redis", return_value=fake_redis), \
             patch.object(tasks, "call_esmfold_api", return_value=self._boltz_pred()), \
             patch.object(tasks, "AGENT_ENABLED", False), \
             patch.object(tasks, "ROSETTA_ENABLED", True), \
             patch.object(tasks, "BOLTZ_ENABLED", False), \
             patch.object(tasks, "run_rosetta_relax", return_value=("ATOM_RELAXED", -123.4)), \
             patch.object(tasks, "_parse_plddt_from_pdb", return_value=[80.0]), \
             patch("orchestrator.scoring.count_clashes", return_value=0):
            result = tasks._run_prediction_core(
                {"run_id": "t-relax", "sequence": "MKTAYIAK", "context": {}}
            )

        ens = result["ensemble_result"]
        assert ens["structure_pdb"] == "ATOM_RELAXED"   # the relax did happen
        assert ens["mean_plddt"] == 80.0
        # ...and none of the provenance was lost on the way through.
        assert ens["affinity_score"] == -1.35
        assert ens["affinity_probability"] == 0.91
        assert ens["backend_version"] == "2.2.1@b1ebfc46ecf5"
        assert ens["seed"] == 7
        assert ens["model_name"] == "boltz2"

    def test_model_copy_carries_every_unlisted_field(self):
        """Direct guard on the mechanism, independent of pipeline wiring."""
        pred = self._boltz_pred()
        updated = pred.model_copy(update={"structure_pdb": "ATOM_NEW"})

        assert updated.structure_pdb == "ATOM_NEW"
        for field in ("affinity_score", "affinity_probability", "backend_version",
                      "seed", "model_name", "plddt_scores", "mean_plddt"):
            assert getattr(updated, field) == getattr(pred, field), f"{field} was dropped"
