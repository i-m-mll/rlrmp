"""Parity tests for cached objective-comparator analysis."""

from types import SimpleNamespace

import pytest
from feedbax.analysis.types import AnalysisInputData

from rlrmp.analysis.objective_comparator import (
    OBJECTIVE_TERMS_EVALUATION_TYPE,
    STANDARD_SPLIT_BANK_LENSES,
    CachedObjectiveComparatorInput,
    ObjectiveComparatorAnalysis,
    build_objective_comparator_sidecar_from_cached,
    objective_comparator_recipe,
)
from rlrmp.eval.objective_terms import objective_term_rows


def _cached() -> dict:
    lenses = {
        name: {
            "status": "available",
            "runs": {
                "robust": {
                    "gru": {"terms": {"total": {"mean": 12.0}}},
                    "extlqg": {"terms": {"total": {"mean": 10.0}}},
                    "gru_vs_extlqg": {"terms": {"total": {"ratio_to_extlqg": 1.2}}},
                }
            },
        }
        for name in STANDARD_SPLIT_BANK_LENSES
    }
    return {
        "source_manifest": "eval:objective",
        "checkpoint_policy": {
            "label": "validation_selected_per_replicate",
            "selection_source": "checkpoint_manifest",
            "selection_role": "audit_only_not_used_for_checkpoint_selection",
        },
        "extlqg_decomposition": {
            "deterministic_initial_state": 8.0,
            "initial_covariance_trace": 1.0,
            "accumulated_noise_scalar": 1.0,
            "total_expected_cost": 10.0,
            "comparable_scalar": 8.0,
        },
        "same_noise_bank_monte_carlo": {
            "status": "available",
            "bank_id": "shared-noise-v1",
            "gru_vs_extlqg": {"terms": {"total": {"ratio_to_extlqg": 1.25}}},
        },
        "per_term_realized_scoring": {
            "status": "available",
            "terms": {
                "running_state_q": {"mean": 4.0},
                "terminal_state_q_f": {"mean": 3.0},
                "command_r": {"mean": 2.0},
                "force_filter_state": {"mean": 2.0},
                "disturbance_integrator_state": {"mean": 1.0},
            },
        },
        "shared_rollout_comparator": {
            "status": "available",
            "bank": {"bank_id": "shared-x0-epsilon-v1", "n_trials": 32},
            "runs": {"robust": {"comparability": {"status": "stress_test_only"}}},
        },
        "standard_split_bank_comparator": {
            "status": "available",
            "fairness": {
                "checkpoint_selection_role": "audit_only_not_used_for_checkpoint_selection",
                "shared_initial_states": True,
                "shared_process_epsilon": True,
                "gru_hidden_state_conditioning": "checkpoint_default",
            },
            "lenses": lenses,
        },
        "rows": [
            {
                "run_id": "robust",
                "n_replicates": 3,
                "objective_comparability": {"status": "comparable_full_qrf"},
                "gru_mean_selected_validation_full_qrf": 12.0,
                "selected_to_extlqg_deterministic_ratio": 1.5,
                "shared_rollout_comparator": {
                    "gru_vs_extlqg": {"terms": {"total": {"ratio_to_extlqg": 1.25}}}
                },
                "standard_split_bank_comparator": {"lenses": lenses},
            }
        ],
    }


def test_cached_schema_requires_every_split_bank_lens_and_fairness() -> None:
    CachedObjectiveComparatorInput.model_validate(_cached())
    missing = _cached()
    del missing["standard_split_bank_comparator"]["lenses"]["x0_plus_epsilon"]
    with pytest.raises(ValueError, match="missing lenses"):
        CachedObjectiveComparatorInput.model_validate(missing)


def test_archived_v6_projection_preserves_full_qrf_and_split_bank_fields() -> None:
    result = build_objective_comparator_sidecar_from_cached(
        _cached(), issue="unit", scope="validation", source_manifest="eval:objective"
    )
    assert result["schema_version"] == "rlrmp.objective_comparator_sidecar.v6"
    assert result["extlqg_decomposition"]["total_expected_cost"] == 10.0
    assert result["same_noise_bank_monte_carlo"]["bank_id"] == "shared-noise-v1"
    assert result["per_term_realized_scoring"]["terms"]["command_r"]["mean"] == 2.0
    split = result["standard_split_bank_comparator"]
    assert tuple(split["lenses"]) == STANDARD_SPLIT_BANK_LENSES
    assert split["fairness"]["gru_hidden_state_conditioning"] == "checkpoint_default"
    row = result["rows"][0]
    assert row["objective_comparability"]["status"] == "comparable_full_qrf"
    assert row["shared_rollout_comparator"]["gru_vs_extlqg"]["terms"]["total"] == {
        "ratio_to_extlqg": 1.25
    }


def test_analysis_consumes_cached_manifest_without_rollout_execution() -> None:
    result = ObjectiveComparatorAnalysis(variant="objective_comparator").compute(
        AnalysisInputData(
            models={}, tasks={}, states={"cached": _cached()}, hps={},
            extras={"params": {"issue": "unit", "scope": "validation"}},
        )
    )
    assert result["rows"][0]["run_id"] == "robust"
    assert result["checkpoint_policy"]["selection_role"] == (
        "audit_only_not_used_for_checkpoint_selection"
    )


def test_objective_recipe_declares_registered_gru_diagnostics_parent() -> None:
    assert objective_comparator_recipe.EVAL_DEPENDENCIES == (OBJECTIVE_TERMS_EVALUATION_TYPE,)
    result = objective_comparator_recipe(
        SimpleNamespace(params={}), None, [SimpleNamespace(states=_cached())]
    )
    assert set(result.analyses) == {"objective_comparator"}


def test_objective_terms_normalizes_cached_term_aliases() -> None:
    rows = objective_term_rows(
        {"rows": [{"run_id": "robust", "objective_terms": {"total": 12.0},
                   "extlqg_terms": {"total": 10.0}}]}
    )
    assert rows[0]["terms"]["total"] == 12.0
    assert rows[0]["reference_terms"]["total"] == 10.0
