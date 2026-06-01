"""Tests for the 30f2313 C&S GRU standard materializer."""

from __future__ import annotations

import numpy as np

from rlrmp.analysis.bridge_certificates import (
    BELLMAN_HESSIAN_RESIDUAL,
    CLOSED_LOOP_TRANSITION_MISMATCH,
    OBSERVATION_HISTORY_TO_ACTION_MAP_MISMATCH,
    STATE_WEIGHTED_ACTION_MISMATCH,
    VALUE_POLICY_GAP,
)
from rlrmp.analysis.cs_gru_standard_materialization import (
    RUN_IDS,
    build_gru_standard_manifest_from_actions,
    materialize_gru_standard_result,
    normalize_gru_hps,
)
from rlrmp.analysis.failure_decomposition import failure_diagnostic_from_standard_row


def _minimal_run_spec() -> dict[str, object]:
    return {
        "seed": 7,
        "stochastic_preset": "cs_stochastic",
        "fidelity_status": {"nn_hidden": "gru"},
        "hps": {
            "model": {
                "hidden_size": 8,
                "n_replicates": 2,
                "population_structure": {
                    "n_input_only": 0,
                    "n_input_readout": 8,
                    "n_readout_only": 0,
                    "n_recurrent_only": 0,
                },
            }
        },
    }


def _components(row: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        component["name"]: component
        for component in row["certificate_components"]  # type: ignore[index]
    }


def test_normalize_gru_hps_maps_serialized_gru_type_to_builder_default() -> None:
    hps = {"hidden_type": "equinox.nn._rnn.GRUCell", "model": {"hidden_size": 8}}

    normalized = normalize_gru_hps(hps)

    assert normalized["hidden_type"] is None
    assert hps["hidden_type"] == "equinox.nn._rnn.GRUCell"


def test_gru_manifest_keeps_same_coordinate_rows_not_applicable() -> None:
    reference = np.ones((2, 3, 2))
    candidate = np.zeros_like(reference)

    manifest = build_gru_standard_manifest_from_actions(
        run_id="cs_stochastic_gru__unit",
        run_spec=_minimal_run_spec(),
        training_summary={"completed_batches": 12},
        candidate_actions=candidate,
        reference_actions=reference,
        action_weight=np.broadcast_to(np.eye(2), (3, 2, 2)),
    )
    row = manifest.to_json_dict()
    by_name = _components(row)

    assert row["spec"]["architecture"] == "gru"
    assert row["spec"]["parameters"]["certificate_mode"] == "empirical_nonlinear"
    assert by_name[STATE_WEIGHTED_ACTION_MISMATCH]["status"] == "available"
    assert by_name[STATE_WEIGHTED_ACTION_MISMATCH]["summary"]["aggregate_mismatch_ratio"] == 1.0
    assert by_name[CLOSED_LOOP_TRANSITION_MISMATCH]["status"] == "not_applicable"
    assert by_name[VALUE_POLICY_GAP]["status"] == "not_applicable"
    assert by_name[BELLMAN_HESSIAN_RESIDUAL]["status"] == "not_applicable"
    assert by_name[OBSERVATION_HISTORY_TO_ACTION_MAP_MISMATCH]["status"] == "missing"
    assert "4D-to-8D" in by_name[OBSERVATION_HISTORY_TO_ACTION_MAP_MISMATCH]["reason"]

    diagnostic = failure_diagnostic_from_standard_row(row, source_group="cs_stochastic_gru")
    assert diagnostic["classification"]["classification"] == "external_rollout_mismatch"
    assert diagnostic["certificate"]["response_map_mismatch"] is None
    assert diagnostic["gradient_diagnostics"]["status"] == "not_applicable"
    assert diagnostic["gain_error_decomposition"]["status"] == "not_applicable"


def test_gru_materializer_does_not_claim_action_evidence_when_models_are_not_loaded() -> None:
    result = materialize_gru_standard_result(run_ids=(RUN_IDS[0],), load_models=False)
    row = result["rows"][0]
    by_name = _components(row)

    assert row["status"] == "standard_certificate_missing_action_evidence"
    assert by_name[STATE_WEIGHTED_ACTION_MISMATCH]["status"] == "missing"
    assert "not evaluated" in by_name[STATE_WEIGHTED_ACTION_MISMATCH]["reason"]
    assert "aggregate_mismatch_ratio" not in by_name[STATE_WEIGHTED_ACTION_MISMATCH]["summary"]
    assert result["failure_decomposition"]["rows"][0]["classification"]["classification"] == (
        "uncertain"
    )
