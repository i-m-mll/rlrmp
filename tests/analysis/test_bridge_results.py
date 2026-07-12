"""Contract tests for structured bridge results and shared array shapes."""

from __future__ import annotations

import numpy as np
import pytest

from rlrmp.analysis.bridge_results import (
    BridgeAnalysisResult,
    BridgeCertificateComponent,
    BridgeRolloutBatch,
    BridgeRunSpec,
    make_bridge_run_id,
)


def _spec() -> BridgeRunSpec:
    return BridgeRunSpec(
        issue_id="a0ad145",
        run_id=make_bridge_run_id("optimal", "free time-varying", "smoke"),
        objective="optimal",
        architecture="free_time_varying",
        controller_label="smoke",
        optimizer_label="none",
        training_distribution="nominal",
        evaluation_lane="deterministic",
        reference_controller="analytical_lqr",
        seed=0,
        gamma_factor=1.4,
        parameters={"horizon": 3},
    )


def test_bridge_analysis_result_exposes_structured_payload() -> None:
    batch = BridgeRolloutBatch(
        plant_states=np.zeros((2, 4, 6)),
        observations=np.zeros((2, 3, 4)),
        estimator_states=np.zeros((2, 4, 6)),
        actions=np.zeros((2, 3, 2)),
        step_costs=np.zeros((2, 3)),
        total_costs=np.zeros((2,)),
    )
    result = BridgeAnalysisResult(
        spec=_spec(),
        status="smoke",
        arrays=batch.array_specs(),
        metrics={"cost_ratio": 1.0},
        artifacts={"arrays": "_artifacts/a0ad145/runs/smoke/arrays.npz"},
        certificate_components=(
            BridgeCertificateComponent.available("state_weighted_action_mismatch", rms=0.0),
            BridgeCertificateComponent.not_applicable(
                "closed_loop_transition_mismatch",
                "smoke fixture has no controller transition matrix",
            ),
        ),
    )

    payload = result.to_payload()

    assert payload["spec"]["run_id"] == "optimal__free_time-varying__smoke"
    assert payload["arrays"][0]["name"] == "plant_states"
    assert payload["certificate_components"][1]["status"] == "not_applicable"


def test_bridge_rollout_batch_rejects_time_mismatch() -> None:
    with pytest.raises(ValueError, match="one more time sample"):
        BridgeRolloutBatch(
            plant_states=np.zeros((2, 3, 6)),
            actions=np.zeros((2, 3, 2)),
        )


def test_bridge_rollout_batch_rejects_optional_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="observations"):
        BridgeRolloutBatch(
            plant_states=np.zeros((2, 4, 6)),
            actions=np.zeros((2, 3, 2)),
            observations=np.zeros((3, 3, 4)),
        )


def test_make_bridge_run_id_requires_nonempty_input() -> None:
    with pytest.raises(ValueError, match="nonempty"):
        make_bridge_run_id("", " ")
