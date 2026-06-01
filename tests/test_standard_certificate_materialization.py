"""Tests for reusable standard-certificate materialization helpers."""

from __future__ import annotations

import numpy as np

from rlrmp.analysis.bridge_certificates import (
    CLOSED_LOOP_TRANSITION_MISMATCH,
    STATE_WEIGHTED_ACTION_MISMATCH,
)
from rlrmp.analysis.bridge_contracts import BridgeRunSpec
from rlrmp.analysis.standard_certificate_materialization import (
    StandardCertificateRowRequest,
    component_by_name,
    component_status_counts,
    materialization_summary,
    build_standard_certificate_manifest,
)


def test_standard_request_uses_standard_certificate_components_as_entrypoint() -> None:
    spec = BridgeRunSpec(
        issue_id="e6a32b8",
        run_id="unit__standard_request",
        objective="diagnostic",
        architecture="free_time_varying",
        controller_label="unit",
        optimizer_label="unit",
        training_distribution="nominal",
        evaluation_lane="diagnostic",
        reference_controller="reference",
    )

    manifest = build_standard_certificate_manifest(
        StandardCertificateRowRequest(
            spec=spec,
            architecture="free_time_varying",
            status="unit",
            component_kwargs={
                "candidate_actions": np.zeros((2, 3, 1)),
                "reference_actions": np.ones((2, 3, 1)),
            },
        )
    )

    by_name = component_by_name(manifest)
    assert by_name[STATE_WEIGHTED_ACTION_MISMATCH].status == "available"
    assert by_name[CLOSED_LOOP_TRANSITION_MISMATCH].status == "missing"
    assert component_status_counts([manifest])[f"{STATE_WEIGHTED_ACTION_MISMATCH}:available"] == 1
    assert materialization_summary([manifest])["status_counts"] == {"unit": 1}
