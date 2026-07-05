from __future__ import annotations

import pytest

from rlrmp.analysis.manifest_queries import (
    certificate_component_summary,
    certificate_component_summary_value,
    standard_row_by_source_run_id,
)


def test_standard_row_and_component_queries_select_expected_payloads() -> None:
    row = {
        "status": "failed",
        "spec": {
            "run_id": "standard__unit",
            "parameters": {"source_run_id": "source__unit"},
        },
        "certificate_components": [
            {
                "name": "observation_history_to_action_map_mismatch",
                "status": "available",
                "summary": {
                    "aggregate_mismatch_ratio": 0.25,
                    "covariance_weighted_aggregate_mismatch_ratio": 0.5,
                },
                "reason": None,
            }
        ],
    }
    manifest = {"rows": [row]}

    assert standard_row_by_source_run_id(manifest, "source__unit") is row
    assert certificate_component_summary(
        row,
        "observation_history_to_action_map_mismatch",
    ) == {
        "status": "available",
        "summary": {
            "aggregate_mismatch_ratio": 0.25,
            "covariance_weighted_aggregate_mismatch_ratio": 0.5,
        },
        "reason": None,
    }
    assert (
        certificate_component_summary_value(
            row,
            "observation_history_to_action_map_mismatch",
            "covariance_weighted_aggregate_mismatch_ratio",
        )
        == 0.5
    )


def test_manifest_queries_return_none_for_absent_entries() -> None:
    manifest = {"rows": []}
    row = {"certificate_components": []}

    assert standard_row_by_source_run_id(manifest, "missing") is None
    assert certificate_component_summary(row, "missing") is None
    assert certificate_component_summary_value(row, "missing", "aggregate_mismatch_ratio") is None


def test_manifest_queries_reject_duplicate_matches() -> None:
    row = {
        "spec": {"parameters": {"source_run_id": "source__unit"}},
        "certificate_components": [
            {"name": "duplicate", "summary": {"value": 1}},
            {"name": "duplicate", "summary": {"value": 2}},
        ],
    }
    manifest = {"rows": [row, row]}

    with pytest.raises(ValueError, match="Expected exactly one standard row"):
        standard_row_by_source_run_id(manifest, "source__unit")
    with pytest.raises(ValueError, match="Expected exactly one certificate component"):
        certificate_component_summary_value(row, "duplicate", "value")
