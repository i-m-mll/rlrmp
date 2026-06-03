"""Tests for objective-comparator sidecars."""

from __future__ import annotations

import json

from rlrmp.analysis.objective_comparator import (
    SCHEMA_VERSION,
    ExtLQGCostDecomposition,
    build_objective_comparator_sidecar,
    render_objective_comparator_markdown,
    write_objective_comparator_sidecar,
)


def _checkpoint_selection() -> dict[str, object]:
    return {
        "schema_version": "rlrmp.validation_selected_gru_checkpoints.v1",
        "selection_policy": "validation-selected test policy",
        "runs": {
            "run_b": [
                {
                    "replicate": 0,
                    "scoring_validation_objective": 44.0,
                    "best_logged_validation_objective": 43.0,
                }
            ],
            "run_a": [
                {
                    "replicate": 0,
                    "scoring_validation_objective": 10.0,
                    "best_logged_validation_objective": 9.0,
                },
                {
                    "replicate": 1,
                    "scoring_validation_objective": 14.0,
                    "best_logged_validation_objective": 11.0,
                },
            ],
        },
    }


def test_extlqg_decomposition_reports_component_sum_and_declared_total() -> None:
    decomposition = ExtLQGCostDecomposition(
        deterministic_initial_state=4.0,
        initial_covariance_trace=3.0,
        accumulated_noise_scalar=2.0,
        total_expected_cost=9.5,
        provenance="unit-test",
    )

    payload = decomposition.to_json()

    assert payload["component_sum"] == 9.0
    assert payload["total_expected_cost"] == 9.5
    assert payload["component_sum_delta"] == 0.5
    assert payload["comparable_scalar"] == 4.0
    assert payload["comparable_scalar_lens"] == "extlqg_deterministic_initial_state_full_qrf"


def test_build_objective_comparator_sidecar_uses_deterministic_comparator_lens() -> None:
    sidecar = build_objective_comparator_sidecar(
        issue="abc1234",
        source_manifest="source.json",
        checkpoint_selection=_checkpoint_selection(),
        extlqg=ExtLQGCostDecomposition(
            deterministic_initial_state=12.0,
            initial_covariance_trace=30.0,
            accumulated_noise_scalar=2.0,
            total_expected_cost=44.0,
            provenance="unit-test",
        ),
        scope="unit scope",
        generated_by="unit",
    )

    assert sidecar["schema_version"] == SCHEMA_VERSION
    assert sidecar["extlqg_decomposition"]["total_expected_cost"] == 44.0
    assert sidecar["same_noise_bank_monte_carlo"]["status"] == "not_implemented"

    first_row = sidecar["rows"][0]
    assert first_row["run_id"] == "run_a"
    assert first_row["gru_mean_selected_validation_full_qrf"] == 12.0
    assert first_row["selected_to_extlqg_deterministic_ratio"] == 1.0
    assert first_row["selected_to_extlqg_total_ratio_not_apples_to_apples"] == 12.0 / 44.0
    assert first_row["extlqg_comparable_lens"] == "extlqg_deterministic_initial_state_full_qrf"


def test_write_objective_comparator_sidecar_serializes_json_and_markdown(tmp_path) -> None:
    sidecar = build_objective_comparator_sidecar(
        issue="abc1234",
        source_manifest="source.json",
        checkpoint_selection=_checkpoint_selection(),
        extlqg=ExtLQGCostDecomposition(
            deterministic_initial_state=12.0,
            initial_covariance_trace=30.0,
            accumulated_noise_scalar=2.0,
            total_expected_cost=44.0,
            provenance="unit-test",
        ),
        scope="unit scope",
        generated_by="unit",
    )
    json_path = tmp_path / "sidecar.json"
    markdown_path = tmp_path / "sidecar.md"

    write_objective_comparator_sidecar(
        sidecar,
        json_path=json_path,
        markdown_path=markdown_path,
    )

    reloaded = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert reloaded["schema_version"] == SCHEMA_VERSION
    assert render_objective_comparator_markdown(sidecar) == markdown
    assert "not directly comparable to GRU validation values" in markdown
    assert "selected/total" in markdown
