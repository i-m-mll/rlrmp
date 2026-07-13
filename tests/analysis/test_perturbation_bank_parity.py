"""Archived row/metric parity oracle for the perturbation-bank strangler."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest
from feedbax.analysis import MatrixMaterializerHarness

from rlrmp.analysis.perturbation_matrix import (
    archived_parity_projection,
    load_perturbation_bank_matrix,
    materialize_perturbation_bank_rows,
    perturbation_bank_matrix_payload,
)
from rlrmp.eval.perturbation_bank import (
    PerturbationBankParams,
    default_cs_perturbation_bank,
    expand_perturbation_bank,
)


FIXTURE = Path(__file__).parents[1] / "fixtures" / "perturbation_bank_archived_parity.json"
pytestmark = pytest.mark.feedbax_contract


def test_archived_representative_rows_and_aggregate_parity() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    actual = archived_parity_projection(payload["rows"])

    assert actual["row_count"] == payload["aggregate"]["row_count"]
    assert actual["evaluated_count"] == payload["aggregate"]["evaluated_count"]
    assert actual["families"] == payload["aggregate"]["families"]
    for metric in (
        "mean_delta_action_norm",
        "mean_delta_position_max",
        "mean_delta_cost_total",
    ):
        assert actual[metric] == pytest.approx(payload["aggregate"][metric])


def test_governed_evaluation_matrix_materializes_role_addressed_family_rows() -> None:
    matrix = load_perturbation_bank_matrix()

    rows = materialize_perturbation_bank_rows(matrix)

    assert [row.row_id for row in rows] == [
        "initial_position",
        "initial_velocity",
        "command_input",
        "process_epsilon",
        "sensory_feedback",
        "target_stream",
    ]
    assert matrix.base.params["source_experiment"] == ""
    assert matrix.base.params["metadata"]["source_binding"] == "caller_specialized"
    assert matrix.base.params["metadata"]["custody"] == "EvaluationRunManifest"
    assert all(
        row.payload.evaluation_type == "rlrmp.eval.perturbation_response_bank" for row in rows
    )
    assert MatrixMaterializerHarness.__module__ == "feedbax.analysis.harness"


def test_typed_bank_preserves_representative_science_families() -> None:
    bank = default_cs_perturbation_bank()
    families = {row["family"] for row in bank["perturbations"]}

    assert len(bank["perturbations"]) == 111
    assert {
        "initial_position_offset",
        "command_input_pulse",
        "process_epsilon_position_xy",
        "sensory_feedback_offset",
        "target_stream_jump",
    } <= families


def test_typed_bank_params_select_one_matrix_condition() -> None:
    bank = expand_perturbation_bank(
        PerturbationBankParams(
            families=("command_input_pulse",),
            axes=("x",),
            signs=(1,),
            timing_bins=("early",),
            amplitude_policy="raw_default",
        )
    )

    assert [row["perturbation_id"] for row in bank["perturbations"]] == [
        "command_input_pulse__early_t5_x_pos"
    ]
    assert bank["condition_source"]["model"] == (
        "rlrmp.eval.perturbation_bank.PerturbationBankParams"
    )


def test_specialized_matrix_preserves_manifest_custody_and_family_partitions() -> None:
    matrix = perturbation_bank_matrix_payload(
        source_experiment="source-experiment",
        run_ids=("run-a", "run-b"),
        labels=("A", "B"),
        n_rollout_trials=13,
        bank_mode="raw",
        calibration_level=None,
        calibration_reach=None,
        feedback_scale_manifest_path=None,
        preferred_checkpoint_manifest_path=Path("results/source/preferred-checkpoints.json"),
    )

    rows = materialize_perturbation_bank_rows(matrix)

    expected_families = {
        "initial_position": ["initial_position_offset"],
        "initial_velocity": ["initial_velocity_offset"],
        "command_input": [
            "command_input_pulse",
            "target_aligned_lateral_command_load_pulse",
        ],
        "process_epsilon": [
            "process_epsilon_position_xy",
            "process_epsilon_velocity_xy",
            "process_epsilon_force_state_xy",
            "process_epsilon_integrator_xy",
        ],
        "sensory_feedback": ["sensory_feedback_offset"],
        "target_stream": ["target_stream_jump"],
    }
    assert {
        row.row_id: row.payload.params["bank_params"]["families"] for row in rows
    } == expected_families
    for row in rows:
        assert row.payload.training_run_ids == ["run-a", "run-b"]
        assert [(ref.kind, ref.id, ref.role) for ref in row.payload.inputs] == [
            ("TrainingRunManifest", "run-a", "training_run"),
            ("TrainingRunManifest", "run-b", "training_run"),
        ]
        assert row.payload.params["checkpoint_selection_mode"] == "fixed_bank_manifest"
        assert row.payload.params["preferred_checkpoint_manifest_path"] == (
            "results/source/preferred-checkpoints.json"
        )


def test_typed_bank_executable_conditions_keep_signed_science_pairs() -> None:
    rows = default_cs_perturbation_bank()["perturbations"]
    executable = [row for row in rows if row["family"] != "target_stream_jump"]
    pairs: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for row in executable:
        provenance = row.get("channel_provenance") or {}
        key = (
            row["family"],
            row["channel"],
            row["axis"],
            row["timing_bin"],
            row.get("epsilon_component"),
            provenance.get("feedback_quantity"),
        )
        pairs[key].append(row["sign"])

    assert pairs
    assert all(sorted(signs) == [-1, 1] for signs in pairs.values())
    target_rows = [row for row in rows if row["family"] == "target_stream_jump"]
    assert [(row["channel"], row["adapter"]) for row in target_rows] == [
        ("target_stream", "not_applicable_current_fixed_target_checkpoint")
    ]
