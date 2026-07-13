"""Archived row/metric parity oracle for the perturbation-bank strangler."""

from __future__ import annotations

import json
import inspect
from pathlib import Path

import pytest
from feedbax.analysis import MatrixMaterializerHarness

from rlrmp.analysis.perturbation_bank import (
    archived_parity_projection,
    load_perturbation_bank_matrix,
    materialize_perturbation_bank_rows,
)
from rlrmp.eval import perturbation_bank as science
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


def test_evaluation_science_has_no_durable_writer_or_private_eval_loop() -> None:
    source = inspect.getsource(science)

    assert "savez_compressed" not in source
    assert "materialize_gru_perturbation_response" not in source
    rollout_source = inspect.getsource(science._evaluate_model_rollout_product)
    assert "eval_ensemble_on_trials" in rollout_source
    assert ".eval_trials" not in rollout_source


def test_retired_perturbation_pipeline_cannot_reaccrete() -> None:
    repo_root = Path(__file__).parents[2]
    assert not (repo_root / "src/rlrmp/analysis/pipelines/gru_perturbation_bank.py").exists()
    assert not (repo_root / "scripts/materialize_gru_perturbation_response.py").exists()
    retired_import = "rlrmp.analysis.pipelines." + "gru_perturbation_bank"
    shared_reconciliation = repo_root / "src/rlrmp/analysis/declarative_materialization.py"
    for root in (repo_root / "src", repo_root / "scripts", repo_root / "tests"):
        for path in root.rglob("*.py"):
            if path == shared_reconciliation:
                continue
            source = path.read_text(encoding="utf-8")
            assert retired_import not in source
