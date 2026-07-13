"""Conformance coverage for compact per-row RLRMP training lowering."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from feedbax.contracts.run_matrix import TrainingRunMatrixSpec
from feedbax.contracts.spec_storage import training_spec_sha256
from feedbax.contracts.training import TrainingRunSpec
from feedbax.training.run_matrix import materialize_adapted_run_matrix

from rlrmp.loss import CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE
from rlrmp.model.feedback_descriptors import DESCRIPTOR_PAYLOAD_KEY
from rlrmp.train.executor.slots import CS_SUPERVISED_METHOD_REF
from rlrmp.train.heterogeneous_training_matrix import (
    ARCHITECTURES,
    DISTRIBUTIONS,
    author_training_run_matrix,
)
from rlrmp.train.matrix_lowering import (
    RLRMP_TRAINING_ARCHITECTURE_LOWERER_ID,
    RLRMP_TRAINING_ARCHITECTURE_LOWERER_VERSION,
    RLRMP_TRAINING_ROW_LOWERER_ID,
    RLRMP_TRAINING_ROW_LOWERER_VERSION,
    RlrmpTrainingAuthoringIntent,
    lower_rlrmp_training_row,
)
from rlrmp.train.matrix_materialization import materialize_rlrmp_training_matrix
from rlrmp.train.training_configs import CsNominalGruConfig


FORCE_FEEDBACK_AXIS = "force_filter_feedback"
PGD_AXIS = "broad_epsilon_pgd_training"


def _compact_matrix(
    tmp_path: Path,
) -> tuple[TrainingRunMatrixSpec, dict[str, Any]]:
    intent = RlrmpTrainingAuthoringIntent(
        config=CsNominalGruConfig(
            issue="5816bf0",
            output_dir=str(tmp_path / "artifacts"),
            spec_dir=str(tmp_path / "spec"),
            dry_run=True,
            smoke=True,
            target_relative_multitarget=True,
            force_filter_feedback=False,
            broad_epsilon_pgd_training=False,
        )
    )
    intent_payload = intent.model_dump(mode="json", exclude_none=True)
    intent_path = tmp_path / "compact-intent.json"
    intent_path.write_text(
        json.dumps(intent_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    matrix = TrainingRunMatrixSpec.model_validate(
        {
            "name": "force feedback x broad-epsilon PGD conformance",
            "issue": "5816bf0",
            "base": {
                "kind": "authored_intent",
                "ref": intent_path.name,
                "content_hash": training_spec_sha256(intent_payload),
                "pin_algorithm": "canonical_json_v1",
            },
            "axes": [
                {
                    "id": FORCE_FEEDBACK_AXIS,
                    "path": f"config.{FORCE_FEEDBACK_AXIS}",
                    "variation": {"kind": "explicit", "values": [False, True]},
                },
                {
                    "id": PGD_AXIS,
                    "path": f"config.{PGD_AXIS}",
                    "variation": {"kind": "explicit", "values": [False, True]},
                },
            ],
            "combination": {"mode": "cross"},
        }
    )
    return matrix, intent_payload


def _validate_training_spec(payload: dict[str, Any], _row_id: str) -> TrainingRunSpec:
    return TrainingRunSpec.model_validate(payload)


def _materialize(matrix: TrainingRunMatrixSpec, tmp_path: Path):
    return materialize_adapted_run_matrix(
        matrix,
        repo_root=tmp_path,
        row_lowerer=lower_rlrmp_training_row,
        row_validator=_validate_training_spec,
    )


def _heterogeneous_matrix(tmp_path: Path) -> TrainingRunMatrixSpec:
    intent = RlrmpTrainingAuthoringIntent(
        config=CsNominalGruConfig(
            issue="5816bf0",
            output_dir="_artifacts/5816bf0/runs/base",
            spec_dir="results/5816bf0/runs/base",
            controller_architecture="gru",
            dry_run=True,
            smoke=True,
            target_relative_multitarget=True,
            force_filter_feedback=True,
            broad_epsilon_pgd_training=False,
        )
    )
    payload = intent.model_dump(mode="json", exclude_none=True)
    base_path = tmp_path / "results/5816bf0/runs/base.intent.json"
    base_path.parent.mkdir(parents=True, exist_ok=True)
    base_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return author_training_run_matrix(
        payload,
        issue="5816bf0",
        base_ref=base_path,
        repo_root=tmp_path,
    )


def test_compact_intent_excludes_compiled_and_escape_hatch_surfaces(
    tmp_path: Path,
) -> None:
    matrix, intent_payload = _compact_matrix(tmp_path)

    assert set(intent_payload) == {"schema_id", "schema_version", "config"}
    assert [axis.path for axis in matrix.axes] == [
        f"config.{FORCE_FEEDBACK_AXIS}",
        f"config.{PGD_AXIS}",
    ]
    assert matrix.rows == []
    assert matrix.deltas == []

    encoded = json.dumps(intent_payload, sort_keys=True)
    for forbidden in (
        "feedbax_training_run_spec",
        "worker_execution",
        "callback",
        "escape_hatch",
        "legacy_run_spec",
    ):
        assert forbidden not in encoded

    for forbidden_key in (
        "graph",
        "task",
        "objective",
        "method_payload",
        "worker_execution",
    ):
        rejected = {**intent_payload, forbidden_key: {}}
        with pytest.raises(ValidationError):
            RlrmpTrainingAuthoringIntent.model_validate(rejected)

    for forbidden_config_key in ("callback", "escape_hatch", "compiled_graph"):
        rejected = deepcopy(intent_payload)
        rejected["config"][forbidden_config_key] = {}
        with pytest.raises(ValidationError):
            RlrmpTrainingAuthoringIntent.model_validate(rejected)

    runtime_static_config = CsNominalGruConfig(
        issue="5816bf0",
        output_dir="_artifacts/5816bf0/runs/static",
        controller_architecture="static_linear",
    )
    with pytest.raises(ValidationError, match="public authored architecture"):
        RlrmpTrainingAuthoringIntent(config=runtime_static_config)


def test_compact_2x2_matrix_relowers_complete_consistent_training_specs(
    tmp_path: Path,
) -> None:
    matrix, _intent_payload = _compact_matrix(tmp_path)
    materialized = _materialize(matrix, tmp_path)
    repeated = _materialize(matrix, tmp_path)

    assert len(materialized.rows) == 4
    assert [row.planned_run_id for row in materialized.rows] == [
        row.planned_run_id for row in repeated.rows
    ]
    assert [row.payload for row in materialized.rows] == [row.payload for row in repeated.rows]
    assert len({row.planned_run_id for row in materialized.rows}) == 4
    assert len({row.provenance.authored_payload_hash for row in materialized.rows}) == 4
    assert len({row.provenance.lowered_execution_payload_hash for row in materialized.rows}) == 4

    seen_coordinates: set[tuple[bool, bool]] = set()
    for row in materialized.rows:
        assert row.coordinate is not None
        coordinates = row.coordinate.values
        force_feedback = bool(coordinates[FORCE_FEEDBACK_AXIS])
        pgd_training = bool(coordinates[PGD_AXIS])
        seen_coordinates.add((force_feedback, pgd_training))

        authored_config = row.authored_payload["config"]
        assert authored_config[FORCE_FEEDBACK_AXIS] is force_feedback
        assert authored_config[PGD_AXIS] is pgd_training
        assert {patch.path for patch in row.overrides} == {
            f"config.{FORCE_FEEDBACK_AXIS}",
            f"config.{PGD_AXIS}",
        }
        assert row.provenance.authored_payload_hash == training_spec_sha256(row.authored_payload)
        assert row.provenance.lowered_execution_payload_hash == training_spec_sha256(row.payload)
        assert row.provenance.axis_coordinates["values"] == coordinates

        spec = row.spec
        assert spec is not None
        feedback_dim = 6 if force_feedback else 4
        component_ids = (
            ["position", "velocity", "force_filter"] if force_feedback else ["position", "velocity"]
        )
        basis_id = (
            "target_relative_delayed_feedback_plus_force_filter"
            if force_feedback
            else "target_relative_delayed_feedback"
        )
        graph = spec.graph.inline
        assert graph is not None
        assert graph["nodes"]["feedback"]["params"]["output_size"] == feedback_dim
        assert graph["nodes"]["sensory"]["params"]["input_shape"] == [feedback_dim]
        descriptors = spec.metadata[DESCRIPTOR_PAYLOAD_KEY]
        assert descriptors["basis_id"] == basis_id
        assert descriptors["component_ids"] == component_ids
        assert descriptors["variable"]["value_schema"]["shape"] == [feedback_dim]
        assert descriptors["basis"]["scope"]["feedback_dim"] == feedback_dim
        assert spec.task.params["extra_inputs"] == ["target", "epsilon"]

        expected_science_ids = ["target_relative"]
        if pgd_training:
            expected_science_ids.append("broad_epsilon_pgd")
        expected_science_ids.append("objective.partial")
        lowerer_identities = row.provenance.lowerer_identities
        assert lowerer_identities[0].lowerer_id == RLRMP_TRAINING_ROW_LOWERER_ID
        assert lowerer_identities[0].lowerer_version == RLRMP_TRAINING_ROW_LOWERER_VERSION
        assert [identity.lowerer_id for identity in lowerer_identities[1:-1]] == (
            expected_science_ids
        )
        assert lowerer_identities[-1].lowerer_id == RLRMP_TRAINING_ARCHITECTURE_LOWERER_ID
        assert lowerer_identities[-1].lowerer_version == (
            RLRMP_TRAINING_ARCHITECTURE_LOWERER_VERSION
        )
        assert {identity.lowerer_version for identity in lowerer_identities[1:]} == {"v1"}

        method_payload = spec.method_payload.payload
        expected_distribution = "broad_epsilon_pgd" if pgd_training else "nominal"
        assert method_payload["training_mode"] == expected_distribution
        pre_step = method_payload.get("pre_step")
        assert (isinstance(pre_step, dict) and pre_step.get("enabled") is True) is (pgd_training)
        if pgd_training:
            assert pre_step["kind"] == "broad_epsilon_pgd"

        objective = spec.objective.payload
        assert objective["loss_objective"] == CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE
        assert objective["loss_summary"]["objective_profile"] == (CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE)
        assert objective["fidelity_status"]["loss_objective"] == (CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE)
        assert spec.method_ref.key == CS_SUPERVISED_METHOD_REF
        worker = spec.worker_execution.model_dump(mode="json", exclude_none=True)
        assert worker["method_contract"]["method_ref"] == CS_SUPERVISED_METHOD_REF
        assert worker["effective_phase"]["phase_program"]["phases"]
        assert worker["effective_phase"]["phase_program"]["transitions"]
        assert worker["metadata"]["native_executor"] == (
            "feedbax.training.executor.execute_training_run_spec"
        )

    assert seen_coordinates == {
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    }


def test_compact_six_row_matrix_dispatches_through_public_architecture_providers(
    tmp_path: Path,
) -> None:
    materialized = materialize_rlrmp_training_matrix(
        _heterogeneous_matrix(tmp_path),
        repo_root=tmp_path,
    )

    assert [row.row_id for row in materialized.rows] == [
        f"{architecture}.{distribution}"
        for architecture in ARCHITECTURES
        for distribution in DISTRIBUTIONS
    ]
    expected_contracts = {
        "gru": ("gru", "empirical_nonlinear"),
        "time_constrained_free_gain": ("static_linear", "static_gain"),
        "linear_recurrence": ("linear_recurrence", "augmented_linear"),
    }
    for row in materialized.rows:
        architecture, distribution = row.row_id.split(".", maxsplit=1)
        robust = distribution == "broad_epsilon_pgd"
        spec = row.spec
        assert spec is not None

        runtime_architecture, certificate_mode = expected_contracts[architecture]
        assert row.authored_payload["config"]["controller_architecture"] == architecture
        assert row.authored_payload["config"]["broad_epsilon_pgd_training"] is robust
        assert spec.method_payload.payload["config"]["controller_architecture"] == (
            runtime_architecture
        )
        assert spec.metadata["controller_architecture"] == architecture
        assert spec.metadata["certificate_mode"] == certificate_mode
        assert spec.metadata["training_distribution"] == ("broad_epsilon" if robust else "nominal")
        assert spec.metadata["training_method_distribution"] == distribution
        assert spec.method_payload.metadata["training_distribution"] == distribution

        artifact_root = f"_artifacts/5816bf0/runs/{row.row_id}"
        tracked_spec_dir = f"results/5816bf0/runs/{row.row_id}"
        assert spec.artifacts.artifact_root == artifact_root
        checkpoint_policy = spec.method_payload.payload["checkpoint_policy"]
        assert checkpoint_policy["artifact_root"] == artifact_root
        assert checkpoint_policy["tracked_spec_dir"] == tracked_spec_dir

        pre_step = spec.method_payload.payload.get("pre_step")
        assert (pre_step is not None and pre_step["enabled"] is True) is robust
        assert row.provenance.lowerer_identities[-1].model_dump() == {
            "lowerer_id": RLRMP_TRAINING_ARCHITECTURE_LOWERER_ID,
            "lowerer_version": RLRMP_TRAINING_ARCHITECTURE_LOWERER_VERSION,
        }

        graph = spec.graph.inline
        assert graph is not None
        if architecture == "time_constrained_free_gain":
            assert spec.training_config.network_type == "static_linear"
            assert graph["nodes"]["net"]["type"] == "AffineFeedbackController"
            assert len(graph["nodes"]["net"]["params"]["gain"][0]) == 6
        elif architecture == "linear_recurrence":
            assert spec.training_config.network_type == "linear_recurrence"
            assert graph["subgraphs"]["net"]["nodes"]["cell"]["type"] == "VanillaRNN"
        else:
            assert spec.training_config.network_type == "gru"
            assert graph["subgraphs"]["net"]["nodes"]["cell"]["type"] == "GRU"
