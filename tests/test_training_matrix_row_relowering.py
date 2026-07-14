"""Conformance coverage for compact per-row RLRMP training lowering."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from feedbax.contracts.run_matrix import TrainingRunMatrixSpec
from feedbax.contracts.spec_storage import training_run_execution_hash, training_spec_sha256
from feedbax.contracts.training import TrainingRunSpec
from feedbax.orchestration.bundle import (
    AuthoredIntentRef,
    BudgetPolicy,
    EnvironmentDeclaration,
    ExecutionCapsuleRef,
    ExecutionIdentityEnvelope,
    ResolvedSnapshotRef,
    RunBundle,
    RunRowSpec,
    RowLaunchSpec,
    SchemaArtifactRef,
)
from feedbax.orchestration.stages import run_preflight_checks
from feedbax.training.optimizers import learning_rate_schedule
from feedbax.training.run_matrix import materialize_adapted_run_matrix

from rlrmp.loss import CS_PARTIAL_FEEDBAX_LOSS_OBJECTIVE
from rlrmp.model.feedback_descriptors import DESCRIPTOR_PAYLOAD_KEY
from rlrmp.runtime.training_run_specs import (
    CsSupervisedMethodPayload,
    cs_supervised_optimizer_spec,
    require_cs_supervised_optimizer,
)
from rlrmp.train.executor.cs_supervised import _learning_rate_schedule
from rlrmp.train.executor.slots import CS_SUPERVISED_METHOD_REF
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


def _schedule_preflight_bundle(spec: TrainingRunSpec, tmp_path: Path) -> RunBundle:
    payload = spec.model_dump(mode="json", exclude_none=True)
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload_path = tmp_path / "preflight-row.json"
    payload_path.write_bytes(payload_bytes)
    payload_hash = hashlib.sha256(payload_bytes).hexdigest()
    resolved_root = "1" * 64
    return RunBundle(
        run_set_id="2026-07-13-ebd5d02",
        driver="local",
        rows=[
            RunRowSpec(
                row_id="fresh-cs-row",
                execution=ExecutionIdentityEnvelope(
                    payload=SchemaArtifactRef(
                        schema_id=spec.schema_id,
                        schema_version=spec.schema_version,
                        artifact_id="test:fresh-cs-row:payload",
                        sha256=payload_hash,
                        uri=str(payload_path),
                    ),
                    authored_intent=AuthoredIntentRef(
                        schema_id="feedbax.spec.training_run_matrix",
                        schema_version="feedbax.spec.training_run_matrix.v3",
                        artifact_id="test:fresh-cs-row:authored",
                        sha256="2" * 64,
                        intent_hash="3" * 64,
                    ),
                    resolved_snapshot=ResolvedSnapshotRef(
                        schema_id="feedbax.spec.training_run_resolved_semantics",
                        schema_version="feedbax.spec.training_run_resolved_semantics.v1",
                        artifact_id="test:fresh-cs-row:resolved",
                        sha256="4" * 64,
                        root_hash=resolved_root,
                    ),
                    execution_capsule=ExecutionCapsuleRef(
                        schema_id="feedbax.manifest.training_run_execution_capsule",
                        schema_version="feedbax.manifest.training_run_execution_capsule.v2",
                        artifact_id="test:fresh-cs-row:capsule",
                        sha256="5" * 64,
                        execution_hash=training_run_execution_hash(resolved_root, []),
                    ),
                    immutable_inputs=[],
                ),
                launch=RowLaunchSpec(command=["true"]),
            )
        ],
        environment=EnvironmentDeclaration(python_version="3.13"),
        budget=BudgetPolicy(max_wall_clock_seconds=10.0),
        orchestration_root=str(tmp_path / "orchestration"),
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

    with pytest.raises(ValidationError):
        CsNominalGruConfig(
            issue="5816bf0",
            output_dir="_artifacts/5816bf0/runs/static",
            controller_architecture="static_linear",
        )


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


def test_fresh_cs_row_passes_public_schedule_realization_preflight(tmp_path: Path) -> None:
    matrix, _intent_payload = _compact_matrix(tmp_path)
    materialized = materialize_rlrmp_training_matrix(matrix, repo_root=tmp_path)
    spec = materialized.rows[0].spec
    assert spec is not None

    checks = {
        check.name: check
        for check in run_preflight_checks(_schedule_preflight_bundle(spec, tmp_path))
    }

    schedule_check = checks["schedule-realization"]
    assert schedule_check.status == "pass"
    observed = schedule_check.observed["fresh-cs-row"]
    assert len(observed) == 1
    assert observed[0]["scheduled"] is True
    assert len(observed[0]["samples"]) >= 4
    assert observed[0]["expected_context"] == {
        "schedule_origin_step": 0,
        "current_step": 0,
        "optimizer_count_at_current_step": 0,
    }
    assert observed[0]["observed_context"] == observed[0]["expected_context"]


def test_cs_typed_optimizer_rejects_runtime_config_drift(tmp_path: Path) -> None:
    matrix, _intent_payload = _compact_matrix(tmp_path)
    materialized = materialize_rlrmp_training_matrix(matrix, repo_root=tmp_path)
    spec = materialized.rows[0].spec
    assert spec is not None
    payload = deepcopy(spec.method_payload.payload)
    payload["optimizer"]["lr_schedule"]["learning_rate_0"] *= 2.0

    with pytest.raises(ValidationError, match="disagrees with governed C&S runtime config"):
        CsSupervisedMethodPayload.model_validate(payload)

    historical_payload = deepcopy(spec.method_payload.payload)
    historical_payload.pop("optimizer")
    assert CsSupervisedMethodPayload.model_validate(historical_payload).optimizer is None
    with pytest.raises(ValueError, match="lacks governed typed optimizer"):
        require_cs_supervised_optimizer(historical_payload)


@pytest.mark.parametrize(
    ("warmup_batches", "schedule_kind"),
    [(3, "warmup_cosine"), (0, "delayed_cosine")],
)
def test_cs_typed_optimizer_numerically_matches_live_schedule(
    warmup_batches: int,
    schedule_kind: str,
) -> None:
    total_batches = 10
    config = {
        "n_train_batches": total_batches,
        "controller_lr": 0.01,
        "lr_warmup_batches": warmup_batches,
        "lr_warmup_init_fraction": 0.2,
        "lr_cosine_alpha": 0.05,
    }
    optimizer = cs_supervised_optimizer_spec(
        config=config,
        n_train_batches=total_batches,
    )
    assert optimizer.params == {"weight_decay": 0.0}
    assert optimizer.lr_schedule is not None
    assert optimizer.lr_schedule.kind == schedule_kind

    typed_schedule = learning_rate_schedule(optimizer.lr_schedule)
    live_schedule = _learning_rate_schedule(
        SimpleNamespace(
            lr_schedule=schedule_kind,
            constant_lr_iterations=warmup_batches,
            n_batches_condition=total_batches,
            learning_rate_0=config["controller_lr"],
            warmup_init_fraction=config["lr_warmup_init_fraction"],
            cosine_annealing_alpha=config["lr_cosine_alpha"],
        )
    )
    sample_steps = {
        0,
        max(warmup_batches - 1, 0),
        warmup_batches,
        warmup_batches + 1,
        total_batches - 1,
        total_batches,
        total_batches + 1,
    }
    for step in sorted(sample_steps):
        assert float(typed_schedule(step)) == pytest.approx(
            float(live_schedule(step)),
            rel=1e-7,
            abs=1e-10,
        )
