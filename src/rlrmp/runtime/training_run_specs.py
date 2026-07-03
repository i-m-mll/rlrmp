"""Feedbax TrainingRunSpec adapters for RLRMP training recipes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from feedbax.contracts.manifest import (
    ArtifactRef,
    Provenance,
    SpecPayload,
    TrainingRunManifest,
    sha256_file,
    spec_payload,
    write_manifest,
)
from feedbax.contracts.training import (
    ArtifactPolicySpec,
    CheckpointProgressPolicySpec,
    DEFAULT_TRAINING_METHOD_REGISTRY,
    ExecutionPolicySpec,
    GraphTopologySourceSpec,
    MethodPayloadEnvelope,
    MethodRefSpec,
    ObjectiveSlotSpec,
    RiskAggregationSpec,
    TaskSpec,
    TrainingConfig,
    TrainingMethodRegistration,
    TrainingRunSpec,
    WorkerExecutionSpec,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_contract,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
)
from feedbax.contracts.worker import (
    AxisSpec,
    CheckpointBarrierSpec,
    CheckpointSlotSpec,
    EffectivePhaseSpec,
    MethodContractSpec,
    OptimizerTargetBinding,
    PhaseProgramSpec,
    PhaseSpec,
    StateSlotSpec,
    UpdateKernelSpec,
    UpdateStepSpec,
    derive_consistency_predicate,
)

from rlrmp.model.feedback_descriptors import (
    DESCRIPTOR_PAYLOAD_KEY,
    controller_feedback_descriptor_from_container,
)
from rlrmp.model.feedbax_graph import graph_spec_payload
from rlrmp.runtime.spec_migrations import (
    FINITE_ADVERSARY_POLICY_METADATA_KIND,
    RUN_SPEC_KIND,
    RUN_SPEC_SCHEMA_ID,
    RUN_SPEC_SCHEMA_VERSION,
    stamp_current_schema,
)


FEEDBAX_TRAINING_RUN_SPEC_KEY = "feedbax_training_run_spec"
RLRMP_RUN_SPEC_PAYLOAD_KEY = "rlrmp_run_spec"

CLOSED_LOOP_DISTILLATION_METHOD_REF = "rlrmp/closed_loop_distillation/v1"
CLOSED_LOOP_DISTILLATION_PAYLOAD_SCHEMA_ID = (
    "rlrmp.spec.training_method.closed_loop_distillation_payload"
)
CLOSED_LOOP_DISTILLATION_PAYLOAD_SCHEMA_VERSION = (
    "rlrmp.spec.training_method.closed_loop_distillation_payload.v1"
)
GUIDED_DISTILLATION_METHOD_REF = "rlrmp/guided_distillation/v1"
GUIDED_DISTILLATION_PAYLOAD_SCHEMA_ID = (
    "rlrmp.spec.training_method.guided_distillation_payload"
)
GUIDED_DISTILLATION_PAYLOAD_SCHEMA_VERSION = (
    "rlrmp.spec.training_method.guided_distillation_payload.v1"
)


class _StrictPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClosedLoopDistillationMethodPayload(_StrictPayloadModel):
    """Governed payload for closed-loop analytical-teacher distillation."""

    teacher_contract: dict[str, Any]
    student_contract: dict[str, Any]
    base_contract: dict[str, Any]
    closed_loop_semantics: dict[str, Any]
    loss_surface: dict[str, Any]
    optimizer: dict[str, Any]
    checkpointing: dict[str, Any]
    finite_adversary_policy_metadata: dict[str, Any] | None = None


class GuidedDistillationMethodPayload(_StrictPayloadModel):
    """Governed payload for guided analytical-teacher distillation."""

    teacher_contract: dict[str, Any]
    teacher_bank: dict[str, Any]
    base_contract: dict[str, Any]
    training_schedule: dict[str, Any]
    distillation_surface: dict[str, Any]
    optimizer: dict[str, Any]
    model_contract: dict[str, Any]
    checkpointing: dict[str, Any]


def training_arg_parser(*args: Any, **kwargs: Any) -> argparse.ArgumentParser:
    """Return an argparse parser outside the scanned training-entry module set."""

    return argparse.ArgumentParser(*args, **kwargs)


def closed_loop_distillation_method_ref() -> MethodRefSpec:
    """Return the RLRMP method ref for closed-loop distillation."""

    return MethodRefSpec(package="rlrmp", name="closed_loop_distillation", version="v1")


def guided_distillation_method_ref() -> MethodRefSpec:
    """Return the RLRMP method ref for guided distillation."""

    return MethodRefSpec(package="rlrmp", name="guided_distillation", version="v1")


def register_rlrmp_distillation_methods() -> None:
    """Register RLRMP distillation method refs with Feedbax's method registry."""

    _register_method_once(
        TrainingMethodRegistration(
            method_ref=CLOSED_LOOP_DISTILLATION_METHOD_REF,
            payload_schema_id=CLOSED_LOOP_DISTILLATION_PAYLOAD_SCHEMA_ID,
            payload_schema_version=CLOSED_LOOP_DISTILLATION_PAYLOAD_SCHEMA_VERSION,
            payload_model=ClosedLoopDistillationMethodPayload,
            contract_factory=closed_loop_distillation_method_contract,
            update_kernels_factory=_closed_loop_distillation_update_kernels,
            rejected_payload_versions=(
                "rlrmp.spec.training_method.closed_loop_distillation_payload.v0",
            ),
            owner="rlrmp.runtime.training_run_specs",
            package="rlrmp",
        )
    )
    _register_method_once(
        TrainingMethodRegistration(
            method_ref=GUIDED_DISTILLATION_METHOD_REF,
            payload_schema_id=GUIDED_DISTILLATION_PAYLOAD_SCHEMA_ID,
            payload_schema_version=GUIDED_DISTILLATION_PAYLOAD_SCHEMA_VERSION,
            payload_model=GuidedDistillationMethodPayload,
            contract_factory=guided_distillation_method_contract,
            update_kernels_factory=_guided_distillation_update_kernels,
            rejected_payload_versions=(
                "rlrmp.spec.training_method.guided_distillation_payload.v0",
            ),
            owner="rlrmp.runtime.training_run_specs",
            package="rlrmp",
        )
    )


def closed_loop_distillation_method_payload(
    run_spec: dict[str, Any],
) -> MethodPayloadEnvelope:
    """Return the governed payload envelope for a closed-loop distillation spec."""

    payload = {
        "teacher_contract": _mapping(run_spec, "teacher_contract"),
        "student_contract": _mapping(run_spec, "student_contract"),
        "base_contract": _mapping(run_spec, "base_contract"),
        "closed_loop_semantics": _mapping(run_spec, "closed_loop_semantics"),
        "loss_surface": _mapping(run_spec, "loss_surface"),
        "optimizer": _optimizer_payload_from_run_spec(run_spec),
        "checkpointing": _mapping(run_spec, "checkpointing"),
        "finite_adversary_policy_metadata": run_spec.get("finite_adversary_policy_metadata"),
    }
    validated = ClosedLoopDistillationMethodPayload.model_validate(payload)
    return MethodPayloadEnvelope(
        schema_id=CLOSED_LOOP_DISTILLATION_PAYLOAD_SCHEMA_ID,
        schema_version=CLOSED_LOOP_DISTILLATION_PAYLOAD_SCHEMA_VERSION,
        payload=validated.model_dump(mode="json", exclude_none=True),
    )


def guided_distillation_method_payload(run_spec: dict[str, Any]) -> MethodPayloadEnvelope:
    """Return the governed payload envelope for a guided distillation spec."""

    payload = {
        "teacher_contract": _mapping(run_spec, "teacher_contract"),
        "teacher_bank": _mapping(run_spec, "teacher_bank"),
        "base_contract": _mapping(run_spec, "base_contract"),
        "training_schedule": _mapping(run_spec, "training_schedule"),
        "distillation_surface": _mapping(run_spec, "distillation_surface"),
        "optimizer": _mapping(run_spec, "optimizer"),
        "model_contract": _mapping(run_spec, "model_contract"),
        "checkpointing": _mapping(run_spec, "checkpointing"),
    }
    validated = GuidedDistillationMethodPayload.model_validate(payload)
    return MethodPayloadEnvelope(
        schema_id=GUIDED_DISTILLATION_PAYLOAD_SCHEMA_ID,
        schema_version=GUIDED_DISTILLATION_PAYLOAD_SCHEMA_VERSION,
        payload=validated.model_dump(mode="json"),
    )


def closed_loop_distillation_method_contract() -> MethodContractSpec:
    """Return worker axes, slots, and phase program for closed-loop distillation."""

    return _distillation_contract(
        method_ref=CLOSED_LOOP_DISTILLATION_METHOD_REF,
        payload_version=CLOSED_LOOP_DISTILLATION_PAYLOAD_SCHEMA_VERSION,
        phase_names=("closed_loop_rollout_distillation",),
        update_step_name="closed_loop_distillation_gradient_update",
        kernel_ref="rlrmp.train.closed_loop_distillation.closed_loop_gradient_update",
        extra_axes=(AxisSpec(name="rollout", role="rollout"),),
        extra_slots=(
            StateSlotSpec(name="teacher_reference", role="auxiliary"),
            StateSlotSpec(name="closed_loop_rollout", role="auxiliary", required=False),
        ),
        metadata={
            "teacher_location": "method_payload.teacher_contract",
            "teacher_is_worker_axis": False,
            "execution_follow_through_deferred_to": "54b0c2e",
        },
    )


def guided_distillation_method_contract() -> MethodContractSpec:
    """Return worker axes, slots, and phase program for guided distillation."""

    return _distillation_contract(
        method_ref=GUIDED_DISTILLATION_METHOD_REF,
        payload_version=GUIDED_DISTILLATION_PAYLOAD_SCHEMA_VERSION,
        phase_names=(
            "teacher_forced_warm_start",
            "mixed_teacher_student_forcing",
            "mostly_student_forced",
        ),
        update_step_name="guided_distillation_gradient_update",
        kernel_ref="rlrmp.train.guided_distillation.guided_gradient_update",
        extra_axes=(AxisSpec(name="jvp_direction", role="member"),),
        extra_slots=(
            StateSlotSpec(name="teacher_bank", role="auxiliary"),
            StateSlotSpec(name="forcing_schedule", role="auxiliary"),
            StateSlotSpec(name="jvp_probes", role="auxiliary", required=False),
        ),
        metadata={
            "teacher_location": "method_payload.teacher_contract",
            "teacher_is_worker_axis": False,
            "phase_schedule_location": "method_payload.training_schedule",
            "execution_follow_through_deferred_to": "54b0c2e",
        },
    )


def build_distillation_training_run_spec(
    run_spec: dict[str, Any],
    *,
    method: str,
    output_dir: Path,
    spec_path: Path,
) -> TrainingRunSpec:
    """Build and validate a composed Feedbax ``TrainingRunSpec`` for distillation."""

    register_rlrmp_distillation_methods()
    method_ref, method_payload, contract = _distillation_method_parts(run_spec, method=method)
    checkpointing = _mapping(run_spec, "checkpointing")
    task_params = _distillation_task_params(run_spec, method=method)
    objective_payload = _distillation_objective_payload(run_spec, method=method)
    training_config = _distillation_training_config(run_spec, method=method)
    effective_phase = EffectivePhaseSpec(
        method_ref=contract.method_ref,
        axes=contract.axes,
        state_slots=contract.state_slots,
        phase_program=contract.phase_program,
        consistency_predicate=derive_consistency_predicate(contract.phase_program),
    )
    return TrainingRunSpec(
        graph=GraphTopologySourceSpec(
            ref=str(_distillation_graph_ref(run_spec)),
            metadata={
                "source": "rlrmp_runtime_adapter",
                "setup_function": _distillation_setup_function(run_spec, method=method),
                "base_run_spec": _mapping(run_spec, "base_contract").get("run_spec"),
            },
        ),
        task=TaskSpec(
            type="rlrmp_distillation",
            params=task_params,
        ),
        training_config=training_config,
        objective=ObjectiveSlotSpec(
            kind="external",
            payload=objective_payload,
            schema_id=f"{method_ref.package}.{method_ref.name}.objective",
            schema_version=f"{method_ref.package}.{method_ref.name}.objective.v1",
            metadata={"lowering": "method_payload_governed_external_loss"},
        ),
        risk_aggregation=RiskAggregationSpec(
            realization="mean",
            replicate="mean",
            metadata={"source": "distillation_adapter"},
        ),
        method_ref=method_ref,
        method_payload=method_payload,
        method_extensions={
            "metadata": {
                "runner": _distillation_runner(run_spec, method=method),
                "rlrmp_extension_payload": RLRMP_RUN_SPEC_PAYLOAD_KEY,
            }
        },
        worker_execution=WorkerExecutionSpec(
            method_contract=contract,
            effective_phase=effective_phase,
            metadata={
                "legacy_runner_retained_until_native_executor": _distillation_runner(
                    run_spec, method=method
                ),
                "teacher_configuration_location": "method_payload",
            },
        ),
        execution=ExecutionPolicySpec(
            mode="dry_run" if run_spec.get("launch_status") in {"not_launched"} else "local",
            require_review=True,
            allow_cloud=False,
            metadata={"full_train_authorized": False},
        ),
        artifacts=ArtifactPolicySpec(
            manifest_root="_artifacts/feedbax_runs",
            artifact_root=str(output_dir),
            custody="local",
            metadata={"tracked_run_spec": str(spec_path)},
        ),
        checkpoint_progress=CheckpointProgressPolicySpec(
            checkpoint_interval=_int_or_none(checkpointing.get("interval_batches")),
            progress_interval=_int_or_none(checkpointing.get("interval_batches")),
            metadata={"latest_pointer": checkpointing.get("latest_pointer")},
        ),
        metadata={
            "composed_with": RLRMP_RUN_SPEC_PAYLOAD_KEY,
            "serialize_do_not_rederive": True,
        },
    )


def attach_distillation_training_specs(
    run_spec: dict[str, Any],
    *,
    method: str,
    output_dir: Path,
    spec_path: Path,
) -> dict[str, Any]:
    """Attach governed RLRMP and Feedbax spec records to a distillation recipe."""

    payload = dict(run_spec)
    extension = stamp_current_schema(
        RUN_SPEC_KIND,
        {
            "issue": str(payload.get("issue", "")),
            "run_id": str(payload.get("run_id", "")),
            "training_script": _mapping(payload, "training_entry").get("script"),
            "method": method,
            "method_ref": _distillation_method_ref_key(method),
            "artifact_output_dir": str(output_dir),
            "method_payload_location": FEEDBAX_TRAINING_RUN_SPEC_KEY,
            "scientific_payload": {
                key: payload.get(key)
                for key in (
                    "teacher_contract",
                    "teacher_bank",
                    "student_contract",
                    "base_contract",
                    "closed_loop_semantics",
                    "training_schedule",
                    "loss_surface",
                    "distillation_surface",
                    "model_contract",
                    "optimizer",
                    "checkpointing",
                )
                if key in payload
            },
        },
    )
    payload[RLRMP_RUN_SPEC_PAYLOAD_KEY] = extension
    feedbax_spec = build_distillation_training_run_spec(
        payload,
        method=method,
        output_dir=output_dir,
        spec_path=spec_path,
    )
    payload[FEEDBAX_TRAINING_RUN_SPEC_KEY] = feedbax_spec.model_dump(
        mode="json",
        exclude_none=True,
    )
    return payload


def validate_distillation_training_run_spec(run_spec: dict[str, Any], *, method: str) -> None:
    """Validate a distillation recipe's composed Feedbax ``TrainingRunSpec``."""

    payload = attach_distillation_training_specs(
        run_spec,
        method=method,
        output_dir=Path(str(run_spec.get("artifact_output_dir", ""))),
        spec_path=Path(_mapping(run_spec, "training_entry").get("run_spec_path", "run.json")),
    )
    TrainingRunSpec.model_validate(payload[FEEDBAX_TRAINING_RUN_SPEC_KEY])


def write_distillation_run_spec(path: Path, run_spec: dict[str, Any], *, method: str) -> dict[str, Any]:
    """Write a distillation recipe only after composing and validating its specs."""

    payload = attach_distillation_training_specs(
        run_spec,
        method=method,
        output_dir=Path(str(run_spec.get("artifact_output_dir", path.parent))),
        spec_path=path,
    )
    TrainingRunSpec.model_validate(payload[FEEDBAX_TRAINING_RUN_SPEC_KEY])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def finite_adversary_policy_metadata_payload(metadata: Any) -> dict[str, Any]:
    """Return governed finite-policy metadata without a standalone serializer."""

    policy_class = str(metadata.policy_class)
    bias_shape = metadata.bias_shape
    return stamp_current_schema(
        FINITE_ADVERSARY_POLICY_METADATA_KIND,
        {
            "policy_class": policy_class,
            "horizon": int(metadata.horizon),
            "feature_dim": int(metadata.feature_dim),
            "epsilon_dim": int(metadata.epsilon_dim),
            "feature_basis": str(metadata.feature_basis),
            "live_feature_source": str(metadata.live_feature_source),
            "shared_across_trials_in_batch": bool(metadata.shared_across_trials_in_batch),
            "time_varying": bool(metadata.time_varying),
            "centered_features": bool(metadata.centered_features),
            "has_bias": bool(metadata.has_bias),
            "gain_shape": list(metadata.gain_shape),
            "bias_shape": None if bias_shape is None else list(bias_shape),
            "zero_feature_behavior": (
                "zero_epsilon" if policy_class == "linear_no_bias" else "bias_epsilon"
            ),
            "semantics": (
                "epsilon_t is evaluated from live perturbed rollout features at time t; "
                "the finite policy parameters are shared across every trial in the batch."
            ),
        },
    )


def _register_method_once(registration: TrainingMethodRegistration) -> None:
    try:
        DEFAULT_TRAINING_METHOD_REGISTRY.resolve(registration.method_ref, path="/method_ref")
    except ValueError:
        DEFAULT_TRAINING_METHOD_REGISTRY.register(registration)


def _distillation_contract(
    *,
    method_ref: str,
    payload_version: str,
    phase_names: tuple[str, ...],
    update_step_name: str,
    kernel_ref: str,
    extra_axes: tuple[AxisSpec, ...],
    extra_slots: tuple[StateSlotSpec, ...],
    metadata: dict[str, Any],
) -> MethodContractSpec:
    axes = [
        AxisSpec(name="batch", role="batch"),
        AxisSpec(name="replicate", role="replicate"),
        *extra_axes,
    ]
    state_slots = [
        StateSlotSpec(name="model", role="model", axis="replicate"),
        StateSlotSpec(name="optimizer", role="optimizer", axis="replicate"),
        StateSlotSpec(name="prng", role="prng", axis="replicate"),
        StateSlotSpec(name="objective", role="objective"),
        *extra_slots,
        StateSlotSpec(name="train_loss", role="metric", axis="replicate", required=False),
    ]
    phases: list[PhaseSpec] = []
    for index, phase_name in enumerate(phase_names):
        phases.append(
            PhaseSpec(
                name=phase_name,
                kind="outer_loop",
                reads=[
                    "model",
                    "optimizer",
                    "prng",
                    "objective",
                    *[slot.name for slot in extra_slots],
                ],
                writes=["model", "optimizer", "prng", "train_loss"],
                update_steps=[update_step_name],
                legal_next=list(phase_names[index + 1 : index + 2]),
                checkpoint_barrier=f"after_{phase_name}",
            )
        )
    program = PhaseProgramSpec(
        phases=phases,
        initial_phase=phase_names[0],
        update_steps=[
            UpdateStepSpec(
                name=update_step_name,
                kind="gradient",
                kernel=UpdateKernelSpec(kernel_ref=kernel_ref),
                reads=[
                    "model",
                    "optimizer",
                    "prng",
                    "objective",
                    *[slot.name for slot in extra_slots],
                ],
                writes=["model", "optimizer", "prng", "train_loss"],
                axes=[axis.name for axis in axes],
                optimizer_binding="optimizer_to_model",
            )
        ],
        optimizer_bindings=[
            OptimizerTargetBinding(
                name="optimizer_to_model",
                optimizer_slot="optimizer",
                target_slot="model",
                projection="after_step",
                phase_scope=list(phase_names),
                objective_reads=["objective"],
            )
        ],
        checkpoint_barriers=[
            CheckpointBarrierSpec(
                name=f"after_{phase_name}",
                phase=phase_name,
                slots=[
                    CheckpointSlotSpec(slot="model", axis="replicate"),
                    CheckpointSlotSpec(slot="optimizer", axis="replicate"),
                    CheckpointSlotSpec(slot="prng", axis="replicate"),
                    CheckpointSlotSpec(slot="train_loss", axis="replicate", required=False),
                ],
            )
            for phase_name in phase_names
        ],
        metadata={"teacher_is_payload_config_not_axis": True},
    )
    return MethodContractSpec(
        method_ref=method_ref,
        method_payload_schema_version=payload_version,
        axes=axes,
        state_slots=state_slots,
        phase_program=program,
        metadata=metadata,
    )


def _closed_loop_distillation_update_kernels(
    _payload: BaseModel | None = None,
) -> dict[str, Any]:
    return {"rlrmp.train.closed_loop_distillation.closed_loop_gradient_update": _no_op_kernel}


def _guided_distillation_update_kernels(_payload: BaseModel | None = None) -> dict[str, Any]:
    return {"rlrmp.train.guided_distillation.guided_gradient_update": _no_op_kernel}


def _no_op_kernel(slots: dict[str, Any], coordinate: Any, context: dict[str, Any]) -> dict[str, Any]:
    del coordinate, context
    return {
        "model": slots["model"],
        "optimizer": slots["optimizer"],
        "prng": slots["prng"],
        "train_loss": 0.0,
    }


def _distillation_method_parts(
    run_spec: dict[str, Any],
    *,
    method: str,
) -> tuple[MethodRefSpec, MethodPayloadEnvelope, MethodContractSpec]:
    if method == "closed_loop_distillation":
        return (
            closed_loop_distillation_method_ref(),
            closed_loop_distillation_method_payload(run_spec),
            closed_loop_distillation_method_contract(),
        )
    if method == "guided_distillation":
        return (
            guided_distillation_method_ref(),
            guided_distillation_method_payload(run_spec),
            guided_distillation_method_contract(),
        )
    raise ValueError(f"unknown RLRMP distillation method {method!r}")


def _distillation_method_ref_key(method: str) -> str:
    if method == "closed_loop_distillation":
        return CLOSED_LOOP_DISTILLATION_METHOD_REF
    if method == "guided_distillation":
        return GUIDED_DISTILLATION_METHOD_REF
    raise ValueError(f"unknown RLRMP distillation method {method!r}")


def _distillation_training_config(run_spec: dict[str, Any], *, method: str) -> TrainingConfig:
    if method == "closed_loop_distillation":
        student = _mapping(run_spec, "student_contract")
        return TrainingConfig(
            n_batches=int(student.get("n_train_batches", 1)),
            batch_size=int(student.get("batch_size", 1)),
            learning_rate=float(student.get("controller_lr", 1e-3)),
            grad_clip=float(student.get("gradient_clip_norm", 1.0)),
            hidden_dim=int(student.get("hidden_size", 0)),
            network_type="gru",
            n_reach_steps=int(_mapping(run_spec, "teacher_contract").get("horizon", 60)),
            effort_weight=float(
                _mapping(_mapping(run_spec, "loss_surface"), "weights").get(
                    "action_force_trajectory",
                    1.0,
                )
            ),
            snapshot_interval=int(_mapping(run_spec, "checkpointing").get("interval_batches", 1)),
        )
    model = _mapping(run_spec, "model_contract")
    optimizer = _mapping(run_spec, "optimizer")
    teacher_bank = _mapping(run_spec, "teacher_bank")
    return TrainingConfig(
        n_batches=int(run_spec.get("n_train_batches", _mapping(run_spec, "training_schedule").get("total_batches", 1))),
        batch_size=int(run_spec.get("batch_size", model.get("batch_size", 1))),
        learning_rate=float(run_spec.get("controller_lr", optimizer.get("controller_lr", 1e-3))),
        grad_clip=float(optimizer.get("gradient_clip_norm", 1.0)),
        hidden_dim=int(model.get("hidden_size", 0)),
        network_type="gru",
        n_reach_steps=int(teacher_bank.get("horizon", 60)),
        effort_weight=float(
            _mapping(_mapping(run_spec, "distillation_surface"), "components")
            .get("clean_action", {})
            .get("weight", 1.0)
        ),
        snapshot_interval=int(_mapping(run_spec, "checkpointing").get("interval_batches", 1)),
    )


def _distillation_task_params(run_spec: dict[str, Any], *, method: str) -> dict[str, Any]:
    if method == "closed_loop_distillation":
        student = _mapping(run_spec, "student_contract")
        return {
            "student_contract": student,
            "closed_loop": True,
            "teacher": _mapping(run_spec, "teacher_contract").get("controller"),
        }
    return {
        "model_contract": _mapping(run_spec, "model_contract"),
        "teacher_bank": _mapping(run_spec, "teacher_bank"),
        "training_schedule": _mapping(run_spec, "training_schedule"),
    }


def _distillation_objective_payload(run_spec: dict[str, Any], *, method: str) -> dict[str, Any]:
    if method == "closed_loop_distillation":
        return {
            "loss_surface": _mapping(run_spec, "loss_surface"),
            "closed_loop_semantics": _mapping(run_spec, "closed_loop_semantics"),
        }
    return {
        "distillation_surface": _mapping(run_spec, "distillation_surface"),
        "training_schedule": _mapping(run_spec, "training_schedule"),
    }


def _distillation_graph_ref(run_spec: dict[str, Any]) -> str:
    base = _mapping(run_spec, "base_contract").get("run_spec")
    return str(base or "runtime://rlrmp.train.task_model.setup_task_model_pair")


def _distillation_setup_function(run_spec: dict[str, Any], *, method: str) -> str:
    key = "student_contract" if method == "closed_loop_distillation" else "model_contract"
    return str(
        _mapping(run_spec, key).get(
            "setup_function",
            "rlrmp.train.task_model.setup_task_model_pair",
        )
    )


def _distillation_runner(run_spec: dict[str, Any], *, method: str) -> str:
    entry = _mapping(run_spec, "training_entry")
    if method == "closed_loop_distillation":
        return str(entry.get("module", "rlrmp.train.closed_loop_distillation"))
    return str(entry.get("trainer", "rlrmp.train.guided_distillation.run_guided_distillation_training"))


def _optimizer_payload_from_run_spec(run_spec: dict[str, Any]) -> dict[str, Any]:
    optimizer = _mapping(run_spec, "optimizer")
    if optimizer:
        return optimizer
    student = _mapping(run_spec, "student_contract")
    return {
        "name": "adamw",
        "controller_lr": student.get("controller_lr"),
        "lr_schedule": student.get("lr_schedule"),
        "lr_warmup_batches": student.get("lr_warmup_batches"),
        "lr_cosine_alpha": student.get("lr_cosine_alpha"),
        "gradient_clip_norm": student.get("gradient_clip_norm"),
    }


def _int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)


def rlrmp_extension_payload(run_spec: dict[str, Any]) -> dict[str, Any]:
    """Return the RLRMP-owned v2 extension payload embedded in tracked recipes."""

    training_summary = _mapping(run_spec, "training_summary")
    model_summary = _mapping(run_spec, "model_summary")
    graph = _mapping(run_spec, "feedbax_graph")
    feedback_descriptors = controller_feedback_descriptor_from_container(
        model_summary,
        feedback_dim=_feedback_dim_from_run_spec(run_spec),
        source="rlrmp_extension_payload",
    )
    payload = {
        "issue": str(run_spec.get("issue", "")),
        "mode": str(run_spec.get("mode", "")),
        "training_script": str(run_spec.get("training_script", "")),
        "loss_objective": run_spec.get("loss_objective"),
        "training_mode": training_summary.get("training_mode"),
        "game_card": run_spec.get("game_card"),
        "model_summary": model_summary,
        "training_summary": training_summary,
        "loss_summary": run_spec.get("loss_summary"),
        "task_timing": run_spec.get("task_timing"),
        "fidelity_status": run_spec.get("fidelity_status"),
        "training_distribution": run_spec.get("training_distribution"),
        "delayed_reach": run_spec.get("delayed_reach"),
        "validation_bins": run_spec.get("validation_bins"),
        "feedbax_graph": graph,
        DESCRIPTOR_PAYLOAD_KEY: feedback_descriptors,
        "hps": run_spec.get("hps"),
    }
    return stamp_current_schema(RUN_SPEC_KIND, payload)


def build_feedbax_training_run_spec(
    run_spec: dict[str, Any],
    *,
    graph_spec: Any,
    output_dir: Path,
    spec_dir: Path,
) -> TrainingRunSpec:
    """Build the composed Feedbax ``TrainingRunSpec`` for one C&S GRU run."""

    training_summary = _mapping(run_spec, "training_summary")
    optimizer = _mapping(run_spec, "optimizer")
    checkpointing = _mapping(run_spec, "checkpointing")
    training_diagnostics = _mapping(run_spec, "training_diagnostics")
    feedback_descriptors = controller_feedback_descriptor_from_container(
        _mapping(run_spec, "model_summary"),
        feedback_dim=_feedback_dim_from_run_spec(run_spec),
        source="feedbax_training_run_spec",
    )
    objective_payload = {
        "loss_summary": run_spec.get("loss_summary"),
        "loss_objective": run_spec.get("loss_objective"),
        "fidelity_status": run_spec.get("fidelity_status"),
        DESCRIPTOR_PAYLOAD_KEY: feedback_descriptors,
    }
    method_metadata = {
        "runner": "rlrmp.train.cs_nominal_gru",
        "rlrmp_training_mode": training_summary.get("training_mode"),
        "rlrmp_loss_objective": run_spec.get("loss_objective"),
        "adversarial_phase": run_spec.get("adversarial_phase"),
        "rlrmp_extension_payload": RLRMP_RUN_SPEC_PAYLOAD_KEY,
    }
    return TrainingRunSpec(
        graph=GraphTopologySourceSpec(
            inline=graph_spec_payload(graph_spec),
            schema_id=getattr(graph_spec, "schema_id", None),
            schema_version=getattr(graph_spec, "schema_version", None),
            metadata={
                "source": "materialized_runtime_graph",
                "sidecar_policy": run_spec.get("feedbax_graph", {}),
                DESCRIPTOR_PAYLOAD_KEY: feedback_descriptors,
                "descriptor_basis_hash": feedback_descriptors["descriptor_basis_hash"],
            },
        ),
        task=TaskSpec(
            type=str(_mapping(run_spec, "task_timing").get("type", "rlrmp_task")),
            params=_mapping(run_spec, "task_timing"),
        ),
        training_config=TrainingConfig(
            n_batches=int(run_spec.get("n_train_batches", training_summary["n_train_batches"])),
            batch_size=int(run_spec.get("batch_size", training_summary["batch_size"])),
            learning_rate=float(run_spec.get("controller_lr", optimizer["learning_rate_0"])),
            grad_clip=(
                1.0
                if optimizer.get("gradient_clip_norm") is None
                else float(optimizer["gradient_clip_norm"])
            ),
            hidden_dim=int(_mapping(run_spec, "model_summary").get("hidden_size", 0)),
            network_type="gru",
            n_reach_steps=int(_mapping(run_spec, "task_timing").get("n_steps", 0)),
            effort_weight=float(
                _mapping(_mapping(run_spec, "loss_summary"), "active_cs_terms")
                .get("control", {})
                .get("scale", 1.0)
            ),
            snapshot_interval=int(checkpointing.get("interval_batches", 1)),
        ),
        objective=ObjectiveSlotSpec(
            kind="external",
            payload=objective_payload,
            schema_id="rlrmp.cs_gru_objective",
            schema_version="rlrmp.cs_gru_objective.v1",
            metadata={"rlrmp_loss_objective": run_spec.get("loss_objective")},
        ),
        risk_aggregation=RiskAggregationSpec(
            realization="mean",
            replicate="mean",
            metadata={"source": "$.training_summary"},
        ),
        method_ref=standard_supervised_method_ref(),
        method_payload=standard_supervised_method_payload(),
        method_extensions={"metadata": method_metadata},
        worker_execution=WorkerExecutionSpec(
            method_contract=standard_supervised_method_contract(),
            effective_phase=standard_supervised_effective_phase_spec(),
            metadata={
                "legacy_runner": "rlrmp.train.cs_nominal_gru.run_full_training",
                "full_feedbax_executor_deferred_to": "54b0c2e",
            },
        ),
        execution=ExecutionPolicySpec(
            mode="local",
            require_review=bool(run_spec.get("full_training_launch") != "requested"),
            allow_cloud=False,
            metadata={"launch_mode": run_spec.get("mode")},
        ),
        artifacts=ArtifactPolicySpec(
            manifest_root="_artifacts/feedbax_runs",
            artifact_root=str(output_dir),
            custody="local",
            metadata={"tracked_spec_dir": str(spec_dir)},
        ),
        checkpoint_progress=CheckpointProgressPolicySpec(
            checkpoint_interval=int(checkpointing.get("interval_batches", 1)),
            progress_interval=(
                None
                if training_diagnostics.get("enabled") is False
                else int(checkpointing.get("interval_batches", 1))
            ),
            metadata={"checkpoint_dir": checkpointing.get("checkpoint_dir")},
        ),
        metadata={
            "composed_with": RLRMP_RUN_SPEC_PAYLOAD_KEY,
            "serialize_do_not_rederive": True,
            DESCRIPTOR_PAYLOAD_KEY: feedback_descriptors,
            "descriptor_basis_hash": feedback_descriptors["descriptor_basis_hash"],
        },
    )


def attach_composed_training_specs(
    run_spec: dict[str, Any],
    *,
    graph_spec: Any,
    output_dir: Path,
    spec_dir: Path,
) -> dict[str, Any]:
    """Attach Feedbax and RLRMP spec records to a tracked run recipe."""

    payload = dict(run_spec)
    extension = rlrmp_extension_payload(payload)
    feedbax_spec = build_feedbax_training_run_spec(
        payload,
        graph_spec=graph_spec,
        output_dir=output_dir,
        spec_dir=spec_dir,
    )
    payload[RLRMP_RUN_SPEC_PAYLOAD_KEY] = extension
    payload[FEEDBAX_TRAINING_RUN_SPEC_KEY] = feedbax_spec.model_dump(
        mode="json",
        exclude_none=True,
    )
    return payload


def feedbax_training_run_spec_from_payload(run_spec: dict[str, Any]) -> TrainingRunSpec:
    """Load the composed Feedbax ``TrainingRunSpec`` from a tracked recipe."""

    return TrainingRunSpec.model_validate(run_spec[FEEDBAX_TRAINING_RUN_SPEC_KEY])


def assert_runtime_graph_matches_training_spec(
    run_spec: dict[str, Any],
    *,
    graph_spec: Any,
) -> None:
    """Raise if the runtime graph diverges from the serialized Feedbax spec."""

    expected = feedbax_training_run_spec_from_payload(run_spec).graph.inline
    actual = graph_spec_payload(graph_spec)
    if expected != actual:
        raise ValueError(
            "Serialized TrainingRunSpec graph does not match the materialized runtime graph"
        )


def write_training_run_manifest_for_spec(
    *,
    run_spec_path: Path,
    run_spec: dict[str, Any],
    manifest_root: Path,
    graph_manifest_path: Path,
    graph_spec_path: Path | None,
) -> Path:
    """Emit the Feedbax ``TrainingRunManifest`` parity record at production time."""

    rel_run_spec = _repo_relative(run_spec_path)
    rel_graph_manifest = _repo_relative(graph_manifest_path)
    artifacts = [
        ArtifactRef(
            role="tracked_run_spec",
            logical_name=run_spec_path.name,
            artifact_id=f"repo://rlrmp/{rel_run_spec}",
            sha256=sha256_file(run_spec_path),
            media_type="application/json",
            storage_backend="rlrmp-results",
            uri=rel_run_spec,
            metadata={"availability": "checked_in", "source_issue": str(run_spec.get("issue"))},
        ),
        ArtifactRef(
            role="model_graph_manifest",
            logical_name=graph_manifest_path.name,
            artifact_id=f"repo://rlrmp/{rel_graph_manifest}",
            sha256=sha256_file(graph_manifest_path),
            media_type="application/json",
            storage_backend="rlrmp-results",
            uri=rel_graph_manifest,
            metadata={"availability": "checked_in", "source_issue": str(run_spec.get("issue"))},
        ),
    ]
    if graph_spec_path is not None:
        rel_graph_spec = _repo_relative(graph_spec_path)
        artifacts.append(
            ArtifactRef(
                role="model_graph_spec",
                logical_name=graph_spec_path.name,
                artifact_id=f"repo://rlrmp/{rel_graph_spec}",
                sha256=sha256_file(graph_spec_path),
                media_type="application/json",
                storage_backend="rlrmp-results",
                uri=rel_graph_spec,
                metadata={"availability": "checked_in", "source_issue": str(run_spec.get("issue"))},
            )
        )

    extension = run_spec[RLRMP_RUN_SPEC_PAYLOAD_KEY]
    training_spec_payload = SpecPayload(
        kind=RUN_SPEC_KIND,
        schema_id=RUN_SPEC_SCHEMA_ID,
        schema_version=RUN_SPEC_SCHEMA_VERSION,
        inline=extension,
        ref=rel_run_spec,
        sha256=sha256_file(run_spec_path),
        source_sha256=sha256_file(run_spec_path),
        metadata={"source_record_role": "tracked_run_spec"},
    )
    training_spec_payload = spec_payload(
        RUN_SPEC_KIND,
        training_spec_payload.inline,
        ref=training_spec_payload.ref,
    ).model_copy(
        update={
            "source_sha256": training_spec_payload.source_sha256,
            "metadata": training_spec_payload.metadata,
        }
    )
    manifest = TrainingRunManifest(
        id=f"feedbax-training-run:rlrmp-{run_spec.get('issue')}-{run_spec_path.stem}",
        status="completed" if run_spec.get("mode") == "full_train" else "pending",
        job_id=str(run_spec_path.stem),
        graph_spec=None,
        training_spec=training_spec_payload,
        provenance=Provenance(
            source_repo="https://github.com/i-m-mll/rlrmp.git",
            source_branch=_string_or_none(
                _mapping(run_spec, "provenance").get("git", {}).get("branch")
            ),
            source_commit=_string_or_none(
                _mapping(run_spec, "provenance").get("git", {}).get("commit")
            ),
            dirty=bool(_mapping(run_spec, "provenance").get("git", {}).get("dirty", False)),
            issues=[str(run_spec.get("issue"))],
            metadata={"producer": "rlrmp.train.cs_nominal_gru.write_run_spec"},
        ),
        artifacts=artifacts,
        summary_metrics={"planned_batches": int(run_spec.get("n_train_batches", 0))},
        metadata={
            "feedbax_training_run_spec": run_spec[FEEDBAX_TRAINING_RUN_SPEC_KEY],
            "rlrmp_layout": {
                "tracked_specs": "results/<issue>/runs/*.json",
                "bulk_artifacts": "_artifacts/<issue>/runs/<variant>/",
                "feedbax_manifest_root": "_artifacts/feedbax_runs/",
            },
        },
    )
    return write_manifest(manifest, root=manifest_root)


def _mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    return value if isinstance(value, dict) else {}


def _feedback_dim_from_run_spec(run_spec: dict[str, Any]) -> int:
    model_summary = _mapping(run_spec, "model_summary")
    feedback = model_summary.get("feedback")
    if isinstance(feedback, dict) and feedback.get("dimension") is not None:
        return int(feedback["dimension"])
    hps = run_spec.get("hps")
    if isinstance(hps, dict):
        model = hps.get("model")
        target = hps.get("target_relative_multitarget")
        if isinstance(model, dict) and model.get("force_filter_feedback") is True:
            return 6
        if isinstance(target, dict) and target.get("force_filter_feedback") is True:
            return 6
    return 4


def _repo_relative(path: Path) -> str:
    from rlrmp.paths import REPO_ROOT

    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _string_or_none(value: Any) -> str | None:
    return None if value is None else str(value)
