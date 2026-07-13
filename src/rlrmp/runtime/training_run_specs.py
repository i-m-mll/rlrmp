"""Feedbax TrainingRunSpec adapters for RLRMP training recipes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from feedbax.contracts.manifest import (
    ArtifactRef,
    Provenance,
    SCHEMA_VERSION,
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
    LrScheduleSpec,
    MethodPayloadEnvelope,
    MethodRefSpec,
    ObjectiveSlotSpec,
    OptimizerSpec,
    RiskAggregationSpec,
    TaskSpec,
    TrainingConfig,
    TrainingMethodRegistration,
    TrainingRunSpec,
    TRAINING_RUN_SPEC_SCHEMA_ID as FEEDBAX_TRAINING_RUN_SPEC_SCHEMA_ID,
    TRAINING_RUN_SPEC_SCHEMA_VERSION as FEEDBAX_TRAINING_RUN_SPEC_SCHEMA_VERSION,
    WorkerExecutionSpec,
)
from feedbax.contracts.worker import (
    AxisSpec,
    CheckpointBarrierSpec,
    CheckpointSlotSpec,
    EffectivePhaseSpec,
    MethodContractSpec,
    MetricGuardSpec,
    OptimizerTargetBinding,
    PhaseProgramSpec,
    PhaseSpec,
    PhaseTransitionSpec,
    ResumeCoordinateSpec,
    StateSlotSpec,
    TrainingBatchProgressSpec,
    UpdateKernelSpec,
    UpdateStepSpec,
    derive_consistency_predicate,
)

from rlrmp.model.feedback_descriptors import (
    DESCRIPTOR_PAYLOAD_KEY,
    controller_feedback_descriptor_from_container,
)
from rlrmp.model.feedbax_graph import graph_spec_payload
from rlrmp.paths import portable_repo_path
from rlrmp.runtime.graph_spec_migrations import migrate_feedbax_graph_payload
from rlrmp.runtime.spec_migrations import (
    FeedbaxTrainingRunSpecMigrationError,
    FINITE_ADVERSARY_POLICY_METADATA_KIND,
    LEGACY_FEEDBAX_STANDARD_SUPERVISED_METHOD_REF,
    RUN_SPEC_KIND,
    RUN_SPEC_SCHEMA_ID,
    RUN_SPEC_SCHEMA_VERSION,
    SEMANTIC_METHOD_EXTENSION_METADATA_KEYS,
    accept_rlrmp_spec_payload,
    ensure_rlrmp_spec_families,
    stamp_current_schema,
)
from rlrmp.train.executor.guards import make_stop_predicate
from rlrmp.train.executor.slots import (
    ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF,
    COMPLETED_BATCHES,
    CS_SUPERVISED_METHOD_REF,
    CS_SUPERVISED_SCHEMA,
    HISTORY_CHUNK_BYTES,
    MODEL,
    OBJECTIVE,
    OPTIMIZER,
    POLICY_ADVERSARY_SUPERVISED_METHOD_REF,
    PRNG,
    TRAIN_LOSS,
    artifact_sink_specs,
    checkpoint_slot_specs,
    supervised_state_slots,
)


FEEDBAX_TRAINING_RUN_SPEC_KEY = "feedbax_training_run_spec"
RLRMP_RUN_SPEC_PAYLOAD_KEY = "rlrmp_run_spec"
COMPACT_RUN_SPEC_MARKER_KEY = "compact_run_spec"
CONSUMED_DATA_IDENTITIES_KEY = "consumed_data_identities"
POST_RUN_SCHEMA_VERSION = "rlrmp.post_run_provenance.v1"
PINNED_MANIFEST_ROOT = "_artifacts/feedbax_runs"
FEEDBAX_PROVIDER_VERSION = "feedbax-provider.v1"
CHECKPOINT_INTERVAL_BATCHES_KEY = "interval_batches"

CLOSED_LOOP_DISTILLATION_METHOD_REF = "rlrmp/closed_loop_distillation/v1"
CLOSED_LOOP_DISTILLATION_PAYLOAD_SCHEMA_ID = (
    "rlrmp.spec.training_method.closed_loop_distillation_payload"
)
CLOSED_LOOP_DISTILLATION_PAYLOAD_SCHEMA_VERSION = (
    "rlrmp.spec.training_method.closed_loop_distillation_payload.v2"
)
GUIDED_DISTILLATION_METHOD_REF = "rlrmp/guided_distillation/v1"
GUIDED_DISTILLATION_PAYLOAD_SCHEMA_ID = "rlrmp.spec.training_method.guided_distillation_payload"
GUIDED_DISTILLATION_PAYLOAD_SCHEMA_VERSION = (
    "rlrmp.spec.training_method.guided_distillation_payload.v2"
)
CS_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID = "rlrmp.spec.training_method.cs_supervised_payload"
CS_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION = "rlrmp.spec.training_method.cs_supervised_payload.v1"
CS_SUPERVISED_CHUNK_KERNEL_REF = "rlrmp.cs_supervised.train_chunk"
CS_SUPERVISED_STOP_PREDICATE_REF = "rlrmp.cs_supervised.training_complete"
CS_SUPERVISED_BARRIER = "after_train_chunk"

_COMPACT_RUN_SPEC_DUPLICATE_PATHS = (
    "game_card",
    "training_distribution.perturbation_training",
    "feedbax_graph",
)
_MISSING = object()


class MissingTrainingRunSpecFieldError(ValueError):
    """Raised when a recording adapter would otherwise fabricate run provenance."""

    def __init__(self, *, field_path: str, spec_identity: str) -> None:
        self.field_path = field_path
        self.spec_identity = spec_identity
        super().__init__(
            "TrainingRunSpec recording adapter missing required field "
            f"{field_path!r} in {spec_identity}"
        )


@dataclass(frozen=True)
class TrainingRunSpecScaffold:
    """Shared top-level policies for RLRMP-authored ``TrainingRunSpec`` records."""

    risk_aggregation: RiskAggregationSpec
    execution: ExecutionPolicySpec
    artifacts: ArtifactPolicySpec
    checkpoint_progress: CheckpointProgressPolicySpec
    metadata: dict[str, Any]

    def build(
        self,
        *,
        graph: GraphTopologySourceSpec,
        task: TaskSpec,
        training_config: TrainingConfig,
        objective: ObjectiveSlotSpec,
        method_ref: MethodRefSpec,
        method_payload: MethodPayloadEnvelope,
        method_extensions: Mapping[str, Any],
        method_contract: MethodContractSpec,
        effective_phase: EffectivePhaseSpec,
        worker_metadata: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> TrainingRunSpec:
        """Build one complete run spec around method-specific fields."""
        return TrainingRunSpec(
            graph=graph,
            task=task,
            training_config=training_config,
            objective=objective,
            risk_aggregation=self.risk_aggregation,
            method_ref=method_ref,
            method_payload=method_payload,
            method_extensions=dict(method_extensions),
            worker_execution=WorkerExecutionSpec(
                method_contract=method_contract,
                effective_phase=effective_phase,
                metadata=dict(worker_metadata),
            ),
            execution=self.execution,
            artifacts=self.artifacts,
            checkpoint_progress=self.checkpoint_progress,
            metadata={**dict(metadata or {}), **self.metadata},
        )


def build_training_run_spec_scaffold(
    *,
    risk_metadata: Mapping[str, Any],
    execution_mode: Literal["dry_run", "local", "remote"],
    require_review: bool,
    allow_cloud: bool,
    execution_metadata: Mapping[str, Any],
    artifact_root: str,
    artifact_metadata: Mapping[str, Any],
    checkpoint_interval: int | None,
    progress_interval: int | None,
    checkpoint_metadata: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> TrainingRunSpecScaffold:
    """Build the shared RLRMP policy preset for a Feedbax training run."""
    return TrainingRunSpecScaffold(
        risk_aggregation=RiskAggregationSpec(
            realization="mean",
            replicate="mean",
            metadata=dict(risk_metadata),
        ),
        execution=ExecutionPolicySpec(
            mode=execution_mode,
            require_review=require_review,
            allow_cloud=allow_cloud,
            metadata=dict(execution_metadata),
        ),
        artifacts=ArtifactPolicySpec(
            manifest_root=PINNED_MANIFEST_ROOT,
            artifact_root=artifact_root,
            custody="local",
            metadata=dict(artifact_metadata),
        ),
        checkpoint_progress=CheckpointProgressPolicySpec(
            checkpoint_interval=checkpoint_interval,
            progress_interval=progress_interval,
            metadata=dict(checkpoint_metadata),
        ),
        metadata={**dict(metadata), "serialize_do_not_rederive": True},
    )


class _StrictPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DistillationBaseContractPayload(_StrictPayloadModel):
    """Shared base-row provenance recorded in distillation method payloads."""

    issue: str
    run_id: str
    run_spec: str
    inherit: list[str]


class DistillationOptimizerPayload(_StrictPayloadModel):
    """Optimizer settings recorded by distillation method payloads."""

    name: str
    controller_lr: float | None = None
    lr_schedule: str
    lr_warmup_batches: int
    lr_cosine_alpha: float
    gradient_clip_norm: float | None
    lr_warmup_init_fraction: float | None = None


class DistillationCheckpointingPayload(_StrictPayloadModel):
    """Checkpoint policy recorded by distillation method payloads."""

    enabled: bool
    interval_batches: int
    resume_flag: str
    latest_pointer: str
    format: str
    default: str | None = None
    contents: list[str] | None = None


class ClosedLoopTeacherContractPayload(_StrictPayloadModel):
    """Analytical teacher contract for closed-loop distillation."""

    issue: str
    controller: str
    teacher_package: str
    teacher_gains_key: str
    required_package_key: str
    matched_inputs: list[str]
    feedback_basis: str
    reference_basis_note: str
    horizon: int | None = None


class ClosedLoopStudentContractPayload(_StrictPayloadModel):
    """Student graph/training contract for closed-loop distillation."""

    setup_function: str
    graph_source: str
    feedback_input_basis: str
    controller_input_dim: int
    force_filter_feedback: bool
    initial_hidden_encoder: bool
    hidden_size: int
    n_replicates: int
    batch_size: int
    n_train_batches: int
    controller_lr: float
    lr_schedule: str
    lr_warmup_batches: int
    lr_cosine_alpha: float
    gradient_clip_norm: float | None
    broad_epsilon_pgd_training: bool
    trainable_dtype: str


class ClosedLoopSemanticsPayload(_StrictPayloadModel):
    """Closed-loop rollout semantics for the analytical-teacher loss."""

    student_rollout: str
    student_actions_feed_future_observations: bool
    teacher_forced_feedback_bank_imitation: bool
    old_guided_trainer_is_main_path: bool
    trainer_contract: str


class ClosedLoopWeightedComponentPayload(_StrictPayloadModel):
    """Enabled/weight pair for closed-loop scalar loss components."""

    enabled: bool
    weight: float


class ClosedLoopDirectionalJvpComponentPayload(ClosedLoopWeightedComponentPayload):
    """Directional JVP component settings for closed-loop distillation."""

    basis: str
    jacobian_shape: list[int]
    implementation: str


class ClosedLoopLossComponentsPayload(_StrictPayloadModel):
    """Named closed-loop distillation loss components."""

    closed_loop_kinematics_trajectory: ClosedLoopWeightedComponentPayload
    velocity_trajectory: ClosedLoopWeightedComponentPayload
    action_force_trajectory: ClosedLoopWeightedComponentPayload
    perturbation_response_trajectory: ClosedLoopWeightedComponentPayload
    directional_input_output_jvp: ClosedLoopDirectionalJvpComponentPayload
    task_qr_rollout: ClosedLoopWeightedComponentPayload


class ClosedLoopLossWeightsPayload(_StrictPayloadModel):
    """Closed-loop loss weight summary."""

    kinematics_trajectory: float
    velocity: float
    action_force_trajectory: float
    perturbation_response_trajectory: float
    directional_input_output_jvp: float
    task_qr_rollout: float
    endpoint: float
    settling: float


class ClosedLoopLossSurfacePayload(_StrictPayloadModel):
    """Closed-loop distillation loss-surface payload."""

    weights: ClosedLoopLossWeightsPayload
    default_task_qr_rollout_loss: str
    task_qr_rollout_loss_can_be_enabled_later: bool
    components: ClosedLoopLossComponentsPayload


class ClosedLoopDistillationMethodPayload(_StrictPayloadModel):
    """Governed payload for closed-loop analytical-teacher distillation."""

    teacher_contract: ClosedLoopTeacherContractPayload
    student_contract: ClosedLoopStudentContractPayload
    base_contract: DistillationBaseContractPayload
    closed_loop_semantics: ClosedLoopSemanticsPayload
    loss_surface: ClosedLoopLossSurfacePayload
    optimizer: DistillationOptimizerPayload
    checkpointing: DistillationCheckpointingPayload
    finite_adversary_policy_metadata: dict[str, Any] | None = None


class GuidedTeacherExternalBasisPayload(_StrictPayloadModel):
    """External teacher/student basis names for guided distillation."""

    feedback_history: str
    action_history: str
    action_output: str


class GuidedTeacherContractPayload(_StrictPayloadModel):
    """Analytical teacher contract for guided distillation."""

    issue: str
    primary_teacher: str
    diagnostic_teacher: str
    teacher_package: str
    teacher_manifest: str
    external_basis: GuidedTeacherExternalBasisPayload
    teacher_gains_key: str | None = None
    student_architecture_boundary: str | None = None


class GuidedTeacherBankPayload(_StrictPayloadModel):
    """Batched analytical-teacher materialization contract."""

    materializer: str
    source: str
    teacher: str
    horizon: int
    sampled_initial_state_std: float
    observation_perturbation_std: float
    action_context: str
    approximation: str
    teacher_gains_key: str | None = None


class GuidedTrainingPhasePayload(_StrictPayloadModel):
    """One phase in the guided distillation forcing schedule."""

    name: str
    start_batch: int
    end_batch: int
    teacher_forcing_fraction: float
    student_forcing_fraction: float


class GuidedTrainingSchedulePayload(_StrictPayloadModel):
    """Guided distillation training schedule."""

    total_batches: int | None = None
    phases: list[GuidedTrainingPhasePayload]


class GuidedDistillationWeightsPayload(_StrictPayloadModel):
    """Guided distillation loss weights."""

    clean_action: float
    perturbation_response: float
    input_output_jvp: float
    student_forced_rollout_anchor: float


class GuidedDistillationConfigPayload(_StrictPayloadModel):
    """Summary of the reusable guided distillation loss config."""

    issue: str
    experiment_issue: str
    teacher_issue: str
    base_issue: str
    base_run_id: str
    teacher_package: str
    primary_teacher: str
    diagnostic_teacher: str
    feedback_basis: str
    action_basis: str
    hidden_state_supervision: bool
    n_jvp_directions: int
    jvp_direction_basis: str
    jvp_direction_sampler: str
    weights: GuidedDistillationWeightsPayload


class GuidedDescribedComponentPayload(_StrictPayloadModel):
    """Enabled/weight/description triple for guided loss components."""

    enabled: bool
    weight: float
    description: str


class GuidedJvpComponentPayload(_StrictPayloadModel):
    """Directional JVP settings for guided distillation."""

    enabled: bool
    weight: float
    n_directions: int
    direction_basis: str
    implementation: str


class GuidedDistillationComponentsPayload(_StrictPayloadModel):
    """Named guided distillation loss components."""

    clean_action: GuidedDescribedComponentPayload
    perturbation_response: GuidedDescribedComponentPayload
    input_output_jvp: GuidedJvpComponentPayload
    student_forced_rollout_anchor: GuidedDescribedComponentPayload


class GuidedDistillationSurfacePayload(_StrictPayloadModel):
    """Guided distillation loss surface."""

    config: GuidedDistillationConfigPayload
    hidden_state_supervision: bool
    components: GuidedDistillationComponentsPayload
    student_action_history_input: bool | None = None


class GuidedModelContractPayload(_StrictPayloadModel):
    """Student model contract for guided distillation."""

    setup_function: str
    checkpoint_format: str
    final_model: str
    initial_hidden_encoder: bool
    force_filter_feedback: bool
    hidden_size: int
    batch_size: int | None = None
    n_replicates: int
    vectorized_replicates: bool
    plant_backend: str
    stochastic_preset: str
    broad_epsilon_pgd_training: bool
    checkpoint_model: str | None = None
    controller_input_dim: int | None = None
    student_action_history_input: bool | None = None
    trainable_dtype: str | None = None
    population_mask_mode: str | None = None


class GuidedDistillationMethodPayload(_StrictPayloadModel):
    """Governed payload for guided analytical-teacher distillation."""

    teacher_contract: GuidedTeacherContractPayload
    teacher_bank: GuidedTeacherBankPayload
    base_contract: DistillationBaseContractPayload
    training_schedule: GuidedTrainingSchedulePayload
    distillation_surface: GuidedDistillationSurfacePayload
    optimizer: DistillationOptimizerPayload
    model_contract: GuidedModelContractPayload
    checkpointing: DistillationCheckpointingPayload


class CsSupervisedPreStepPayload(_StrictPayloadModel):
    """Optional pre-step perturbation optimizer applied before supervised descent."""

    kind: str
    enabled: bool
    config: dict[str, Any] | None = None


class CsSupervisedCheckpointPolicyPayload(_StrictPayloadModel):
    """Checkpoint and artifact policy owned by the cs-supervised method payload."""

    checkpoint_interval_batches: int
    artifact_root: str
    tracked_spec_dir: str


class CsSupervisedMethodPayload(_StrictPayloadModel):
    """Governed payload for plain and PGD-pre-step C&S supervised training."""

    config: dict[str, Any] | None = None
    training_mode: str
    n_train_batches: int
    batch_size: int
    # Optional only so historical v1 payloads remain readable. Every current
    # builder emits this field and every architecture transform requires it.
    optimizer: OptimizerSpec | None = None
    optimizer_policy: dict[str, Any]
    gradient_clip_norm: float | None = None
    training_diagnostics: dict[str, Any]
    pre_step: CsSupervisedPreStepPayload | None = None
    checkpoint_policy: CsSupervisedCheckpointPolicyPayload

    @model_validator(mode="after")
    def _validate_optimizer_matches_runtime_config(self) -> "CsSupervisedMethodPayload":
        if self.optimizer is None:
            return self
        if self.config is None:
            raise ValueError("/optimizer requires governed /config for schedule validation")
        expected = cs_supervised_optimizer_spec(
            config=self.config,
            n_train_batches=self.n_train_batches,
        )
        if self.optimizer.model_dump(mode="json") != expected.model_dump(mode="json"):
            raise ValueError(
                "/optimizer disagrees with governed C&S runtime config; regenerate the "
                "method payload instead of launching a divergent schedule"
            )
        return self


def cs_supervised_optimizer_spec(
    *,
    config: Mapping[str, Any],
    n_train_batches: int,
) -> OptimizerSpec:
    """Build the typed optimizer that exactly mirrors the live C&S runtime."""

    def required(name: str) -> Any:
        if name not in config:
            raise ValueError(f"C&S runtime config missing optimizer field {name!r}")
        return config[name]

    runtime_batches = int(required("n_train_batches"))
    if runtime_batches != int(n_train_batches):
        raise ValueError(
            "C&S method payload n_train_batches disagrees with governed runtime config"
        )
    warmup_batches = int(required("lr_warmup_batches"))
    return OptimizerSpec(
        type="adamw",
        params={"weight_decay": 0.0},
        lr_schedule=LrScheduleSpec(
            origin={"kind": "run_start"},
            kind="warmup_cosine" if warmup_batches > 0 else "delayed_cosine",
            learning_rate_0=float(required("controller_lr")),
            total_steps=runtime_batches,
            constant_lr_iterations=warmup_batches,
            warmup_init_fraction=float(required("lr_warmup_init_fraction")),
            cosine_annealing_alpha=float(required("lr_cosine_alpha")),
        ),
    )


def require_cs_supervised_optimizer(payload: Mapping[str, Any]) -> OptimizerSpec:
    """Validate a current C&S payload and return its governed typed optimizer."""

    validated = CsSupervisedMethodPayload.model_validate(payload)
    if validated.optimizer is None:
        raise ValueError(
            "canonical C&S base lacks governed typed optimizer; regenerate it through "
            "the current C&S run-spec builder"
        )
    return validated.optimizer


def training_arg_parser(*args: Any, **kwargs: Any) -> argparse.ArgumentParser:
    """Return an argparse parser outside the scanned training-entry module set."""

    return argparse.ArgumentParser(*args, **kwargs)


def cs_supervised_method_ref() -> MethodRefSpec:
    """Return the RLRMP method ref for native C&S supervised training."""

    return MethodRefSpec(package="rlrmp", name="cs_supervised", version="v1")


def closed_loop_distillation_method_ref() -> MethodRefSpec:
    """Return the RLRMP method ref for closed-loop distillation."""

    return MethodRefSpec(package="rlrmp", name="closed_loop_distillation", version="v1")


def guided_distillation_method_ref() -> MethodRefSpec:
    """Return the RLRMP method ref for guided distillation."""

    return MethodRefSpec(package="rlrmp", name="guided_distillation", version="v1")


def register_rlrmp_cs_supervised_method() -> None:
    """Register the native RLRMP C&S supervised method with Feedbax."""

    _register_method_once(
        TrainingMethodRegistration(
            method_ref=CS_SUPERVISED_METHOD_REF,
            payload_schema_id=CS_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
            payload_schema_version=CS_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
            payload_model=CsSupervisedMethodPayload,
            contract_factory=cs_supervised_method_contract,
            update_kernels_factory=_cs_supervised_update_kernels,
            guard_predicates_factory=_cs_supervised_guard_predicates,
            rejected_payload_versions=("rlrmp.spec.training_method.cs_supervised_payload.v0",),
            owner="rlrmp.runtime.training_run_specs",
            package="rlrmp",
            requires_execution_preparation=True,
        )
    )


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
                "rlrmp.spec.training_method.closed_loop_distillation_payload.v1",
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
                "rlrmp.spec.training_method.guided_distillation_payload.v1",
            ),
            owner="rlrmp.runtime.training_run_specs",
            package="rlrmp",
        )
    )


def _validation_missing_field_path(exc: ValidationError) -> str | None:
    for error in exc.errors():
        if error.get("type") == "missing":
            return ".".join(str(part) for part in error["loc"])
    return None


def _payload_validation_error(
    exc: ValidationError,
    *,
    run_spec: dict[str, Any],
    spec_path: Path | None = None,
) -> ValidationError:
    field_path = _validation_missing_field_path(exc)
    if field_path is not None:
        raise MissingTrainingRunSpecFieldError(
            field_path=field_path,
            spec_identity=_recording_spec_identity(run_spec, spec_path=spec_path),
        ) from exc
    return exc


def _closed_loop_distillation_payload_model(
    run_spec: dict[str, Any],
    *,
    spec_path: Path | None = None,
) -> ClosedLoopDistillationMethodPayload:
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
    try:
        return ClosedLoopDistillationMethodPayload.model_validate(payload)
    except ValidationError as exc:
        raise _payload_validation_error(exc, run_spec=run_spec, spec_path=spec_path)


def _guided_distillation_payload_model(
    run_spec: dict[str, Any],
    *,
    spec_path: Path | None = None,
) -> GuidedDistillationMethodPayload:
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
    try:
        return GuidedDistillationMethodPayload.model_validate(payload)
    except ValidationError as exc:
        raise _payload_validation_error(exc, run_spec=run_spec, spec_path=spec_path)


def closed_loop_distillation_method_payload(
    run_spec: dict[str, Any],
) -> MethodPayloadEnvelope:
    """Return the governed payload envelope for a closed-loop distillation spec."""

    validated = _closed_loop_distillation_payload_model(run_spec)
    return MethodPayloadEnvelope(
        schema_id=CLOSED_LOOP_DISTILLATION_PAYLOAD_SCHEMA_ID,
        schema_version=CLOSED_LOOP_DISTILLATION_PAYLOAD_SCHEMA_VERSION,
        payload=validated.model_dump(mode="json", exclude_none=True),
    )


def guided_distillation_method_payload(run_spec: dict[str, Any]) -> MethodPayloadEnvelope:
    """Return the governed payload envelope for a guided distillation spec."""

    validated = _guided_distillation_payload_model(run_spec)
    return MethodPayloadEnvelope(
        schema_id=GUIDED_DISTILLATION_PAYLOAD_SCHEMA_ID,
        schema_version=GUIDED_DISTILLATION_PAYLOAD_SCHEMA_VERSION,
        payload=validated.model_dump(mode="json", exclude_none=True),
    )


def cs_supervised_method_payload(
    run_spec: dict[str, Any],
    *,
    output_dir: Path,
    spec_dir: Path,
) -> MethodPayloadEnvelope:
    """Return the governed payload envelope for native C&S supervised training."""

    training_summary = _mapping(run_spec, "training_summary")
    hps = _mapping(run_spec, "hps")
    pgd_config = _mapping(hps, "broad_epsilon_pgd_training")
    pre_step = None
    if "enabled" in pgd_config and bool(pgd_config["enabled"]):
        pre_step = CsSupervisedPreStepPayload(
            kind="broad_epsilon_pgd_pre_step",
            enabled=True,
            config=pgd_config,
        )
    # The generic Feedbax CLI sees only this nested TrainingRunSpec, not the
    # outer RLRMP authoring document. Keep the validated runtime configuration
    # in the governed payload so plugin preparation can build non-JSON slots.
    from rlrmp.train.executor.cs_supervised import _args_values_from_run_spec

    boundary_run_spec = {
        **run_spec,
        "artifact_output_dir": str(output_dir),
        "spec_dir": str(spec_dir),
    }
    config = _args_values_from_run_spec(boundary_run_spec)
    n_train_batches = int(_required_recording_field(run_spec, "training_summary.n_train_batches"))
    payload = CsSupervisedMethodPayload(
        config=config,
        training_mode=str(_required_recording_field(run_spec, "training_summary.training_mode")),
        n_train_batches=n_train_batches,
        batch_size=int(_required_recording_field(run_spec, "training_summary.batch_size")),
        optimizer=cs_supervised_optimizer_spec(
            config=config,
            n_train_batches=n_train_batches,
        ),
        optimizer_policy={
            "optimizer": _mapping(run_spec, "optimizer"),
            "controller_lr": training_summary.get("controller_lr"),
            "lr_schedule": training_summary.get("lr_schedule"),
        },
        gradient_clip_norm=training_summary.get("gradient_clip_norm"),
        training_diagnostics=_mapping(run_spec, "training_diagnostics"),
        pre_step=pre_step,
        checkpoint_policy=CsSupervisedCheckpointPolicyPayload(
            checkpoint_interval_batches=int(
                _required_recording_field(run_spec, "checkpointing.interval_batches")
            ),
            artifact_root=portable_repo_path(output_dir),
            tracked_spec_dir=portable_repo_path(spec_dir),
        ),
    )
    return MethodPayloadEnvelope(
        schema_id=CS_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
        schema_version=CS_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
        payload=payload.model_dump(mode="json", exclude_none=True),
    )


def cs_supervised_method_contract() -> MethodContractSpec:
    """Return the chunked native-executor contract for C&S supervised training."""

    program = PhaseProgramSpec(
        phases=[
            PhaseSpec(
                name="train_chunk",
                kind="custom",
                reads=[MODEL, OPTIMIZER, PRNG, COMPLETED_BATCHES, OBJECTIVE],
                writes=[
                    MODEL,
                    OPTIMIZER,
                    PRNG,
                    COMPLETED_BATCHES,
                    TRAIN_LOSS,
                    HISTORY_CHUNK_BYTES,
                ],
                update_steps=["cs_supervised_chunk"],
                legal_next=["done", "train_chunk"],
                checkpoint_barrier=CS_SUPERVISED_BARRIER,
                loop_axis="batch",
                metadata={"native_kernel_granularity": "checkpoint_interval_chunk"},
            ),
            PhaseSpec(
                name="done",
                kind="evaluation",
                reads=[MODEL, OPTIMIZER, PRNG, COMPLETED_BATCHES, TRAIN_LOSS],
            ),
        ],
        initial_phase="train_chunk",
        transitions=[
            PhaseTransitionSpec(
                source="train_chunk",
                target="done",
                barrier=CS_SUPERVISED_BARRIER,
                guard=MetricGuardSpec(
                    predicate_ref=CS_SUPERVISED_STOP_PREDICATE_REF,
                    metric_slots=[TRAIN_LOSS],
                    bookkeeping_slots=[COMPLETED_BATCHES],
                ),
            ),
            PhaseTransitionSpec(
                source="train_chunk",
                target="train_chunk",
                barrier=CS_SUPERVISED_BARRIER,
            ),
        ],
        update_steps=[
            UpdateStepSpec(
                name="cs_supervised_chunk",
                kind="gradient",
                kernel=UpdateKernelSpec(kernel_ref=CS_SUPERVISED_CHUNK_KERNEL_REF),
                reads=[MODEL, OPTIMIZER, PRNG, COMPLETED_BATCHES, OBJECTIVE],
                writes=[
                    MODEL,
                    OPTIMIZER,
                    PRNG,
                    COMPLETED_BATCHES,
                    TRAIN_LOSS,
                    HISTORY_CHUNK_BYTES,
                ],
                axes=["batch", "replicate"],
                optimizer_binding="optimizer_to_model",
                metadata={"direction": "minimize", "pre_step": "method_payload.pre_step"},
            )
        ],
        optimizer_bindings=[
            OptimizerTargetBinding(
                name="optimizer_to_model",
                optimizer_slot=OPTIMIZER,
                target_slot=MODEL,
                direction="minimize",
                projection="after_step",
                phase_scope=["train_chunk"],
                objective_reads=[OBJECTIVE],
            )
        ],
        checkpoint_barriers=[
            CheckpointBarrierSpec(
                name=CS_SUPERVISED_BARRIER,
                phase="train_chunk",
                slots=[
                    slot
                    for slot in checkpoint_slot_specs(CS_SUPERVISED_SCHEMA)
                    if slot.slot != TRAIN_LOSS
                ],
                artifact_sinks=artifact_sink_specs(CS_SUPERVISED_SCHEMA),
            )
        ],
        batch_progress=TrainingBatchProgressSpec(slot=COMPLETED_BATCHES),
        metadata={
            "phase_program_identity": "rlrmp.cs_supervised.chunked_supervised.v1",
            "checkpoint_barrier_policy": "after_each_training_chunk",
        },
    )
    return MethodContractSpec(
        method_ref=CS_SUPERVISED_METHOD_REF,
        method_payload_schema_version=CS_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
        axes=[
            AxisSpec(name="batch", role="batch"),
            AxisSpec(name="replicate", role="replicate"),
        ],
        state_slots=supervised_state_slots(CS_SUPERVISED_SCHEMA),
        phase_program=program,
        objective_reducers=[
            {"axis": "batch", "owner": "objective", "path": "/objective/payload/loss_summary"}
        ],
        worker_reducers=[
            {"axis": "replicate", "owner": "worker", "path": "/risk_aggregation/replicate"}
        ],
        metadata={
            "runtime_context": "rlrmp_runtime.components.cs_supervised",
            "supported_modes": ["plain_supervised", "broad_epsilon_pgd_pre_step"],
        },
    )


def cs_supervised_effective_phase_spec(
    contract: MethodContractSpec | None = None,
) -> EffectivePhaseSpec:
    """Return the effective phase spec corresponding to the cs-supervised contract."""

    active_contract = contract or cs_supervised_method_contract()
    return EffectivePhaseSpec(
        method_ref=active_contract.method_ref,
        axes=active_contract.axes,
        state_slots=active_contract.state_slots,
        phase_program=active_contract.phase_program,
        consistency_predicate=derive_consistency_predicate(active_contract.phase_program),
    )


def closed_loop_distillation_method_contract() -> MethodContractSpec:
    """Return worker axes, slots, and phase program for closed-loop distillation."""

    return _distillation_contract(
        method_ref=CLOSED_LOOP_DISTILLATION_METHOD_REF,
        payload_version=CLOSED_LOOP_DISTILLATION_PAYLOAD_SCHEMA_VERSION,
        phase_names=("closed_loop_rollout_distillation",),
        update_step_name="closed_loop_distillation_gradient_update",
        kernel_ref="rlrmp.train.distillation_native.closed_loop_gradient_update",
        extra_axes=(AxisSpec(name="rollout", role="rollout"),),
        extra_slots=(
            StateSlotSpec(name="teacher_reference", role="auxiliary"),
            StateSlotSpec(name="closed_loop_rollout", role="auxiliary", required=False),
        ),
        metadata={
            "teacher_location": "method_payload.teacher_contract",
            "teacher_is_worker_axis": False,
            "native_executor": "rlrmp.train.distillation_native",
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
        kernel_ref="rlrmp.train.distillation_native.guided_gradient_update",
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
            "native_executor": "rlrmp.train.distillation_native",
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
    training_config = _distillation_training_config(run_spec, method=method, spec_path=spec_path)
    effective_phase = EffectivePhaseSpec(
        method_ref=contract.method_ref,
        axes=contract.axes,
        state_slots=contract.state_slots,
        phase_program=contract.phase_program,
        consistency_predicate=derive_consistency_predicate(contract.phase_program),
    )
    scaffold = build_training_run_spec_scaffold(
        risk_metadata={"source": "distillation_adapter"},
        execution_mode=(
            "dry_run" if run_spec.get("launch_status") in {"not_launched"} else "local"
        ),
        require_review=True,
        allow_cloud=False,
        execution_metadata={"full_train_authorized": False},
        artifact_root=str(output_dir),
        artifact_metadata={"tracked_run_spec": str(spec_path)},
        checkpoint_interval=training_config.snapshot_interval,
        progress_interval=training_config.snapshot_interval,
        checkpoint_metadata={"latest_pointer": checkpointing.get("latest_pointer")},
        metadata={"composed_with": RLRMP_RUN_SPEC_PAYLOAD_KEY},
    )
    return scaffold.build(
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
        method_ref=method_ref,
        method_payload=method_payload,
        method_extensions={
            "metadata": {
                "runner": _distillation_runner(run_spec, method=method),
                "rlrmp_extension_payload": RLRMP_RUN_SPEC_PAYLOAD_KEY,
            }
        },
        method_contract=contract,
        effective_phase=effective_phase,
        worker_metadata={
            "native_executor": "rlrmp.train.distillation_native",
            "teacher_configuration_location": "method_payload",
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


def write_distillation_run_spec(
    path: Path, run_spec: dict[str, Any], *, method: str
) -> dict[str, Any]:
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


def _method_ref_key_from_payload(method_ref: Any) -> str | None:
    if isinstance(method_ref, MethodRefSpec):
        return method_ref.key
    if isinstance(method_ref, Mapping):
        package = method_ref.get("package")
        name = method_ref.get("name")
        version = method_ref.get("version")
        if package is not None and name is not None and version is not None:
            return f"{package}/{name}/{version}"
    if isinstance(method_ref, str):
        return method_ref
    return None


def _mapping_path_value(mapping: Mapping[str, Any], path: str) -> Any:
    """Return a dotted mapping path, or a private sentinel when it is absent."""

    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING
        value = value[part]
    return value


def hydrate_compact_run_spec_envelope(run_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Hydrate a compact tracked recipe from its authoritative RLRMP extension.

    Compact recipes retain a small identity and pointer surface at the root while
    storing the complete executable RLRMP payload in ``rlrmp_run_spec``. The
    composed Feedbax payload remains a separate immutable root record because it
    is deliberately excluded from the RLRMP extension.
    """

    outer = dict(run_spec)
    if COMPACT_RUN_SPEC_MARKER_KEY not in outer:
        return outer
    if outer[COMPACT_RUN_SPEC_MARKER_KEY] is not True:
        raise ValueError("compact C&S GRU run spec compact_run_spec must be the boolean true")

    extension = outer.get(RLRMP_RUN_SPEC_PAYLOAD_KEY)
    if not isinstance(extension, Mapping):
        raise ValueError(
            "compact C&S GRU run spec rlrmp_run_spec must be an authoritative JSON object"
        )
    accepted = accept_rlrmp_spec_payload(
        RUN_SPEC_KIND,
        extension,
        source_version=extension.get("schema_version"),
        path=RLRMP_RUN_SPEC_PAYLOAD_KEY,
    )
    canonical = accepted.payload
    if not isinstance(canonical, Mapping):
        raise ValueError("compact C&S GRU run spec rlrmp_run_spec must hydrate to a JSON object")

    for path in _COMPACT_RUN_SPEC_DUPLICATE_PATHS:
        root_value = _mapping_path_value(outer, path)
        extension_value = _mapping_path_value(canonical, path)
        if root_value is _MISSING or extension_value is _MISSING:
            raise ValueError(
                "compact C&S GRU run spec must duplicate "
                f"{path!r} at both the root and rlrmp_run_spec"
            )
        if root_value != extension_value:
            raise ValueError(
                f"compact C&S GRU run spec root value disagrees with rlrmp_run_spec at {path!r}"
            )

    feedbax_spec = outer.get(FEEDBAX_TRAINING_RUN_SPEC_KEY)
    if not isinstance(feedbax_spec, Mapping):
        raise ValueError(
            "compact C&S GRU run spec feedbax_training_run_spec must be an immutable JSON object"
        )

    hydrated = dict(canonical)
    hydrated[RLRMP_RUN_SPEC_PAYLOAD_KEY] = dict(extension)
    hydrated[FEEDBAX_TRAINING_RUN_SPEC_KEY] = dict(feedbax_spec)
    return hydrated


def _ensure_known_rlrmp_method_registered_for_payload(spec_payload: Mapping[str, Any]) -> None:
    """Register rlrmp-owned native methods before Feedbax validates method refs."""

    method_ref = _method_ref_key_from_payload(spec_payload.get("method_ref"))
    if method_ref == CS_SUPERVISED_METHOD_REF:
        register_rlrmp_cs_supervised_method()
    elif method_ref == ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF:
        from rlrmp.train.adaptive_epsilon_native import (
            ensure_adaptive_epsilon_training_method_registered,
        )

        ensure_adaptive_epsilon_training_method_registered()
    elif method_ref == POLICY_ADVERSARY_SUPERVISED_METHOD_REF:
        from rlrmp.train.policy_adversary_native import (
            ensure_policy_adversary_training_method_registered,
        )

        ensure_policy_adversary_training_method_registered()


def _semantic_method_metadata_keys(spec_payload: Mapping[str, Any]) -> set[str]:
    extensions = spec_payload.get("method_extensions")
    if not isinstance(extensions, Mapping):
        return set()
    metadata = extensions.get("metadata")
    if not isinstance(metadata, Mapping):
        return set()
    return set(metadata) & SEMANTIC_METHOD_EXTENSION_METADATA_KEYS


def _method_extension_provenance_metadata(spec_payload: Mapping[str, Any]) -> dict[str, Any]:
    extensions = spec_payload.get("method_extensions")
    if not isinstance(extensions, Mapping):
        return {"rlrmp_extension_payload": RLRMP_RUN_SPEC_PAYLOAD_KEY}
    metadata = extensions.get("metadata")
    if not isinstance(metadata, Mapping):
        return {"rlrmp_extension_payload": RLRMP_RUN_SPEC_PAYLOAD_KEY}
    provenance = {
        key: value
        for key, value in metadata.items()
        if key not in SEMANTIC_METHOD_EXTENSION_METADATA_KEYS
    }
    provenance.setdefault("rlrmp_extension_payload", RLRMP_RUN_SPEC_PAYLOAD_KEY)
    return provenance


def _spec_payload_artifact_root(
    spec_payload: Mapping[str, Any], run_spec: Mapping[str, Any]
) -> Path:
    artifacts = spec_payload.get("artifacts")
    if isinstance(artifacts, Mapping) and artifacts.get("artifact_root") is not None:
        return Path(str(artifacts["artifact_root"]))
    return Path(str(run_spec.get("artifact_output_dir", "")))


def _spec_payload_spec_dir(spec_payload: Mapping[str, Any]) -> Path:
    artifacts = spec_payload.get("artifacts")
    metadata: Any = None
    if isinstance(artifacts, Mapping):
        metadata = artifacts.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("tracked_spec_dir") is not None:
        return Path(str(metadata["tracked_spec_dir"]))
    return Path("")


def _migrate_legacy_standard_supervised_training_run_spec(
    run_spec: dict[str, Any],
    spec_payload: Mapping[str, Any],
) -> dict[str, Any]:
    register_rlrmp_cs_supervised_method()
    payload = dict(spec_payload)
    payload["schema_id"] = FEEDBAX_TRAINING_RUN_SPEC_SCHEMA_ID
    payload["schema_version"] = FEEDBAX_TRAINING_RUN_SPEC_SCHEMA_VERSION
    method_payload = cs_supervised_method_payload(
        run_spec,
        output_dir=_spec_payload_artifact_root(payload, run_spec),
        spec_dir=_spec_payload_spec_dir(payload),
    )
    method_contract = cs_supervised_method_contract()
    effective_phase = cs_supervised_effective_phase_spec(method_contract)
    graph = payload.get("graph")
    graph_payload = graph.get("inline") if isinstance(graph, Mapping) else {}
    if not isinstance(graph_payload, Mapping):
        graph_payload = {}
    fingerprint = _effective_phase_fingerprint(
        graph_payload=graph_payload,
        effective_phase=effective_phase,
        method_payload=method_payload.model_dump(mode="json", exclude_none=True),
    )
    payload["method_ref"] = cs_supervised_method_ref().model_dump(
        mode="json",
        exclude_none=True,
    )
    payload["method_payload"] = method_payload.model_dump(mode="json", exclude_none=True)
    payload["method_extensions"] = {"metadata": _method_extension_provenance_metadata(spec_payload)}
    payload["worker_execution"] = WorkerExecutionSpec(
        method_contract=method_contract,
        effective_phase=effective_phase,
        metadata={
            "native_executor": "feedbax.training.executor.execute_training_run_spec",
            "effective_phase_fingerprint": fingerprint,
            "migrated_from_method_ref": LEGACY_FEEDBAX_STANDARD_SUPERVISED_METHOD_REF,
        },
    ).model_dump(mode="json", exclude_none=True)
    metadata = dict(payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {})
    metadata["effective_phase_fingerprint"] = fingerprint
    metadata["feedbax_training_run_spec_migration"] = {
        "source_method_ref": LEGACY_FEEDBAX_STANDARD_SUPERVISED_METHOD_REF,
        "target_method_ref": CS_SUPERVISED_METHOD_REF,
        "semantic_metadata_keys_removed": sorted(_semantic_method_metadata_keys(spec_payload)),
    }
    payload["metadata"] = metadata
    return payload


def _migrate_feedbax_training_run_spec_payload(
    run_spec: dict[str, Any],
    spec_payload: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _migrate_feedbax_training_run_spec_graph_payload(spec_payload)
    method_ref = _method_ref_key_from_payload(spec_payload.get("method_ref"))
    semantic_keys = _semantic_method_metadata_keys(spec_payload)
    if method_ref == LEGACY_FEEDBAX_STANDARD_SUPERVISED_METHOD_REF and semantic_keys:
        return _migrate_legacy_standard_supervised_training_run_spec(run_spec, payload)
    if method_ref == LEGACY_FEEDBAX_STANDARD_SUPERVISED_METHOD_REF:
        raise FeedbaxTrainingRunSpecMigrationError(
            "Embedded feedbax_training_run_spec uses legacy "
            f"method_ref={LEGACY_FEEDBAX_STANDARD_SUPERVISED_METHOD_REF!r} without "
            "the semantic method_extensions.metadata keys required for safe migration. "
            "See Mandible issue dfa0cd5; restore the original metadata or regenerate "
            "the run spec through the current native method builder."
        )
    if semantic_keys:
        raise FeedbaxTrainingRunSpecMigrationError(
            "Embedded feedbax_training_run_spec carries semantic method identity in "
            f"method_extensions.metadata for unsupported method_ref={method_ref!r}: "
            f"keys={sorted(semantic_keys)}. Add an explicit migration or regenerate "
            "the run spec through the current native method builder."
        )
    return payload


def _migrate_feedbax_training_run_spec_graph_payload(
    spec_payload: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(spec_payload)
    graph = payload.get("graph")
    if not isinstance(graph, Mapping):
        return payload
    inline = graph.get("inline")
    if not isinstance(inline, Mapping):
        return payload

    migrated_inline = migrate_feedbax_graph_payload(inline)
    migrated_graph = dict(graph)
    migrated_graph["inline"] = migrated_inline
    migrated_graph["schema_id"] = migrated_inline.get("schema_id")
    migrated_graph["schema_version"] = migrated_inline.get("schema_version")
    payload["graph"] = migrated_graph
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
        StateSlotSpec(name="completed_batches", role="auxiliary"),
        StateSlotSpec(name="objective", role="objective"),
        *extra_slots,
        StateSlotSpec(name="train_loss", role="metric", axis="replicate", required=False),
    ]
    phases: list[PhaseSpec] = []
    for phase_name in phase_names:
        phases.append(
            PhaseSpec(
                name=phase_name,
                kind="outer_loop",
                reads=[
                    "model",
                    "optimizer",
                    "prng",
                    "completed_batches",
                    "objective",
                    *[slot.name for slot in extra_slots],
                ],
                writes=["model", "optimizer", "prng", "completed_batches", "train_loss"],
                update_steps=[update_step_name],
                legal_next=[],
                checkpoint_barrier=f"after_{phase_name}",
            )
        )
    phases.append(
        PhaseSpec(
            name="done",
            kind="evaluation",
            reads=["model", "optimizer", "prng", "completed_batches", "train_loss"],
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
                    "completed_batches",
                    "objective",
                    *[slot.name for slot in extra_slots],
                ],
                writes=["model", "optimizer", "prng", "completed_batches", "train_loss"],
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
                    CheckpointSlotSpec(slot="prng", axis="replicate"),
                    CheckpointSlotSpec(slot="completed_batches"),
                ],
                resume_coordinate=ResumeCoordinateSpec(
                    phase="done",
                    completed_barrier=f"after_{phase_name}",
                ),
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
    payload: BaseModel | None = None,
) -> dict[str, Any]:
    from rlrmp.train.distillation_native import distillation_update_kernels

    return dict(distillation_update_kernels("closed_loop_distillation", payload))


def _guided_distillation_update_kernels(payload: BaseModel | None = None) -> dict[str, Any]:
    from rlrmp.train.distillation_native import distillation_update_kernels

    return dict(distillation_update_kernels("guided_distillation", payload))


def _cs_supervised_update_kernels(payload: BaseModel | None = None) -> dict[str, Any]:
    from rlrmp.train.cs_nominal_gru import cs_supervised_update_kernels

    return cs_supervised_update_kernels(payload)


def _cs_supervised_guard_predicates(payload: BaseModel | None = None) -> dict[str, Any]:
    return {CS_SUPERVISED_STOP_PREDICATE_REF: make_stop_predicate(payload)}


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


def _distillation_method_for_run_spec(run_spec: dict[str, Any]) -> str:
    keys = set(run_spec)
    if {"student_contract", "closed_loop_semantics", "loss_surface"} <= keys:
        return "closed_loop_distillation"
    if {"model_contract", "teacher_bank", "training_schedule", "distillation_surface"} <= keys:
        return "guided_distillation"
    raise ValueError("run spec does not carry a recognized RLRMP distillation payload")


def _recording_spec_identity(
    run_spec: dict[str, Any],
    *,
    spec_path: Path | None = None,
    spec_dir: Path | None = None,
) -> str:
    """Return a concise source identity for fail-closed recording errors."""

    if spec_path is not None:
        return str(spec_path)
    entry_path = _mapping(run_spec, "training_entry").get("run_spec_path")
    if entry_path:
        return str(entry_path)
    issue = run_spec.get("issue")
    run_id = run_spec.get("run_id")
    if issue is not None and run_id is not None:
        return f"issue={issue} run_id={run_id}"
    if spec_dir is not None:
        return str(spec_dir)
    return "<unknown run spec>"


def _value_at_path(mapping: dict[str, Any], field_path: str) -> tuple[Any, bool]:
    value: Any = mapping
    for part in field_path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None, False
        value = value[part]
    return value, True


def _required_recording_field(
    run_spec: dict[str, Any],
    field_path: str,
    *,
    spec_path: Path | None = None,
    spec_dir: Path | None = None,
) -> Any:
    value, found = _value_at_path(run_spec, field_path)
    if found:
        return value
    raise MissingTrainingRunSpecFieldError(
        field_path=field_path,
        spec_identity=_recording_spec_identity(
            run_spec,
            spec_path=spec_path,
            spec_dir=spec_dir,
        ),
    )


def _required_recording_field_from(
    run_spec: dict[str, Any],
    field_paths: tuple[str, ...],
    *,
    spec_path: Path | None = None,
    spec_dir: Path | None = None,
) -> Any:
    for field_path in field_paths:
        value, found = _value_at_path(run_spec, field_path)
        if found:
            return value
    raise MissingTrainingRunSpecFieldError(
        field_path=" or ".join(field_paths),
        spec_identity=_recording_spec_identity(
            run_spec,
            spec_path=spec_path,
            spec_dir=spec_dir,
        ),
    )


def _required_float_or_none_recording_field(
    run_spec: dict[str, Any],
    field_path: str,
    *,
    spec_path: Path | None = None,
    spec_dir: Path | None = None,
) -> float | None:
    value = _required_recording_field(
        run_spec,
        field_path,
        spec_path=spec_path,
        spec_dir=spec_dir,
    )
    return None if value is None else float(value)


def _required_typed_recording_field(
    run_spec: dict[str, Any],
    field_path: str,
    value: Any,
    *,
    spec_path: Path | None = None,
    missing_field_path: str | None = None,
) -> Any:
    _, found = _value_at_path(run_spec, field_path)
    if found:
        return value
    raise MissingTrainingRunSpecFieldError(
        field_path=missing_field_path or field_path,
        spec_identity=_recording_spec_identity(run_spec, spec_path=spec_path),
    )


def _distillation_training_config(
    run_spec: dict[str, Any],
    *,
    method: str,
    spec_path: Path | None = None,
) -> TrainingConfig:
    if method == "closed_loop_distillation":
        payload = _closed_loop_distillation_payload_model(run_spec, spec_path=spec_path)
        return TrainingConfig(
            n_batches=int(payload.student_contract.n_train_batches),
            batch_size=int(payload.student_contract.batch_size),
            learning_rate=float(payload.student_contract.controller_lr),
            grad_clip=(
                None
                if payload.student_contract.gradient_clip_norm is None
                else float(payload.student_contract.gradient_clip_norm)
            ),
            hidden_dim=int(payload.student_contract.hidden_size),
            network_type="gru",
            n_reach_steps=int(
                _required_typed_recording_field(
                    run_spec,
                    "teacher_contract.horizon",
                    payload.teacher_contract.horizon,
                    spec_path=spec_path,
                )
            ),
            effort_weight=float(payload.loss_surface.weights.action_force_trajectory),
            snapshot_interval=int(payload.checkpointing.interval_batches),
        )
    payload = _guided_distillation_payload_model(run_spec, spec_path=spec_path)
    if "n_train_batches" in run_spec:
        n_batches = run_spec["n_train_batches"]
    else:
        n_batches = _required_typed_recording_field(
            run_spec,
            "training_schedule.total_batches",
            payload.training_schedule.total_batches,
            spec_path=spec_path,
            missing_field_path="n_train_batches or training_schedule.total_batches",
        )
    if "batch_size" in run_spec:
        batch_size = run_spec["batch_size"]
    else:
        batch_size = _required_typed_recording_field(
            run_spec,
            "model_contract.batch_size",
            payload.model_contract.batch_size,
            spec_path=spec_path,
            missing_field_path="batch_size or model_contract.batch_size",
        )
    if "controller_lr" in run_spec:
        learning_rate = run_spec["controller_lr"]
    else:
        learning_rate = _required_typed_recording_field(
            run_spec,
            "optimizer.controller_lr",
            payload.optimizer.controller_lr,
            spec_path=spec_path,
            missing_field_path="controller_lr or optimizer.controller_lr",
        )
    return TrainingConfig(
        n_batches=int(n_batches),
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        grad_clip=(
            None
            if payload.optimizer.gradient_clip_norm is None
            else float(payload.optimizer.gradient_clip_norm)
        ),
        hidden_dim=int(payload.model_contract.hidden_size),
        network_type="gru",
        n_reach_steps=int(payload.teacher_bank.horizon),
        effort_weight=float(payload.distillation_surface.components.clean_action.weight),
        snapshot_interval=int(payload.checkpointing.interval_batches),
    )


def _distillation_task_params(run_spec: dict[str, Any], *, method: str) -> dict[str, Any]:
    if method == "closed_loop_distillation":
        payload = _closed_loop_distillation_payload_model(run_spec)
        return {
            "student_contract": payload.student_contract.model_dump(
                mode="json",
                exclude_none=True,
            ),
            "closed_loop": True,
            "teacher": payload.teacher_contract.controller,
        }
    payload = _guided_distillation_payload_model(run_spec)
    return {
        "model_contract": payload.model_contract.model_dump(mode="json", exclude_none=True),
        "teacher_bank": payload.teacher_bank.model_dump(mode="json", exclude_none=True),
        "training_schedule": payload.training_schedule.model_dump(
            mode="json",
            exclude_none=True,
        ),
    }


def _distillation_objective_payload(run_spec: dict[str, Any], *, method: str) -> dict[str, Any]:
    if method == "closed_loop_distillation":
        payload = _closed_loop_distillation_payload_model(run_spec)
        return {
            "loss_surface": payload.loss_surface.model_dump(mode="json", exclude_none=True),
            "closed_loop_semantics": payload.closed_loop_semantics.model_dump(
                mode="json",
                exclude_none=True,
            ),
        }
    payload = _guided_distillation_payload_model(run_spec)
    return {
        "distillation_surface": payload.distillation_surface.model_dump(
            mode="json",
            exclude_none=True,
        ),
        "training_schedule": payload.training_schedule.model_dump(
            mode="json",
            exclude_none=True,
        ),
    }


def _distillation_graph_ref(run_spec: dict[str, Any]) -> str:
    method = _distillation_method_for_run_spec(run_spec)
    if method == "closed_loop_distillation":
        base = _closed_loop_distillation_payload_model(run_spec).base_contract.run_spec
    else:
        base = _guided_distillation_payload_model(run_spec).base_contract.run_spec
    return str(base)


def _distillation_setup_function(run_spec: dict[str, Any], *, method: str) -> str:
    if method == "closed_loop_distillation":
        return _closed_loop_distillation_payload_model(run_spec).student_contract.setup_function
    return _guided_distillation_payload_model(run_spec).model_contract.setup_function


def _distillation_runner(run_spec: dict[str, Any], *, method: str) -> str:
    entry = _mapping(run_spec, "training_entry")
    field = "module" if method == "closed_loop_distillation" else "trainer"
    try:
        runner = entry[field]
    except KeyError as exc:
        raise ValueError(f"{method} training_entry requires {field!r}") from exc
    return str(runner)


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


def _cs_training_config(
    run_spec: dict[str, Any],
    *,
    spec_dir: Path | None = None,
) -> TrainingConfig:
    return TrainingConfig(
        n_batches=int(
            _required_recording_field_from(
                run_spec,
                ("n_train_batches", "training_summary.n_train_batches"),
                spec_dir=spec_dir,
            )
        ),
        batch_size=int(
            _required_recording_field_from(
                run_spec,
                ("batch_size", "training_summary.batch_size"),
                spec_dir=spec_dir,
            )
        ),
        learning_rate=float(
            _required_recording_field_from(
                run_spec,
                ("controller_lr", "optimizer.learning_rate_0"),
                spec_dir=spec_dir,
            )
        ),
        grad_clip=_required_float_or_none_recording_field(
            run_spec,
            "optimizer.gradient_clip_norm",
            spec_dir=spec_dir,
        ),
        hidden_dim=int(
            _required_recording_field(
                run_spec,
                "model_summary.hidden_size",
                spec_dir=spec_dir,
            )
        ),
        network_type="gru",
        n_reach_steps=int(
            _required_recording_field(
                run_spec,
                "task_timing.n_steps",
                spec_dir=spec_dir,
            )
        ),
        effort_weight=float(
            # C&S records effort either as the legacy active control term or as the
            # canonical output-command loss weight for full-Q/R rows; both are source data.
            _required_recording_field_from(
                run_spec,
                (
                    "loss_summary.active_cs_terms.control.scale",
                    "loss_summary.weights.nn_output",
                ),
                spec_dir=spec_dir,
            )
        ),
        snapshot_interval=int(
            _required_recording_field(
                run_spec,
                f"checkpointing.{CHECKPOINT_INTERVAL_BATCHES_KEY}",
                spec_dir=spec_dir,
            )
        ),
    )


def rlrmp_extension_payload(run_spec: dict[str, Any]) -> dict[str, Any]:
    """Return the full RLRMP-owned v2 run payload embedded in manifests."""

    payload = {
        key: value
        for key, value in run_spec.items()
        if key not in {RLRMP_RUN_SPEC_PAYLOAD_KEY, FEEDBAX_TRAINING_RUN_SPEC_KEY}
    }
    source_schema_version = payload.get("schema_version")
    payload["schema_id"] = RUN_SPEC_SCHEMA_ID
    payload["schema_version"] = RUN_SPEC_SCHEMA_VERSION
    if source_schema_version is not None and source_schema_version != RUN_SPEC_SCHEMA_VERSION:
        payload.setdefault("source_schema_version", source_schema_version)
    payload.setdefault(CONSUMED_DATA_IDENTITIES_KEY, [])
    model_summary = _mapping(run_spec, "model_summary")
    payload.setdefault(
        DESCRIPTOR_PAYLOAD_KEY,
        controller_feedback_descriptor_from_container(
            model_summary,
            feedback_dim=_feedback_dim_from_run_spec(run_spec),
            source="rlrmp_extension_payload",
        ),
    )
    return stamp_current_schema(RUN_SPEC_KIND, payload)


def add_consumed_data_identity(
    run_spec: dict[str, Any],
    *,
    role: str,
    schema: str,
    hash: str,
) -> dict[str, Any]:
    """Return ``run_spec`` with one consumed data-product identity recorded."""

    if not role.strip():
        raise ValueError("consumed data identity role must not be empty")
    if not schema.strip():
        raise ValueError("consumed data identity schema must not be empty")
    if not hash.strip():
        raise ValueError("consumed data identity hash must not be empty")
    payload = dict(run_spec)
    existing = payload.get(CONSUMED_DATA_IDENTITIES_KEY, [])
    if not isinstance(existing, list):
        raise TypeError(f"{CONSUMED_DATA_IDENTITIES_KEY} must be a list")
    entry = {"role": role, "schema": schema, "hash": hash}
    payload[CONSUMED_DATA_IDENTITIES_KEY] = (
        [*existing, entry] if entry not in existing else existing
    )
    return payload


def build_feedbax_training_run_spec(
    run_spec: dict[str, Any],
    *,
    graph_spec: Any,
    output_dir: Path,
    spec_dir: Path,
) -> TrainingRunSpec:
    """Build the composed Feedbax ``TrainingRunSpec`` for one C&S GRU run."""

    register_rlrmp_cs_supervised_method()
    checkpointing = _mapping(run_spec, "checkpointing")
    training_diagnostics = _mapping(run_spec, "training_diagnostics")
    training_config = _cs_training_config(run_spec, spec_dir=spec_dir)
    graph_payload = graph_spec_payload(graph_spec)
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
    method_payload = cs_supervised_method_payload(
        run_spec,
        output_dir=output_dir,
        spec_dir=spec_dir,
    )
    method_contract = cs_supervised_method_contract()
    effective_phase = cs_supervised_effective_phase_spec(method_contract)
    effective_phase_fingerprint = _effective_phase_fingerprint(
        graph_payload=graph_payload,
        effective_phase=effective_phase,
        method_payload=method_payload.model_dump(mode="json", exclude_none=True),
    )
    method_metadata = {
        "runner": "rlrmp.train.cs_nominal_gru",
        "rlrmp_extension_payload": RLRMP_RUN_SPEC_PAYLOAD_KEY,
    }
    task = TaskSpec(
        type=str(_mapping(run_spec, "task_timing").get("type", "rlrmp_task")),
        params=_mapping(run_spec, "task_timing"),
    )
    objective = ObjectiveSlotSpec(
        kind="external",
        payload=objective_payload,
        schema_id="rlrmp.cs_gru_objective",
        schema_version="rlrmp.cs_gru_objective.v1",
        metadata={"rlrmp_loss_objective": run_spec.get("loss_objective")},
    )
    metadata = {
        "composed_with": RLRMP_RUN_SPEC_PAYLOAD_KEY,
        DESCRIPTOR_PAYLOAD_KEY: feedback_descriptors,
        "descriptor_basis_hash": feedback_descriptors["descriptor_basis_hash"],
    }
    scaffold = build_training_run_spec_scaffold(
        risk_metadata={"source": "$.training_summary"},
        execution_mode="local",
        require_review=bool(run_spec.get("full_training_launch") != "requested"),
        allow_cloud=False,
        execution_metadata={"launch_mode": run_spec.get("mode")},
        artifact_root=portable_repo_path(output_dir),
        artifact_metadata={"tracked_spec_dir": portable_repo_path(spec_dir)},
        checkpoint_interval=training_config.snapshot_interval,
        # Training diagnostics are optional progress metadata, not run topology.
        progress_interval=(
            None
            if training_diagnostics.get("enabled") is False
            else training_config.snapshot_interval
        ),
        checkpoint_metadata={"checkpoint_dir": checkpointing.get("checkpoint_dir")},
        metadata=metadata,
    )
    if _adaptive_epsilon_run_spec_enabled(run_spec):
        from rlrmp.train.adaptive_epsilon_native import (
            build_adaptive_epsilon_training_run_spec,
        )

        return build_adaptive_epsilon_training_run_spec(
            run_spec,
            graph_spec=graph_spec,
            output_dir=output_dir,
            spec_dir=spec_dir,
            training_config=training_config,
            objective=objective,
            task=task,
            risk_aggregation=scaffold.risk_aggregation,
            method_extensions={"metadata": method_metadata},
            execution=scaffold.execution,
            artifacts=scaffold.artifacts,
            checkpoint_progress=scaffold.checkpoint_progress,
            metadata=scaffold.metadata,
        )
    if _policy_adversary_run_spec_enabled(run_spec):
        from rlrmp.train.policy_adversary_native import (
            build_policy_adversary_training_run_spec,
        )

        return build_policy_adversary_training_run_spec(
            run_spec,
            graph_spec=graph_spec,
            output_dir=output_dir,
            spec_dir=spec_dir,
            training_config=training_config,
            objective=objective,
            task=task,
            risk_aggregation=scaffold.risk_aggregation,
            method_extensions={"metadata": method_metadata},
            execution=scaffold.execution,
            artifacts=scaffold.artifacts,
            checkpoint_progress=scaffold.checkpoint_progress,
            metadata=scaffold.metadata,
        )
    return scaffold.build(
        graph=GraphTopologySourceSpec(
            inline=graph_payload,
            schema_id=getattr(graph_spec, "schema_id", None),
            schema_version=getattr(graph_spec, "schema_version", None),
            metadata={
                "source": "materialized_runtime_graph",
                "sidecar_policy": run_spec.get("feedbax_graph", {}),
                DESCRIPTOR_PAYLOAD_KEY: feedback_descriptors,
                "descriptor_basis_hash": feedback_descriptors["descriptor_basis_hash"],
            },
        ),
        task=task,
        training_config=training_config,
        objective=objective,
        method_ref=cs_supervised_method_ref(),
        method_payload=method_payload,
        method_extensions={"metadata": method_metadata},
        method_contract=method_contract,
        effective_phase=effective_phase,
        worker_metadata={
            "native_executor": "feedbax.training.executor.execute_training_run_spec",
            "effective_phase_fingerprint": effective_phase_fingerprint,
        },
        metadata={
            "effective_phase_fingerprint": effective_phase_fingerprint,
            "resume_context": {
                "schedule_origin_step": 0,
                "current_step": 0,
                "optimizer_count_at_current_step": 0,
            },
            "optimizer_build_context": {
                "schedule_origin_step": 0,
                "current_step": 0,
                "optimizer_count_at_current_step": 0,
            },
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

    hydrated = hydrate_compact_run_spec_envelope(run_spec)
    raw_spec = hydrated[FEEDBAX_TRAINING_RUN_SPEC_KEY]
    if not isinstance(raw_spec, Mapping):
        raise TypeError(f"{FEEDBAX_TRAINING_RUN_SPEC_KEY} must be a JSON object")
    spec_payload = _migrate_feedbax_training_run_spec_payload(hydrated, raw_spec)
    _ensure_known_rlrmp_method_registered_for_payload(spec_payload)
    return TrainingRunSpec.model_validate(spec_payload)


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

    ensure_rlrmp_spec_families()
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

    extension = dict(run_spec[RLRMP_RUN_SPEC_PAYLOAD_KEY])
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


def attach_post_run_provenance(
    run_spec: dict[str, Any],
    *,
    run_spec_path: Path,
    artifact_dir: Path,
    manifest_root: Path,
    graph_manifest_path: Path | None = None,
    graph_spec_path: Path | None = None,
) -> dict[str, Any]:
    """Attach the production-time post-run provenance stamp to a run payload."""

    payload = dict(run_spec)
    payload["post_run_provenance"] = {
        "schema_version": POST_RUN_SCHEMA_VERSION,
        "tool": "rlrmp.runtime.training_run_specs",
        "rlrmp": _git_record(_repo_root()),
        "feedbax": _git_record(_feedbax_repo()),
        "schemas": {
            "post_run_provenance": POST_RUN_SCHEMA_VERSION,
            "feedbax_manifest": SCHEMA_VERSION,
            "feedbax_provider": FEEDBAX_PROVIDER_VERSION,
        },
        "feedbax_manifest_root": {
            "path": PINNED_MANIFEST_ROOT,
            "absolute_path_sha256": hashlib.sha256(
                str(manifest_root.resolve()).encode()
            ).hexdigest(),
            "env": "FEEDBAX_RUNS_DIR",
        },
        "feedbax_graph": _graph_metadata(
            run_spec=payload,
            run_spec_path=run_spec_path,
            artifact_dir=artifact_dir,
            graph_manifest_path=graph_manifest_path,
            graph_spec_path=graph_spec_path,
        ),
    }
    return payload


def _mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    return value if isinstance(value, dict) else {}


def _effective_phase_fingerprint(
    *,
    graph_payload: Mapping[str, Any],
    effective_phase: EffectivePhaseSpec,
    method_payload: Mapping[str, Any],
) -> str:
    parity = {
        "graph": graph_payload,
        "effective_phase": effective_phase.model_dump(mode="json", exclude_none=True),
        "method_payload": method_payload,
    }
    encoded = json.dumps(parity, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _policy_adversary_run_spec_enabled(run_spec: Mapping[str, Any]) -> bool:
    hps = run_spec.get("hps")
    if not isinstance(hps, Mapping):
        return False
    policy_adversary = hps.get("policy_adversary_training")
    return isinstance(policy_adversary, Mapping) and policy_adversary.get("enabled") is True


def _adaptive_epsilon_run_spec_enabled(run_spec: Mapping[str, Any]) -> bool:
    hps = run_spec.get("hps")
    if not isinstance(hps, Mapping):
        return False
    adaptive = hps.get("adaptive_epsilon_curriculum")
    return isinstance(adaptive, Mapping) and adaptive.get("enabled") is True


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


def _repo_root() -> Path:
    from rlrmp.paths import REPO_ROOT

    return REPO_ROOT


def _git_value(repo: Path | None, *args: str) -> str | None:
    if repo is None:
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return None


def _git_record(repo: Path | None) -> dict[str, Any]:
    status = _git_value(repo, "status", "--short")
    return {
        "commit": _git_value(repo, "rev-parse", "HEAD"),
        "branch": _git_value(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "remote": _git_value(repo, "config", "--get", "remote.origin.url"),
    }


def _feedbax_repo() -> Path | None:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback is unused here.
        return None
    pyproject = _repo_root() / "pyproject.toml"
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except OSError:
        return None
    source = data.get("tool", {}).get("uv", {}).get("sources", {}).get("feedbax", {}).get("path")
    if not source:
        return None
    path = Path(str(source)).expanduser()
    if not path.is_absolute():
        path = _repo_root() / path
    return path if path.exists() else None


def _graph_metadata(
    *,
    run_spec: dict[str, Any],
    run_spec_path: Path,
    artifact_dir: Path,
    graph_manifest_path: Path | None,
    graph_spec_path: Path | None,
) -> dict[str, Any]:
    graph = _mapping(run_spec, "feedbax_graph")
    graph_version = None
    if graph_spec_path is not None and graph_spec_path.is_file():
        try:
            payload = json.loads(graph_spec_path.read_text(encoding="utf-8"))
            graph_version = (
                payload.get("schema_version") or payload.get("version") or payload.get("$schema")
            )
        except (OSError, json.JSONDecodeError):
            graph_version = None
    return {
        "graph_spec_path": graph.get("graph_spec_path"),
        "graph_spec_sha256": (
            sha256_file(graph_spec_path)
            if graph_spec_path is not None and graph_spec_path.is_file()
            else None
        ),
        "graph_spec_version": graph_version,
        "graph_manifest_path": graph.get("manifest_path")
        or (None if graph_manifest_path is None else graph_manifest_path.name),
        "graph_manifest_sha256": (
            sha256_file(graph_manifest_path)
            if graph_manifest_path is not None and graph_manifest_path.is_file()
            else None
        ),
        "tracked_run_spec": _repo_relative(run_spec_path),
        "artifact_dir": _repo_relative(artifact_dir),
    }
