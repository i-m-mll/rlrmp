"""Hyperparameter construction for the minimax adversarial trainer.

Bug: 8404108 — extracted from ``scripts/train_minimax.py`` so analysis /
eval / diagnostic scripts can reconstruct the same hyperparameter tree
from a saved ``config.json`` without ``sys.path``-injecting the training
script.

The corresponding training loop lives in ``scripts/train_minimax.py``; only
the hyperparameter construction is library-grade.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import equinox as eqx
from feedbax.contracts.training import (
    DEFAULT_TRAINING_METHOD_REGISTRY,
    ArtifactPolicySpec,
    CheckpointProgressPolicySpec,
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
    PhaseTransitionSpec,
    ProgressCoordinate,
    ResumeCoordinateSpec,
    StateSlotSpec,
    UpdateKernelSpec,
    UpdateStepSpec,
    derive_consistency_predicate,
)
from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from rlrmp.model.feedbax_graph import graph_spec_payload
from rlrmp.runtime.spec_migrations import (
    RUN_SPEC_KIND,
    RUN_SPEC_SCHEMA_ID,
    RUN_SPEC_SCHEMA_VERSION,
    stamp_current_schema,
)
from rlrmp.model.trainable import staged_network_trainable_paths

__all__ = [
    "MINIMAX_METHOD_REF",
    "MINIMAX_METHOD_PAYLOAD_SCHEMA_ID",
    "MINIMAX_METHOD_PAYLOAD_SCHEMA_VERSION",
    "MinimaxConfig",
    "MinimaxMethodPayload",
    "build_hps",
    "build_minimax_training_run_spec",
    "ensure_minimax_training_method_registered",
    "legacy_cli_args_to_minimax_config",
    "minimax_config_namespace",
    "minimax_effective_phase_fingerprint",
    "minimax_training_run_spec_from_file",
    "minimax_training_run_spec_to_config",
    "validate_minimax_run_spec",
    "validate_minimax_run_spec_file",
]


MINIMAX_METHOD_REF = "rlrmp/minimax/v1"
MINIMAX_METHOD_PAYLOAD_SCHEMA_ID = "rlrmp.spec.training_method.minimax_payload"
# Schema version intentionally remains v1: this refactor gives the unchanged
# wire shape typed owners, and no durable minimax payload corpus existed when it
# landed.
MINIMAX_METHOD_PAYLOAD_SCHEMA_VERSION = "rlrmp.spec.training_method.minimax_payload.v1"


class MinimaxConfig(BaseModel):
    """Flat minimax method config whose fields own all authoring defaults."""

    model_config = ConfigDict(extra="forbid")

    n_warmup_batches: int = Field(2000, ge=0)
    n_adversary_batches: int = Field(8000, ge=0)
    n_adversary_steps: int = Field(5, gt=0)
    batch_size: int = Field(250, gt=0)
    adv_batch_size: int | None = None
    n_replicates: int = Field(5, gt=0)
    seed: int = 42

    controller_lr: float = 1e-4
    adversary_lr: float = 3e-4
    loss_update_enabled: bool = False
    loss_update_ratio: float = 0.5

    adversary_type: Literal["gaussian_bump", "linear_dynamics"] = "gaussian_bump"
    n_adversaries: int = Field(1, gt=0)
    n_bumps: int = 3
    force_max: float = 1.0
    linear_dynamics_eta_max: float = 0.1
    linear_dynamics_pgd_steps: int = 5
    linear_dynamics_lr: float = 1e-2

    hidden_type: Literal["gru", "vanilla_rnn", "linear", "linear_tracker"] = "gru"
    sisu_gating: Literal["additive", "multiplicative"] = "additive"

    nn_output: float = 1e-5
    nn_hidden: float = 1e-5
    nn_hidden_derivative: float = 0.0
    nn_output_jerk: float = 0.0
    nn_output_pre_go: float = 0.0
    nn_hidden_derivative_pre_go: float = 0.0
    effector_hold_pos: float = 10.0
    effector_hold_vel: float = 10.0
    effector_final_vel: float = 0.0
    effector_vel_late: float = 0.1
    effector_pos_running: float = 1.0
    effector_pos_late_weight: float = 0.5
    effector_pos_late_final_scale: float = 2.0
    effector_pos_late_start_step: int = 80

    effector_pos_running_schedule: Literal["flat", "powerlaw", "movement_ramp"] = "flat"
    effector_hold_pos_schedule: Literal["flat", "powerlaw"] = "flat"
    position_powerlaw_power: float = 6.0
    movement_ramp_shape: Literal["linear", "cosine", "power"] = "linear"
    movement_ramp_duration_steps: int = 60
    movement_ramp_power: float = 2.0

    p_catch_trial: float = 0.5

    warmup_model: str | None = None
    output_dir: str = "_artifacts/minimax/minimax_test"
    spec_dir: str | None = None
    jax_cache_dir: str | None = None
    jax_explain_cache_misses: bool = False
    checkpoint: bool = False
    checkpoint_every: int = 500
    resume: bool = False
    fused: bool = True
    streaming_loss: bool = False

    @model_validator(mode="after")
    def _validate_config(self) -> "MinimaxConfig":
        if self.adversary_type == "linear_dynamics" and not self.fused:
            raise ValueError("linear_dynamics minimax requires fused execution")
        return self


class MinimaxWarmupSpec(BaseModel):
    """Typed warmup phase view derived from :class:`MinimaxConfig`."""

    model_config = ConfigDict(extra="forbid")

    n_batches: int = Field(ge=0)
    optimizer: Literal["adamw"] = "adamw"
    direction: Literal["minimize"] = "minimize"
    target: Literal["controller"] = "controller"


class MinimaxAdversarialSpec(BaseModel):
    """Typed adversarial phase view derived from :class:`MinimaxConfig`."""

    model_config = ConfigDict(extra="forbid")

    n_batches: int = Field(ge=0)
    inner_steps: int = Field(gt=0)
    active_member: str = "batch_idx % n_adversaries"
    kernel_variant: Literal["fused", "decomposed"]
    frozen_controller: bool = True
    outer_direction: Literal["minimize"] = "minimize"
    inner_direction: Literal["maximize"] = "maximize"


class MinimaxProjectionSpec(BaseModel):
    """Typed projection policy view derived from :class:`MinimaxConfig`."""

    model_config = ConfigDict(extra="forbid")

    target: str
    operator: Literal["frobenius_ball"] = "frobenius_ball"
    radius: float
    radius_source: str = "linear_dynamics_eta_max"
    timing: str = "after_each_adversary_step"
    phase_scope: str = "adversarial"


class MinimaxOptimizerPolicySpec(BaseModel):
    """Typed optimizer policy view derived from :class:`MinimaxConfig`."""

    model_config = ConfigDict(extra="forbid")

    controller_lr: float
    adversary_lr: float
    controller_optimizer: str = "adamw"
    adversary_optimizer: str = "adam"


class MinimaxCheckpointPolicySpec(BaseModel):
    """Typed checkpoint policy view derived from :class:`MinimaxConfig`."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_every: int
    barriers: list[str]
    custody: str = "feedbax"


class MinimaxOutputArtifactsSpec(BaseModel):
    """Typed output artifact policy view derived from :class:`MinimaxConfig`."""

    model_config = ConfigDict(extra="forbid")

    artifact_root: str
    tracked_spec_dir: str
    training_run_spec: str = "feedbax_training_run_spec"


class MinimaxMethodPayload(BaseModel):
    """RLRMP-governed minimax method payload embedded in TrainingRunSpec."""

    model_config = ConfigDict(extra="forbid")

    config: MinimaxConfig
    adversary_type: Literal["gaussian_bump", "linear_dynamics"]
    warmup: MinimaxWarmupSpec
    adversarial: MinimaxAdversarialSpec
    projection: MinimaxProjectionSpec
    optimizer_policy: MinimaxOptimizerPolicySpec
    checkpoint_policy: MinimaxCheckpointPolicySpec
    output_artifacts: MinimaxOutputArtifactsSpec
    rlrmp_extension_payload: str = "rlrmp_run_spec"

    @field_serializer("config")
    def _serialize_config(self, config: MinimaxConfig, info: Any) -> dict[str, Any]:
        return config.model_dump(mode=info.mode, exclude_none=False)

    @model_validator(mode="after")
    def _validate_payload(self) -> "MinimaxMethodPayload":
        config = self.config
        expected_projection_target = "adversary_population[active_member].delta_A"
        expected_adversary_lr = (
            config.linear_dynamics_lr
            if config.adversary_type == "linear_dynamics"
            else config.adversary_lr
        )
        checks = {
            "adversary_type": self.adversary_type == config.adversary_type,
            "warmup.n_batches": self.warmup.n_batches == config.n_warmup_batches,
            "adversarial.n_batches": (
                self.adversarial.n_batches == config.n_adversary_batches
            ),
            "adversarial.inner_steps": (
                self.adversarial.inner_steps == config.n_adversary_steps
            ),
            "adversarial.kernel_variant": (
                self.adversarial.kernel_variant == ("fused" if config.fused else "decomposed")
            ),
            "projection.radius": self.projection.radius == config.linear_dynamics_eta_max,
            "projection.target": self.projection.target == expected_projection_target,
            "optimizer_policy.controller_lr": (
                self.optimizer_policy.controller_lr == config.controller_lr
            ),
            "optimizer_policy.adversary_lr": (
                self.optimizer_policy.adversary_lr == expected_adversary_lr
            ),
            "checkpoint_policy.checkpoint_every": (
                self.checkpoint_policy.checkpoint_every == config.checkpoint_every
            ),
        }
        mismatches = [name for name, ok in checks.items() if not ok]
        if mismatches:
            raise ValueError(
                "minimax method payload derived views disagree with config: "
                + ", ".join(mismatches)
            )
        if self.adversary_type == "linear_dynamics" and self.projection.target != (
            expected_projection_target
        ):
            raise ValueError(
                "linear_dynamics minimax payload must project "
                "adversary_population[active_member].delta_A"
            )
        return self


def legacy_cli_args_to_minimax_config(argv: Sequence[str]) -> dict[str, Any]:
    """Translate legacy minimax CLI flags into a validated config mapping.

    This preserves existing command-line launch ergonomics while making CLI
    parsing an authoring step. The training runner consumes the resulting
    ``TrainingRunSpec``, never a raw parser namespace.
    """

    raw_overrides: dict[str, Any] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in {"-h", "--help"}:
            raise SystemExit(_minimax_cli_help())
        if not token.startswith("--"):
            raise ValueError(f"unexpected positional argument for minimax spec authoring: {token}")
        raw_name = token[2:]
        value_text: str | None = None
        if "=" in raw_name:
            raw_name, value_text = raw_name.split("=", 1)
        negate = raw_name.startswith("no-")
        name = raw_name[3:] if negate else raw_name
        field = name.replace("-", "_")
        if field not in MinimaxConfig.model_fields:
            raise ValueError(f"unknown minimax option: --{raw_name}")
        if MinimaxConfig.model_fields[field].annotation is bool:
            raw_overrides[field] = not negate
            if value_text is not None:
                raw_overrides[field] = _parse_bool(value_text, field=field)
        else:
            if negate:
                raise ValueError(f"--no-{name} is only valid for boolean minimax options")
            if value_text is None:
                index += 1
                if index >= len(argv):
                    raise ValueError(f"missing value for --{name}")
                value_text = argv[index]
            raw_overrides[field] = value_text
        index += 1
    return MinimaxConfig.model_validate(raw_overrides).model_dump(mode="python")


def minimax_config_namespace(config: Mapping[str, Any]) -> SimpleNamespace:
    """Return attribute-style access for an already validated minimax config."""

    normalized = MinimaxConfig.model_validate(dict(config))
    return SimpleNamespace(**normalized.model_dump(mode="python"))


def build_minimax_training_run_spec(
    config: Mapping[str, Any],
    *,
    graph_spec: Any,
    output_dir: Path,
    spec_dir: Path,
    git: Mapping[str, Any] | None = None,
    gpu_info: Mapping[str, Any] | None = None,
    feedbax_graph: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the tracked minimax recipe with a composed Feedbax TrainingRunSpec."""

    ensure_minimax_training_method_registered()
    normalized = MinimaxConfig.model_validate(dict(config))
    payload = _legacy_minimax_run_spec_payload(
        normalized,
        git=dict(git or {}),
        gpu_info=dict(gpu_info or {}),
        feedbax_graph=dict(feedbax_graph or {}),
    )
    extension = stamp_current_schema(RUN_SPEC_KIND, _rlrmp_minimax_extension_payload(payload))
    method_payload = _minimax_method_payload(normalized, output_dir=output_dir, spec_dir=spec_dir)
    contract = minimax_method_contract()
    effective_phase = minimax_effective_phase_spec(contract)
    fingerprint = minimax_effective_phase_fingerprint(
        effective_phase=effective_phase,
        graph_payload=graph_spec_payload(graph_spec),
        method_payload=method_payload.model_dump(mode="json", exclude_none=True),
    )
    feedbax_spec = TrainingRunSpec(
        graph=GraphTopologySourceSpec(
            inline=graph_spec_payload(graph_spec),
            schema_id=getattr(graph_spec, "schema_id", None),
            schema_version=getattr(graph_spec, "schema_version", None),
            metadata={
                "source": "requested_serialized_graph_spec",
                "declarative_adversary_injection": (
                    normalized.adversary_type == "linear_dynamics"
                ),
                "component_parameter_binding": (
                    "linear_dynamics_adversary_params"
                    if normalized.adversary_type == "linear_dynamics"
                    else None
                ),
            },
        ),
        task=TaskSpec(type="rlrmp_minimax_delayed_reach", params={"dt": 0.01}),
        training_config=TrainingConfig(
            n_batches=normalized.n_warmup_batches + normalized.n_adversary_batches,
            batch_size=normalized.batch_size,
            learning_rate=normalized.controller_lr,
            grad_clip=1.0,
            hidden_dim=180,
            network_type=normalized.hidden_type,
            n_reach_steps=140,
            effort_weight=normalized.nn_output,
            snapshot_interval=max(1, normalized.checkpoint_every or 1),
        ),
        objective=ObjectiveSlotSpec(
            kind="external",
            payload={
                "objective_ref": "rlrmp.minimax_controller_loss",
                "tail_reductions": {
                    "trial": "mean",
                    "replicate": "mean",
                    "active_adversary_member": "selected_by_batch_modulo_population",
                },
            },
            schema_id="rlrmp.minimax_objective",
            schema_version="rlrmp.minimax_objective.v1",
            metadata={"lowering_owner": "rlrmp.train.minimax"},
        ),
        risk_aggregation=RiskAggregationSpec(
            realization="mean",
            replicate="mean",
            metadata={"population_member": "active_member_only"},
        ),
        method_ref=MethodRefSpec(package="rlrmp", name="minimax", version="v1"),
        method_payload=method_payload,
        method_extensions={
            "metadata": {
                "rlrmp_extension_payload": "rlrmp_run_spec",
                "scientific_semantics_owner": "rlrmp.train.adversary",
            }
        },
        worker_execution=WorkerExecutionSpec(
            method_contract=contract,
            effective_phase=effective_phase,
            metadata={
                "effective_phase_fingerprint": fingerprint,
                "pre_execution_parity": "compare_requested_serialized_spec",
                "legacy_loop_backend": "scripts.train_minimax.run_training",
            },
        ),
        execution=ExecutionPolicySpec(
            mode="local",
            require_review=False,
            allow_cloud=False,
            metadata={"entrypoint": "scripts/train_minimax.py"},
        ),
        artifacts=ArtifactPolicySpec(
            manifest_root="_artifacts/feedbax_runs",
            artifact_root=str(output_dir),
            custody="local",
            metadata={
                "tracked_spec_dir": str(spec_dir),
                "bulk_outputs": str(output_dir),
            },
        ),
        checkpoint_progress=CheckpointProgressPolicySpec(
            checkpoint_interval=max(1, normalized.checkpoint_every or 1),
            progress_interval=max(1, normalized.checkpoint_every or 1),
            metadata={
                "effective_phase_fingerprint": fingerprint,
                "checkpoint_custody_owner": "feedbax",
                "rlrmp_legacy_checkpoint_writer": "deferred_to_799fcb9",
            },
        ),
        metadata={
            "composed_with": "rlrmp_run_spec",
            "serialize_do_not_rederive": True,
            "effective_phase_fingerprint": fingerprint,
        },
    )
    payload["rlrmp_run_spec"] = extension
    payload["feedbax_training_run_spec"] = feedbax_spec.model_dump(
        mode="json",
        exclude_none=True,
    )
    validate_minimax_run_spec(payload, spec_dir=spec_dir, require_graph_sidecars=False)
    return payload


def minimax_training_run_spec_from_file(path: Path | str) -> TrainingRunSpec:
    """Load a tracked minimax recipe and return its validated TrainingRunSpec."""

    ensure_minimax_training_method_registered()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_minimax_run_spec(payload, spec_dir=Path(path).parent)
    return TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])


def minimax_training_run_spec_to_config(spec: TrainingRunSpec) -> dict[str, Any]:
    """Recover the governed minimax config from a validated TrainingRunSpec."""

    ensure_minimax_training_method_registered()
    validated = DEFAULT_TRAINING_METHOD_REGISTRY.validate_payload(
        spec.method_ref,
        spec.method_payload,
        path="/method_payload",
    )
    if not isinstance(validated, MinimaxMethodPayload):
        raise TypeError("TrainingRunSpec does not carry an RLRMP minimax method payload")
    return validated.config.model_dump(mode="python")


def validate_minimax_run_spec(
    run_spec: dict[str, Any],
    *,
    spec_dir: Path,
    require_graph_sidecars: bool = True,
) -> None:
    """Validate the tracked minimax run-spec contract."""

    del require_graph_sidecars
    ensure_minimax_training_method_registered()
    required = {
        "schema_id",
        "schema_version",
        "mode",
        "training_script",
        "training_summary",
        "adversary",
        "phase_program",
        "projection",
        "optimizer",
        "checkpointing",
        "feedbax_graph",
        "rlrmp_run_spec",
        "feedbax_training_run_spec",
    }
    missing = sorted(required - set(run_spec))
    if missing:
        raise ValueError("minimax run spec is missing required keys: " + ", ".join(missing))
    if run_spec["schema_id"] != RUN_SPEC_SCHEMA_ID:
        raise ValueError(f"unsupported minimax run spec schema_id: {run_spec['schema_id']!r}")
    if run_spec["schema_version"] != RUN_SPEC_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported minimax run spec schema_version: {run_spec['schema_version']!r}"
        )
    training_spec = TrainingRunSpec.model_validate(run_spec["feedbax_training_run_spec"])
    if training_spec.method_ref.key != MINIMAX_METHOD_REF:
        raise ValueError(
            "minimax run spec TrainingRunSpec method_ref mismatch: "
            f"{training_spec.method_ref.key!r}"
        )
    expected = training_spec.metadata.get("effective_phase_fingerprint")
    actual = minimax_effective_phase_fingerprint(
        effective_phase=training_spec.worker_execution.effective_phase,
        graph_payload=training_spec.graph.inline or {},
        method_payload=training_spec.method_payload.model_dump(mode="json", exclude_none=True),
    )
    if expected != actual:
        raise ValueError(
            "minimax TrainingRunSpec effective-phase fingerprint mismatch: "
            f"stored={expected!r}, actual={actual!r}"
        )
    if spec_dir is None:
        raise ValueError("spec_dir is required for minimax run spec validation")


def validate_minimax_run_spec_file(path: Path | str) -> None:
    """Load and validate a tracked minimax ``run.json`` file."""

    run_spec_path = Path(path)
    validate_minimax_run_spec(
        json.loads(run_spec_path.read_text(encoding="utf-8")),
        spec_dir=run_spec_path.parent,
    )


def ensure_minimax_training_method_registered() -> None:
    """Install the RLRMP minimax method row into Feedbax's default registry."""

    if MINIMAX_METHOD_REF in DEFAULT_TRAINING_METHOD_REGISTRY.available_keys():
        return
    DEFAULT_TRAINING_METHOD_REGISTRY.register(
        TrainingMethodRegistration(
            method_ref=MINIMAX_METHOD_REF,
            payload_schema_id=MINIMAX_METHOD_PAYLOAD_SCHEMA_ID,
            payload_schema_version=MINIMAX_METHOD_PAYLOAD_SCHEMA_VERSION,
            payload_model=MinimaxMethodPayload,
            contract_factory=minimax_method_contract,
            update_kernels_factory=_minimax_update_kernels,
            rejected_payload_versions=("rlrmp.spec.training_method.minimax_payload.v0",),
            owner="rlrmp.train.minimax",
            package="rlrmp",
        )
    )


def minimax_method_contract() -> MethodContractSpec:
    """Return the declarative warmup/adversarial minimax phase program."""

    program = PhaseProgramSpec(
        phases=[
            PhaseSpec(
                name="warmup",
                kind="warmup",
                reads=["controller", "controller_optimizer", "rng", "objective"],
                writes=["controller", "controller_optimizer", "rng", "controller_loss"],
                update_steps=["warmup_controller_descent"],
                legal_next=["adversarial"],
                checkpoint_barrier="after_warmup",
                loop_axis="batch",
                metadata={
                    "activation_binding": "linear_dynamics_adversary_params.active=false",
                    "frozen_slots": ["adversary_population", "adversary_optimizer"],
                },
            ),
            PhaseSpec(
                name="adversarial",
                kind="adversarial",
                reads=[
                    "controller",
                    "controller_optimizer",
                    "adversary_population",
                    "adversary_optimizer",
                    "trial_batch",
                    "rng",
                    "objective",
                ],
                writes=[
                    "controller",
                    "controller_optimizer",
                    "adversary_population",
                    "adversary_optimizer",
                    "rng",
                    "controller_loss",
                    "adversary_loss",
                ],
                update_steps=[
                    "inner_adversary_ascent",
                    "adversary_projection",
                    "outer_controller_descent",
                ],
                checkpoint_barrier="after_adversarial",
                loop_axis="batch",
                metadata={
                    "active_member": "global_step % n_adversaries",
                    "frozen_controller_boundary": (
                        "stop_gradient(controller) during inner_adversary_ascent"
                    ),
                    "kernel_variants": {
                        "fused": "same reads/writes/barriers as decomposed",
                        "decomposed": "same reads/writes/barriers as fused",
                    },
                },
            ),
        ],
        initial_phase="warmup",
        transitions=[
            PhaseTransitionSpec(
                source="warmup",
                target="adversarial",
                barrier="after_warmup",
            )
        ],
        update_steps=[
            UpdateStepSpec(
                name="warmup_controller_descent",
                kind="gradient",
                kernel=UpdateKernelSpec(kernel_ref="rlrmp.minimax.warmup_controller_descent"),
                reads=["controller", "controller_optimizer", "rng", "objective"],
                writes=["controller", "controller_optimizer", "rng", "controller_loss"],
                axes=["batch", "replicate"],
                optimizer_binding="controller_optimizer_to_controller_warmup",
                metadata={"direction": "minimize"},
            ),
            UpdateStepSpec(
                name="inner_adversary_ascent",
                kind="gradient",
                kernel=UpdateKernelSpec(kernel_ref="rlrmp.minimax.inner_adversary_ascent"),
                reads=["controller", "adversary_population", "adversary_optimizer", "rng"],
                writes=["adversary_population", "adversary_optimizer", "rng", "adversary_loss"],
                axes=["inner_step", "adversary_member", "replicate"],
                optimizer_binding="adversary_optimizer_to_active_member",
                metadata={"direction": "maximize"},
            ),
            UpdateStepSpec(
                name="adversary_projection",
                kind="projection",
                kernel=UpdateKernelSpec(kernel_ref="rlrmp.minimax.frobenius_ball_projection"),
                reads=["adversary_population"],
                writes=["adversary_population"],
                axes=["adversary_member", "replicate"],
                metadata={
                    "target": "adversary_population[active_member].delta_A",
                    "operator": "frobenius_ball",
                    "radius_source": "method_payload.projection.radius",
                    "applies_to": "adversary_params_not_optimizer_moments",
                },
            ),
            UpdateStepSpec(
                name="outer_controller_descent",
                kind="gradient",
                kernel=UpdateKernelSpec(kernel_ref="rlrmp.minimax.outer_controller_descent"),
                reads=["controller", "controller_optimizer", "adversary_population", "rng"],
                writes=["controller", "controller_optimizer", "rng", "controller_loss"],
                axes=["batch", "replicate"],
                optimizer_binding="controller_optimizer_to_controller_adversarial",
                metadata={"direction": "minimize"},
            ),
        ],
        optimizer_bindings=[
            OptimizerTargetBinding(
                name="controller_optimizer_to_controller_warmup",
                optimizer_slot="controller_optimizer",
                target_slot="controller",
                direction="minimize",
                projection="after_step",
                phase_scope=["warmup"],
                objective_reads=["objective"],
            ),
            OptimizerTargetBinding(
                name="adversary_optimizer_to_active_member",
                optimizer_slot="adversary_optimizer",
                target_slot="adversary_population",
                target_selector="adversary_population[active_member]",
                direction="maximize",
                projection="after_step",
                phase_scope=["adversarial"],
                objective_reads=["objective"],
                metadata={"active_member": "global_step % n_adversaries"},
            ),
            OptimizerTargetBinding(
                name="controller_optimizer_to_controller_adversarial",
                optimizer_slot="controller_optimizer",
                target_slot="controller",
                direction="minimize",
                projection="after_step",
                phase_scope=["adversarial"],
                objective_reads=["objective"],
            ),
        ],
        checkpoint_barriers=[
            CheckpointBarrierSpec(
                name="after_warmup",
                phase="warmup",
                slots=[
                    CheckpointSlotSpec(slot="controller", axis="replicate"),
                    CheckpointSlotSpec(slot="controller_optimizer", axis="replicate"),
                    CheckpointSlotSpec(slot="adversary_population", axis="adversary_member"),
                    CheckpointSlotSpec(slot="adversary_optimizer", axis="adversary_member"),
                    CheckpointSlotSpec(slot="rng"),
                ],
                resume_coordinate=ResumeCoordinateSpec(
                    phase="adversarial",
                    completed_barrier="after_warmup",
                    global_step=0,
                ),
            ),
            CheckpointBarrierSpec(
                name="after_adversarial",
                phase="adversarial",
                slots=[
                    CheckpointSlotSpec(slot="controller", axis="replicate"),
                    CheckpointSlotSpec(slot="controller_optimizer", axis="replicate"),
                    CheckpointSlotSpec(slot="adversary_population", axis="adversary_member"),
                    CheckpointSlotSpec(slot="adversary_optimizer", axis="adversary_member"),
                    CheckpointSlotSpec(slot="rng"),
                ],
            ),
        ],
        metadata={
            "phase_program_identity": "rlrmp.minimax.warmup_then_adversarial.v1",
            "checkpoint_barrier_policy": "after_warmup_and_after_adversarial",
        },
    )
    return MethodContractSpec(
        method_ref=MINIMAX_METHOD_REF,
        method_payload_schema_version=MINIMAX_METHOD_PAYLOAD_SCHEMA_VERSION,
        axes=[
            AxisSpec(name="batch", role="batch"),
            AxisSpec(name="replicate", role="replicate"),
            AxisSpec(name="adversary_member", role="member"),
            AxisSpec(name="inner_step", role="epoch"),
        ],
        state_slots=[
            StateSlotSpec(name="controller", role="model", axis="replicate"),
            StateSlotSpec(name="controller_optimizer", role="optimizer", axis="replicate"),
            StateSlotSpec(
                name="adversary_population",
                role="population",
                axis="adversary_member",
                metadata={"member_payload": "GaussianBumpAdversary|LinearDynamicsAdversary"},
            ),
            StateSlotSpec(
                name="adversary_optimizer",
                role="optimizer",
                axis="adversary_member",
            ),
            StateSlotSpec(name="trial_batch", role="environment", lifetime="per-outer-step-init"),
            StateSlotSpec(name="rng", role="prng"),
            StateSlotSpec(name="objective", role="objective"),
            StateSlotSpec(name="controller_loss", role="metric", required=False),
            StateSlotSpec(name="adversary_loss", role="metric", required=False),
        ],
        phase_program=program,
        objective_reducers=[
            {
                "axis": "batch",
                "owner": "objective",
                "path": "/objective/payload/tail_reductions/trial",
            }
        ],
        worker_reducers=[
            {
                "axis": "replicate",
                "owner": "worker",
                "path": "/risk_aggregation/replicate",
            }
        ],
        metadata={
            "projection_binding": "adversary_population[active_member].delta_A",
            "component_parameter_source": "linear_dynamics_adversary_params",
        },
    )


def minimax_effective_phase_spec(contract: MethodContractSpec | None = None):
    """Return the EffectivePhaseSpec corresponding to the minimax contract."""

    active_contract = contract or minimax_method_contract()
    return EffectivePhaseSpec(
        method_ref=active_contract.method_ref,
        axes=active_contract.axes,
        state_slots=active_contract.state_slots,
        phase_program=active_contract.phase_program,
        consistency_predicate=derive_consistency_predicate(active_contract.phase_program),
    )


def minimax_effective_phase_fingerprint(
    *,
    effective_phase: Any,
    graph_payload: Mapping[str, Any],
    method_payload: Mapping[str, Any],
) -> str:
    """Hash the executor-relevant minimax parity surface."""

    parity = {
        "graph": graph_payload,
        "effective_phase": effective_phase.model_dump(mode="json", exclude_none=True),
        "method_payload": method_payload,
    }
    encoded = json.dumps(parity, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _minimax_update_kernels(
    _payload: BaseModel | None = None,
) -> Mapping[str, Any]:
    def _identity_kernel(
        slots: Mapping[str, Any],
        coordinate: ProgressCoordinate,
        context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del context
        return {
            "controller": slots.get("controller"),
            "controller_optimizer": slots.get("controller_optimizer"),
            "adversary_population": slots.get("adversary_population"),
            "adversary_optimizer": slots.get("adversary_optimizer"),
            "rng": slots.get("rng"),
            "controller_loss": float(coordinate.global_step),
            "adversary_loss": float(coordinate.global_step),
        }

    return {
        "rlrmp.minimax.warmup_controller_descent": _identity_kernel,
        "rlrmp.minimax.inner_adversary_ascent": _identity_kernel,
        "rlrmp.minimax.frobenius_ball_projection": _identity_kernel,
        "rlrmp.minimax.outer_controller_descent": _identity_kernel,
    }


def _minimax_method_payload(
    config: Mapping[str, Any] | MinimaxConfig,
    *,
    output_dir: Path,
    spec_dir: Path,
) -> MethodPayloadEnvelope:
    normalized = (
        config if isinstance(config, MinimaxConfig) else MinimaxConfig.model_validate(dict(config))
    )
    payload = MinimaxMethodPayload(
        config=normalized,
        adversary_type=normalized.adversary_type,
        warmup=MinimaxWarmupSpec(n_batches=normalized.n_warmup_batches),
        adversarial=MinimaxAdversarialSpec(
            n_batches=normalized.n_adversary_batches,
            inner_steps=normalized.n_adversary_steps,
            kernel_variant="fused" if normalized.fused else "decomposed",
        ),
        projection=MinimaxProjectionSpec(
            target="adversary_population[active_member].delta_A",
            radius=normalized.linear_dynamics_eta_max,
        ),
        optimizer_policy=MinimaxOptimizerPolicySpec(
            controller_lr=normalized.controller_lr,
            adversary_lr=(
                normalized.linear_dynamics_lr
                if normalized.adversary_type == "linear_dynamics"
                else normalized.adversary_lr
            ),
        ),
        checkpoint_policy=MinimaxCheckpointPolicySpec(
            checkpoint_every=normalized.checkpoint_every,
            barriers=["after_warmup", "after_adversarial"],
        ),
        output_artifacts=MinimaxOutputArtifactsSpec(
            artifact_root=str(output_dir),
            tracked_spec_dir=str(spec_dir),
        ),
    )
    return MethodPayloadEnvelope(
        schema_id=MINIMAX_METHOD_PAYLOAD_SCHEMA_ID,
        schema_version=MINIMAX_METHOD_PAYLOAD_SCHEMA_VERSION,
        payload=payload.model_dump(mode="json", exclude_none=True),
    )


def _legacy_minimax_run_spec_payload(
    config: MinimaxConfig,
    *,
    git: Mapping[str, Any],
    gpu_info: Mapping[str, Any],
    feedbax_graph: Mapping[str, Any],
) -> dict[str, Any]:
    training_mode = "minimax"
    if config.adversary_type == "linear_dynamics":
        training_mode += "+linear_dynamics"
    config_dict = config.model_dump(mode="python")
    return stamp_current_schema(
        RUN_SPEC_KIND,
        {
            **config_dict,
            "mode": "train_minimax",
            "training_script": "scripts/train_minimax.py",
            "git": dict(git),
            "gpu_info": dict(gpu_info),
            "feedbax_graph": dict(feedbax_graph),
            "training_summary": {
                "training_mode": training_mode,
                "n_warmup_batches": config.n_warmup_batches,
                "n_adversary_batches": config.n_adversary_batches,
                "batch_size": config.batch_size,
                "adv_batch_size": config.adv_batch_size,
                "n_replicates": config.n_replicates,
            },
            "adversary": {
                "type": config.adversary_type,
                "n_adversaries": config.n_adversaries,
                "n_inner_steps": config.n_adversary_steps,
                "n_bumps": config.n_bumps,
                "force_max": config.force_max,
                "linear_dynamics": {
                    "eta_max": config.linear_dynamics_eta_max,
                    "pgd_steps": config.linear_dynamics_pgd_steps,
                    "learning_rate": config.linear_dynamics_lr,
                },
            },
            "phase_program": {
                "warmup": "controller_descent",
                "adversarial": [
                    "inner_adversary_ascent",
                    "frobenius_ball_projection",
                    "outer_controller_descent",
                ],
                "kernel_variant": "fused" if config.fused else "decomposed",
            },
            "projection": {
                "target": "adversary_population[active_member].delta_A",
                "operator": "frobenius_ball",
                "radius_source": "linear_dynamics_eta_max",
                "radius": config.linear_dynamics_eta_max,
            },
            "optimizer": {
                "controller": {"type": "adamw", "learning_rate": config.controller_lr},
                "adversary": {
                    "type": "adam",
                    "learning_rate": (
                        config.linear_dynamics_lr
                        if config.adversary_type == "linear_dynamics"
                        else config.adversary_lr
                    ),
                },
            },
            "checkpointing": {
                "checkpoint_every": config.checkpoint_every,
                "resume": config.resume,
                "custody": "feedbax",
            },
        },
    )


def _rlrmp_minimax_extension_payload(run_spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "issue": "54b0c2e",
        "mode": run_spec.get("mode"),
        "training_script": run_spec.get("training_script"),
        "training_summary": run_spec.get("training_summary"),
        "adversary": run_spec.get("adversary"),
        "phase_program": run_spec.get("phase_program"),
        "projection": run_spec.get("projection"),
        "optimizer": run_spec.get("optimizer"),
        "checkpointing": run_spec.get("checkpointing"),
        "feedbax_graph": run_spec.get("feedbax_graph"),
    }


def _parse_bool(value: str, *, field: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{field} expects a boolean value, found {value!r}")


def _minimax_cli_help() -> str:
    options = "\n".join(
        f"  --{name.replace('_', '-')} (default: {field.default!r})"
        for name, field in sorted(MinimaxConfig.model_fields.items())
    )
    return "Minimax adversarial training spec authoring options:\n" + options


ensure_minimax_training_method_registered()


def _trainable_paths_for_hidden_type(hidden_type: str, sisu_gating: str) -> list[str]:
    if hidden_type == "linear":
        return ["nodes.net.gain"]
    if hidden_type == "linear_tracker":
        return ["nodes.net.gain", "nodes.net.feedforward"]
    return staged_network_trainable_paths(sisu_gating=sisu_gating)


def _resolve_hidden_type(hidden_type_str: str, dt: float):
    """Map a CLI hidden-type string to the corresponding recurrent cell class/partial.

    Args:
        hidden_type_str: One of ``"gru"``, ``"vanilla_rnn"``, ``"linear"``,
            ``"linear_tracker"``.
        dt: Simulation timestep (used to set ``alpha = dt / tau`` for
            ``VanillaRNNCell``).

    Returns:
        A class or partial-applied constructor compatible with
        ``point_mass_nn``'s ``hidden_type`` parameter (i.e. callable as
        ``hidden_type(input_size, hidden_size, use_bias=..., key=...)``), or the
        sentinel string for linear controllers.
    """
    if hidden_type_str == "gru":
        return eqx.nn.GRUCell
    elif hidden_type_str == "vanilla_rnn":
        from rlrmp.model import VanillaRNNCell
        # tau=0.1 s (100 ms) => alpha=dt/tau=0.1 at dt=0.01 — matches cortical-neuron
        # time constant in motor-control RNN literature (Yang 2019, Sussillo 2015).
        return partial(VanillaRNNCell, dt=dt, tau=0.1)
    elif hidden_type_str in ("linear", "linear_tracker"):
        # Sentinel string forwarded to setup_task_model_pair, which dispatches to
        # create_point_mass_linear_ensemble. Linear controllers have no recurrent
        # cell — they replace SimpleStagedNetwork entirely. Bug: 410d7ac.
        return hidden_type_str
    else:
        raise ValueError(f"Unknown hidden_type: {hidden_type_str!r}")


def build_hps(args: Any) -> TreeNamespace:
    """Construct minimax-trainer hyperparameters from CLI args.

    Uses the same task config as :func:`rlrmp.train.standard.build_hps`
    (running_cost loss mode), so the two trainers produce comparable models.
    """
    dt = 0.01
    hps_dict = {
        "method": "pai-asf",
        "dt": dt,
        # n_batches_condition drives setup_task_model_pair's loss schedule;
        # set to total training length so late-ramp terms are calibrated correctly.
        "n_batches_condition": args.n_warmup_batches + args.n_adversary_batches,
        "n_batches_baseline": 0,
        "batch_size": getattr(args, "batch_size", 250),
        "learning_rate_0": args.controller_lr,
        "n_scaleup_batches": 0,
        "constant_lr_iterations": 0,
        "cosine_annealing_alpha": 1.0,
        "weight_decay": 0.0,
        "state_reset_iterations": [],
        "intervention_scaleup_batches": [0, 0],
        "model": {
            "n_replicates": getattr(args, "n_replicates", 5),
            "effector_mass": 1.0,
            "hidden_size": 180,
            "feedback_delay_steps": 5,
            "feedback_noise_std": 0.01,
            "motor_noise_std": 0.01,
            "sensory_noise_std": getattr(args, "sensory_noise_std", None),
            "additive_motor_noise_std": getattr(args, "additive_motor_noise_std", None),
            "signal_dependent_motor_noise_std": getattr(
                args,
                "signal_dependent_motor_noise_std",
                None,
            ),
            "plant_process_force_noise_std": getattr(
                args,
                "plant_process_force_noise_std",
                0.0,
            ),
            "damping": 10.0,
            "tau_rise": 0.05,
            "population_structure": {
                "n_input_only": 60,
                "n_readout_only": 60,
                "n_recurrent_only": 60,
                "n_input_readout": 0,
            },
        },
        "task": {
            "type": "delayed_reach",
            "n_steps": 140,
            "workspace": [[-1.0, -1.0], [1.0, 1.0]],
            "eval_grid_n": 1,
            "eval_n_directions": 8,
            "eval_reach_length": 0.5,
            "train_endpoint_mode": "center_out",
            # Drop pure-hold to 0 steps; target-on now 100-300 ms (10-30 steps
            # at dt=0.01 s), matching Shahbazi 2025 §4.2. Bug: 2bc95fd
            "epoch_len_ranges": [[0, 1], [10, 30]],
            "target_on_epochs": [1, 2],
            "hold_epochs": [0, 1],
            "move_epochs": [2],
            "p_catch_trial": getattr(args, "p_catch_trial", 0.5),
        },
        "pert": {
            "type": (
                "dynamics_matrix"
                if getattr(args, "adversary_type", "gaussian_bump") == "linear_dynamics"
                else "gusts"
            ),
            # Warm-start uses pert_std=1.0 (normal gusts).
            "std": 1.0,
            "duration_mean": 8,
            "n_expected": 3,
        },
        "loss": {
            "weights": {
                "goal_hit_in_window": 0.0,
                "effector_pos": 0.0,
                "effector_pos_running": getattr(args, "effector_pos_running", 1.0),
                "effector_pos_mid": 0.0,
                "effector_vel_mid": 0.0,
                "effector_pos_late": getattr(args, "effector_pos_late_weight", 0.5),
                "effector_vel_late": getattr(args, "effector_vel_late", 0.1),
                "effector_hold_pos": getattr(args, "effector_hold_pos", 10.0),
                "effector_hold_vel": getattr(args, "effector_hold_vel", 10.0),
                # Terminal-step velocity penalty (historical simple_reach_loss
                # shape). Fires only at t=T; strong "come-to-rest" signal.
                # Default 0.0 = disabled (preserves baseline behaviour).
                # Activate via --effector-final-vel 1.0. Bug: 2bc95fd
                "effector_final_vel": getattr(args, "effector_final_vel", 0.0),
                "nn_output": getattr(args, "nn_output", 1e-5),
                "nn_hidden": getattr(args, "nn_hidden", 1e-5),
                # Compositional ||h_t - h_{t-1}||² hidden-state smoothness
                # term, off-by-default. Enable via --nn-hidden-derivative
                # (e.g. 1e-3 per Shahbazi et al. 2025 Eq. 1). Bug: efc4d68
                "nn_hidden_derivative": getattr(args, "nn_hidden_derivative", 0.0),
                # Compositional ||v_{t+1} - 2 v_t + v_{t-1}||² output-jerk
                # term, off-by-default. Enable via --nn-output-jerk
                # (e.g. 1e5 per Shahbazi et al. 2025 Eq. 1). Bug: efc4d68
                # (feedbax 7e1d257)
                "nn_output_jerk": getattr(args, "nn_output_jerk", 0.0),
                # Pre-go controller-output penalty (epochs 0+1, before the go
                # cue). Wraps the standard nn_output squared-L2 term in
                # EpochMaskedLoss; off-by-default. Enable via
                # --nn-output-pre-go (suggested 1e-2 ≈ 1000x the post-aggregated
                # nn_output weight). Bug: efc4d68 (feedbax 50507a9)
                "nn_output_pre_go": getattr(args, "nn_output_pre_go", 0.0),
                # Pre-go hidden-state-derivative penalty (epochs 0+1).
                # Companion to the motor-pre-go term — included so the
                # "suppress preparation too" comparator is one flag away.
                # Off-by-default. Bug: efc4d68 (feedbax 50507a9)
                "nn_hidden_derivative_pre_go": getattr(
                    args, "nn_hidden_derivative_pre_go", 0.0
                ),
            },
            "effector_pos_late": {
                "start_step_after_go": getattr(
                    args, "effector_pos_late_start_step", 80
                ),
                "final_scale_factor": getattr(
                    args, "effector_pos_late_final_scale", 2.0
                ),
            },
            "effector_vel_late": {
                "start_step_after_go": 80,
                "final_scale_factor": 1.0,
            },
            # Power-law schedule: "flat" (default) or "powerlaw" ((t/T-1)^power).
            # Bug: 2e1a6ad
            "effector_pos_running_schedule": getattr(
                args, "effector_pos_running_schedule", "flat"
            ),
            "effector_hold_pos_schedule": getattr(
                args, "effector_hold_pos_schedule", "flat"
            ),
            "position_powerlaw_power": getattr(args, "position_powerlaw_power", 6.0),
            "movement_ramp_shape": getattr(args, "movement_ramp_shape", "linear"),
            "movement_ramp_duration_steps": getattr(
                args, "movement_ramp_duration_steps", 60
            ),
            "movement_ramp_power": getattr(args, "movement_ramp_power", 2.0),
        },
        "loss_update": {
            "enabled": args.loss_update_enabled,
            "target_ratio": args.loss_update_ratio,
            "alpha": 0.005,
            "control_term": "nn_output",
            "goal_term": ["effector_pos_running", "effector_pos_late"],
            "start_iteration": 0,
        },
        "where": {
            0: _trainable_paths_for_hidden_type(args.hidden_type, args.sisu_gating),
        },
        # hidden_type is a callable (class or partial), not serialisable to JSON.
        # It is resolved here from the CLI string and stored directly in the namespace.
        "hidden_type": _resolve_hidden_type(args.hidden_type, dt),
        "sisu_gating": args.sisu_gating,
    }
    return dict_to_namespace(hps_dict, to_type=TreeNamespace)
