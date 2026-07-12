"""Native minimax method contract and hyperparameter construction."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import equinox as eqx
from feedbax.contracts.training import (
    DEFAULT_TRAINING_METHOD_REGISTRY,
    GraphTopologySourceSpec,
    MethodPayloadEnvelope,
    MethodRefSpec,
    ObjectiveSlotSpec,
    TaskSpec,
    TrainingConfig,
    TrainingMethodRegistration,
    TrainingRunSpec,
)
from feedbax.training.checkpoint_custody import load_latest_checkpoint
from feedbax.contracts.worker import (
    AxisSpec,
    CheckpointBarrierSpec,
    EffectivePhaseSpec,
    MetricGuardSpec,
    MethodContractSpec,
    OptimizerTargetBinding,
    PhaseProgramSpec,
    PhaseSpec,
    PhaseTransitionSpec,
    StateSlotSpec,
    UpdateKernelSpec,
    UpdateStepSpec,
    derive_consistency_predicate,
)
from feedbax.config.namespace import TreeNamespace, dict_to_namespace
from pydantic import BaseModel, ConfigDict

from rlrmp.model.feedbax_graph import graph_spec_payload
from rlrmp.model.trainable import staged_network_trainable_paths
from rlrmp.runtime.jax_config import assert_jax_x64_disabled
from rlrmp.runtime.training_run_specs import build_training_run_spec_scaffold
from rlrmp.train.executor.slots import minimax_checkpoint_slot_specs
from rlrmp.train.resume_control import emit_launch_continuation, resolve_launch_continuation
from rlrmp.train.training_configs import MinimaxConfig

logger = logging.getLogger("rlrmp.train.minimax_native")

__all__ = [
    "MINIMAX_METHOD_REF",
    "MINIMAX_METHOD_PAYLOAD_SCHEMA_ID",
    "MINIMAX_METHOD_PAYLOAD_SCHEMA_VERSION",
    "MinimaxConfig",
    "MinimaxMethodPayload",
    "build_hps",
    "build_minimax_training_run_spec",
    "build_minimax_native_initial_slots",
    "ensure_minimax_training_method_registered",
    "execute_minimax_training_run_spec_native",
    "verify_minimax_checkpoint_resume",
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


class MinimaxMethodPayload(BaseModel):
    """Minimal native method payload owned by the unified config model."""

    model_config = ConfigDict(extra="forbid")

    config: MinimaxConfig


def build_minimax_training_run_spec(
    config: Mapping[str, Any] | MinimaxConfig,
    *,
    graph_spec: Any,
    output_dir: Path,
    spec_dir: Path,
    feedbax_graph: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the tracked minimax recipe with a composed Feedbax TrainingRunSpec."""

    ensure_minimax_training_method_registered()
    normalized = (
        config if isinstance(config, MinimaxConfig) else MinimaxConfig.model_validate(dict(config))
    )
    payload: dict[str, Any] = {
        "schema_id": "rlrmp.minimax.native_run_spec",
        "schema_version": "rlrmp.minimax.native_run_spec.v1",
        "config": normalized.model_dump(mode="json"),
        "feedbax_graph": dict(feedbax_graph or {}),
    }
    method_payload = MethodPayloadEnvelope(
        schema_id=MINIMAX_METHOD_PAYLOAD_SCHEMA_ID,
        schema_version=MINIMAX_METHOD_PAYLOAD_SCHEMA_VERSION,
        payload=MinimaxMethodPayload(config=normalized).model_dump(mode="json"),
    )
    contract = minimax_method_contract()
    effective_phase = minimax_effective_phase_spec(contract)
    fingerprint = minimax_effective_phase_fingerprint(
        effective_phase=effective_phase,
        graph_payload=graph_spec_payload(graph_spec),
        method_payload=method_payload.model_dump(mode="json", exclude_none=True),
    )
    checkpoint_interval = max(1, normalized.checkpoint_every or 1)
    scaffold = build_training_run_spec_scaffold(
        risk_metadata={"population_member": "active_member_only"},
        execution_mode="local",
        require_review=False,
        allow_cloud=False,
        execution_metadata={"entrypoint": "scripts/train_minimax.py"},
        artifact_root=str(output_dir),
        artifact_metadata={
            "tracked_spec_dir": str(spec_dir),
            "bulk_outputs": str(output_dir),
        },
        checkpoint_interval=checkpoint_interval,
        progress_interval=checkpoint_interval,
        checkpoint_metadata={
            "effective_phase_fingerprint": fingerprint,
            "checkpoint_custody_owner": "feedbax",
        },
        metadata={
            "composed_with": "rlrmp_run_spec",
            "effective_phase_fingerprint": fingerprint,
        },
    )
    feedbax_spec = scaffold.build(
        graph=GraphTopologySourceSpec(
            inline=graph_spec_payload(graph_spec),
            schema_id=getattr(graph_spec, "schema_id", None),
            schema_version=getattr(graph_spec, "schema_version", None),
            metadata={
                "source": "requested_serialized_graph_spec",
                "declarative_adversary_injection": (normalized.adversary_type == "linear_dynamics"),
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
            metadata={"lowering_owner": "rlrmp.train.minimax_native"},
        ),
        method_ref=MethodRefSpec(package="rlrmp", name="minimax", version="v1"),
        method_payload=method_payload,
        method_extensions={
            "metadata": {
                "scientific_semantics_owner": "rlrmp.train.adversary",
            }
        },
        method_contract=contract,
        effective_phase=effective_phase,
        worker_metadata={
            "effective_phase_fingerprint": fingerprint,
            "pre_execution_parity": "compare_requested_serialized_spec",
        },
    )
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


def build_minimax_native_initial_slots(
    *,
    run_spec: Mapping[str, Any] | TrainingRunSpec,
    hps: TreeNamespace,
    args: Any,
    key: Any,
):
    """Build minimax native-executor initial slots and runtime context."""

    from rlrmp.train.minimax_native.kernels import build_minimax_native_initial_slots as _build

    return _build(run_spec=run_spec, hps=hps, args=args, key=key)


def execute_minimax_training_run_spec_native(
    spec: TrainingRunSpec | Mapping[str, Any],
    *,
    run_id: str | None = None,
    key: Any | None = None,
    manifest_root: Path | str | None = None,
    checkpoint_root: Path | str | None = None,
    resume: bool = False,
    stop_after_barrier: str | None = None,
    **kwargs: Any,
):
    """Execute an RLRMP minimax TrainingRunSpec through Feedbax's native executor."""

    import jax.random as jr
    from feedbax.training.executor import execute_training_run_spec
    from rlrmp.train.executor.adapters import RLRMP_RUNTIME_CONTEXT_KEY
    from rlrmp.train.minimax_native.kernels import MinimaxExternalObjectiveLossService

    ensure_minimax_training_method_registered()
    training_spec = (
        spec if isinstance(spec, TrainingRunSpec) else TrainingRunSpec.model_validate(spec)
    )
    config = MinimaxConfig.model_validate(minimax_training_run_spec_to_config(training_spec))
    assert_jax_x64_disabled("minimax resume verification", allow_x64=config.allow_x64)
    hps = build_hps(config)
    initial_slots, runtime = build_minimax_native_initial_slots(
        run_spec=training_spec,
        hps=hps,
        args=config,
        key=key if key is not None else jr.PRNGKey(config.seed),
    )
    return execute_training_run_spec(
        training_spec,
        run_id=run_id,
        initial_slots=initial_slots,
        kernel_context={RLRMP_RUNTIME_CONTEXT_KEY: runtime},
        manifest_root=manifest_root,
        checkpoint_root=checkpoint_root,
        loss_service=kwargs.pop("loss_service", MinimaxExternalObjectiveLossService()),
        resume=resume,
        stop_after_barrier=stop_after_barrier,
        **kwargs,
    )


def verify_minimax_checkpoint_resume(
    spec: TrainingRunSpec | Mapping[str, Any],
) -> dict[str, Any]:
    """Load and strictly validate a minimax checkpoint without training."""

    import jax.random as jr

    training_spec = (
        spec if isinstance(spec, TrainingRunSpec) else TrainingRunSpec.model_validate(spec)
    )
    config = MinimaxConfig.model_validate(minimax_training_run_spec_to_config(training_spec))
    checkpoint_root = Path(config.output_dir) / "checkpoints_adversarial"
    continuation = resolve_launch_continuation(
        checkpoint_root=checkpoint_root,
        resume_requested=True,
        allow_fresh_start=False,
        stop_target_batches=config.n_warmup_batches + config.n_adversary_batches,
        completed_batches_from_latest=lambda path: _completed_minimax_batches(path, config),
    )
    emit_launch_continuation(continuation, logger=logger)
    initial_slots, _runtime = build_minimax_native_initial_slots(
        run_spec=training_spec,
        hps=build_hps(config),
        args=config,
        key=jr.PRNGKey(config.seed),
    )
    loaded = load_latest_checkpoint(
        checkpoint_root,
        expected_run_spec=training_spec,
        expected_phase_program=(
            training_spec.worker_execution.method_contract.phase_program
        ),
        expected_slots=initial_slots,
    )
    return {
        "verified_resume": True,
        "checkpoint_root": str(checkpoint_root),
        "transaction_id": loaded.manifest.transaction_id,
        "completed_batches": continuation.completed_batches,
        "continuation_batches": continuation.continuation_batches,
    }


def _completed_minimax_batches(path: Path, config: MinimaxConfig) -> int:
    coordinate = json.loads(path.read_text(encoding="utf-8")).get(
        "completed_coordinate", {}
    )
    total = config.n_warmup_batches + config.n_adversary_batches
    if coordinate.get("phase") == "done":
        return total
    if coordinate.get("completed_barrier") == "after_warmup":
        return config.n_warmup_batches
    if coordinate.get("completed_barrier") == "after_adversarial":
        return min(total, config.n_warmup_batches + int(coordinate.get("global_step", 0)))
    raise ValueError(f"unsupported minimax checkpoint coordinate in {path}: {coordinate!r}")


def validate_minimax_run_spec(
    run_spec: dict[str, Any],
    *,
    spec_dir: Path,
    require_graph_sidecars: bool = True,
) -> None:
    """Validate the tracked minimax run-spec contract."""

    del spec_dir, require_graph_sidecars
    ensure_minimax_training_method_registered()
    required = {
        "schema_id",
        "schema_version",
        "config",
        "feedbax_graph",
        "feedbax_training_run_spec",
    }
    missing = sorted(required - set(run_spec))
    if missing:
        raise ValueError("minimax run spec is missing required keys: " + ", ".join(missing))
    if run_spec["schema_id"] != "rlrmp.minimax.native_run_spec":
        raise ValueError(f"unsupported minimax run spec schema_id: {run_spec['schema_id']!r}")
    if run_spec["schema_version"] != "rlrmp.minimax.native_run_spec.v1":
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
            guard_predicates_factory=_minimax_guard_predicates,
            rejected_payload_versions=("rlrmp.spec.training_method.minimax_payload.v0",),
            owner="rlrmp.train.minimax_native",
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
                legal_next=["done", "adversarial"],
                checkpoint_barrier="after_adversarial",
                loop_axis="batch",
                metadata={
                    "activation_binding": "linear_dynamics_adversary_params.active=false",
                    "native_kernel_granularity": "full_warmup_phase",
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
                    "trial_batch",
                    "rng",
                    "controller_loss",
                    "adversary_loss",
                ],
                update_steps=[
                    "inner_adversary_ascent",
                    "adversary_projection",
                    "outer_controller_descent",
                ],
                legal_next=["done", "adversarial"],
                checkpoint_barrier="after_adversarial",
                loop_axis="batch",
                metadata={
                    "active_member": "program_step % n_adversaries",
                    "frozen_controller_boundary": (
                        "stop_gradient(controller) during inner_adversary_ascent"
                    ),
                    "kernel_variants": {
                        "fused": "same reads/writes/barriers as decomposed",
                        "decomposed": "same reads/writes/barriers as fused",
                    },
                    "native_kernel_granularity": "one_adversarial_batch",
                },
            ),
            PhaseSpec(
                name="done",
                kind="evaluation",
                reads=[
                    "controller",
                    "controller_optimizer",
                    "adversary_population",
                    "adversary_optimizer",
                    "rng",
                    "controller_loss",
                    "adversary_loss",
                ],
            ),
        ],
        initial_phase="warmup",
        transitions=[
            PhaseTransitionSpec(
                source="warmup",
                target="done",
                barrier="after_adversarial",
                guard=MetricGuardSpec(
                    predicate_ref="rlrmp.minimax.no_adversarial_batches",
                    metric_slots=["controller_loss"],
                ),
            ),
            PhaseTransitionSpec(
                source="warmup",
                target="adversarial",
                barrier="after_adversarial",
            ),
            PhaseTransitionSpec(
                source="adversarial",
                target="done",
                barrier="after_adversarial",
                guard=MetricGuardSpec(
                    predicate_ref="rlrmp.minimax.adversarial_complete",
                    metric_slots=["controller_loss", "adversary_loss"],
                ),
            ),
            PhaseTransitionSpec(
                source="adversarial",
                target="adversarial",
                barrier="after_adversarial",
            ),
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
                writes=[
                    "adversary_population",
                    "adversary_optimizer",
                    "trial_batch",
                    "rng",
                    "adversary_loss",
                ],
                axes=["inner_step", "adversary_member", "replicate"],
                optimizer_binding="adversary_optimizer_to_active_member",
                metadata={"direction": "maximize"},
            ),
            UpdateStepSpec(
                name="adversary_projection",
                kind="projection",
                kernel=UpdateKernelSpec(kernel_ref="rlrmp.minimax.frobenius_ball_projection"),
                reads=["adversary_population", "trial_batch"],
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
                reads=[
                    "controller",
                    "controller_optimizer",
                    "adversary_population",
                    "trial_batch",
                ],
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
                metadata={"active_member": "program_step % n_adversaries"},
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
                name="after_adversarial",
                phase="adversarial",
                slots=minimax_checkpoint_slot_specs(),
            ),
        ],
        metadata={
            "phase_program_identity": "rlrmp.minimax.warmup_then_adversarial.v2",
            "checkpoint_barrier_policy": "shared_after_each_minimax_program_step",
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
    encoded = json.dumps(parity, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _minimax_update_kernels(
    payload: BaseModel | None = None,
) -> Mapping[str, Any]:
    from rlrmp.train.minimax_native.kernels import minimax_update_kernels

    return minimax_update_kernels(payload)


def _minimax_guard_predicates(payload: BaseModel | None = None) -> Mapping[str, Any]:
    from rlrmp.train.minimax_native.kernels import minimax_guard_predicates

    return minimax_guard_predicates(payload)


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


def _build_hps_from_config(args: Any) -> TreeNamespace:
    """Materialize task/model hyperparameters from the validated config.

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
                "nn_hidden_derivative_pre_go": getattr(args, "nn_hidden_derivative_pre_go", 0.0),
            },
            "effector_pos_late": {
                "start_step_after_go": getattr(args, "effector_pos_late_start_step", 80),
                "final_scale_factor": getattr(args, "effector_pos_late_final_scale", 2.0),
            },
            "effector_vel_late": {
                "start_step_after_go": 80,
                "final_scale_factor": 1.0,
            },
            # Power-law schedule: "flat" (default) or "powerlaw" ((t/T-1)^power).
            # Bug: 2e1a6ad
            "effector_pos_running_schedule": getattr(args, "effector_pos_running_schedule", "flat"),
            "effector_hold_pos_schedule": getattr(args, "effector_hold_pos_schedule", "flat"),
            "position_powerlaw_power": getattr(args, "position_powerlaw_power", 6.0),
            "movement_ramp_shape": getattr(args, "movement_ramp_shape", "linear"),
            "movement_ramp_duration_steps": getattr(args, "movement_ramp_duration_steps", 60),
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


def build_hps(args: Any) -> TreeNamespace:
    """Validate the unified minimax config and materialize trainer hyperparameters."""

    raw = vars(args) if hasattr(args, "__dict__") else dict(args)
    config_payload = {name: raw[name] for name in MinimaxConfig.model_fields if name in raw}
    config = MinimaxConfig.model_validate(config_payload)
    return _build_hps_from_config(SimpleNamespace(**(config.model_dump() | raw)))
