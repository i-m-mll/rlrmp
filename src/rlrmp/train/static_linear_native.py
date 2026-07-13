"""Authoring contract for canonical static-linear C&S training rows."""

from __future__ import annotations

from typing import Any, Literal

from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.training import MethodPayloadEnvelope, TrainingRunSpec

from rlrmp.model.cs_lss_static_linear import STATIC_LINEAR_CONTROLLER_KIND
from rlrmp.runtime.training_run_specs import (
    CS_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
    CS_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
    cs_supervised_effective_phase_spec,
    cs_supervised_method_contract,
    cs_supervised_method_ref,
)


STATIC_LINEAR_CERTIFICATE_MODE = "static_gain"
STATIC_LINEAR_BRIDGE_ARCHITECTURE = "time_constrained_free_gain"
STATIC_LINEAR_FEEDBACK_BASIS = "target_relative_delayed_feedback_plus_force_filter"
STATIC_LINEAR_NATIVE_METHOD = "rlrmp/cs_supervised/v1"
STATIC_LINEAR_RUNNER = "rlrmp.train.orchestrated_row"
STATIC_LINEAR_TRAINING_DISTRIBUTIONS = ("nominal", "broad_epsilon_pgd")
STATIC_LINEAR_CERTIFICATE_COMPONENT_INPUTS = (
    "states",
    "action_states",
    "candidate_gain",
    "reference_gain",
    "candidate_transition",
    "reference_transition",
    "candidate_value_matrices",
    "reference_value_matrices",
    "bellman_hessian",
)


def _bridge_training_distribution(
    training_distribution: Literal["nominal", "broad_epsilon_pgd"],
) -> Literal["nominal", "broad_epsilon"]:
    """Map the execution arm to the grouped-certificate vocabulary."""

    return "broad_epsilon" if training_distribution == "broad_epsilon_pgd" else "nominal"


def _certificate_contract(
    training_distribution: Literal["nominal", "broad_epsilon_pgd"],
) -> dict[str, Any]:
    """Return the manifest-native static certificate handoff declaration."""

    return {
        "architecture": STATIC_LINEAR_BRIDGE_ARCHITECTURE,
        "mode": STATIC_LINEAR_CERTIFICATE_MODE,
        "training_distribution": _bridge_training_distribution(training_distribution),
        "state_basis": "coupled_closed_loop_state",
        "action_basis": STATIC_LINEAR_FEEDBACK_BASIS,
        "trainable_paths": ["nodes.net.gain"],
        "memory": "none",
        "required_component_inputs": list(STATIC_LINEAR_CERTIFICATE_COMPONENT_INPUTS),
        "candidate_gain_source": {
            "parent_role": "training_checkpoint_custody",
            "slot": "model",
            "trainable_path": "nodes.net.gain",
        },
    }


def _without_source_checkpoint_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Remove provenance that would turn a canonical base into a continuation."""

    inherited_keys = {
        "checkpoint_dir",
        "latest_checkpoint",
        "latest_pointer",
        "lr_continuation_schedule",
        "source_checkpoint_root",
        "source_checkpoint_transaction_id",
    }
    return {key: value for key, value in metadata.items() if key not in inherited_keys}


def _validate_canonical_static_graph(graph_spec: GraphSpec) -> None:
    """Require the C&S target-relative proprioceptive action basis."""

    net = graph_spec.nodes.get("net")
    if net is None or net.type != "AffineFeedbackController":
        raise ValueError("static-linear base graph must contain an AffineFeedbackController net")
    if net.params.get("bias") is not None or net.params.get("feedforward") is not None:
        raise ValueError("canonical static-linear controller must be gain-only")
    gain = net.params.get("gain")
    if not isinstance(gain, list) or len(gain) != 2 or any(len(row) != 6 for row in gain):
        raise ValueError("canonical static-linear gain must have shape (2, 6)")

    feedback = graph_spec.nodes.get("feedback")
    channels = [] if feedback is None else feedback.params.get("channels", [])
    transforms = [channel.get("transform") for channel in channels]
    if transforms != ["target_minus", "negate", "identity"]:
        raise ValueError(
            "canonical static-linear feedback must be target-relative delayed position, "
            "negated delayed velocity, and delayed force/filter state"
        )


def static_linear_runtime_config(config: dict[str, Any]) -> tuple[Any, Any]:
    """Materialize runtime args and force the canonical static controller kind."""

    from rlrmp.train.config_materialization import _config_namespace, build_hps

    args = _config_namespace(config)
    return args, build_hps(args) | {"hidden_type": STATIC_LINEAR_CONTROLLER_KIND}


def author_static_linear_training_base(
    base: TrainingRunSpec,
    *,
    graph_spec: GraphSpec,
    training_distribution: Literal["nominal", "broad_epsilon_pgd"],
) -> TrainingRunSpec:
    """Return a content-pinnable static-linear base from a canonical C&S base.

    This authoring transform is deliberately outside matrix compilation. The
    resulting document is already a complete ``TrainingRunSpec``: matrices may
    vary seeds, optimizer values, and artifact routes without patching expanded
    execution payloads or invoking an experiment-local callback.
    """

    if training_distribution not in STATIC_LINEAR_TRAINING_DISTRIBUTIONS:
        raise ValueError(f"unsupported static-linear distribution {training_distribution!r}")
    _validate_canonical_static_graph(graph_spec)

    source_payload = dict(base.method_payload.payload)
    config = dict(source_payload.get("config") or {})
    config["controller_architecture"] = STATIC_LINEAR_CONTROLLER_KIND
    robust_enabled = training_distribution == "broad_epsilon_pgd"
    config["broad_epsilon_pgd_training"] = robust_enabled
    config["adaptive_epsilon_curriculum"] = False
    config["policy_adversary_training"] = False
    config["resume"] = False
    config["allow_fresh_start"] = True
    payload: dict[str, Any] = {
        "config": config,
        "training_mode": training_distribution,
        "n_train_batches": int(base.training_config.n_batches),
        "batch_size": int(base.training_config.batch_size),
        "optimizer_policy": {
            "controller_lr": float(base.training_config.learning_rate),
            "lr_schedule": config.get("lr_schedule"),
        },
        "gradient_clip_norm": base.training_config.grad_clip,
        "training_diagnostics": {
            "enabled": bool(config.get("training_diagnostics", True)),
            "custody": "checkpoint_barrier_artifact_sink",
        },
        "checkpoint_policy": {
            "checkpoint_interval_batches": int(
                base.checkpoint_progress.checkpoint_interval
                or base.training_config.snapshot_interval
            ),
            "artifact_root": base.artifacts.artifact_root,
            "tracked_spec_dir": str(base.artifacts.metadata.get("tracked_spec_dir", "results")),
        },
    }
    pre_step: dict[str, Any] = {}
    if robust_enabled:
        pre_step.update(
            {
                "kind": "broad_epsilon_pgd",
                "enabled": True,
                "config": dict(config.get("broad_epsilon_pgd") or {}),
            }
        )
        payload["pre_step"] = pre_step
    else:
        payload.pop("pre_step", None)

    method_payload = MethodPayloadEnvelope(
        schema_id=CS_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
        schema_version=CS_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
        payload=payload,
        metadata={
            **base.method_payload.metadata,
            "controller_architecture": STATIC_LINEAR_BRIDGE_ARCHITECTURE,
            "controller_kind": STATIC_LINEAR_CONTROLLER_KIND,
            "native_method": STATIC_LINEAR_NATIVE_METHOD,
            "runner": STATIC_LINEAR_RUNNER,
            "training_distribution": training_distribution,
        },
    )
    graph = base.graph.model_copy(
        update={
            "inline": graph_spec.model_dump(mode="json", exclude_none=True),
            "ref": None,
            "metadata": {
                **base.graph.metadata,
                "controller_architecture": STATIC_LINEAR_BRIDGE_ARCHITECTURE,
                "controller_kind": STATIC_LINEAR_CONTROLLER_KIND,
                "certificate_mode": STATIC_LINEAR_CERTIFICATE_MODE,
                "native_method": STATIC_LINEAR_NATIVE_METHOD,
                "runner": STATIC_LINEAR_RUNNER,
            },
        }
    )
    training_config = base.training_config.model_copy(
        update={"network_type": STATIC_LINEAR_CONTROLLER_KIND}
    )
    method_contract = cs_supervised_method_contract()
    worker_execution = base.worker_execution.model_copy(
        update={
            "method_contract": method_contract,
            "effective_phase": cs_supervised_effective_phase_spec(method_contract),
            "checkpoint_slots": None,
            "resume": None,
            "progress": None,
            "metadata": {
                **base.worker_execution.metadata,
                "native_executor": "feedbax.training.executor.execute_training_run_spec",
                "kernel_owner": "rlrmp.train.cs_nominal_gru",
                "controller_architecture": STATIC_LINEAR_BRIDGE_ARCHITECTURE,
                "controller_kind": STATIC_LINEAR_CONTROLLER_KIND,
                "native_method": STATIC_LINEAR_NATIVE_METHOD,
                "runner": STATIC_LINEAR_RUNNER,
            },
        }
    )
    return base.model_copy(
        update={
            "graph": graph,
            "training_config": training_config,
            "method_ref": cs_supervised_method_ref(),
            "method_payload": method_payload,
            "method_extensions": base.method_extensions.model_copy(
                update={
                    "metadata": {
                        **base.method_extensions.metadata,
                        "runner": STATIC_LINEAR_RUNNER,
                        "controller_architecture": STATIC_LINEAR_BRIDGE_ARCHITECTURE,
                        "controller_kind": STATIC_LINEAR_CONTROLLER_KIND,
                        "native_method": STATIC_LINEAR_NATIVE_METHOD,
                    }
                }
            ),
            "worker_execution": worker_execution,
            "checkpoint_progress": base.checkpoint_progress.model_copy(
                update={
                    "resume_from": None,
                    "checkpoint_slots": None,
                    "continuation": None,
                    "metadata": _without_source_checkpoint_metadata(
                        base.checkpoint_progress.metadata
                    ),
                }
            ),
            "metadata": {
                **_without_source_checkpoint_metadata(base.metadata),
                "architecture": STATIC_LINEAR_BRIDGE_ARCHITECTURE,
                "controller_architecture": STATIC_LINEAR_BRIDGE_ARCHITECTURE,
                "controller_kind": STATIC_LINEAR_CONTROLLER_KIND,
                "certificate_mode": STATIC_LINEAR_CERTIFICATE_MODE,
                "training_distribution": _bridge_training_distribution(training_distribution),
                "training_method_distribution": training_distribution,
                "certificate_contract": _certificate_contract(training_distribution),
                "native_method": STATIC_LINEAR_NATIVE_METHOD,
                "runner": STATIC_LINEAR_RUNNER,
                "serialize_do_not_rederive": True,
            },
        }
    )


def author_static_linear_training_base_from_canonical(
    base: TrainingRunSpec,
    *,
    training_distribution: Literal["nominal", "broad_epsilon_pgd"] = "nominal",
) -> TrainingRunSpec:
    """Lower a canonical C&S base into a complete static-linear base document."""

    from rlrmp.train.run_spec_authoring import build_training_run_graph_spec

    payload = dict(base.method_payload.payload)
    config = dict(payload.get("config") or {})
    if not config:
        raise ValueError("canonical C&S base lacks governed runtime config")
    config["controller_architecture"] = STATIC_LINEAR_CONTROLLER_KIND
    config["broad_epsilon_pgd_training"] = training_distribution == "broad_epsilon_pgd"
    config["adaptive_epsilon_curriculum"] = False
    config["policy_adversary_training"] = False
    _args, hps = static_linear_runtime_config(config)
    seed = int(config.get("seed", base.metadata.get("seed", 0)))
    graph_spec = build_training_run_graph_spec(hps, seed=seed)
    return author_static_linear_training_base(
        base,
        graph_spec=graph_spec,
        training_distribution=training_distribution,
    )


def static_linear_architecture_metadata(spec: TrainingRunSpec) -> dict[str, Any]:
    """Return the manifest handoff metadata for a validated static-linear base."""

    if spec.training_config.network_type != STATIC_LINEAR_CONTROLLER_KIND:
        raise ValueError("TrainingRunSpec is not a static-linear authored base")
    if spec.metadata.get("controller_architecture") != STATIC_LINEAR_BRIDGE_ARCHITECTURE:
        raise ValueError("static-linear base lacks the canonical bridge architecture")
    return {
        "architecture": STATIC_LINEAR_BRIDGE_ARCHITECTURE,
        "controller_architecture": STATIC_LINEAR_BRIDGE_ARCHITECTURE,
        "controller_kind": STATIC_LINEAR_CONTROLLER_KIND,
        "certificate_mode": STATIC_LINEAR_CERTIFICATE_MODE,
        "training_distribution": spec.metadata["training_distribution"],
        "certificate_contract": dict(spec.metadata["certificate_contract"]),
    }


__all__ = [
    "STATIC_LINEAR_BRIDGE_ARCHITECTURE",
    "STATIC_LINEAR_CERTIFICATE_MODE",
    "STATIC_LINEAR_CERTIFICATE_COMPONENT_INPUTS",
    "STATIC_LINEAR_FEEDBACK_BASIS",
    "STATIC_LINEAR_NATIVE_METHOD",
    "STATIC_LINEAR_RUNNER",
    "STATIC_LINEAR_TRAINING_DISTRIBUTIONS",
    "author_static_linear_training_base",
    "author_static_linear_training_base_from_canonical",
    "static_linear_runtime_config",
    "static_linear_architecture_metadata",
]
