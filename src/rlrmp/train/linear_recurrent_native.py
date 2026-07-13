"""Authoring contract for canonical linear-recurrent C&S training rows."""

from __future__ import annotations

from typing import Any, Literal

from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.training import MethodPayloadEnvelope, TrainingRunSpec

from rlrmp.runtime.training_run_specs import (
    CS_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
    CS_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
    cs_supervised_effective_phase_spec,
    cs_supervised_method_contract,
    cs_supervised_method_ref,
)


LINEAR_RECURRENT_ARCHITECTURE = "linear_recurrence"
LINEAR_RECURRENT_CERTIFICATE_MODE = "augmented_linear"
LINEAR_RECURRENT_HIDDEN_TYPE = "VanillaRNN"
LINEAR_RECURRENT_KERNEL_OWNER = "rlrmp.train.cs_nominal_gru"
LINEAR_RECURRENT_NATIVE_METHOD = "rlrmp/cs_supervised/v1"
LINEAR_RECURRENT_RUNNER = "rlrmp.train.orchestrated_row"
LINEAR_RECURRENT_TRAINING_DISTRIBUTIONS = ("nominal", "broad_epsilon_pgd")

_COMPONENT_INPUTS = {
    "augmented_states": {
        "source": "EvaluationRunManifest.cached_states",
        "construction": (
            "concatenate(controller_visible_target_relative_post_step_coupled_state, "
            "previous_step_hidden_state)"
        ),
        "basis": [
            "controller_visible_target_relative_post_step_coupled_state",
            "previous_step_hidden_state",
        ],
        "state_history_timing": "feedbax_post_step_history_pair",
        "layout": "batch_time_state",
    },
    "candidate_augmented_action_sensitivity": {
        "source": "trained_controller_graph",
        "construction": (
            "[readout.weight @ alpha * cell.weight_ih @ observation_map, "
            "readout.weight @ ((1 - alpha) * I + alpha * cell.weight_hh)]"
        ),
        "layout": "time_action_augmented_state",
    },
    "reference_augmented_action_sensitivity": {
        "source": "standard_certificate_reference",
        "construction": "reference action map embedded in [plant_state, hidden_state] basis",
        "layout": "time_action_augmented_state",
        "owner": "evaluation",
    },
    "candidate_transition": {
        "source": "trained_controller_graph_plus_mechanics",
        "construction": (
            "[[A_target_relative + B @ K_x, B @ K_h], "
            "[alpha * cell.weight_ih @ target_relative_observation_map, "
            "(1 - alpha) * I + alpha * cell.weight_hh]]; "
            "K_x=readout.weight @ alpha * cell.weight_ih @ "
            "target_relative_observation_map; "
            "K_h=readout.weight @ ((1 - alpha) * I + alpha * cell.weight_hh); "
            "alpha=cell.dt/cell.tau"
        ),
        "layout": "time_augmented_state_augmented_state",
    },
    "reference_transition": {
        "source": "standard_certificate_reference",
        "layout": "time_augmented_state_augmented_state",
        "owner": "evaluation",
    },
    "candidate_value_matrices": {
        "source": "augmented_closed_loop_value_solution",
        "layout": "time_augmented_state_augmented_state",
        "owner": "evaluation",
    },
    "reference_value_matrices": {
        "source": "standard_certificate_reference",
        "layout": "time_augmented_state_augmented_state",
        "owner": "evaluation",
    },
    "bellman_hessian": {
        "source": "standard_certificate_reference",
        "layout": "time_action_action",
        "owner": "evaluation",
    },
    "recurrence_diagnostics": {
        "source": "trained_controller_graph",
        "fields": ["recurrent_spectral_radius", "hidden_dim", "observation_dim"],
    },
}


def author_linear_recurrent_training_base(
    base: TrainingRunSpec,
    *,
    graph_spec: GraphSpec,
    training_distribution: Literal["nominal", "broad_epsilon_pgd"],
) -> TrainingRunSpec:
    """Derive a complete recurrent base from the canonical authored C&S base.

    This pure authoring transform preserves the C&S plant, task, objective,
    stochastic runtime, optimizer, checkpoint, and custody contracts.  It only
    replaces the controller architecture with Feedbax's registered
    ``VanillaRNN`` cell declared with identity activation, so the learned
    recurrence is genuinely linear.
    """

    if training_distribution not in LINEAR_RECURRENT_TRAINING_DISTRIBUTIONS:
        raise ValueError(f"unsupported linear-recurrent distribution {training_distribution!r}")
    _validate_linear_recurrent_graph(graph_spec)

    source_payload = dict(base.method_payload.payload)
    config = dict(source_payload.get("config") or {})
    if not config:
        raise ValueError("canonical C&S base lacks governed runtime config")
    robust_enabled = training_distribution == "broad_epsilon_pgd"
    config.update(
        {
            "controller_architecture": LINEAR_RECURRENT_ARCHITECTURE,
            "broad_epsilon_pgd_training": robust_enabled,
            "adaptive_epsilon_curriculum": False,
            "policy_adversary_training": False,
            "initial_hidden_encoder": False,
            "resume": False,
            "allow_fresh_start": True,
        }
    )
    for inherited_key in (
        "source_checkpoint_root",
        "source_checkpoint_transaction_id",
        "lr_continuation_mode",
        "lr_continuation_schedule",
    ):
        config.pop(inherited_key, None)
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
    if robust_enabled:
        payload["pre_step"] = {
            "kind": "broad_epsilon_pgd",
            "enabled": True,
            "config": dict(config.get("broad_epsilon_pgd") or {}),
        }

    method_payload = MethodPayloadEnvelope(
        schema_id=CS_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
        schema_version=CS_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
        payload=payload,
        metadata={
            **base.method_payload.metadata,
            "architecture": LINEAR_RECURRENT_ARCHITECTURE,
            "controller_architecture": LINEAR_RECURRENT_ARCHITECTURE,
            "native_method": LINEAR_RECURRENT_NATIVE_METHOD,
            "runner": LINEAR_RECURRENT_RUNNER,
            "training_distribution": training_distribution,
        },
    )
    graph = base.graph.model_copy(
        update={
            "inline": graph_spec.model_dump(mode="json", exclude_none=True),
            "ref": None,
            "metadata": {
                **base.graph.metadata,
                "architecture": LINEAR_RECURRENT_ARCHITECTURE,
                "controller_architecture": LINEAR_RECURRENT_ARCHITECTURE,
                "certificate_mode": LINEAR_RECURRENT_CERTIFICATE_MODE,
                "native_method": LINEAR_RECURRENT_NATIVE_METHOD,
                "runner": LINEAR_RECURRENT_RUNNER,
            },
        }
    )
    method_contract = cs_supervised_method_contract()
    method_ref = cs_supervised_method_ref()
    return base.model_copy(
        update={
            "graph": graph,
            "training_config": base.training_config.model_copy(
                update={"network_type": LINEAR_RECURRENT_ARCHITECTURE}
            ),
            "method_ref": method_ref,
            "method_payload": method_payload,
            "method_extensions": base.method_extensions.model_copy(
                update={
                    "metadata": {
                        **base.method_extensions.metadata,
                        "runner": LINEAR_RECURRENT_RUNNER,
                        "architecture": LINEAR_RECURRENT_ARCHITECTURE,
                        "controller_architecture": LINEAR_RECURRENT_ARCHITECTURE,
                        "native_method": LINEAR_RECURRENT_NATIVE_METHOD,
                    }
                }
            ),
            "worker_execution": base.worker_execution.model_copy(
                update={
                    "method_contract": method_contract,
                    "effective_phase": cs_supervised_effective_phase_spec(method_contract),
                    "checkpoint_slots": None,
                    "resume": None,
                    "progress": None,
                    "metadata": {
                        **base.worker_execution.metadata,
                        "native_executor": ("feedbax.training.executor.execute_training_run_spec"),
                        "kernel_owner": LINEAR_RECURRENT_KERNEL_OWNER,
                        "architecture": LINEAR_RECURRENT_ARCHITECTURE,
                        "controller_architecture": LINEAR_RECURRENT_ARCHITECTURE,
                        "native_method": LINEAR_RECURRENT_NATIVE_METHOD,
                        "runner": LINEAR_RECURRENT_RUNNER,
                    },
                }
            ),
            "checkpoint_progress": base.checkpoint_progress.model_copy(
                update={
                    "resume_from": None,
                    "checkpoint_slots": None,
                    "continuation": None,
                }
            ),
            "metadata": {
                **{
                    key: value
                    for key, value in base.metadata.items()
                    if key
                    not in {
                        "source_checkpoint_root",
                        "source_checkpoint_transaction_id",
                        "lr_continuation_schedule",
                    }
                },
                "architecture": LINEAR_RECURRENT_ARCHITECTURE,
                "controller_architecture": LINEAR_RECURRENT_ARCHITECTURE,
                "native_method": LINEAR_RECURRENT_NATIVE_METHOD,
                "runner": LINEAR_RECURRENT_RUNNER,
                "training_distribution": ("broad_epsilon" if robust_enabled else "nominal"),
                "training_method_distribution": training_distribution,
                "certificate_mode": LINEAR_RECURRENT_CERTIFICATE_MODE,
                "certificate_contract": {
                    "architecture": LINEAR_RECURRENT_ARCHITECTURE,
                    "mode": LINEAR_RECURRENT_CERTIFICATE_MODE,
                    "augmented_state_basis": [
                        "controller_visible_target_relative_post_step_coupled_state",
                        "previous_step_hidden_state",
                    ],
                    "state_history_timing": "feedbax_post_step_history_pair",
                    "affine_terms": "forbidden",
                    "cell_bias": "absent",
                    "readout_bias": "absent",
                    "initial_hidden_state": "zero",
                    "recurrence": "zero_bias_leaky_identity_activation",
                    "component_provider": "rlrmp.eval.linear_recurrent_augmented",
                    "component_inputs": _COMPONENT_INPUTS,
                    "static_gain_coercion": "forbidden",
                },
                "serialize_do_not_rederive": True,
            },
        }
    )


def author_linear_recurrent_training_base_from_canonical(
    base: TrainingRunSpec,
    *,
    training_distribution: Literal["nominal", "broad_epsilon_pgd"] = "nominal",
) -> TrainingRunSpec:
    """Build a complete recurrent base through the canonical C&S graph road."""

    from rlrmp.train.execution_preparation import _runtime_config
    from rlrmp.train.run_spec_authoring import build_training_run_graph_spec

    source_payload = dict(base.method_payload.payload)
    config = dict(source_payload.get("config") or {})
    if not config:
        raise ValueError("canonical C&S base lacks governed runtime config")
    config.update(
        {
            "controller_architecture": LINEAR_RECURRENT_ARCHITECTURE,
            "broad_epsilon_pgd_training": (training_distribution == "broad_epsilon_pgd"),
            "adaptive_epsilon_curriculum": False,
            "policy_adversary_training": False,
            "initial_hidden_encoder": False,
        }
    )
    _args, hps = _runtime_config(config)
    seed = int(config.get("seed", base.metadata.get("seed", 0)))
    graph_spec = build_training_run_graph_spec(hps, seed=seed)
    return author_linear_recurrent_training_base(
        base,
        graph_spec=graph_spec,
        training_distribution=training_distribution,
    )


def linear_recurrent_architecture_metadata(spec: TrainingRunSpec) -> dict[str, Any]:
    """Return the architecture contract declared by a recurrent training base.

    This metadata describes how a later evaluation provider must construct the
    numeric augmented-certificate inputs. It is not itself ``component_kwargs``:
    those are cached evaluation arrays consumed by ``StandardCertificateRowRequest``.
    """

    if spec.metadata.get("controller_architecture") != LINEAR_RECURRENT_ARCHITECTURE:
        raise ValueError("TrainingRunSpec is not a linear-recurrent authored base")
    return {
        "architecture": LINEAR_RECURRENT_ARCHITECTURE,
        "training_distribution": spec.metadata["training_distribution"],
        "certificate_mode": LINEAR_RECURRENT_CERTIFICATE_MODE,
        "component_provider": dict(spec.metadata["certificate_contract"])["component_provider"],
        "component_input_contract": dict(spec.metadata["certificate_contract"])["component_inputs"],
    }


def _validate_linear_recurrent_graph(graph_spec: GraphSpec) -> None:
    net = graph_spec.nodes.get("net")
    if net is None or net.type != "Subgraph":
        raise ValueError("linear-recurrent C&S graph must contain a native Subgraph net")
    subgraph = graph_spec.subgraphs.get("net")
    if subgraph is None:
        raise ValueError("linear-recurrent C&S graph must inline its recurrent subgraph")
    cell = subgraph.nodes.get("cell")
    if cell is None or cell.type != LINEAR_RECURRENT_HIDDEN_TYPE:
        raise ValueError("linear-recurrent graph must use Feedbax VanillaRNN")
    if cell.params.get("activation") != "identity":
        raise ValueError("linear-recurrent graph cell activation must be identity")
    if cell.params.get("use_bias") is not False:
        raise ValueError("linear-recurrent graph cell bias must be disabled")
    if "h0_encoder" in subgraph.nodes:
        raise ValueError("linear-recurrent graph must not contain an h0 encoder")
    if "h0_context" in subgraph.input_ports or "h0_context" in subgraph.input_bindings:
        raise ValueError("linear-recurrent graph must initialize hidden state from zero")
    readout = subgraph.nodes.get("readout")
    if readout is None or readout.type != "Linear":
        raise ValueError("linear-recurrent graph must use a linear action readout")
    if readout.params.get("use_bias") is not False:
        raise ValueError("linear-recurrent graph readout bias must be disabled")


__all__ = [
    "LINEAR_RECURRENT_ARCHITECTURE",
    "LINEAR_RECURRENT_CERTIFICATE_MODE",
    "LINEAR_RECURRENT_HIDDEN_TYPE",
    "LINEAR_RECURRENT_KERNEL_OWNER",
    "LINEAR_RECURRENT_NATIVE_METHOD",
    "LINEAR_RECURRENT_RUNNER",
    "LINEAR_RECURRENT_TRAINING_DISTRIBUTIONS",
    "author_linear_recurrent_training_base",
    "author_linear_recurrent_training_base_from_canonical",
    "linear_recurrent_architecture_metadata",
]
