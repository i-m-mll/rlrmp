"""Canonical GRU training-base authoring for C&S experiments."""

from __future__ import annotations

from typing import Any, Literal

from feedbax.contracts.training import MethodPayloadEnvelope, TrainingRunSpec

from rlrmp.runtime.training_run_specs import (
    CS_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
    CS_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
    cs_supervised_effective_phase_spec,
    cs_supervised_method_contract,
    cs_supervised_method_ref,
    require_cs_supervised_optimizer,
)


TrainingDistribution = Literal["nominal", "broad_epsilon_pgd"]

GRU_CONTROLLER_ARCHITECTURE = "gru"
GRU_KERNEL_OWNER = "rlrmp.train.cs_nominal_gru"
GRU_NATIVE_METHOD = "rlrmp/cs_supervised/v1"
GRU_RUNNER = "rlrmp.train.orchestrated_row"
TRAINING_DISTRIBUTIONS: tuple[TrainingDistribution, ...] = (
    "nominal",
    "broad_epsilon_pgd",
)


def author_gru_training_base(
    base: TrainingRunSpec,
    *,
    training_distribution: TrainingDistribution,
) -> TrainingRunSpec:
    """Derive a complete GRU row through the registered C&S method contract."""

    source_payload = dict(base.method_payload.payload)
    config = dict(source_payload.get("config") or {})
    if not config:
        raise ValueError("canonical C&S base lacks governed runtime config")
    robust_enabled = training_distribution == "broad_epsilon_pgd"
    config.update(
        {
            "controller_architecture": GRU_CONTROLLER_ARCHITECTURE,
            "broad_epsilon_pgd_training": robust_enabled,
            "adaptive_epsilon_curriculum": False,
            "policy_adversary_training": False,
        }
    )
    payload: dict[str, Any] = {
        "config": config,
        "training_mode": training_distribution,
        "n_train_batches": int(base.training_config.n_batches),
        "batch_size": int(base.training_config.batch_size),
        "optimizer": require_cs_supervised_optimizer(source_payload).model_dump(mode="json"),
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
            "controller_architecture": GRU_CONTROLLER_ARCHITECTURE,
            "native_method": GRU_NATIVE_METHOD,
            "runner": GRU_RUNNER,
            "training_distribution": training_distribution,
        },
    )
    method_contract = cs_supervised_method_contract()
    return base.model_copy(
        update={
            "graph": base.graph.model_copy(
                update={
                    "metadata": {
                        **base.graph.metadata,
                        "controller_architecture": GRU_CONTROLLER_ARCHITECTURE,
                        "native_method": GRU_NATIVE_METHOD,
                        "runner": GRU_RUNNER,
                    }
                }
            ),
            "method_ref": cs_supervised_method_ref(),
            "method_payload": method_payload,
            "method_extensions": base.method_extensions.model_copy(
                update={
                    "metadata": {
                        **base.method_extensions.metadata,
                        "controller_architecture": GRU_CONTROLLER_ARCHITECTURE,
                        "native_method": GRU_NATIVE_METHOD,
                        "runner": GRU_RUNNER,
                    }
                }
            ),
            "worker_execution": base.worker_execution.model_copy(
                update={
                    "method_contract": method_contract,
                    "effective_phase": cs_supervised_effective_phase_spec(method_contract),
                    "metadata": {
                        **base.worker_execution.metadata,
                        "native_executor": "feedbax.training.executor.execute_training_run_spec",
                        "kernel_owner": GRU_KERNEL_OWNER,
                        "controller_architecture": GRU_CONTROLLER_ARCHITECTURE,
                        "native_method": GRU_NATIVE_METHOD,
                        "runner": GRU_RUNNER,
                    },
                }
            ),
            "metadata": {
                **base.metadata,
                "architecture": GRU_CONTROLLER_ARCHITECTURE,
                "controller_architecture": GRU_CONTROLLER_ARCHITECTURE,
                "certificate_mode": "empirical_nonlinear",
                "training_distribution": "broad_epsilon" if robust_enabled else "nominal",
                "training_method_distribution": training_distribution,
                "native_method": GRU_NATIVE_METHOD,
                "runner": GRU_RUNNER,
                "serialize_do_not_rederive": True,
            },
        }
    )


__all__ = [
    "GRU_CONTROLLER_ARCHITECTURE",
    "GRU_KERNEL_OWNER",
    "GRU_NATIVE_METHOD",
    "GRU_RUNNER",
    "TRAINING_DISTRIBUTIONS",
    "TrainingDistribution",
    "author_gru_training_base",
]
