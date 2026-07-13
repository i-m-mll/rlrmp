"""Compact canonical heterogeneous C&S training-matrix authoring."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.run_matrix import MatrixRow, TrainingRunMatrixSpec
from feedbax.contracts.spec_storage import training_spec_sha256
from feedbax.contracts.training import MethodPayloadEnvelope, TrainingRunSpec

from rlrmp.runtime.training_run_specs import (
    CS_SUPERVISED_METHOD_PAYLOAD_SCHEMA_ID,
    CS_SUPERVISED_METHOD_PAYLOAD_SCHEMA_VERSION,
    cs_supervised_effective_phase_spec,
    cs_supervised_method_contract,
    cs_supervised_method_ref,
)


TrainingArchitecture = Literal["gru", "time_constrained_free_gain", "linear_recurrence"]
TrainingDistribution = Literal["nominal", "broad_epsilon_pgd"]

ARCHITECTURES: tuple[TrainingArchitecture, ...] = (
    "gru",
    "time_constrained_free_gain",
    "linear_recurrence",
)
DISTRIBUTIONS: tuple[TrainingDistribution, ...] = ("nominal", "broad_epsilon_pgd")
COMPACT_ROW_OVERRIDE_PATHS = frozenset(
    {
        "config.controller_architecture",
        "config.broad_epsilon_pgd_training",
        "config.output_dir",
        "config.spec_dir",
    }
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
            "controller_architecture": "gru",
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
            "controller_architecture": "gru",
            "training_distribution": training_distribution,
        },
    )
    method_contract = cs_supervised_method_contract()
    return base.model_copy(
        update={
            "method_ref": cs_supervised_method_ref(),
            "method_payload": method_payload,
            "worker_execution": base.worker_execution.model_copy(
                update={
                    "method_contract": method_contract,
                    "effective_phase": cs_supervised_effective_phase_spec(method_contract),
                    "metadata": {
                        **base.worker_execution.metadata,
                        "native_executor": "feedbax.training.executor.execute_training_run_spec",
                        "controller_architecture": "gru",
                    },
                }
            ),
            "metadata": {
                **base.metadata,
                "architecture": "gru",
                "controller_architecture": "gru",
                "certificate_mode": "empirical_nonlinear",
                "training_distribution": "broad_epsilon" if robust_enabled else "nominal",
                "training_method_distribution": training_distribution,
                "native_method": "rlrmp/cs_supervised/v1",
                "serialize_do_not_rederive": True,
            },
        }
    )


def author_training_run_matrix(
    base_intent: Mapping[str, Any],
    *,
    issue: str,
    base_ref: Path,
    repo_root: Path,
) -> TrainingRunMatrixSpec:
    """Author six compact rows for the generic RLRMP row-lowering contract.

    The base remains authored intent. Rows select only architecture, training
    distribution, and disjoint custody roots; the generic lowerer owns graph,
    task, method-payload, and worker-execution construction.
    """

    config = base_intent.get("config")
    if not isinstance(config, Mapping) or "controller_architecture" not in config:
        raise ValueError("compact heterogeneous base requires typed config.controller_architecture")
    relative_ref = base_ref.resolve().relative_to(repo_root.resolve())
    rows: list[MatrixRow] = []
    for architecture in ARCHITECTURES:
        for distribution in DISTRIBUTIONS:
            row_id = f"{architecture}.{distribution}"
            artifact_root = f"_artifacts/{issue}/runs/{row_id}"
            tracked_spec_dir = f"results/{issue}/runs/{row_id}"
            rows.append(
                MatrixRow(
                    row_id=row_id,
                    overrides=[
                        OverridePatch(
                            op="replace",
                            path="config.controller_architecture",
                            value=architecture,
                        ),
                        OverridePatch(
                            op="replace",
                            path="config.broad_epsilon_pgd_training",
                            value=distribution == "broad_epsilon_pgd",
                        ),
                        OverridePatch(
                            op="replace",
                            path="config.output_dir",
                            value=artifact_root,
                        ),
                        OverridePatch(
                            op="replace",
                            path="config.spec_dir",
                            value=tracked_spec_dir,
                        ),
                    ],
                    metadata={
                        "controller_architecture": architecture,
                        "training_distribution": distribution,
                    },
                )
            )
    return TrainingRunMatrixSpec(
        name="Canonical C&S architecture by training-distribution matrix",
        issue=issue,
        base={
            "kind": "authored_intent",
            "ref": str(relative_ref),
            "content_hash": training_spec_sha256(base_intent),
            "symbolic_name": f"{issue}.canonical_cs_training",
        },
        rows=rows,
        tags=["rlrmp", "cs2019", "heterogeneous_architecture", "canonical_training"],
        metadata={
            "authoring_stage": "compact_intent_before_registered_row_lowering",
            "required_row_lowering_contract": "rlrmp.heterogeneous_cs_architecture.v1",
            "expanded_payload_patching": False,
            "experiment_local_callbacks": False,
            "compiler_edits": False,
        },
    )


__all__ = [
    "ARCHITECTURES",
    "COMPACT_ROW_OVERRIDE_PATHS",
    "DISTRIBUTIONS",
    "TrainingArchitecture",
    "TrainingDistribution",
    "author_gru_training_base",
    "author_training_run_matrix",
]
