"""Feedbax TrainingRunSpec adapters for RLRMP training recipes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    ExecutionPolicySpec,
    GraphTopologySourceSpec,
    ObjectiveSlotSpec,
    RiskAggregationSpec,
    TaskSpec,
    TrainingConfig,
    TrainingRunSpec,
    WorkerExecutionSpec,
    standard_supervised_effective_phase_spec,
    standard_supervised_method_contract,
    standard_supervised_method_payload,
    standard_supervised_method_ref,
)

from rlrmp.model.feedbax_graph import graph_spec_payload
from rlrmp.runtime.spec_migrations import (
    RUN_SPEC_KIND,
    RUN_SPEC_SCHEMA_ID,
    RUN_SPEC_SCHEMA_VERSION,
    stamp_current_schema,
)


FEEDBAX_TRAINING_RUN_SPEC_KEY = "feedbax_training_run_spec"
RLRMP_RUN_SPEC_PAYLOAD_KEY = "rlrmp_run_spec"


def rlrmp_extension_payload(run_spec: dict[str, Any]) -> dict[str, Any]:
    """Return the RLRMP-owned v2 extension payload embedded in tracked recipes."""

    training_summary = _mapping(run_spec, "training_summary")
    model_summary = _mapping(run_spec, "model_summary")
    graph = _mapping(run_spec, "feedbax_graph")
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
    objective_payload = {
        "loss_summary": run_spec.get("loss_summary"),
        "loss_objective": run_spec.get("loss_objective"),
        "fidelity_status": run_spec.get("fidelity_status"),
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
            source_branch=_string_or_none(_mapping(run_spec, "provenance").get("git", {}).get("branch")),
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


def _repo_relative(path: Path) -> str:
    from rlrmp.paths import REPO_ROOT

    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _string_or_none(value: Any) -> str | None:
    return None if value is None else str(value)
