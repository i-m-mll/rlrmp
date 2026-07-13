"""Real execution and resume evidence for canonical linear C&S training bases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import jax
import jax.tree as jt
import numpy as np
import pytest
from feedbax.contracts.graph import GraphSpec
from feedbax.contracts.manifest import TrainingRunManifest
from feedbax.contracts.training import TrainingRunSpec
from feedbax.training import (
    DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY,
    ExecutionPreparationRequest,
)
from feedbax.training.executor import TrainingRunExecutionResult, execute_training_run_spec
from feedbax.training.interruption import CancellationDecision

from rlrmp.runtime.checkpoint_fork_gate import register_rlrmp_training_methods
from rlrmp.train.execution_preparation import register_execution_preparations
from rlrmp.train.linear_recurrent_native import (
    author_linear_recurrent_training_base_from_canonical,
)
from rlrmp.train.resume_control import LaunchContinuation, attach_cs_supervised_checkpoint_continuation
from rlrmp.train.static_linear_native import author_static_linear_training_base_from_canonical
from rlrmp.train.training_base_routes import route_training_base


REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_BASE = REPO_ROOT / "results/c6c5997/runs/flat_3e-5-epsilon-ramp.json"
Architecture = Literal["static_linear", "linear_recurrence"]


def _tiny_authored_spec(
    architecture: Architecture,
    *,
    tmp_path: Path,
    row_id: str,
    training_distribution: Literal["nominal", "broad_epsilon_pgd"] = "nominal",
) -> TrainingRunSpec:
    """Author an executable two-batch base without inherited continuation state."""

    register_rlrmp_training_methods()
    register_execution_preparations(DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY)
    payload = json.loads(CANONICAL_BASE.read_text(encoding="utf-8"))
    base = TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])
    config = dict(base.method_payload.payload["config"])
    config.update(
        {
            "allow_fresh_start": True,
            "batch_size": 1,
            "broad_epsilon_pgd_steps": 1,
            "checkpoint_interval_batches": 1,
            "constant_lr_iterations": 1,
            "disable_progress": True,
            "full_train": True,
            "hidden_size": 2,
            "issue": "427d0d8",
            "lr_warmup_batches": 1,
            "n_batches_condition": 2,
            "n_replicates": 1,
            "n_train_batches": 2,
            "output_dir": str(tmp_path / row_id / "bulk"),
            "quiet_progress": True,
            "resume": False,
            "spec_dir": str(tmp_path / row_id / "spec"),
            "training_diagnostics": False,
        }
    )
    base = base.model_copy(
        update={
            "training_config": base.training_config.model_copy(
                update={
                    "n_batches": 2,
                    "batch_size": 1,
                    "hidden_dim": 2,
                    "snapshot_interval": 1,
                }
            ),
            "checkpoint_progress": base.checkpoint_progress.model_copy(
                update={"checkpoint_interval": 1, "continuation": None}
            ),
            "method_payload": base.method_payload.model_copy(
                update={
                    "payload": {
                        **base.method_payload.payload,
                        "config": config,
                        "n_train_batches": 2,
                        "batch_size": 1,
                    }
                }
            ),
        }
    )
    if architecture == "static_linear":
        authored = author_static_linear_training_base_from_canonical(
            base,
            training_distribution=training_distribution,
        )
    else:
        authored = author_linear_recurrent_training_base_from_canonical(
            base,
            training_distribution=training_distribution,
        )
    routed = route_training_base(authored, issue="427d0d8", row_id=row_id)
    return routed.model_copy(
        update={
            "artifacts": routed.artifacts.model_copy(
                update={
                    "artifact_root": str(tmp_path / row_id / "custody"),
                    "manifest_root": str(tmp_path / row_id / "declared-manifests"),
                }
            )
        }
    )


def _execute(
    spec: TrainingRunSpec,
    *,
    run_id: str,
    manifest_root: Path,
    checkpoint_root: Path,
    resume: bool = False,
    stop_after_first_checkpoint: bool = False,
) -> TrainingRunExecutionResult:
    preparation = DEFAULT_EXECUTION_PREPARATION_PROVIDER_REGISTRY.prepare(
        ExecutionPreparationRequest(run_spec=spec, run_id=run_id, resume=resume)
    )
    cancellation_probe = None
    if stop_after_first_checkpoint:
        decision = CancellationDecision("stop", "test", 1.0)
        cancellation_probe = lambda coordinate: (  # noqa: E731
            decision if coordinate.program_step == 1 else None
        )
    return execute_training_run_spec(
        spec,
        run_id=run_id,
        initial_slots=preparation.initial_slots,
        kernel_context=preparation.kernel_context,
        loss_service=preparation.loss_service,
        manifest_root=manifest_root,
        checkpoint_root=checkpoint_root,
        resume=resume,
        resume_slot_transform=preparation.resume_slot_transform,
        cancellation_probe=cancellation_probe,
        issues=["427d0d8"],
    )


def _assert_slot_tree_equal(left: Any, right: Any) -> None:
    assert jt.structure(left) == jt.structure(right)
    for left_leaf, right_leaf in zip(jt.leaves(left), jt.leaves(right), strict=True):
        if hasattr(left_leaf, "shape") or hasattr(right_leaf, "shape"):
            np.testing.assert_allclose(
                np.asarray(jax.device_get(left_leaf)),
                np.asarray(jax.device_get(right_leaf)),
                rtol=0,
                atol=1e-7,
            )
        else:
            assert left_leaf == right_leaf


@pytest.mark.parametrize("architecture", ["static_linear", "linear_recurrence"])
def test_registered_broad_epsilon_pgd_linear_rows_execute(
    architecture: Architecture,
    tmp_path: Path,
) -> None:
    """Execute each robust linear row through the same registered native method."""

    spec = _tiny_authored_spec(
        architecture,
        tmp_path=tmp_path,
        row_id=f"{architecture}-robust",
        training_distribution="broad_epsilon_pgd",
    )
    result = _execute(
        spec,
        run_id=f"{architecture}-robust",
        manifest_root=tmp_path / architecture / "robust-manifests",
        checkpoint_root=tmp_path / architecture / "robust-checkpoints",
    )

    assert result.status == result.manifest.status == "completed"
    assert result.manifest.completed_batches == 2
    assert result.manifest.training_spec is not None
    training_spec = result.manifest.training_spec.inline
    assert training_spec["method_ref"] == {
        "package": "rlrmp",
        "name": "cs_supervised",
        "version": "v1",
    }
    assert training_spec["method_payload"]["payload"]["pre_step"]["kind"] == (
        "broad_epsilon_pgd"
    )
    assert result.manifest.checkpoint_custody


@pytest.mark.parametrize("architecture", ["static_linear", "linear_recurrence"])
def test_registered_linear_rows_emit_standard_manifest_and_resume_exactly(
    architecture: Architecture,
    tmp_path: Path,
) -> None:
    """Run, interrupt, and resume each linear architecture through public contracts."""

    full_spec = _tiny_authored_spec(
        architecture,
        tmp_path=tmp_path,
        row_id=f"{architecture}-full",
    )
    resumed_spec = _tiny_authored_spec(
        architecture,
        tmp_path=tmp_path,
        row_id=f"{architecture}-resumed",
    )
    full = _execute(
        full_spec,
        run_id=f"{architecture}-full",
        manifest_root=tmp_path / architecture / "full-manifests",
        checkpoint_root=tmp_path / architecture / "full-checkpoints",
    )
    partial = _execute(
        resumed_spec,
        run_id=f"{architecture}-partial",
        manifest_root=tmp_path / architecture / "partial-manifests",
        checkpoint_root=tmp_path / architecture / "resumed-checkpoints",
        stop_after_first_checkpoint=True,
    )
    continuation = LaunchContinuation(
        resume=True,
        resume_source=str(tmp_path / architecture / "resumed-checkpoints" / "latest.json"),
        completed_batches=1,
        stop_target_batches=2,
        continuation_batches=1,
    )
    continuation_spec = attach_cs_supervised_checkpoint_continuation(
        resumed_spec,
        continuation,
    )
    resumed = _execute(
        continuation_spec,
        run_id=f"{architecture}-resumed",
        manifest_root=tmp_path / architecture / "resumed-manifests",
        checkpoint_root=tmp_path / architecture / "resumed-checkpoints",
        resume=True,
    )

    assert isinstance(full.manifest, TrainingRunManifest)
    assert full.status == full.manifest.status == "completed"
    assert partial.status == partial.manifest.status == "cancelled"
    assert resumed.status == resumed.manifest.status == "completed"
    assert partial.manifest.completed_batches == 1
    assert full.manifest.completed_batches == resumed.manifest.completed_batches == 2
    assert full.manifest.checkpoint_custody
    assert resumed.manifest.checkpoint_custody
    assert all(ref.role == "training_checkpoint_custody" for ref in resumed.manifest.checkpoint_custody)
    assert resumed.manifest.provenance.issues == ["427d0d8"]
    assert resumed.manifest.training_spec is not None
    handoff = resumed.manifest.training_spec.inline["metadata"]
    assert handoff["certificate_mode"] == (
        "static_gain" if architecture == "static_linear" else "augmented_linear"
    )
    assert handoff["execution_start"] == "fresh"
    assert handoff.get("source_checkpoint_root") is None
    assert handoff.get("source_checkpoint_transaction_id") is None
    assert resumed.manifest.artifacts
    assert all(artifact.sha256 and artifact.uri for artifact in resumed.manifest.artifacts)
    custody_artifacts = [
        artifact
        for artifact in resumed.manifest.artifacts
        if artifact.role in {"rlrmp_training_artifact", "training_history"}
    ]
    assert custody_artifacts
    assert all(artifact.artifact_id for artifact in custody_artifacts)

    for slot in ("model", "optimizer", "prng", "completed_batches"):
        _assert_slot_tree_equal(full.final_slots[slot], resumed.final_slots[slot])

    if architecture == "linear_recurrence":
        graph = GraphSpec.model_validate(continuation_spec.graph.inline)
        cell = graph.subgraphs["net"].nodes["cell"]
        readout = graph.subgraphs["net"].nodes["readout"]
        contract = handoff["certificate_contract"]
        assert cell.params["activation"] == "identity"
        assert cell.params["use_bias"] is False
        assert readout.params["use_bias"] is False
        assert contract["augmented_state_basis"] == [
            "controller_visible_target_relative_post_step_coupled_state",
            "previous_step_hidden_state",
        ]
        assert contract["state_history_timing"] == "feedbax_post_step_history_pair"
        assert contract["static_gain_coercion"] == "forbidden"
        components = contract["component_inputs"]
        assert components["augmented_states"]["source"] == (
            "EvaluationRunManifest.cached_states"
        )
        assert components["candidate_augmented_action_sensitivity"]["source"] == (
            "trained_controller_graph"
        )
        assert components["candidate_transition"]["source"] == (
            "trained_controller_graph_plus_mechanics"
        )
