"""Semantic invariants for native minimax and distillation executor lifecycles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import pytest
from feedbax.contracts.training import TrainingRunSpec

from rlrmp.model.feedbax_graph import build_rlrmp_feedbax_graph_bundle
from rlrmp.train.distillation_entry import load_distillation_run_spec
from rlrmp.train.distillation_native import execute_distillation_training_run_spec_native
from rlrmp.train.executor.equivalence import assert_paired_equivalent, run_paired_equivalence
from rlrmp.train.minimax_native import (
    MinimaxControllerState,
    build_hps,
    build_minimax_training_run_spec,
    execute_minimax_training_run_spec_native,
)
from rlrmp.train.task_model import build_task_base
from rlrmp.train.training_configs import (
    ClosedLoopDistillationConfig,
    GuidedDistillationConfig,
    MinimaxConfig,
)


@pytest.fixture(autouse=True)
def _executor_tests_use_float32():
    """Keep deterministic training comparisons independent of global x64 state."""

    with jax.enable_x64(False):
        yield


def _minimax_spec(tmp_path: Path, adversary_type: str) -> TrainingRunSpec:
    config = MinimaxConfig(
        adversary_type=adversary_type,
        n_warmup_batches=0,
        n_adversary_batches=2,
        n_adversary_steps=1,
        batch_size=1,
        adv_batch_size=1,
        n_replicates=1,
        output_dir=str(tmp_path / adversary_type),
    )
    hps = build_hps(config)
    graph_bundle = build_rlrmp_feedbax_graph_bundle(
        hps,
        task=build_task_base(hps),
        n_extra_inputs=1,
        hidden_type=hps.hidden_type,
        sisu_gating=hps.sisu_gating,
        key=jr.PRNGKey(config.seed),
    )
    payload = build_minimax_training_run_spec(
        config,
        graph_spec=graph_bundle.graph_spec,
        output_dir=Path(config.output_dir),
        spec_dir=tmp_path / "spec",
        feedbax_graph={"graph_spec_path": "graph_spec.json", "manifest_path": "manifest.json"},
    )
    return TrainingRunSpec.model_validate(payload["feedbax_training_run_spec"])


def _minimax_comparable(slots: dict[str, Any]) -> dict[str, Any]:
    controller = slots["controller"]
    assert isinstance(controller, MinimaxControllerState)
    comparable = {
        "controller": controller.per_replicate_leaves,
        "controller_optimizer": slots["controller_optimizer"],
        "adversary_population": slots["adversary_population"],
        "adversary_optimizer": slots["adversary_optimizer"],
        "rng": slots["rng"],
        "controller_loss": slots["controller_loss"],
        "adversary_loss": slots["adversary_loss"],
    }
    return jt.map(
        lambda leaf: (
            leaf.astype(jnp.int8)
            if getattr(getattr(leaf, "dtype", None), "kind", None) == "b"
            else leaf
        ),
        comparable,
    )


def _assert_finite_numeric_tree(tree: Any) -> None:
    numeric_leaves = [
        jnp.asarray(leaf)
        for leaf in jt.leaves(tree)
        if getattr(getattr(leaf, "dtype", None), "kind", None) in set("biufc")
        or isinstance(leaf, (bool, int, float, complex))
    ]
    assert numeric_leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in numeric_leaves)


def _numeric_arrays_by_path(tree: Any) -> dict[str, Any]:
    """Select live numeric state while ignoring non-array PyTree metadata."""

    arrays: dict[str, Any] = {}
    for path, leaf in jt.flatten_with_path(tree)[0]:
        try:
            array = jnp.asarray(leaf)
        except (TypeError, ValueError):
            continue
        if array.dtype.kind in set("biufc"):
            arrays[jax.tree_util.keystr(path)] = array
    return arrays


@pytest.mark.parametrize(
    ("adversary_type", "seed"),
    [("gaussian_bump", 1), ("linear_dynamics", 3)],
)
def test_minimax_resume_is_semantically_equivalent_to_uninterrupted_execution(
    tmp_path: Path,
    adversary_type: str,
    seed: int,
) -> None:
    spec = _minimax_spec(tmp_path, adversary_type)
    full = execute_minimax_training_run_spec_native(
        spec,
        run_id=f"minimax-{adversary_type}-full",
        key=jr.PRNGKey(seed),
        manifest_root=tmp_path / "manifests" / "full",
        checkpoint_root=tmp_path / "checkpoints" / "full",
        manifest_conflict_policy="reuse-identical",
    )
    checkpoint_root = tmp_path / "checkpoints" / "resume"
    stopped = execute_minimax_training_run_spec_native(
        spec,
        run_id=f"minimax-{adversary_type}-resume",
        key=jr.PRNGKey(seed),
        manifest_root=tmp_path / "manifests" / "stopped",
        checkpoint_root=checkpoint_root,
        stop_after_barrier="after_adversarial",
        manifest_conflict_policy="reuse-identical",
    )
    resumed = execute_minimax_training_run_spec_native(
        spec,
        run_id=f"minimax-{adversary_type}-resume",
        key=jr.PRNGKey(seed),
        manifest_root=tmp_path / "manifests" / "resumed",
        checkpoint_root=checkpoint_root,
        resume=True,
        manifest_conflict_policy="reuse-identical",
    )

    report = run_paired_equivalence(
        f"minimax.{adversary_type}.resume",
        lambda: full.final_slots,
        lambda: resumed.final_slots,
        comparable=_minimax_comparable,
        left_label="uninterrupted",
        right_label="resumed",
    )
    assert stopped.final_coordinate.completed_barrier == "after_adversarial"
    assert full.final_coordinate.phase == resumed.final_coordinate.phase == "done"
    assert_paired_equivalent(report)
    _assert_finite_numeric_tree(_minimax_comparable(full.final_slots))
    assert float(full.final_slots["controller_loss"]) != 0.0
    assert float(full.final_slots["adversary_loss"]) != 0.0


@pytest.mark.parametrize(
    ("method", "barrier"),
    [
        ("guided_distillation", "after_teacher_forced_warm_start"),
        ("closed_loop_distillation", "after_closed_loop_rollout_distillation"),
    ],
)
def test_distillation_resume_is_semantically_equivalent_to_uninterrupted_execution(
    tmp_path: Path,
    method: str,
    barrier: str,
) -> None:
    if method == "guided_distillation":
        config = GuidedDistillationConfig(
            run_spec="tests/fixtures/legacy_payloads/guided_distillation_run_spec.json",
            n_batches=1,
            batch_size=1,
            n_replicates=1,
            hidden_size=6,
            n_jvp_directions=1,
            output_dir=str(tmp_path / "guided"),
            checkpoint=False,
        )
    else:
        config = ClosedLoopDistillationConfig(
            run_spec=Path(
                "tests/fixtures/legacy_payloads/closed_loop_distillation_run_spec.json"
            ),
            n_batches=1,
            batch_size=1,
            n_replicates=1,
            hidden_size=6,
            output_dir=str(tmp_path / "closed"),
        )
    spec = load_distillation_run_spec(config, method=method)
    full = execute_distillation_training_run_spec_native(
        spec,
        method=method,
        run_id=f"{method}-full",
        key=jr.PRNGKey(3),
        manifest_root=tmp_path / "manifests" / "full",
        checkpoint_root=tmp_path / "checkpoints" / "full",
        manifest_conflict_policy="reuse-identical",
    )
    checkpoint_root = tmp_path / "checkpoints" / "resume"
    stopped = execute_distillation_training_run_spec_native(
        spec,
        method=method,
        run_id=f"{method}-resume",
        key=jr.PRNGKey(3),
        manifest_root=tmp_path / "manifests" / "stopped",
        checkpoint_root=checkpoint_root,
        stop_after_barrier=barrier,
        manifest_conflict_policy="reuse-identical",
    )
    resumed = execute_distillation_training_run_spec_native(
        spec,
        method=method,
        run_id=f"{method}-resume",
        key=jr.PRNGKey(3),
        manifest_root=tmp_path / "manifests" / "resumed",
        checkpoint_root=checkpoint_root,
        resume=True,
        manifest_conflict_policy="reuse-identical",
    )

    execution_keys = ("model", "optimizer", "prng", "completed_batches", "train_loss")
    durable_resume_keys = ("model", "prng", "completed_batches")

    def execution_state(slots: dict[str, Any]) -> dict[str, Any]:
        return {key: slots[key] for key in execution_keys}

    def durable_resume_state(slots: dict[str, Any]) -> dict[str, Any]:
        return {key: slots[key] for key in durable_resume_keys}

    checkpoint_report = run_paired_equivalence(
        f"{method}.checkpoint",
        lambda: full.final_slots,
        lambda: stopped.final_slots,
        comparable=lambda slots: _numeric_arrays_by_path(execution_state(slots)),
        left_label="uninterrupted",
        right_label="stopped_after_terminal_update",
    )
    resume_report = run_paired_equivalence(
        f"{method}.resume",
        lambda: full.final_slots,
        lambda: resumed.final_slots,
        comparable=lambda slots: _numeric_arrays_by_path(durable_resume_state(slots)),
        left_label="uninterrupted",
        right_label="resumed",
    )
    assert stopped.final_coordinate.completed_barrier == barrier
    assert full.final_coordinate.completed_barrier == barrier
    assert resumed.final_coordinate.phase == "done"
    assert_paired_equivalent(checkpoint_report)
    assert_paired_equivalent(resume_report)
    _assert_finite_numeric_tree(execution_state(full.final_slots))
    assert int(full.final_slots["completed_batches"]) == 1
    assert float(full.final_slots["train_loss"]) != 0.0
