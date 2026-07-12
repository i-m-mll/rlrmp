"""Generated-config entry tests for native closed-loop distillation."""

from __future__ import annotations

import importlib.util

import jax.numpy as jnp
import pytest

from rlrmp.train.distillation_entry import (
    load_distillation_run_spec,
    run_distillation_config,
)
from rlrmp.train.distillation_native import closed_loop_kernel
from rlrmp.train.training_configs import ClosedLoopDistillationConfig


def test_closed_loop_typed_config_loads_and_refreshes_tracked_native_spec() -> None:
    config = ClosedLoopDistillationConfig(n_batches=3, batch_size=2, n_replicates=1)
    spec = load_distillation_run_spec(config, method="closed_loop_distillation")

    assert spec["student_contract"]["n_train_batches"] == 3
    assert spec["student_contract"]["batch_size"] == 2
    assert spec["student_contract"]["n_replicates"] == 1
    assert spec["training_entry"]["module"] == "rlrmp.train.distillation_entry"
    assert spec["schema_version"] == "rlrmp.closed_loop_distillation.training_entry.v2"


def test_closed_loop_reference_math_remains_a_distinct_kernel() -> None:
    reference = closed_loop_kernel.ExtLQGClosedLoopReference(
        plant_a=jnp.eye(2),
        plant_b=jnp.eye(2),
        controller_gains=jnp.zeros((2, 2, 2)),
        observation_matrix=jnp.eye(2),
        feedback_gains=jnp.zeros((2, 2, 2)),
        state_dim=2,
    )
    rollout = reference.rollout(
        initial_vector=jnp.zeros((1, 2)),
        target_pos=jnp.zeros((1, 2)),
        n_steps=2,
    )
    assert rollout["position"].shape == (1, 2, 2)
    assert rollout["action"].shape == (1, 2, 2)


def test_closed_loop_full_train_requires_both_confirmation_flags() -> None:
    config = ClosedLoopDistillationConfig(full_train=True)

    with pytest.raises(PermissionError, match="requires both"):
        run_distillation_config(config, method="closed_loop_distillation")


def test_retired_closed_loop_public_module_is_absent() -> None:
    assert importlib.util.find_spec("rlrmp.train.closed_loop_distillation") is None
