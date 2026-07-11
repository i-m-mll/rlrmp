"""Regression coverage for RLRMP's Feedbax training-method plugin hook."""

from __future__ import annotations

from importlib.metadata import EntryPoint

from feedbax.contracts.training import default_training_method_registry
from feedbax.plugins.discovery import load_training_method_plugins

from rlrmp.runtime.training_run_specs import (
    CLOSED_LOOP_DISTILLATION_METHOD_REF,
    CS_SUPERVISED_METHOD_REF,
    GUIDED_DISTILLATION_METHOD_REF,
)
from rlrmp.train.adaptive_epsilon_native import ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF
from rlrmp.train.minimax_native import MINIMAX_METHOD_REF
from rlrmp.train.policy_adversary_native import POLICY_ADVERSARY_SUPERVISED_METHOD_REF


RLRMP_NATIVE_METHOD_REFS = frozenset(
    {
        CS_SUPERVISED_METHOD_REF,
        CLOSED_LOOP_DISTILLATION_METHOD_REF,
        GUIDED_DISTILLATION_METHOD_REF,
        ADAPTIVE_EPSILON_CURRICULUM_METHOD_REF,
        POLICY_ADVERSARY_SUPERVISED_METHOD_REF,
        MINIMAX_METHOD_REF,
    }
)


def test_rlrmp_plugin_entry_point_registers_all_native_training_methods() -> None:
    """Every RLRMP-owned native method must load before generic validation."""
    registry = default_training_method_registry()
    entry_point = EntryPoint(
        name="rlrmp",
        value="rlrmp:register_experiment_package",
        group="feedbax.plugins",
    )

    load_training_method_plugins(registry=registry, entry_points=[entry_point])

    assert RLRMP_NATIVE_METHOD_REFS <= set(registry.available_keys())
    for method_ref in RLRMP_NATIVE_METHOD_REFS:
        assert registry.resolve(method_ref, path="/method_ref").method_ref == method_ref
