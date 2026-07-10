"""Canonical minimax method contract, execution, and numerical kernels."""

# ruff: noqa: F401

from importlib import import_module
from typing import Any

from rlrmp.train.minimax_native.method import (
    MINIMAX_METHOD_PAYLOAD_SCHEMA_ID,
    MINIMAX_METHOD_PAYLOAD_SCHEMA_VERSION,
    MINIMAX_METHOD_REF,
    MinimaxConfig,
    MinimaxMethodPayload,
    build_hps,
    build_minimax_native_initial_slots,
    build_minimax_training_run_spec,
    ensure_minimax_training_method_registered,
    execute_minimax_training_run_spec_native,
    minimax_effective_phase_fingerprint,
    minimax_effective_phase_spec,
    minimax_method_contract,
    minimax_training_run_spec_from_file,
    minimax_training_run_spec_to_config,
    validate_minimax_run_spec,
    validate_minimax_run_spec_file,
)

_KERNEL_EXPORTS = frozenset(
    {
        "ADVERSARIAL_COMPLETE_REF",
        "INNER_ASCENT_KERNEL_REF",
        "NO_ADVERSARIAL_BATCHES_REF",
        "OUTER_DESCENT_KERNEL_REF",
        "PROJECTION_KERNEL_REF",
        "WARMUP_KERNEL_REF",
        "MinimaxControllerLayout",
        "MinimaxControllerState",
        "MinimaxExternalObjectiveLoss",
        "MinimaxExternalObjectiveLossService",
        "MinimaxNativeRuntime",
        "MinimaxPreparedBatch",
        "_controller_layout",
        "_controller_state_from_model",
        "_inject_adversary_delta_A",
        "_model_from_controller_state",
        "_prepare_adversarial_batch",
        "_vmapped_controller_descent",
        "_vmapped_gaussian_adversary_ascent",
        "_vmapped_linear_adversary_ascent",
        "minimax_guard_predicates",
        "minimax_update_kernels",
    }
)


def __getattr__(name: str) -> Any:
    """Load heavy JAX kernel ownership only when execution needs it."""

    if name not in _KERNEL_EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module("rlrmp.train.minimax_native.kernels"), name)
    globals()[name] = value
    return value


__all__ = [
    "ADVERSARIAL_COMPLETE_REF",
    "INNER_ASCENT_KERNEL_REF",
    "MINIMAX_METHOD_PAYLOAD_SCHEMA_ID",
    "MINIMAX_METHOD_PAYLOAD_SCHEMA_VERSION",
    "MINIMAX_METHOD_REF",
    "NO_ADVERSARIAL_BATCHES_REF",
    "OUTER_DESCENT_KERNEL_REF",
    "PROJECTION_KERNEL_REF",
    "WARMUP_KERNEL_REF",
    "MinimaxConfig",
    "MinimaxControllerLayout",
    "MinimaxControllerState",
    "MinimaxExternalObjectiveLoss",
    "MinimaxExternalObjectiveLossService",
    "MinimaxMethodPayload",
    "MinimaxNativeRuntime",
    "MinimaxPreparedBatch",
    "build_hps",
    "build_minimax_native_initial_slots",
    "build_minimax_training_run_spec",
    "ensure_minimax_training_method_registered",
    "execute_minimax_training_run_spec_native",
    "minimax_effective_phase_fingerprint",
    "minimax_effective_phase_spec",
    "minimax_guard_predicates",
    "minimax_method_contract",
    "minimax_training_run_spec_from_file",
    "minimax_training_run_spec_to_config",
    "minimax_update_kernels",
    "validate_minimax_run_spec",
    "validate_minimax_run_spec_file",
]
