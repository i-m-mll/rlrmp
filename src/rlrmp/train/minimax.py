"""Thin public adapter for the canonical minimax-native capability."""

# ruff: noqa: F401

from rlrmp.train.minimax_native import (
    MINIMAX_METHOD_PAYLOAD_SCHEMA_ID,
    MINIMAX_METHOD_PAYLOAD_SCHEMA_VERSION,
    MINIMAX_METHOD_REF,
    MinimaxConfig,
    MinimaxMethodPayload,
    _minimax_method_payload,
    build_hps,
    build_minimax_native_initial_slots,
    build_minimax_training_run_spec,
    ensure_minimax_training_method_registered,
    execute_minimax_training_run_spec_native,
    legacy_cli_args_to_minimax_config,
    minimax_config_namespace,
    minimax_effective_phase_fingerprint,
    minimax_effective_phase_spec,
    minimax_method_contract,
    minimax_training_run_spec_from_file,
    minimax_training_run_spec_to_config,
    validate_minimax_run_spec,
    validate_minimax_run_spec_file,
)

__all__ = [name for name in globals() if not name.startswith("__")]
