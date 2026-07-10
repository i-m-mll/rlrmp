"""Canonical native capability for guided and closed-loop distillation."""

# ruff: noqa: F401

from rlrmp.train.distillation_native.executor import (
    CLOSED_LOOP_KERNEL_REF,
    GUIDED_KERNEL_REF,
    ClosedLoopNativeRuntime,
    DistillationExternalObjectiveLoss,
    DistillationExternalObjectiveLossService,
    GuidedNativeRuntime,
    _closed_loop_training_chunk,
    _guided_training_chunk,
    build_distillation_native_initial_slots,
    distillation_update_kernels,
    execute_distillation_training_run_spec_native,
    guided_distillation_train_step,
    native_distillation_model_from_slot,
)
from rlrmp.train.distillation_native.losses import (
    CSH0DistillationConfig,
    DistillationLossResult,
    DistillationLossWeights,
    PolicyMap,
    batched_directional_jvps,
    clean_action_imitation_loss,
    cs_h0_distillation_config,
    guided_distillation_loss,
    input_output_jvp_matching_loss,
    mean_squared_error,
    perturbation_response_imitation_loss,
    student_forced_rollout_anchor_loss,
)

__all__ = [name for name in globals() if not name.startswith("_")]
