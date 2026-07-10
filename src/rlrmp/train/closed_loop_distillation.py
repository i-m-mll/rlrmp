"""Thin CLI adapter for the canonical closed-loop distillation capability."""

# ruff: noqa: F401

from rlrmp.train.distillation_native.closed_loop import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPEC_PATH,
    ISSUE_ID,
    RUN_ID,
    ClosedLoopDistillationLoss,
    ClosedLoopLossWeights,
    ExtLQGClosedLoopReference,
    FullTrainingApprovalRequiredError,
    build_closed_loop_distillation_spec,
    build_closed_loop_loss,
    build_closed_loop_trainer,
    default_preflight_command,
    full_train_command,
    main,
    output_dir_for,
    run_closed_loop_distillation_training_native,
    run_spec_path_for,
    smoke_directional_jvp,
    smoke_train_command,
    validate_run_spec,
    write_run_spec,
)

__all__ = [name for name in globals() if not name.startswith("__")]
