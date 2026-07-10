"""Thin CLI adapter for the canonical guided-distillation native capability."""

# ruff: noqa: F401

from rlrmp.train.distillation_native.guided import (
    BASE_RUN_ID,
    DEFAULT_POPULATION_MASK_MODE,
    DEFAULT_SPEC_PATH,
    DEFAULT_TEACHER_GAINS_KEY,
    DEFAULT_TRAINABLE_DTYPE,
    HINF_STANDARD_GRAPH_RUN_ID,
    ISSUE_ID,
    LEGACY_ACTION_HISTORY_RUN_ID,
    RUN_ID,
    build_distillation_spec,
    build_parser,
    default_distillation_command,
    forcing_fraction_for_batch,
    full_train_command,
    load_teacher_package,
    main,
    materialize_teacher_batch,
    output_dir_for,
    run_guided_distillation_training,
    run_spec_path_for,
    smoke_distillation_loss,
)

__all__ = [name for name in globals() if not name.startswith("__")]
