"""Generic evaluation primitives for rlrmp.

Helpers that operate on trained-model checkpoints and trial specs to produce
states and kinematic metrics. These primitives are training-method-agnostic:
they do not care whether a model was produced by standard backprop, minimax
adversarial training, CVaR, APT, or any other method.

For minimax-trained checkpoint I/O (which has a specific on-disk layout —
``warmup_model.eqx`` / ``adversarial_model.eqx`` / ``trained_adversary.eqx``)
see :mod:`rlrmp.eval.minimax_io`.

:copyright: Copyright 2023-2026 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from rlrmp.eval.ensemble import N_REPLICATES, eval_ensemble_on_trials
from rlrmp.eval.checkpoint_selection import (
    ReplicateCheckpointSelection,
    build_fixed_bank_checkpoint_selection_manifest,
    build_validation_checkpoint_selection_manifest,
    load_validation_selected_checkpoint_model,
)
from rlrmp.eval.kinematics import compute_kinematics
from rlrmp.eval.pert import eval_at_pert0, eval_at_pert_scale
from rlrmp.eval.recipes import (
    BROAD_EPSILON_EVALUATION_TYPE,
    CENTER_OUT_ENSEMBLE_EVALUATION_TYPE,
    DELAYED_REACH_BANK_EVALUATION_TYPE,
    FEEDBACK_ABLATION_EVALUATION_TYPE,
    PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE,
    WORST_CASE_EPSILON_EVALUATION_TYPE,
    register_rlrmp_evaluation_recipes,
)
from rlrmp.eval.sisu import set_sisu

__all__ = [
    "BROAD_EPSILON_EVALUATION_TYPE",
    "CENTER_OUT_ENSEMBLE_EVALUATION_TYPE",
    "DELAYED_REACH_BANK_EVALUATION_TYPE",
    "FEEDBACK_ABLATION_EVALUATION_TYPE",
    "N_REPLICATES",
    "PERTURBATION_RESPONSE_BANK_EVALUATION_TYPE",
    "ReplicateCheckpointSelection",
    "WORST_CASE_EPSILON_EVALUATION_TYPE",
    "compute_kinematics",
    "build_fixed_bank_checkpoint_selection_manifest",
    "build_validation_checkpoint_selection_manifest",
    "eval_at_pert0",
    "eval_at_pert_scale",
    "eval_ensemble_on_trials",
    "load_validation_selected_checkpoint_model",
    "register_rlrmp_evaluation_recipes",
    "set_sisu",
]
