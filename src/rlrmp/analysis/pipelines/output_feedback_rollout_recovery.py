"""Manifest-backed output-feedback rollout-recovery materialization."""

from rlrmp.eval.output_feedback_rollout_recovery import (
    ISSUE_ID,
    RolloutRecoveryMaterialization,
    RolloutRecoveryResult,
    materialize_output_feedback_rollout_recovery,
    render_markdown,
    result_summary,
)

__all__ = [
    "ISSUE_ID",
    "RolloutRecoveryMaterialization",
    "RolloutRecoveryResult",
    "materialize_output_feedback_rollout_recovery",
    "render_markdown",
    "result_summary",
]
