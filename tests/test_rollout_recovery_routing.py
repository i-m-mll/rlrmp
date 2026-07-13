"""Structural guards for governed rollout-recovery routing (issue 56aad38)."""

from __future__ import annotations

from pathlib import Path

from rlrmp.eval import output_feedback_rollout_recovery


REPO_ROOT = Path(__file__).resolve().parents[1]
RETIRED_FULL_FIT_CALLERS = (
    "scripts/materialize_output_feedback_failure_decomposition.py",
    "scripts/materialize_output_feedback_sweep_certificates.py",
)
def test_retired_full_fit_callers_are_absent() -> None:
    assert not [path for path in RETIRED_FULL_FIT_CALLERS if (REPO_ROOT / path).exists()]


def test_canonical_module_exposes_rollout_recovery_helpers() -> None:
    assert callable(output_feedback_rollout_recovery.observer_error_coverage_conditions)
    assert callable(output_feedback_rollout_recovery.run_output_feedback_rollout_recovery)


def test_rollout_recovery_facade_is_absent() -> None:
    facade = REPO_ROOT / "src/rlrmp/analysis/pipelines/output_feedback_rollout_recovery.py"
    assert not facade.exists()

    searched_paths = (
        REPO_ROOT / "src",
        REPO_ROOT / "scripts",
        REPO_ROOT / "tests",
        REPO_ROOT / "results",
    )
    residuals = []
    retired_module = "rlrmp.analysis.pipelines." + "output_feedback_rollout_recovery"
    for root in searched_paths:
        for path in root.rglob("*.py"):
            if retired_module in path.read_text(encoding="utf-8"):
                residuals.append(path.relative_to(REPO_ROOT).as_posix())
    assert not residuals
