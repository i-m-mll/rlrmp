"""Focused tests for the 1697bdc critical-lambda search helpers."""

from __future__ import annotations
from rlrmp.io import load_named_python_module

from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "1697bdc"
    / "scripts"
    / "materialize_critical_lambda_search.py"
)


@pytest.fixture(scope="module")
def search_module():
    return load_named_python_module('critical_lambda_search_1697bdc', SCRIPT)


def make_point(search_module, multiplier: float, *, valid: bool):
    return search_module.make_point(
        run_id="row",
        mechanism="direct_epsilon",
        optimizer="pgd_projected_epsilon",
        phase="probe",
        point_index=0,
        lambda_multiplier=multiplier,
        lambda_value=multiplier * 10.0,
        objective_gain_over_zero=1.0 if valid else 1.0,
        task_loss_gain_over_zero=2.0,
        energy_penalty=1.0,
        energy_mean=0.1,
        max_norm_over_cap=0.98 if valid else 1.02,
        mean_norm_over_cap=0.5,
        cap_bound_fraction=0.0 if valid else 1.0,
        finite_status="finite",
        gradient_status="finite",
        gradient_norm=0.0,
        optimizer_success=True,
        optimizer_status="test",
        optimizer_iterations=1,
        optimizer_evaluations=1,
        details={},
    )


def test_validity_uses_strict_practical_interior(search_module) -> None:
    valid = make_point(search_module, 4.0, valid=True)
    near_cap = search_module.make_point(
        run_id="row",
        mechanism="direct_epsilon",
        optimizer="pgd_projected_epsilon",
        phase="probe",
        point_index=0,
        lambda_multiplier=2.0,
        lambda_value=20.0,
        objective_gain_over_zero=1.0,
        task_loss_gain_over_zero=2.0,
        energy_penalty=1.0,
        energy_mean=0.1,
        max_norm_over_cap=0.995,
        mean_norm_over_cap=0.5,
        cap_bound_fraction=0.0,
        finite_status="finite",
        gradient_status="finite",
        gradient_norm=0.0,
        optimizer_success=True,
        optimizer_status="test",
        optimizer_iterations=1,
        optimizer_evaluations=1,
        details={},
    )

    assert valid.valid
    assert valid.failure_mode == "valid"
    assert not near_cap.valid
    assert near_cap.failure_mode == "near_cap"


def test_find_first_valid_bracket(search_module) -> None:
    low = make_point(search_module, 1.0, valid=False)
    high = make_point(search_module, 2.0, valid=True)

    assert search_module.find_first_valid_bracket([high, low]) == (low, high)


def test_log_bisect_moves_high_on_valid_midpoint(search_module) -> None:
    low = make_point(search_module, 1.0, valid=False)
    high = make_point(search_module, 4.0, valid=True)

    def evaluator(multiplier: float, phase: str, index: int):
        return make_point(search_module, multiplier, valid=multiplier >= 2.0)

    points, final_low, final_high = search_module.log_bisect(
        evaluator,
        low,
        high,
        2,
        rel_tol=1.5,
        max_steps=4,
    )

    assert points
    assert final_low.lambda_multiplier < final_high.lambda_multiplier
    assert final_high.valid
    assert final_high.lambda_multiplier <= 2.0
