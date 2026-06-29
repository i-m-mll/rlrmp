"""Focused tests for the f3c5db9 frozen Adam tuning materializer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "f3c5db9"
    / "scripts"
    / "materialize_frozen_adam_audit_tuning.py"
)


@pytest.fixture(scope="module")
def tuning_module():
    spec = importlib.util.spec_from_file_location("frozen_adam_audit_tuning_f3c5db9", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_match_requires_finite_useful_and_interior(tuning_module) -> None:
    row = {
        "valid": True,
        "finite_status": "finite",
        "gradient_status": "finite",
        "useful": True,
        "interior": True,
    }
    assert tuning_module.matches_reference_region(row)

    row["interior"] = False
    assert not tuning_module.matches_reference_region(row)


def test_select_best_row_prefers_stage1_before_gain(tuning_module) -> None:
    rows = [
        {
            "match_reference_region": True,
            "stage": "stage2_known_direction_init",
            "objective_gain_over_zero": 100.0,
            "max_norm_over_cap": 0.2,
        },
        {
            "match_reference_region": True,
            "stage": "stage1_grid",
            "objective_gain_over_zero": 10.0,
            "max_norm_over_cap": 0.8,
        },
    ]

    assert tuning_module.select_best_row(rows)["stage"] == "stage1_grid"


def test_common_stage1_settings_require_every_row_and_mechanism(tuning_module) -> None:
    rows = []
    for run_id in tuning_module.RUN_IDS:
        for mechanism in tuning_module.MECHANISMS:
            rows.append(
                {
                    "run_id": run_id,
                    "mechanism": mechanism,
                    "stage": "stage1_grid",
                    "match_reference_region": True,
                    "adam_steps": 12,
                    "adam_learning_rate": 1e-5,
                }
            )
    rows.append(
        {
            "run_id": tuning_module.RUN_IDS[0],
            "mechanism": tuning_module.MECHANISMS[0],
            "stage": "stage1_grid",
            "match_reference_region": True,
            "adam_steps": 32,
            "adam_learning_rate": 1e-4,
        }
    )

    assert tuning_module.common_stage1_settings(rows) == [
        {"adam_steps": 12, "adam_learning_rate": 1e-5}
    ]


def test_reference_summary_requires_all_mechanisms(tuning_module, tmp_path) -> None:
    path = tmp_path / "reference.json"
    path.write_text(
        '{"summary": [{"run_id": "open_loop_small", "mechanism": "direct_epsilon", '
        '"optimizer": "pgd_projected_epsilon", "lowest_valid_lambda_multiplier": 2.0}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing reference summary rows"):
        tuning_module.load_reference_summary(path)
