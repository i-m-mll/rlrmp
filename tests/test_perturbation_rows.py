"""Tests for canonical perturbation-bank row schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rlrmp.analysis.perturbation_rows import (
    _CHANNEL_FIELD_RULES,
    PerturbationChannel,
    PerturbationSpec,
)
from rlrmp.eval.perturbation_bank import default_cs_perturbation_bank
from rlrmp.analysis.pipelines.gru_steady_state_perturbation_bank import (
    default_feedback_perturbations,
)


FIXTURES = Path(__file__).parent / "fixtures"


def _feedback_scale_manifest() -> dict[str, Any]:
    return {
        "runs": {
            "fixture_run": {
                "controller_feedback_scales": {
                    "status": "available",
                    "feedback_dim": 6,
                    "components": {
                        "position": {"reference_scale": 0.15, "units": "m"},
                        "velocity": {"reference_scale": 0.75, "units": "m/s"},
                        "force_filter": {"reference_scale": 7.5, "units": "N"},
                    },
                },
            },
        }
    }


def _row_for_channel(channel: PerturbationChannel) -> dict[str, Any]:
    common = {
        "perturbation_id": f"{channel}_row",
        "channel": channel,
        "family": f"{channel}_family",
        "amplitude": 1.0,
        "units": "test_units",
        "axis": "x",
        "basis": "test_basis",
        "sign": 1,
        "timing": {"start_time_index": 0, "duration_steps": 1},
        "adapter": "test_adapter",
        "description": "Test row.",
    }
    if channel == "initial_state":
        common["family"] = "initial_position_offset"
        common["timing"] = {"epoch": "initial_condition", "time_index": 0}
    if channel == "process_epsilon":
        common["epsilon_index"] = 0
        common["epsilon_component"] = "position_x"
    if channel in {"sensory_feedback", "delayed_observation"}:
        common["channel_provenance"] = {"feedback_payload_index": 0}
    if channel == "target_stream":
        common["family"] = "target_stream_jump"
        common["timing"] = {"epoch": "adapter_defined"}
    return common


def test_default_perturbation_banks_round_trip_through_schema(
) -> None:
    banks = [
        default_cs_perturbation_bank(),
        default_cs_perturbation_bank(
            mode="calibrated",
            calibration_level="small",
            feedback_scale_manifest=_feedback_scale_manifest(),
        ),
    ]
    for bank in banks:
        for row in bank["perturbations"]:
            assert PerturbationSpec.from_mapping(row).to_json() == row


def test_steady_state_battery_rows_round_trip_through_schema() -> None:
    for perturbation in default_feedback_perturbations(feedback_dim=6):
        row = perturbation.to_bank_row(feedback_dim=6, pulse_start=3, pulse_duration=2)
        assert PerturbationSpec.from_mapping(row).to_json() == row


@pytest.mark.parametrize("channel", tuple(_CHANNEL_FIELD_RULES))
def test_channel_required_fields_name_channel_and_field(channel: PerturbationChannel) -> None:
    required = _CHANNEL_FIELD_RULES[channel].required
    for field in sorted(required):
        row = _row_for_channel(channel)
        row.pop(field)
        with pytest.raises(ValueError) as excinfo:
            PerturbationSpec.from_mapping(row).validate()
        message = str(excinfo.value)
        assert field in message
        assert (channel in message) or field == "channel"


def test_initial_state_family_constraint_names_channel_and_family() -> None:
    row = _row_for_channel("initial_state")
    row["family"] = "process_epsilon_pulse"

    with pytest.raises(ValueError, match="initial_state.*family"):
        PerturbationSpec.from_mapping(row).validate()


def test_historical_flattened_calibration_rows_are_accepted() -> None:
    fixture = json.loads((FIXTURES / "historical_perturbation_bank_rows.json").read_text())

    for row in fixture["perturbations"]:
        spec = PerturbationSpec.from_mapping(row)
        spec.validate()
        assert spec.to_json() == row
        assert list(spec.to_json()) == list(row)
