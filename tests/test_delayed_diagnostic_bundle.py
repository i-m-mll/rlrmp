from __future__ import annotations

import json

import numpy as np

from rlrmp.analysis.pipelines.delayed_diagnostic_bundle import (
    DecaySignalSpec,
    DirectionGroupSpec,
    build_delayed_diagnostic_bundle,
    materialize_delayed_diagnostic_bundle,
    summarize_direction_split,
    summarize_peak_decay,
)
from rlrmp.runtime.spec_migrations import (
    DELAYED_DIAGNOSTIC_BUNDLE_KIND,
    DELAYED_DIAGNOSTIC_BUNDLE_SCHEMA_VERSION,
    accept_rlrmp_spec_payload,
)


def test_direction_split_summarizes_declared_multi_direction_groups() -> None:
    velocity = np.zeros((2, 4, 5, 2), dtype=np.float64)
    velocity[:, 0, :, 0] = [0.0, 1.0, 2.0, 1.0, 0.0]
    velocity[:, 1, :, 1] = [0.0, 0.5, 1.0, 0.5, 0.0]
    velocity[:, 2, :, 0] = [0.0, 0.25, 0.5, 0.25, 0.0]
    velocity[:, 3, :, 1] = [0.0, 1.5, 3.0, 1.5, 0.0]
    reach_direction = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    result = summarize_direction_split(
        velocity=velocity,
        direction_index=np.asarray([0, 1, 2, 3]),
        reach_direction=reach_direction,
        direction_groups=[
            DirectionGroupSpec("early_dirs", (0, 1)),
            DirectionGroupSpec("late_dirs", (2, 3)),
        ],
        dt=0.01,
        bank_metadata={"direction_count": 4},
    )

    assert result["status"] == "available"
    assert result["scope"] == "multi_direction_target_radial_velocity_split"
    assert result["direction_index_unique"] == [0, 1, 2, 3]
    early = result["groups"]["early_dirs"]
    late = result["groups"]["late_dirs"]
    assert early["n_samples"] == 4
    assert late["n_samples"] == 4
    assert early["peak_mean_forward_velocity_m_s"] == 1.5
    assert late["peak_mean_forward_velocity_m_s"] == 1.75
    assert early["time_of_peak_mean_forward_velocity_s"] == 0.02


def test_direction_split_marks_single_direction_not_applicable() -> None:
    velocity = np.zeros((3, 1, 4, 2), dtype=np.float64)

    result = summarize_direction_split(
        velocity=velocity,
        direction_index=np.zeros((1,), dtype=np.int64),
        reach_direction=np.asarray([[1.0, 0.0]]),
        direction_groups=[DirectionGroupSpec("all", (0,))],
        dt=0.01,
        context={"run_id": "fixed_target"},
    )

    assert result["status"] == "not_applicable"
    assert result["reason"] == "single_direction_context"
    assert "single-direction" in result["note"]


def test_peak_decay_records_signal_context_and_checkpoint_sweep() -> None:
    velocity = np.asarray(
        [
            [0.0, 1.0, 4.0, 3.0, 2.0, 1.0],
            [0.0, 2.0, 4.0, 2.0, 1.0, 0.5],
        ]
    )
    command = np.asarray(
        [
            [2.0, 2.0, 1.5, 1.0, 0.5, 0.25],
            [2.0, 2.0, 1.0, 0.5, 0.25, 0.0],
        ]
    )

    result = summarize_peak_decay(
        signals={
            "velocity_radial": velocity,
            "command_radial_positive": command,
        },
        signal_specs=[
            DecaySignalSpec(
                "velocity_radial",
                role="velocity",
                source="target_radial_velocity",
                units="m/s",
                baseline_window=(1, 3),
            ),
            DecaySignalSpec(
                "command_radial_positive",
                role="command",
                source="state.net.output projected on reach direction",
                units="a.u.",
                baseline_window=(0, 2),
                threshold_start_step=2,
            ),
        ],
        checkpoint_signals={
            "checkpoint_0001000": {"velocity_radial": velocity * 0.5},
            "checkpoint_0002000": {
                "velocity_radial": velocity,
                "command_radial_positive": command,
            },
        },
        dt=0.01,
        thresholds=(0.9,),
        support_windows=((0, 3), (3, 6)),
        context={"bank": "uniform_20dir_0p15m_no_catch"},
    )

    assert result["status"] == "available"
    velocity_summary = result["signals"]["velocity_radial"]
    assert velocity_summary["signal"]["role"] == "velocity"
    assert velocity_summary["peak_step"] == 2
    assert velocity_summary["decay_crossings"]["0.9"]["step"] == 3
    assert result["checkpoint_sweep"]["status"] == "available"
    first_checkpoint = result["checkpoint_sweep"]["rows"][0]
    assert first_checkpoint["signals"]["command_radial_positive"]["status"] == "not_materialized"
    assert first_checkpoint["signals"]["command_radial_positive"]["reason"] == (
        "signal_absent_for_checkpoint"
    )


def test_delayed_bundle_materializes_schema_stamped_manifest(tmp_path) -> None:
    direction_split = {
        "status": "not_applicable",
        "reason": "single_direction_context",
    }
    peak_decay = {
        "status": "available",
        "signals": {},
        "checkpoint_sweep": {"status": "not_materialized", "rows": []},
    }
    output_path = tmp_path / "delayed_bundle.json"

    payload = materialize_delayed_diagnostic_bundle(
        issue="21f4638",
        scope="unit",
        output_path=output_path,
        direction_split=direction_split,
        peak_decay=peak_decay,
        source_inputs=[{"role": "unit_fixture", "path": "tests/test_delayed_diagnostic_bundle.py"}],
        repo_root=tmp_path,
    )

    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == payload
    assert loaded["schema_version"] == DELAYED_DIAGNOSTIC_BUNDLE_SCHEMA_VERSION
    assert (tmp_path / "delayed_bundle_regeneration_spec.json").is_file()
    accepted = accept_rlrmp_spec_payload(DELAYED_DIAGNOSTIC_BUNDLE_KIND, loaded)
    assert accepted.target_version == DELAYED_DIAGNOSTIC_BUNDLE_SCHEMA_VERSION


def test_build_bundle_defaults_missing_components_to_not_materialized() -> None:
    payload = build_delayed_diagnostic_bundle(issue="21f4638", scope="unit")

    assert payload["direction_split"]["status"] == "not_materialized"
    assert payload["peak_decay"]["status"] == "not_materialized"
    assert payload["selection_role"] == "audit_only_not_used_for_checkpoint_selection"
