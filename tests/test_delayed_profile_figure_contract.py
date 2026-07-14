"""Native delayed-profile figure and registered payload contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from feedbax.contracts.figures import FigureSpec

from rlrmp.eval.recipes import (
    DELAYED_VELOCITY_PROFILE_PAYLOAD_SCHEMA_VERSION,
    delayed_velocity_profile_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
def test_registered_delayed_payload_owns_profile_facets_and_bands() -> None:
    payload = delayed_velocity_profile_payload(
        {
            "banks": {
                "no_catch": [
                    {
                        "experiment": "example",
                        "run_id": "row-a",
                        "label": "Row A",
                        "time_s": [-0.1, 0.0, 0.1],
                        "mean": [0.0, 0.2, 0.4],
                        "std": [0.01, 0.02, 0.03],
                        "replicate_mean": [[0.0, 0.2, 0.4]],
                        "replicate_std": [[0.02, 0.03, 0.04]],
                        "alignment": {"time_basis": "go_cue_aligned_canonical_movement_window"},
                    }
                ],
                "catch": [
                    {
                        "experiment": "example",
                        "run_id": "row-a",
                        "label": "Row A",
                        "time_s": [-0.1, 0.0, 0.1],
                        "mean": [0.0, 0.01, 0.0],
                        "std": [0.0, 0.005, 0.0],
                    }
                ],
            }
        }
    )

    assert payload is not None
    assert payload["schema_version"] == DELAYED_VELOCITY_PROFILE_PAYLOAD_SCHEMA_VERSION
    facet = payload["facets"]["condition"]["example/row-a"]
    assert facet["display_name"] == "Row A"
    assert [series["bank_kind"] for series in facet["forward_velocity"]["series"]] == [
        "no_catch",
        "catch",
    ]
    no_catch = facet["forward_velocity"]["series"][0]["profile"]
    assert no_catch["time"] == [-0.1, 0.0, 0.1]
    assert no_catch["upper"] == pytest.approx([0.01, 0.22, 0.43])
    assert no_catch["lower"] == pytest.approx([-0.01, 0.18, 0.37])
    assert len(facet["forward_velocity_by_replicate"]["series"]) == 1


@pytest.mark.parametrize(
    ("issue", "topic"),
    (
        ("40e1911", "delayed_timing_hold_lane_velocity_profiles"),
        ("ef9c882", "start_pos_hold_norm_velocity_profiles"),
        ("ef9c882", "start_pos_hold_norm_matched_ffpert_velocity_profiles"),
    ),
)
def test_delayed_profile_specs_preserve_shared_axes_and_replicate_semantics(
    issue: str,
    topic: str,
) -> None:
    path = REPO_ROOT / "results" / issue / "figures" / topic / "spec.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    spec = FigureSpec.model_validate(payload)

    assert spec.panels[0].axes_labels.model_dump() == {
        "x": "Time relative to go cue (s)",
        "y": "Target-radial velocity (m/s)",
    }
    assert payload["metadata"]["shared_yaxes"] == "all"
    assert payload["metadata"]["replicate_profile_key"] == (
        "forward_velocity_by_replicate"
    )
