"""Declarative closure guards for the 2bc95fd six-cell figure bundle."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from rlrmp.analysis.matrix.history_payload import (
    HISTORY_PAYLOAD_SCHEMA_VERSION,
    history_payload_recipe,
)
REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_ROOT = REPO_ROOT / "results" / "2bc95fd" / "figures"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_profiles_preserve_shared_y_and_intrinsic_axes() -> None:
    expected_y = {
        "forward_velocity_profiles": "Forward velocity (m/s)",
        "hold_drift_profiles": "Forward position drift (mm)",
    }
    for topic, y_label in expected_y.items():
        payload = _json(FIGURE_ROOT / topic / "spec.json")
        assert payload["template"] == "rlrmp.profile_comparison"
        assert payload["panels"][0]["axes_labels"] == {
            "x": "Time from go cue (s)",
            "y": y_label,
        }
        assert payload["metadata"]["shared_yaxes"] == "all"
        assert payload["facet_bindings"]["condition"]["item"] == "manifest"


def test_rmse_figure_preserves_primary_secondary_and_auxiliary_metrics() -> None:
    payload = _json(FIGURE_ROOT / "rmse_ratio_comparison" / "spec.json")
    assert payload["facet_bindings"]["metric"]["path"].endswith("facets.summary_metrics")
    assert payload["metadata"]["metric_order"] == [
        "vel_rmse_ratio",
        "pos_rmse_ratio",
        "cv_peak_vel",
    ]


def test_registered_history_payload_keeps_facets_data_bound() -> None:
    spec = SimpleNamespace(params={"history_key": "motor_histories"})
    resolved = SimpleNamespace(
        ref=SimpleNamespace(id="combo-evaluation"),
        states={
            "motor_histories": {
                "Trial 0": {
                    "time": [0.0, 0.01],
                    "series": {"Rep 0": [0.2, 0.3], "Rep 1": [0.1, 0.4]},
                    "summary": {"hold_to_move_ratio": 0.17691540718078613},
                }
            }
        },
    )

    result = history_payload_recipe(spec, REPO_ROOT, [resolved])
    payload = result.analyses["history_payload"].compute(result.data)

    assert payload["schema_version"] == HISTORY_PAYLOAD_SCHEMA_VERSION
    assert payload["facets"]["Trial 0"]["x"] == [0.0, 0.01]
    assert payload["facets"]["Trial 0"]["series"]["Rep 1"] == [0.1, 0.4]
    assert payload["facets"]["Trial 0"]["summary"]["hold_to_move_ratio"] == (
        0.17691540718078613
    )
