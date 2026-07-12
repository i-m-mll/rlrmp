"""Declarative closure guards for the 2bc95fd six-cell figure bundle."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from feedbax.contracts.figures import FigureSpec

from rlrmp.analysis.matrix.history_payload import (
    HISTORY_PAYLOAD_SCHEMA_VERSION,
    history_payload_recipe,
)
from rlrmp.data_products.envelope import read_data_product


REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_ROOT = REPO_ROOT / "results" / "2bc95fd" / "figures"
ORACLE_ROOT = REPO_ROOT / "results" / "2bc95fd" / "data_products"
TOPICS = {
    "forward_velocity_profiles": "rlrmp.profile_comparison",
    "hold_drift_profiles": "rlrmp.profile_comparison",
    "peak_velocity_distributions": "rlrmp.distribution_comparison",
    "rmse_ratio_comparison": "rlrmp.metric_comparison",
    "training_loss": "rlrmp.history_comparison",
    "training_loss_per_term": "rlrmp.history_comparison",
    "combo_hold_motor_diagnostic": "rlrmp.history_comparison",
}
RETIRED_PRODUCERS = (
    "analyse_anti_anticipation_6cell_variance.py",
    "diagnose_combo_hold_motor.py",
    "plot_training_loss_6cell.py",
)


pytestmark = pytest.mark.feedbax_contract


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_seven_topics_are_native_manifest_data_bound_specs() -> None:
    for topic, template in TOPICS.items():
        payload = _json(FIGURE_ROOT / topic / "spec.json")
        spec = FigureSpec.model_validate(payload)
        assert spec.template == template
        assert spec.assembler is None
        assert spec.slot_bindings
        assert spec.facet_bindings
        assert spec.figure_routing["experiment"] == "2bc95fd"
        assert spec.figure_routing["topic"] == topic
        assert all(binding.item == "manifest" for binding in spec.facet_bindings.values())
        assert payload["metadata"]["parity_oracle"] == (
            f"results/2bc95fd/data_products/figure_parity_{topic}.json"
        )


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


def test_archived_oracles_preserve_labels_and_summary_values() -> None:
    product = read_data_product(ORACLE_ROOT / "figure_parity_oracles.json")
    assert product.product_schema_id == "rlrmp.figure_parity_oracles"
    assert len(product.artifacts) == 7
    for artifact in product.artifacts:
        assert artifact.uri is not None
        assert hashlib.sha256((REPO_ROOT / artifact.uri).read_bytes()).hexdigest() == (
            artifact.sha256
        )

    peak = _json(ORACLE_ROOT / "figure_parity_peak_velocity_distributions.json")
    rmse = _json(ORACLE_ROOT / "figure_parity_rmse_ratio_comparison.json")
    loss = _json(ORACLE_ROOT / "figure_parity_training_loss.json")
    combo = _json(ORACLE_ROOT / "figure_parity_combo_hold_motor_diagnostic.json")

    cells = peak["plot_kwargs"]["cells"]
    assert len(cells) == 6
    assert set(cells) == set(peak["cell_stats"])
    assert set(cells) == set(rmse["rmse_ratios"])
    assert set(cells) == set(loss["end_of_training_stats"])
    assert peak["cell_stats"]["gru__jerk_motor_smooth_combo"]["mean_peak_velocity"] == (
        1.221550703048706
    )
    assert rmse["rmse_ratios"]["gru__jerk_motor_smooth_combo"]["vel_rmse_ratio"] == (
        0.042346216916084055
    )
    assert loss["end_of_training_stats"]["gru__jerk_smooth_high"]["final_mean"] == (
        3.2848129272460938
    )
    assert combo["summary"]["hold_to_move_ratio"] == 0.17691540718078613


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


def test_legacy_2bc_producers_are_deleted() -> None:
    scripts = REPO_ROOT / "results" / "2bc95fd" / "scripts"
    for producer in RETIRED_PRODUCERS:
        assert not (scripts / producer).exists()
