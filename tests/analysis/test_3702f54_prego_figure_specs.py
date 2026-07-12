"""Declarative closure guards for the 3702f54 pre-go figure bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from feedbax.analysis.figures import execute_figure_spec
from feedbax.contracts.figures import FigureSpec
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    AnalysisRunSpec,
    ParentRef,
    spec_payload,
    write_manifest,
)

from rlrmp.data_products.envelope import read_data_product
from rlrmp.figures import register_rlrmp_figure_surfaces, standard_matrix_payload


REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_ROOT = REPO_ROOT / "results" / "3702f54" / "figures"
ORACLE_ROOT = REPO_ROOT / "results" / "3702f54" / "data_products"
PRODUCER = REPO_ROOT / "results" / "3702f54" / "scripts" / "analyse_pregomatrix.py"
TOPICS = {
    "forward_velocity_profiles": "rlrmp.profile_comparison",
    "hold_drift_profiles": "rlrmp.profile_comparison",
    "peak_velocity_distributions": "rlrmp.distribution_comparison",
    "summary_metrics": "rlrmp.metric_comparison",
    "training_loss_per_term": "rlrmp.history_comparison",
}


pytestmark = pytest.mark.feedbax_contract


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_five_topics_are_native_manifest_data_bound_specs() -> None:
    for topic, template in TOPICS.items():
        payload = _json(FIGURE_ROOT / topic / "spec.json")
        spec = FigureSpec.model_validate(payload)
        assert spec.template == template
        assert spec.assembler is None
        assert spec.slot_bindings
        assert spec.facet_bindings
        assert spec.figure_routing["experiment"] == "3702f54"
        assert spec.figure_routing["topic"] == topic
        assert all(binding.item == "manifest" for binding in spec.facet_bindings.values())
        assert payload["metadata"]["parity_oracle"] == (
            f"results/3702f54/data_products/figure_parity_{topic}.json"
        )


def test_profiles_preserve_intrinsic_axes_and_shared_y_semantics() -> None:
    expected_y = {
        "forward_velocity_profiles": "Forward velocity (m/s)",
        "hold_drift_profiles": "Forward position drift (mm)",
    }
    for topic, y_label in expected_y.items():
        payload = _json(FIGURE_ROOT / topic / "spec.json")
        assert payload["panels"][0]["axes_labels"] == {
            "x": "Time from go cue (s)",
            "y": y_label,
        }
        assert payload["metadata"]["shared_yaxes"] == "all"
        assert payload["facet_bindings"]["condition"]["item"] == "manifest"


def test_summary_spec_preserves_the_four_headline_metrics() -> None:
    payload = _json(FIGURE_ROOT / "summary_metrics" / "spec.json")
    assert payload["metadata"]["metric_order"] == [
        "within_cell_vel_rmse",
        "mean_hold_drift_mm",
        "mean_pre_go_rms_mm",
        "mean_peak_velocity",
    ]
    assert payload["metadata"]["pre_go_rms_target_mm"] == 0.5


def test_all_five_tracked_specs_execute_to_completed_figure_manifests(
    tmp_path: Path,
) -> None:
    register_rlrmp_figure_surfaces()
    cells = [
        {
            "run_id": "baseline",
            "display_name": "Baseline",
            "color": "#1f77b4",
            "forward_velocity": {
                "time": [-0.01, 0.0, 0.01],
                "mean": [0.0, 0.2, 0.4],
                "lower": [0.0, 0.1, 0.3],
                "upper": [0.0, 0.3, 0.5],
            },
            "hold_drift": {
                "time": [-0.01, 0.0],
                "mean": [0.1, 0.2],
                "lower": [0.05, 0.1],
                "upper": [0.15, 0.3],
            },
            "peak_velocity": [0.8, 0.9],
            "summary_metrics": {
                "within_cell_vel_rmse": 0.04,
                "mean_hold_drift_mm": 2.3,
                "mean_pre_go_rms_mm": 1.0,
                "mean_peak_velocity": 0.85,
            },
            "training_loss_per_term": {
                "step": [1, 10],
                "terms": {"effector_pos": [1.0, 0.2], "nn_output": [0.5, 0.1]},
            },
        },
        {
            "run_id": "prego",
            "display_name": "Pre-go penalty",
            "color": "#ff7f0e",
            "forward_velocity": {
                "time": [-0.01, 0.0, 0.01],
                "mean": [0.0, 0.15, 0.35],
                "lower": [0.0, 0.1, 0.3],
                "upper": [0.0, 0.2, 0.4],
            },
            "hold_drift": {
                "time": [-0.01, 0.0],
                "mean": [0.01, 0.02],
                "lower": [0.0, 0.01],
                "upper": [0.02, 0.03],
            },
            "peak_velocity": [0.7, 0.75],
            "summary_metrics": {
                "within_cell_vel_rmse": 0.03,
                "mean_hold_drift_mm": 0.1,
                "mean_pre_go_rms_mm": 0.08,
                "mean_peak_velocity": 0.725,
            },
            "training_loss_per_term": {
                "step": [1, 10],
                "terms": {"effector_pos": [0.9, 0.15], "nn_output_pre_go": [0.4, 0.05]},
            },
        },
    ]
    payload = standard_matrix_payload(
        cells,
        params={
            "metric_order": [
                "within_cell_vel_rmse",
                "mean_hold_drift_mm",
                "mean_pre_go_rms_mm",
                "mean_peak_velocity",
            ]
        },
    )
    analysis_manifest = AnalysisRunManifest(
        id="3702f54-representative-payload",
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            AnalysisRunSpec(analysis_type="rlrmp.standard_matrix_payload").model_dump(
                mode="json"
            ),
        ),
        metadata={"figure_payload": payload},
    )
    write_manifest(analysis_manifest, root=tmp_path)

    for topic in TOPICS:
        spec = FigureSpec.model_validate(_json(FIGURE_ROOT / topic / "spec.json"))
        spec = spec.model_copy(
            update={
                "inputs": [
                    ParentRef(
                        kind="AnalysisRunManifest",
                        id=analysis_manifest.id,
                        role="standard_matrix_payload",
                    )
                ]
            }
        )
        figure_manifest, manifest_path = execute_figure_spec(spec, root=tmp_path)

        assert figure_manifest.status == "completed"
        assert manifest_path.is_file()
        render_artifacts = [
            artifact
            for artifact in figure_manifest.artifacts
            if artifact.role == "figure_render"
        ]
        assert render_artifacts
        assert all(artifact.uri and Path(artifact.uri).is_file() for artifact in render_artifacts)
        if topic in {"forward_velocity_profiles", "hold_drift_profiles"}:
            assert figure_manifest.figure_spec.inline["metadata"]["shared_yaxes"] == "all"


def test_archived_oracles_preserve_labels_hashes_and_summary_values() -> None:
    product = read_data_product(ORACLE_ROOT / "figure_parity_oracles.json")
    assert product.product_schema_id == "rlrmp.figure_parity_oracles"
    assert len(product.artifacts) == 5
    for artifact in product.artifacts:
        assert artifact.uri is not None
        assert hashlib.sha256((REPO_ROOT / artifact.uri).read_bytes()).hexdigest() == (
            artifact.sha256
        )

    peak = _json(ORACLE_ROOT / "figure_parity_peak_velocity_distributions.json")
    summary = _json(ORACLE_ROOT / "figure_parity_summary_metrics.json")
    cells = peak["plot_kwargs"]["cells"]
    assert len(cells) == 10
    assert set(cells) == set(peak["cell_stats"])
    assert set(cells) == set(summary["cell_stats"])
    assert peak["cell_stats"]["lit__post_nojerk"]["mean_peak_velocity"] == (
        0.9685674905776978
    )
    assert summary["cell_stats"]["full_trial_pl__pos10_prego_1"][
        "mean_pre_go_rms_mm"
    ] == 0.07584098726511002


def test_legacy_figure_builders_and_save_sites_are_deleted() -> None:
    assert not PRODUCER.exists()
