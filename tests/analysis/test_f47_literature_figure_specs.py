"""Declarative closure guards for the f47abb1 literature-replication figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from feedbax.analysis.figures import execute_figure_spec
from feedbax.analysis.specs import AnalysisRunSpec
from feedbax.contracts.figures import FigureSpec
from feedbax.contracts.manifest import (
    AnalysisRunManifest,
    ParentRef,
    spec_payload,
    write_manifest,
)
import pytest

from rlrmp.data_products.envelope import read_data_product
from rlrmp.figures import (
    STANDARD_MATRIX_PAYLOAD_SCHEMA_VERSION,
    register_rlrmp_figure_surfaces,
    standard_matrix_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_ROOT = REPO_ROOT / "results" / "f47abb1" / "figures"
ORACLE_ROOT = REPO_ROOT / "results" / "f47abb1" / "data_products"
TOPICS = {
    "forward_velocity_profiles": "rlrmp.profile_comparison",
    "hold_drift_profiles": "rlrmp.profile_comparison",
    "peak_velocity_distributions": "rlrmp.distribution_comparison",
    "rmse_ratio_comparison": "rlrmp.metric_comparison",
    "training_loss": "rlrmp.history_comparison",
    "training_loss_per_term": "rlrmp.history_comparison",
}
RETIRED_PRODUCERS = (
    "analyse_lit_replication_6cell.py",
    "plot_training_loss_lit_replication.py",
)


pytestmark = pytest.mark.feedbax_contract


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_six_topics_are_native_manifest_data_bound_specs() -> None:
    for topic, template in TOPICS.items():
        payload = _json(FIGURE_ROOT / topic / "spec.json")
        spec = FigureSpec.model_validate(payload)
        assert spec.template == template
        assert spec.assembler is None
        assert spec.slot_bindings
        assert spec.facet_bindings
        assert spec.figure_routing["experiment"] == "f47abb1"
        assert spec.figure_routing["topic"] == topic
        assert all(binding.item == "manifest" for binding in spec.facet_bindings.values())
        assert payload["metadata"]["parity_oracle"] == (
            f"results/f47abb1/data_products/figure_parity_{topic}.json"
        )


def test_one_standard_matrix_payload_owns_all_six_figure_facets() -> None:
    payload = standard_matrix_payload(
        [_representative_cell()],
        {"metric_order": ["vel_rmse_ratio", "pos_rmse_ratio", "cv_peak_vel"]},
    )
    assert payload["schema_version"] == STANDARD_MATRIX_PAYLOAD_SCHEMA_VERSION
    assert set(payload["facets"]) >= {
        "forward_velocity_profiles",
        "hold_drift_profiles",
        "peak_velocity_distributions",
        "summary_metrics",
        "training_loss",
        "training_loss_per_term",
    }
    assert payload["facets"]["forward_velocity_profiles"]["lit__flat_jerk"] == (
        _representative_cell()
    )


def test_all_six_tracked_specs_execute_to_completed_figure_manifests(
    tmp_path: Path,
) -> None:
    register_rlrmp_figure_surfaces()
    payload = standard_matrix_payload(
        [_representative_cell()],
        {"metric_order": ["vel_rmse_ratio", "pos_rmse_ratio", "cv_peak_vel"]},
    )
    analysis = AnalysisRunManifest(
        id="f47abb1-standard-matrix-representative",
        status="completed",
        analysis_spec=spec_payload(
            "AnalysisRunSpec",
            AnalysisRunSpec(analysis_type="rlrmp.standard_matrix").model_dump(mode="json"),
        ),
        metadata={"figure_payload": payload},
    )
    write_manifest(analysis, root=tmp_path)
    parent = ParentRef(
        kind="AnalysisRunManifest",
        id=analysis.id,
        role="standard_matrix_analysis",
    )

    manifests = []
    for topic in TOPICS:
        tracked = FigureSpec.model_validate(_json(FIGURE_ROOT / topic / "spec.json"))
        manifest, manifest_path = execute_figure_spec(
            tracked.model_copy(update={"inputs": [parent]}),
            root=tmp_path,
            issues=["28876c1"],
        )
        assert manifest.status == "completed"
        assert manifest_path.is_file()
        assert manifest.resolved_inputs == [parent]
        renders = [artifact for artifact in manifest.artifacts if artifact.role == "figure_render"]
        assert renders
        for artifact in renders:
            assert artifact.uri is not None
            render_path = Path(artifact.uri)
            assert render_path.is_file()
            assert hashlib.sha256(render_path.read_bytes()).hexdigest() == artifact.sha256
        manifests.append(manifest)

    assert len(manifests) == 6
    assert len({manifest.id for manifest in manifests}) == 6


def _representative_cell() -> dict:
    return {
        "run_id": "lit__flat_jerk",
        "display_name": "Flat + jerk",
        "color": "#1f77b4",
        "forward_velocity": {"series": [{"profile": {"time": [0.0], "mean": [0.5]}}]},
        "hold_drift": {"series": [{"profile": {"time": [-0.01], "mean": [0.2]}}]},
        "peak_velocity": [0.5, 0.6],
        "summary_metrics": {
            "vel_rmse_ratio": 1.1348947253917163,
            "pos_rmse_ratio": 1.1874486696322288,
            "cv_peak_vel": 0.09163960128569927,
        },
        "training_loss": {"x": [1, 2], "series": {"total": [6.2, 6.0]}},
        "training_loss_per_term": {"x": [1, 2], "series": {"position": [4.0, 3.8]}},
    }


def test_profiles_preserve_shared_y_and_intrinsic_axes() -> None:
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


def test_rmse_metrics_preserve_primary_secondary_and_auxiliary_order() -> None:
    payload = _json(FIGURE_ROOT / "rmse_ratio_comparison" / "spec.json")
    assert payload["metadata"]["metric_order"] == [
        "vel_rmse_ratio",
        "pos_rmse_ratio",
        "cv_peak_vel",
    ]


def test_governed_oracles_preserve_archived_literature_values() -> None:
    product = read_data_product(ORACLE_ROOT / "figure_parity_oracles.json")
    assert product.product_schema_id == "rlrmp.figure_parity_oracles"
    assert len(product.artifacts) == 6
    for artifact in product.artifacts:
        assert artifact.uri is not None
        assert hashlib.sha256((REPO_ROOT / artifact.uri).read_bytes()).hexdigest() == (
            artifact.sha256
        )

    peak = _json(ORACLE_ROOT / "figure_parity_peak_velocity_distributions.json")
    rmse = _json(ORACLE_ROOT / "figure_parity_rmse_ratio_comparison.json")
    loss = _json(ORACLE_ROOT / "figure_parity_training_loss.json")
    assert peak["cell_stats"]["lit__full_nojerk"]["mean_peak_velocity"] == (
        0.9638093709945679
    )
    assert rmse["rmse_ratios"]["lit__post_nojerk"]["vel_rmse_ratio"] == (
        1.1127220754450111
    )
    assert loss["end_of_training_stats"]["lit__full_nojerk"]["final_mean"] == (
        0.040267013013362885
    )


def test_imperative_f47_producers_and_exclusive_driver_hooks_are_deleted() -> None:
    scripts = REPO_ROOT / "results" / "f47abb1" / "scripts"
    for producer in RETIRED_PRODUCERS:
        assert not (scripts / producer).exists()
    assert not (REPO_ROOT / "src/rlrmp/analysis/multi_cell_driver.py").exists()


def test_eval_layer_does_not_import_the_legacy_pilot_figure_pipeline() -> None:
    source = (REPO_ROOT / "src/rlrmp/eval/perturbation_bank.py").read_text(encoding="utf-8")
    assert "rlrmp.analysis.pipelines.gru_pilot_figures" not in source
    assert "rlrmp.eval.trial_inputs" in source
