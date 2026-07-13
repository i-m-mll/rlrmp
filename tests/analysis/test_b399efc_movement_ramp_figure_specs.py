"""Declarative closure guards for the b399efc movement-ramp figures."""

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
from rlrmp.figures import register_rlrmp_figure_surfaces, standard_matrix_payload


REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_ROOT = REPO_ROOT / "results" / "b399efc" / "figures"
TOPICS = {
    "forward_velocity_profiles": "rlrmp.profile_comparison",
    "hold_drift_profiles": "rlrmp.profile_comparison",
    "peak_velocity_distributions": "rlrmp.distribution_comparison",
    "summary_metrics": "rlrmp.metric_comparison",
    "training_loss": "rlrmp.history_comparison",
    "training_loss_per_term": "rlrmp.history_comparison",
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _representative_cell() -> dict:
    return {
        "run_id": "movement_ramp__power6",
        "display_name": "power6",
        "color": "#9467bd",
        "forward_velocity": {
            "time": [-0.01, 0.0, 0.01],
            "mean": [0.0, 0.2, 0.4],
            "lower": [0.0, 0.1, 0.3],
            "upper": [0.0, 0.3, 0.5],
        },
        "hold_drift": {
            "time": [-0.01, 0.0],
            "mean": [0.01, 0.02],
            "lower": [0.0, 0.01],
            "upper": [0.02, 0.03],
        },
        "peak_velocity": [1.3, 1.4],
        "summary_metrics": {
            "within_cell_vel_rmse": 0.12869667150080205,
            "mean_peak_velocity": 1.3973475694656372,
            "mean_time_to_peak_steps": 37.1,
            "mean_hold_drift_mm": 0.016355032101273537,
        },
        "training_loss": {"step": [1, 10], "series": {"total": [1.0, 0.15]}},
        "training_loss_per_term": {
            "step": [1, 10],
            "terms": {"effector_pos": [0.9, 0.1], "nn_output": [0.2, 0.05]},
        },
    }


def test_all_six_specs_execute_to_hash_verified_completed_manifests(tmp_path: Path) -> None:
    register_rlrmp_figure_surfaces()
    payload = standard_matrix_payload(
        [_representative_cell()],
        {"metric_order": list(_representative_cell()["summary_metrics"])},
    )
    analysis = AnalysisRunManifest(
        id="b399efc-standard-matrix-representative",
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
        role="standard_matrix_payload",
    )

    manifests = []
    for topic in TOPICS:
        tracked = FigureSpec.model_validate(_json(FIGURE_ROOT / topic / "spec.json"))
        manifest, manifest_path = execute_figure_spec(
            tracked.model_copy(update={"inputs": [parent]}),
            root=tmp_path,
            issues=["c9b1db0"],
        )
        assert manifest.status == "completed"
        assert manifest_path.is_file()
        renders = [item for item in manifest.artifacts if item.role == "figure_render"]
        assert renders
        for artifact in renders:
            assert artifact.uri is not None
            path = Path(artifact.uri)
            assert path.is_file()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact.sha256
        manifests.append(manifest)
    assert len(manifests) == 6
    assert len({manifest.id for manifest in manifests}) == 6


def test_profiles_preserve_shared_y_and_intrinsic_axes() -> None:
    expected = {
        "forward_velocity_profiles": "Forward velocity (m/s)",
        "hold_drift_profiles": "Forward position drift (mm)",
    }
    for topic, y_label in expected.items():
        payload = _json(FIGURE_ROOT / topic / "spec.json")
        assert payload["panels"][0]["axes_labels"] == {
            "x": "Time from go cue (s)",
            "y": y_label,
        }
        assert payload["metadata"]["shared_yaxes"] == "all"


def test_summary_preserves_movement_ramp_metric_order() -> None:
    payload = _json(FIGURE_ROOT / "summary_metrics" / "spec.json")
    assert payload["metadata"]["metric_order"] == [
        "within_cell_vel_rmse",
        "mean_peak_velocity",
        "mean_time_to_peak_steps",
        "mean_hold_drift_mm",
    ]
