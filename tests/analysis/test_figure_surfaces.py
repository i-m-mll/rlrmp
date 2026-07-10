"""Focused contracts for RLRMP's declarative figure surfaces."""

from __future__ import annotations

import json
from pathlib import Path

import rlrmp
from feedbax.analysis.figures import execute_figure_spec
from feedbax.config.yaml import get_yaml_loader
from feedbax.contracts.figures import FigureSpec
from feedbax.plot.constructors import get_figure_constructor, get_figure_template

from rlrmp.figures import effector_trajectory_spec, register_rlrmp_figure_surfaces


def _render_payload(manifest) -> dict:
    artifact = next(
        artifact
        for artifact in manifest.artifacts
        if artifact.role == "figure_render" and artifact.media_type == "application/json"
    )
    return json.loads(Path(artifact.uri).read_text(encoding="utf-8"))


def test_compact_lqg_piece_spec_is_a_live_native_consumer(tmp_path: Path) -> None:
    register_rlrmp_figure_surfaces()
    spec_path = (
        Path(rlrmp.__file__).parent
        / "config"
        / "figure_specs"
        / "lqg_baseline_profile.yml"
    )
    text = spec_path.read_text(encoding="utf-8")
    assert sum(bool(line.strip()) for line in text.splitlines()) <= 20
    payload = get_yaml_loader(typ="safe").load(text)
    payload["metadata"] = {
        "facets": {
            "forward_velocity_profiles": {
                "trained": {
                    "run_id": "trained",
                    "display_name": "Trained policy",
                    "color": "#1f77b4",
                    "forward_velocity": {
                        "time": [0.0, 1.0],
                        "mean": [0.0, 0.9],
                        "lower": [0.0, 0.8],
                        "upper": [0.0, 1.0],
                    },
                }
            }
        },
        "baseline": {
            "label": "LQG (analytical)",
            "color": "rgb(17,24,39)",
            "profile": {"time": [0.0, 1.0], "y": [[0.0, 0.8]]},
        },
    }
    spec = FigureSpec.model_validate(payload)

    manifest, _path = execute_figure_spec(spec, root=tmp_path, issues=["9977ff0"])

    assert manifest.status == "completed"
    assert [piece.name for piece in manifest.resolved_pieces] == [
        "rlrmp.lqg_baseline_rollout"
    ]
    assert any(
        record.name == "lqg-baseline" and record.status == "included"
        for record in manifest.binding_records
    )
    assert "LQG (analytical)" in [trace.get("name") for trace in _render_payload(manifest)["data"]]
    assert all(
        get_figure_constructor(key).tier != "custom_figure"
        for key in manifest.constructor_versions
    )


def test_effector_template_executes_native_slot_facet_composition(tmp_path: Path) -> None:
    register_rlrmp_figure_surfaces()
    spec = effector_trajectory_spec(
        name="effector-native",
        variables={
            "position": {"trajectories": [[[0.0, 0.0], [0.5, 1.0], [1.0, 1.0]]]},
            "velocity": {"trajectories": [[[0.0, 0.0], [0.25, 0.5], [0.0, 1.0]]]},
        },
    )

    manifest, _path = execute_figure_spec(spec, root=tmp_path, issues=["9977ff0"])

    template = get_figure_template("rlrmp.effector_trajectories_2d")
    assert get_figure_constructor(template.assembler).tier == "figure"
    assert manifest.status == "completed"
    assert set(manifest.constructor_versions) >= {
        "feedbax.trajectory_2d",
        "feedbax.endpoint_markers",
        "rlrmp.comparison_grid",
        "feedbax.trajectories_2d_row",
    }
    assert len([record for record in manifest.binding_records if record.status == "included"]) == 4
    render = _render_payload(manifest)
    assert len(render["data"]) >= 6
    assert all(
        get_figure_constructor(key).tier != "custom_figure"
        for key in manifest.constructor_versions
    )
