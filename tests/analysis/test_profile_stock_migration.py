"""Contracts for the analytical/nominal profile-stock migration."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from feedbax.analysis.figures import execute_figure_spec
from feedbax.contracts.figures import FigureSpec
from feedbax.contracts.manifest import ArtifactRef
import pytest

from rlrmp.profile_payloads import (
    PROFILE_PAYLOAD_SCHEMA_ID,
    ProfilePayloadSpec,
    ProfileSeriesColumns,
    load_profile_payload_spec,
    materialize_profile_payload,
    register_profile_payload_piece,
)
from rlrmp.figures import register_rlrmp_figure_surfaces


pytestmark = pytest.mark.feedbax_contract

REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILIES = {
    "376d023": {
        "topic": "6d_analytical_velocity_profiles",
        "labels": [
            "6D extLQG analytical",
            "6D output-feedback H-infinity analytical",
            "020a65b h0 no-PGD",
            "020a65b h0 PGD",
        ],
        "spread": "sd",
        "axes": ("Time (s)", "Forward velocity (m/s)"),
    },
    "a378b34": {
        "topic": "nominal_velocity_profile_comparison",
        "labels": ["6D extLQG", "Distilled h0 GRU"],
        "spread": "sem",
        "axes": ("Time from trial start (s)", "Target-radial velocity (m/s)"),
    },
    "e148f33": {
        "topic": "nominal_velocity_profile_comparison",
        "labels": [
            "adaptive_curriculum_3500to1000 final",
            "08483d5 no-PGD 12k baseline",
            "6D analytical extLQG nominal",
            "6D output-feedback H-infinity nominal",
        ],
        "spread": "sd",
        "axes": ("Time (s)", "Forward velocity (m/s)"),
    },
}


def test_profile_stock_specs_are_native_and_data_free() -> None:
    for issue, expected in FAMILIES.items():
        topic = expected["topic"]
        figure_path = REPO_ROOT / "results" / issue / "figures" / topic / "spec.json"
        payload_path = (
            REPO_ROOT / "results" / issue / "notes" / "profile_payload_regeneration_spec.json"
        )

        raw_figure = json.loads(figure_path.read_text(encoding="utf-8"))
        figure = FigureSpec.model_validate(raw_figure)
        payload = load_profile_payload_spec(payload_path)

        assert figure.template == "rlrmp.profile_comparison"
        assert figure.inputs == []
        assert set(raw_figure) & {"data", "profile_summaries", "runtime_provenance"} == set()
        assert [series.label for series in payload.series] == expected["labels"]
        assert {series.spread_kind for series in payload.series} == {expected["spread"]}
        assert (figure.panels[0].axes_labels.x, figure.panels[0].axes_labels.y) == expected["axes"]
        assert figure.facet_bindings["condition"].path == "conditions"
        assert list(figure.metadata["conditions"]) == [payload.condition]


def test_profile_payload_materializes_and_executes_through_piece_custody(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "profiles.csv"
    csv_path.write_text(
        "time_s,reference_mean,reference_sd,model_mean,model_sd\n"
        "0.0,0.0,0.1,0.0,0.2\n"
        "0.1,1.0,0.1,0.8,0.2\n",
        encoding="utf-8",
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text('{"status":"archived parity oracle"}\n', encoding="utf-8")
    output_path = tmp_path / "payload.json"
    spec = ProfilePayloadSpec(
        piece_name="rlrmp.test_profile_payload",
        issue="fffffff",
        topic="profiles",
        condition="Nominal",
        source_csv=_artifact(csv_path, "profile_csv", "text/csv"),
        source_summary=_artifact(summary_path, "profile_summary", "application/json"),
        output=ArtifactRef(
            role="profile_figure_payload",
            logical_name="payload.json",
            media_type="application/json",
            uri=str(output_path),
        ),
        series=[
            ProfileSeriesColumns(
                label="Reference",
                mean_column="reference_mean",
                spread_column="reference_sd",
                spread_kind="sd",
                color="#111827",
            ),
            ProfileSeriesColumns(
                label="Model",
                mean_column="model_mean",
                spread_column="model_sd",
                spread_kind="sd",
                color="#2563eb",
            ),
        ],
    )

    payload = materialize_profile_payload(spec, repo_root=tmp_path)
    assert payload["schema_id"] == PROFILE_PAYLOAD_SCHEMA_ID
    assert payload["summary"] == {"status": "archived parity oracle"}
    reference = payload["cell"]["forward_velocity"]["series"][0]
    assert reference["profile"]["lower"] == [-0.1, 0.9]
    assert reference["profile"]["upper"] == [0.1, 1.1]

    register_rlrmp_figure_surfaces()
    register_profile_payload_piece(spec, repo_root=tmp_path)
    figure = FigureSpec(
        name="profile-payload-custody",
        template="rlrmp.profile_comparison",
        slot_bindings={
            "profiles": {
                "name": "profiles",
                "constructor": "rlrmp.profile_series",
                "piece": spec.piece_name,
                "required": True,
            }
        },
        panels=[
            {
                "name": "profile",
                "title": {"item": "condition"},
                "axes_labels": {"x": "Time (s)", "y": "Velocity (m/s)"},
            }
        ],
        facet_bindings={"condition": {"item": "params", "path": "conditions"}},
        metadata={"conditions": {"Nominal": {}}},
    )
    manifest, _path = execute_figure_spec(figure, root=tmp_path / "feedbax")

    assert manifest.status == "completed"
    assert [piece.name for piece in manifest.resolved_pieces] == [spec.piece_name]
    assert [record.status for record in manifest.binding_records] == ["included"]


def test_imperative_profile_stock_producers_are_deleted() -> None:
    paths = (
        "results/376d023/scripts/materialize_6d_analytical_velocity_profiles.py",
        "results/a378b34/scripts/materialize_nominal_velocity_profile_comparison.py",
        "results/e148f33/scripts/materialize_nominal_velocity_profile_comparison.py",
    )
    assert not any((REPO_ROOT / path).exists() for path in paths)


def _artifact(path: Path, role: str, media_type: str) -> ArtifactRef:
    return ArtifactRef(
        role=role,
        logical_name=path.name,
        sha256=sha256(path.read_bytes()).hexdigest(),
        media_type=media_type,
        uri=str(path),
    )
