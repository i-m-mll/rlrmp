"""Contracts for the analytical/nominal profile-stock migration."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from feedbax.analysis.figures import execute_figure_spec
from feedbax.contracts.figures import FigureSpec
from feedbax.contracts.manifest import ArtifactRef

from rlrmp.profile_payloads import (
    PROFILE_PAYLOAD_SCHEMA_ID,
    ProfilePayloadSpec,
    ProfileSeriesColumns,
    materialize_profile_payload,
    register_profile_payload_piece,
)
from rlrmp.figures import register_rlrmp_figure_surfaces


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
def _artifact(path: Path, role: str, media_type: str) -> ArtifactRef:
    return ArtifactRef(
        role=role,
        logical_name=path.name,
        sha256=sha256(path.read_bytes()).hexdigest(),
        media_type=media_type,
        uri=str(path),
    )
