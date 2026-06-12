"""Tests for bridge manifest aggregation helpers."""

from __future__ import annotations

import pytest

from rlrmp.analysis.pipelines.bridge_aggregation import (
    BRIDGE_SUMMARY_FORMAT,
    BridgeManifestValidationError,
    bridge_manifest_row,
    read_bridge_summary,
    render_bridge_summary_markdown,
    summarize_bridge_manifests,
    validate_bridge_manifest,
    write_bridge_summary,
)
from rlrmp.analysis.pipelines.bridge_contracts import (
    BridgeCertificateComponent,
    BridgeRunManifest,
    BridgeRunSpec,
    make_bridge_run_id,
    write_bridge_manifest,
)


def _spec(label: str = "controller") -> BridgeRunSpec:
    return BridgeRunSpec(
        issue_id="8703ca0",
        run_id=make_bridge_run_id("bridge", label),
        objective="diagnostic",
        architecture="linear_recurrence",
        controller_label=label,
        optimizer_label="smoke",
        training_distribution="nominal",
        evaluation_lane="diagnostic",
        reference_controller="analytical_lqr",
        seed=3,
        gamma_factor=1.4,
        parameters={"horizon": 4, "nested": {"alpha": 0.25}},
    )


def _manifest(
    label: str = "controller",
    *,
    artifacts: dict[str, str] | None = None,
    components: tuple[BridgeCertificateComponent, ...] | None = None,
) -> BridgeRunManifest:
    return BridgeRunManifest(
        spec=_spec(label),
        status="smoke",
        metrics={"cost": {"mean": 1.25}, "terminal_error_m": 0.03},
        artifacts=artifacts
        if artifacts is not None
        else {"arrays": "_artifacts/8703ca0/arrays.npz"},
        certificate_components=components
        if components is not None
        else (
            BridgeCertificateComponent.available("recurrence_rollout", rms_error=0.0),
            BridgeCertificateComponent.not_applicable(
                "closed_loop_transition",
                "reference controller exposes no learned recurrence",
            ),
        ),
    )


def test_summarize_bridge_manifests_validates_and_flattens_rows(tmp_path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_bridge_manifest(_manifest("first"), first_path)
    write_bridge_manifest(_manifest("second"), second_path)

    summary = summarize_bridge_manifests(
        [first_path, second_path],
        required_artifact_labels=["arrays"],
        required_certificate_labels=["recurrence_rollout", "closed_loop_transition"],
    )

    assert summary["format"] == BRIDGE_SUMMARY_FORMAT
    assert [row["run_id"] for row in summary["rows"]] == ["bridge__first", "bridge__second"]
    assert summary["rows"][0]["source_path"] == str(first_path)
    assert summary["rows"][0]["parameter.nested.alpha"] == 0.25
    assert summary["rows"][0]["metric.cost.mean"] == 1.25
    assert summary["rows"][0]["artifact.arrays"] == "_artifacts/8703ca0/arrays.npz"
    assert summary["rows"][0]["certificate.recurrence_rollout.status"] == "available"
    assert summary["rows"][0]["certificate.recurrence_rollout.summary.rms_error"] == 0.0
    assert summary["rows"][0]["certificate.closed_loop_transition.status"] == "not_applicable"
    assert "reference controller" in summary["rows"][0]["certificate.closed_loop_transition.reason"]


def test_validate_bridge_manifest_rejects_missing_required_artifact() -> None:
    manifest = _manifest(artifacts={})

    with pytest.raises(BridgeManifestValidationError, match="missing required artifact label"):
        validate_bridge_manifest(manifest, required_artifact_labels=["arrays"])


def test_validate_bridge_manifest_rejects_missing_required_certificate_status() -> None:
    manifest = _manifest(
        components=(
            BridgeCertificateComponent(
                name="recurrence_rollout",
                status="missing",
                reason="not computed yet",
            ),
        )
    )

    with pytest.raises(BridgeManifestValidationError, match="status 'missing'"):
        validate_bridge_manifest(manifest, required_certificate_labels=["recurrence_rollout"])

    validate_bridge_manifest(
        manifest,
        required_certificate_labels=["recurrence_rollout"],
        allow_missing_certificate_components=True,
    )


def test_bridge_manifest_row_rejects_duplicate_flattened_labels() -> None:
    manifest = BridgeRunManifest(
        spec=BridgeRunSpec(
            **{
                **_spec().to_json_dict(),
                "parameters": {"nested.alpha": 1.0, "nested": {"alpha": 2.0}},
            }
        ),
        status="smoke",
    )

    with pytest.raises(BridgeManifestValidationError, match="duplicate bridge row label"):
        bridge_manifest_row(manifest)


def test_summary_json_and_markdown_outputs_are_stable(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    summary_path = tmp_path / "summary.json"
    write_bridge_manifest(_manifest("pipe"), path)
    summary = summarize_bridge_manifests([path])

    write_bridge_summary(summary, summary_path)
    observed = read_bridge_summary(summary_path)
    markdown = render_bridge_summary_markdown(observed["rows"])

    assert observed == summary
    assert markdown.splitlines()[0].startswith("| run_id | status | objective | architecture |")
    assert "| bridge__pipe | smoke | diagnostic | linear_recurrence |" in markdown
    assert "metric.cost.mean" in markdown
    assert "certificate.recurrence_rollout.status" in markdown
