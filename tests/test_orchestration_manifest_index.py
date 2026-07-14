from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest

from feedbax.analysis import (
    EvaluationInputPathError,
    StagedExactParents,
    resolve_evaluation_inputs,
)
from feedbax.analysis.bundles import iter_candidate_manifests
from feedbax.contracts.manifest import EvaluationRunSpec, TrainingRunManifest
from feedbax.persistence import (
    ImmutableArtifactBlobProvider,
    ImmutableArtifactBlobProviderSpec,
    open_immutable_artifact_blob_provider,
)
from rlrmp.train.orchestration_manifest_index import (
    OrchestrationManifestIndexError,
    OrchestrationManifestIndexSpec,
    RegisteredRunSetLocation,
    build_orchestration_manifest_index,
    load_orchestration_manifest_index,
    materialize_orchestration_manifest_index,
    select_exact_training_manifest_refs,
    write_orchestration_manifest_index,
)

SYNTHETIC_ROWS = tuple(
    (f"synthetic-set-{index}", f"synthetic-row-{index}", f"synthetic-training-run-{index}")
    for index in range(4)
)
CORE_CHECK_IDS = (
    "checkpoint_cadence",
    "completed_batches",
    "environment_fingerprint",
    "execution_identity",
    "events_terminal",
    "lr_trace",
    "manifest_valid",
    "seeds",
)
REAL_M1_ROWS = (
    (
        "2026-07-13-b5e80253",
        "force_visible__nominal_seed42_smoke100",
        "feedbax-training-run:13ba53f325a05f24be910385774c1872",
        "36412dcf4db037094151f506afa9c2c86d24e9fae91b9bcb6e7fb34cefd6ea5a",
    ),
    (
        "2026-07-13-1ac1bcee",
        "force_hidden__nominal_seed42_smoke100",
        "feedbax-training-run:97c76892178bd32eadcc8eefb834bfd6",
        "2fe9f046392f66c72ddfdcc2a60ba36e6c0a5784f30952d2ba7360b8f31b843f",
    ),
    (
        "2026-07-13-bd90c6fc",
        "force_visible__broad_pgd_seed42_smoke100",
        "feedbax-training-run:99ef061bf8b05f8761db7483e75a2512",
        "0c475344e96ae9a902bc98ddcdb2862e01d4b08c6d070cc2325adce8fb5a4d80",
    ),
    (
        "2026-07-13-43c9cd35",
        "force_hidden__broad_pgd_seed42_smoke100",
        "feedbax-training-run:6ad196b423dec55afbf1816bc012c76d",
        "c87b389fcd4e61afc4de761c39c9f02598c70bffd366a3f7aa73cf4b6e3cd503",
    ),
)


@dataclass
class RunSetFixture:
    location: RegisteredRunSetLocation
    source: Path
    custody: Path
    certificate: Path
    provider: ImmutableArtifactBlobProvider


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_run_set(
    tmp_path: Path,
    *,
    run_set_id: str = "synthetic-set-a",
    row_id: str = "synthetic-row-a",
    manifest_id: str = "synthetic-training-run-a",
    manifest_status: str = "completed",
    registration_status: str = "completed",
    certificate_status: str = "pass",
) -> RunSetFixture:
    run_set_dir = tmp_path / "orchestration" / run_set_id
    source = run_set_dir / "collected" / row_id / "manifest.json"
    source.parent.mkdir(parents=True)
    manifest = TrainingRunManifest(
        id=manifest_id,
        status=manifest_status,
        run_set_id=run_set_id,
        job_id=manifest_id,
        metadata={
            "training_row_provenance": {
                "row_id": row_id,
                "planned_run_id": manifest_id,
            }
        },
    )
    source.write_text(manifest.model_dump_json(indent=2, exclude_none=True) + "\n")
    bundle = {
        "run_set_id": run_set_id,
        "rows": [
            {
                "row_id": row_id,
                "execution": {
                    "row_provenance": {
                        "row_id": row_id,
                        "planned_run_id": manifest_id,
                    }
                },
            }
        ],
    }
    (run_set_dir / "bundle.json").write_text(json.dumps(bundle))
    certificate = run_set_dir / "conformance.json"
    checks = [{"check_id": check_id, "status": certificate_status} for check_id in CORE_CHECK_IDS]
    certificate.write_text(
        json.dumps(
            {
                "schema_id": "feedbax.run_conformance",
                "schema_version": "feedbax.run_conformance.v1",
                "run_set_id": run_set_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "overall": certificate_status,
                "rows": {row_id: {"checks": checks}},
            },
            sort_keys=True,
        )
        + "\n"
    )
    registration = {
        "run_set_id": run_set_id,
        "status": registration_status,
        "certificate_overall": certificate_status,
        "certificate_sha256": _sha256(certificate),
    }
    (run_set_dir / "registration.json").write_text(json.dumps(registration))
    provider = open_immutable_artifact_blob_provider(
        ImmutableArtifactBlobProviderSpec(),
        explicit_root=tmp_path / "durable-custody",
    )
    artifact = provider.store_bytes(
        source.read_bytes(),
        role="training_run_manifest",
        logical_name=manifest_id,
        media_type="application/json",
    )
    assert artifact.uri is not None
    digest = _sha256(source)
    custody = provider.root / "artifacts" / "sha256" / digest[:2] / digest
    return RunSetFixture(
        location=RegisteredRunSetLocation(
            run_set_dir=run_set_dir,
            expected_row_ids=(row_id,),
            manifest_uris={row_id: artifact.uri},
        ),
        source=source,
        custody=custody,
        certificate=certificate,
        provider=provider,
    )


def _combined_provider(fixtures: list[RunSetFixture]) -> ImmutableArtifactBlobProvider:
    roots = {fixture.provider.root for fixture in fixtures}
    assert len(roots) == 1
    return fixtures[0].provider


def _build(fixture: RunSetFixture) -> OrchestrationManifestIndexSpec:
    return build_orchestration_manifest_index(
        [fixture.location],
        index_id="synthetic-index",
        bundle_id="synthetic-bundle",
        manifest_provider=fixture.provider,
    )


def test_four_synthetic_refs_are_deterministic_and_materialize_regular_files(
    tmp_path: Path,
) -> None:
    fixtures = [
        _write_run_set(tmp_path, run_set_id=run_set, row_id=row, manifest_id=manifest)
        for run_set, row, manifest in SYNTHETIC_ROWS
    ]
    provider = _combined_provider(fixtures)
    forward = build_orchestration_manifest_index(
        [fixture.location for fixture in fixtures],
        index_id="synthetic-four-index",
        bundle_id="synthetic-four-bundle",
        manifest_provider=provider,
    )
    reordered = build_orchestration_manifest_index(
        [fixtures[2].location, fixtures[0].location, fixtures[3].location, fixtures[1].location],
        index_id="synthetic-four-index",
        bundle_id="synthetic-four-bundle",
        manifest_provider=provider,
    )
    assert forward.canonical_bytes() == reordered.canonical_bytes()
    expected_rows = [(run_set, row) for run_set, row, _manifest in SYNTHETIC_ROWS]
    assert len(select_exact_training_manifest_refs(forward, expected_rows=expected_rows)) == 4
    before = {fixture.custody: _sha256(fixture.custody) for fixture in fixtures}
    materialized = materialize_orchestration_manifest_index(
        forward,
        manifest_provider=provider,
        target_root=tmp_path / "clean-checkout" / "feedbax_runs",
        expected_rows=expected_rows,
    )
    assert all(path.is_file() and not path.is_symlink() for path in materialized.manifest_paths)
    assert {_sha256(path) for path in materialized.manifest_paths} == set(before.values())
    assert all(_sha256(path) == digest for path, digest in before.items())
    assert materialized.index_path.read_bytes() == forward.canonical_bytes()
    discovered = iter_candidate_manifests(materialized.root, manifest_kind="TrainingRunManifest")
    assert {manifest.id for manifest in discovered} == {ref.id for ref in forward.refs}
    assert isinstance(materialized.exact_parents, StagedExactParents)
    assert [entry.parent for entry in materialized.exact_parents.parents] == list(
        materialized.parent_refs
    )
    for entry in materialized.exact_parents.parents:
        execution_ref = entry.parent.model_copy(update={"uri": entry.execution_uri})
        resolved = resolve_evaluation_inputs(
            EvaluationRunSpec(evaluation_type="test.exact", inputs=[execution_ref]),
            manifest_root=materialized.root,
            require_unique_manifest_id=False,
        )
        assert resolved[0].ref == execution_ref
        assert resolved[0].sha256 == entry.parent.metadata["manifest_sha256"]


def test_four_real_m1_refs_materialize_after_transient_source_removal(tmp_path: Path) -> None:
    """Exercise the reviewed custody/resolver APIs below the pending staged CLI binding."""

    checkout_root = Path(__file__).resolve().parents[1]
    orchestration_root = checkout_root / "_artifacts" / "orchestration"
    source_dirs = [orchestration_root / run_set_id for run_set_id, *_rest in REAL_M1_ROWS]
    if not all(source_dir.is_dir() for source_dir in source_dirs):
        pytest.skip("the four preserved M1 orchestration run sets are not linked in this checkout")

    provider = open_immutable_artifact_blob_provider(
        ImmutableArtifactBlobProviderSpec(),
        explicit_root=tmp_path / "manifest-custody",
    )
    copied_root = tmp_path / "construction-sources"
    locations: list[RegisteredRunSetLocation] = []
    expected_facts: dict[tuple[str, str], tuple[str, str]] = {}
    for run_set_id, row_id, manifest_id, expected_sha256 in REAL_M1_ROWS:
        copied_run_set = copied_root / run_set_id
        shutil.copytree(orchestration_root / run_set_id, copied_run_set)
        manifest_path = copied_run_set / "collected" / row_id / "manifest.json"
        assert _sha256(manifest_path) == expected_sha256
        artifact = provider.store_bytes(
            manifest_path.read_bytes(),
            role="training_run_manifest",
            logical_name=manifest_id,
            media_type="application/json",
        )
        assert artifact.uri == f"artifact://sha256/{expected_sha256}"
        locations.append(
            RegisteredRunSetLocation(
                run_set_dir=copied_run_set,
                expected_row_ids=(row_id,),
                manifest_uris={row_id: artifact.uri},
            )
        )
        expected_facts[(run_set_id, row_id)] = (manifest_id, expected_sha256)

    index = build_orchestration_manifest_index(
        locations,
        index_id="m1-completed-training-manifests-v1",
        bundle_id="m1-four-completed-training-runs",
        manifest_provider=provider,
    )
    reordered = build_orchestration_manifest_index(
        [locations[2], locations[0], locations[3], locations[1]],
        index_id="m1-completed-training-manifests-v1",
        bundle_id="m1-four-completed-training-runs",
        manifest_provider=provider,
    )
    assert reordered == index
    assert reordered.canonical_bytes() == index.canonical_bytes()
    assert {
        (ref.metadata.run_set_id, ref.metadata.row_id): (
            ref.id,
            ref.metadata.manifest_sha256,
        )
        for ref in index.refs
    } == expected_facts

    cancelled_run_set_id = "2026-07-13-a1c88ba4"
    cancelled_row_id = "force_visible__nominal_seed42_smoke100"
    cancelled_run_set = orchestration_root / cancelled_run_set_id
    if not cancelled_run_set.is_dir():
        pytest.skip("the preserved cancelled M1 run set is not linked in this checkout")
    cancelled_path = cancelled_run_set / "collected" / cancelled_row_id / "manifest.json"
    cancelled_manifest = TrainingRunManifest.model_validate_json(cancelled_path.read_bytes())
    cancelled_sha256 = _sha256(cancelled_path)
    assert cancelled_manifest.status == "cancelled"
    assert cancelled_manifest.metadata["training_row_provenance"]["row_id"] == cancelled_row_id
    assert not any(
        ref.metadata.run_set_id == cancelled_run_set_id
        or ref.metadata.manifest_sha256 == cancelled_sha256
        for ref in index.refs
    )
    # The cancelled attempt reused a later completed run's planned manifest ID.
    # Exact run-set + row + digest identity prevents that shared ID from
    # conflating the cancelled bytes with the completed frozen ref.
    assert any(ref.id == cancelled_manifest.id for ref in index.refs)
    cancelled_artifact = provider.store_bytes(
        cancelled_path.read_bytes(),
        role="training_run_manifest",
        logical_name=cancelled_manifest.id,
        media_type="application/json",
    )
    assert cancelled_artifact.uri is not None
    with pytest.raises(OrchestrationManifestIndexError, match="is not completed"):
        build_orchestration_manifest_index(
            [
                RegisteredRunSetLocation(
                    run_set_dir=cancelled_run_set,
                    expected_row_ids=(cancelled_row_id,),
                    manifest_uris={cancelled_row_id: cancelled_artifact.uri},
                )
            ],
            index_id="must-reject-cancelled-m1",
            bundle_id="must-reject-cancelled-m1",
            manifest_provider=provider,
        )

    shutil.rmtree(copied_root)
    expected_rows = [(ref.metadata.run_set_id, ref.metadata.row_id) for ref in index.refs]
    materialized = materialize_orchestration_manifest_index(
        index,
        manifest_provider=provider,
        target_root=tmp_path / "clean-checkout" / "feedbax_runs",
        expected_rows=expected_rows,
    )
    assert all(
        path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
        for path in materialized.manifest_paths
    )
    assert [entry.parent for entry in materialized.exact_parents.parents] == [
        ref.to_parent_ref() for ref in index.refs
    ]
    for entry in materialized.exact_parents.parents:
        execution_ref = entry.parent.model_copy(update={"uri": entry.execution_uri})
        resolved = resolve_evaluation_inputs(
            EvaluationRunSpec(evaluation_type="test.m1.exact", inputs=[execution_ref]),
            manifest_root=materialized.root,
            require_unique_manifest_id=False,
        )
        assert resolved[0].id == entry.parent.id
        assert resolved[0].sha256 == entry.parent.metadata["manifest_sha256"]


def test_tracked_index_round_trip_is_canonical_and_ref_only(tmp_path: Path) -> None:
    fixture = _write_run_set(tmp_path)
    index = _build(fixture)
    path = write_orchestration_manifest_index(
        index, tmp_path / "tracked" / "index.json", manifest_provider=fixture.provider
    )
    loaded = load_orchestration_manifest_index(path, manifest_provider=fixture.provider)
    assert loaded.canonical_bytes() == path.read_bytes()
    payload = loaded.model_dump(mode="json")
    assert set(payload) == {"schema_id", "schema_version", "index_id", "bundle_id", "refs"}
    assert loaded.refs[0].metadata.conformance_overall == "pass"


def test_selection_missing_extra_and_duplicate_fail_closed(tmp_path: Path) -> None:
    fixture = _write_run_set(tmp_path)
    index = _build(fixture)
    with pytest.raises(OrchestrationManifestIndexError, match="missing=.*missing"):
        select_exact_training_manifest_refs(
            index,
            expected_rows=[("synthetic-set-a", "synthetic-row-a"), ("set", "missing")],
        )
    with pytest.raises(OrchestrationManifestIndexError, match="extra=.*synthetic-row-a"):
        select_exact_training_manifest_refs(index, expected_rows=[])
    with pytest.raises(ValueError, match="duplicate or ambiguous run-set/row"):
        OrchestrationManifestIndexSpec(
            index_id="duplicate", bundle_id="bundle", refs=(index.refs[0], index.refs[0])
        )


def test_selection_rejects_order_different_from_frozen_index(tmp_path: Path) -> None:
    first = _write_run_set(tmp_path, run_set_id="set-a", row_id="row-a")
    second = _write_run_set(
        tmp_path,
        run_set_id="set-b",
        row_id="row-b",
        manifest_id="synthetic-training-run-b",
    )
    index = build_orchestration_manifest_index(
        [second.location, first.location],
        index_id="ordered",
        bundle_id="bundle",
        manifest_provider=_combined_provider([first, second]),
    )
    with pytest.raises(OrchestrationManifestIndexError, match="row order mismatch"):
        select_exact_training_manifest_refs(
            index,
            expected_rows=[("set-b", "row-b"), ("set-a", "row-a")],
        )


def test_same_id_different_bytes_and_tampering_fail_closed(tmp_path: Path) -> None:
    first = _write_run_set(tmp_path, run_set_id="set-a", row_id="row-a")
    second = _write_run_set(
        tmp_path, run_set_id="set-b", row_id="row-b", manifest_id="synthetic-training-run-a"
    )
    with pytest.raises(OrchestrationManifestIndexError, match="same TrainingRunManifest id"):
        build_orchestration_manifest_index(
            [first.location, second.location],
            index_id="duplicate",
            bundle_id="bundle",
            manifest_provider=_combined_provider([first, second]),
        )
    index = _build(first)
    first.custody.write_bytes(first.custody.read_bytes() + b" ")
    with pytest.raises(OrchestrationManifestIndexError, match="size mismatch|hash mismatch"):
        materialize_orchestration_manifest_index(
            index,
            manifest_provider=first.provider,
            target_root=tmp_path / "output",
            expected_rows=[("set-a", "row-a")],
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("manifest_status", ["cancelled", "failed", "running"])
def test_noncompleted_manifest_fails_closed(tmp_path: Path, manifest_status: str) -> None:
    fixture = _write_run_set(tmp_path, manifest_status=manifest_status)
    with pytest.raises(OrchestrationManifestIndexError, match="is not completed"):
        _build(fixture)


def test_registration_and_certificate_hash_fail_closed(tmp_path: Path) -> None:
    fixture = _write_run_set(tmp_path, registration_status="failed")
    with pytest.raises(OrchestrationManifestIndexError, match="registration.*not completed"):
        _build(fixture)
    fixture = _write_run_set(tmp_path / "hash")
    fixture.certificate.write_bytes(fixture.certificate.read_bytes() + b" ")
    with pytest.raises(OrchestrationManifestIndexError, match="certificate hash mismatch"):
        _build(fixture)


@pytest.mark.parametrize("mode", ["missing", "extra", "empty", "malformed"])
def test_certificate_row_and_check_contract_fails_closed(tmp_path: Path, mode: str) -> None:
    fixture = _write_run_set(tmp_path)
    payload = json.loads(fixture.certificate.read_text())
    row_id = "synthetic-row-a"
    if mode == "missing":
        payload["rows"] = {}
    elif mode == "extra":
        payload["rows"]["extra-row"] = payload["rows"][row_id]
    elif mode == "empty":
        payload["rows"][row_id]["checks"] = []
    else:
        payload["rows"][row_id]["checks"] = [{"check_id": "manifest_valid"}]
    fixture.certificate.write_text(json.dumps(payload))
    registration = json.loads((fixture.location.run_set_dir / "registration.json").read_text())
    registration["certificate_sha256"] = _sha256(fixture.certificate)
    (fixture.location.run_set_dir / "registration.json").write_text(json.dumps(registration))
    with pytest.raises(OrchestrationManifestIndexError, match="certificate|checks|core"):
        _build(fixture)


def test_identity_fields_are_required_and_equal(tmp_path: Path) -> None:
    fixture = _write_run_set(tmp_path)
    bundle_path = fixture.location.run_set_dir / "bundle.json"
    bundle = json.loads(bundle_path.read_text())
    del bundle["run_set_id"]
    bundle_path.write_text(json.dumps(bundle))
    with pytest.raises(OrchestrationManifestIndexError, match="bundle requires.*run_set_id"):
        _build(fixture)

    fixture = _write_run_set(tmp_path / "row")
    bundle_path = fixture.location.run_set_dir / "bundle.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["rows"][0]["execution"]["row_provenance"]["row_id"] = "other"
    bundle_path.write_text(json.dumps(bundle))
    with pytest.raises(OrchestrationManifestIndexError, match="row_provenance.row_id"):
        _build(fixture)


def test_materialization_rejects_safe_key_collision_before_publication(tmp_path: Path) -> None:
    first = _write_run_set(
        tmp_path, run_set_id="set-a", row_id="row-a", manifest_id="synthetic:collision"
    )
    second = _write_run_set(
        tmp_path, run_set_id="set-b", row_id="row-b", manifest_id="synthetic/collision"
    )
    provider = _combined_provider([first, second])
    index = build_orchestration_manifest_index(
        [first.location, second.location],
        index_id="collision",
        bundle_id="bundle",
        manifest_provider=provider,
    )
    target = tmp_path / "published"
    with pytest.raises(OrchestrationManifestIndexError, match="collide"):
        materialize_orchestration_manifest_index(
            index,
            manifest_provider=provider,
            target_root=target,
            expected_rows=[("set-a", "row-a"), ("set-b", "row-b")],
        )
    assert not target.exists()
    assert not list(tmp_path.glob(".published.staging-*"))


def test_public_resolver_rejects_symlinked_manifest(tmp_path: Path) -> None:
    fixture = _write_run_set(tmp_path)
    index = _build(fixture)
    root = tmp_path / "symlink-root"
    relative = Path("manifests") / "training_runs" / "linked.json"
    path = root / relative
    path.parent.mkdir(parents=True)
    path.symlink_to(fixture.source)
    ref = index.refs[0].to_parent_ref().model_copy(update={"uri": relative.as_posix()})
    with pytest.raises(EvaluationInputPathError, match="symlink"):
        resolve_evaluation_inputs(
            EvaluationRunSpec(evaluation_type="test.exact", inputs=[ref]),
            manifest_root=root,
            require_unique_manifest_id=False,
        )


def test_missing_durable_uri_fails_before_publication(tmp_path: Path) -> None:
    fixture = _write_run_set(tmp_path)
    missing = RegisteredRunSetLocation(
        run_set_dir=fixture.location.run_set_dir,
        expected_row_ids=("synthetic-row-a",),
        manifest_uris={},
    )
    with pytest.raises(OrchestrationManifestIndexError, match="manifest URI rows"):
        build_orchestration_manifest_index(
            [missing],
            index_id="missing",
            bundle_id="bundle",
            manifest_provider=fixture.provider,
        )
