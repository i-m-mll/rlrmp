from __future__ import annotations

from pathlib import Path

from feedbax.manifest import TrainingRunManifest, load_manifest, sha256_file
from feedbax.manifest_index import index_manifest_file


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "results" / "9455785"
RUN_SPEC = FIXTURE_ROOT / "runs" / "artifact_normalization_fixture__sample.json"
MANIFEST = (
    FIXTURE_ROOT
    / "manifests"
    / "training_runs"
    / "feedbax_training_run_rlrmp_artifact_normalization_fixture.json"
)


def test_artifact_manifest_fixture_loads_with_feedbax_contract() -> None:
    manifest = load_manifest(MANIFEST)

    assert isinstance(manifest, TrainingRunManifest)
    assert manifest.id == "feedbax-training-run:rlrmp-artifact-normalization-fixture"
    assert manifest.provider_version == "feedbax-provider.v1"
    assert {"9455785", "577806f", "64d3059", "58f07a0"} <= set(manifest.provenance.issues)
    assert {parent.kind for parent in manifest.provenance.parents} == {
        "MandibleArtifactProvider",
        "ProviderManifest",
    }


def test_artifact_manifest_fixture_normalizes_repo_relative_paths() -> None:
    manifest = load_manifest(MANIFEST)
    artifacts = {artifact.role: artifact for artifact in manifest.artifacts}

    tracked_spec = artifacts["tracked_run_spec"]
    bulk_checkpoint = artifacts["training_checkpoint"]

    assert tracked_spec.uri == str(RUN_SPEC.relative_to(REPO_ROOT))
    assert tracked_spec.sha256 == sha256_file(RUN_SPEC)
    assert tracked_spec.metadata["availability"] == "checked_in"
    assert tracked_spec.metadata["normalized_from"].startswith("/Users/example/")

    assert bulk_checkpoint.uri == (
        "_artifacts/9455785/runs/artifact_normalization_fixture__sample/trained_model.eqx"
    )
    assert bulk_checkpoint.sha256 is None
    assert bulk_checkpoint.metadata["availability"] == "reference_only"
    assert bulk_checkpoint.metadata["manifest_root"] == "_artifacts/feedbax_runs/"
    assert bulk_checkpoint.metadata["normalized_from"].startswith("/Users/example/")

    for artifact in manifest.artifacts:
        assert artifact.uri is not None
        assert not Path(artifact.uri).is_absolute()


def test_artifact_manifest_fixture_indexes_with_feedbax(tmp_path: Path) -> None:
    db_path = tmp_path / "feedbax.sqlite"

    index_manifest_file(MANIFEST, db_path=db_path)

    assert db_path.is_file()
