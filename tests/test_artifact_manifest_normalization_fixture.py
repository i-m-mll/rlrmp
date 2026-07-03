from __future__ import annotations

import json
from pathlib import Path

import pytest
from feedbax.contracts.manifest import TrainingRunManifest, load_manifest, sha256_file
from feedbax.persistence.manifest_index import index_manifest_file

from rlrmp.runtime.spec_migrations import RUN_SPEC_SCHEMA_VERSION, ensure_rlrmp_spec_families


pytestmark = pytest.mark.feedbax_contract


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "results" / "9455785"
RUN_SPEC = FIXTURE_ROOT / "runs" / "artifact_normalization_fixture__sample.json"
MANIFEST = (
    FIXTURE_ROOT
    / "manifests"
    / "training_runs"
    / "feedbax_training_run_rlrmp_artifact_normalization_fixture.json"
)

ensure_rlrmp_spec_families()


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


def test_artifact_manifest_fixture_preserves_run_spec_parity() -> None:
    run_spec = json.loads(RUN_SPEC.read_text(encoding="utf-8"))
    manifest = load_manifest(MANIFEST)
    training_spec = manifest.training_spec

    assert training_spec is not None
    assert training_spec.kind == "RLRMPRunSpec"
    assert training_spec.schema_id == "rlrmp.run_spec"
    assert training_spec.schema_version == RUN_SPEC_SCHEMA_VERSION
    assert training_spec.ref == str(RUN_SPEC.relative_to(REPO_ROOT))
    assert training_spec.sha256 is not None
    assert training_spec.source_sha256 == sha256_file(RUN_SPEC)
    assert training_spec.metadata == {"source_record_role": "tracked_run_spec"}
    assert training_spec.inline == {
        "schema_id": "rlrmp.run_spec",
        "schema_version": RUN_SPEC_SCHEMA_VERSION,
        "migration_policy": "migrated_active_v1_to_v2",
        "run": run_spec["run"],
        "issue": run_spec["issue"],
        "training_mode": run_spec["training_summary"]["training_mode"],
        "n_train_batches": run_spec["training_summary"]["n_train_batches"],
    }

    assert manifest.provenance.metadata["rlrmp_layout"] == {
        "tracked_specs": "results/<issue>/runs/*.json",
        "bulk_artifacts": "_artifacts/<issue>/runs/<variant>/",
        "feedbax_manifest_root": run_spec["artifacts"]["feedbax_manifest_root"] + "/",
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
