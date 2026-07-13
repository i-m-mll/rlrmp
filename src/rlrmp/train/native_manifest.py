"""RLRMP-owned companion contract for native Feedbax training manifests."""

from __future__ import annotations

from typing import Any, Literal

from feedbax.contracts import (
    TrainingManifestMetadataProjection,
    TrainingManifestMetadataProjectionRegistration,
)
from feedbax.contracts.spec_storage import training_spec_sha256
from feedbax.contracts.training import TrainingMethodRegistry
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rlrmp.runtime.spec_migrations import (
    RUN_SPEC_KIND,
    RUN_SPEC_SCHEMA_ID,
    RUN_SPEC_SCHEMA_VERSION,
    accept_rlrmp_spec_payload,
)
from rlrmp.runtime.training_run_specs import (
    feedbax_training_run_spec_from_rlrmp_record,
)


RLRMP_NATIVE_MANIFEST_COMPANION_KEY = "rlrmp_native_manifest"
RLRMP_NATIVE_MANIFEST_COMPANION_SCHEMA_ID = "rlrmp.spec.native_training_manifest_companion"
RLRMP_NATIVE_MANIFEST_COMPANION_SCHEMA_VERSION = "rlrmp.spec.native_training_manifest_companion.v1"
RLRMP_TRAINING_MANIFEST_METADATA_PROJECTION_SCHEMA_ID = (
    "rlrmp.spec.training_manifest_metadata_projection"
)
RLRMP_TRAINING_MANIFEST_METADATA_PROJECTION_SCHEMA_VERSION = (
    "rlrmp.spec.training_manifest_metadata_projection.v1"
)


class NativeManifestTrainingDiagnostics(BaseModel):
    """Selector projection for native training-diagnostics consumers."""

    model_config = ConfigDict(extra="forbid", strict=True)

    enabled: bool


class RlrmpNativeManifestMetadata(BaseModel):
    """RLRMP metadata projected onto the Feedbax-owned root manifest."""

    model_config = ConfigDict(extra="forbid", strict=True)

    training_diagnostics: NativeManifestTrainingDiagnostics
    gru_postrun_candidate: bool = False

    @model_validator(mode="after")
    def _validate_gru_candidate(self) -> "RlrmpNativeManifestMetadata":
        if self.gru_postrun_candidate and not self.training_diagnostics.enabled:
            raise ValueError("/gru_postrun_candidate requires /training_diagnostics/enabled=true")
        return self


class RlrmpNativeManifestCompanion(BaseModel):
    """Digest-bound RLRMP identity and selector inputs for one native manifest."""

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["rlrmp.spec.native_training_manifest_companion"] = (
        RLRMP_NATIVE_MANIFEST_COMPANION_SCHEMA_ID
    )
    schema_version: Literal["rlrmp.spec.native_training_manifest_companion.v1"] = (
        RLRMP_NATIVE_MANIFEST_COMPANION_SCHEMA_VERSION
    )
    training_spec_payload: dict[str, Any]
    training_spec_payload_kind: Literal["RLRMPRunSpec"] = RUN_SPEC_KIND
    training_spec_payload_schema_id: Literal["rlrmp.run_spec"] = RUN_SPEC_SCHEMA_ID
    training_spec_payload_schema_version: Literal["rlrmp.run_spec.v2"] = RUN_SPEC_SCHEMA_VERSION
    training_spec_payload_ref: str = Field(min_length=1)
    manifest_metadata: RlrmpNativeManifestMetadata

    @model_validator(mode="after")
    def _validate_external_payload(self) -> "RlrmpNativeManifestCompanion":
        accepted = accept_rlrmp_spec_payload(
            RUN_SPEC_KIND,
            self.training_spec_payload,
            source_version=self.training_spec_payload_schema_version,
            path=(f"{RLRMP_NATIVE_MANIFEST_COMPANION_KEY}.training_spec_payload"),
        )
        if accepted.payload != self.training_spec_payload:
            raise ValueError(
                "/training_spec_payload must already be the canonical current RLRMPRunSpec"
            )
        feedbax_training_run_spec_from_rlrmp_record(self.training_spec_payload)
        diagnostics = self.training_spec_payload.get("training_diagnostics")
        if not isinstance(diagnostics, dict):
            raise ValueError("/training_spec_payload/training_diagnostics must be an object")
        if diagnostics.get("enabled") is not self.manifest_metadata.training_diagnostics.enabled:
            raise ValueError(
                "/manifest_metadata/training_diagnostics/enabled must match the "
                "RLRMPRunSpec payload"
            )
        controller_kind = self.training_spec_payload.get("model_summary", {}).get("controller_kind")
        expected_candidate = bool(
            controller_kind == "gru" and self.manifest_metadata.training_diagnostics.enabled
        )
        if self.manifest_metadata.gru_postrun_candidate is not expected_candidate:
            raise ValueError(
                "/manifest_metadata/gru_postrun_candidate must be true exactly for "
                "enabled GRU diagnostics"
            )
        return self

    def external_training_payload_kwargs(self) -> dict[str, Any]:
        """Return the public Feedbax external-training-payload arguments."""

        return {
            "training_spec_payload": self.training_spec_payload,
            "training_spec_payload_kind": self.training_spec_payload_kind,
            "training_spec_payload_schema_id": self.training_spec_payload_schema_id,
            "training_spec_payload_schema_version": (self.training_spec_payload_schema_version),
            "training_spec_payload_ref": self.training_spec_payload_ref,
        }

    def manifest_metadata_projection(self) -> TrainingManifestMetadataProjection:
        """Return the hash-bound Feedbax projection request for this payload."""

        return TrainingManifestMetadataProjection(
            source_payload_kind=self.training_spec_payload_kind,
            source_payload_schema_id=self.training_spec_payload_schema_id,
            source_payload_schema_version=self.training_spec_payload_schema_version,
            source_payload_sha256=training_spec_sha256(self.training_spec_payload),
            projection_schema_id=(RLRMP_TRAINING_MANIFEST_METADATA_PROJECTION_SCHEMA_ID),
            projection_schema_version=(RLRMP_TRAINING_MANIFEST_METADATA_PROJECTION_SCHEMA_VERSION),
            values=self.manifest_metadata.model_dump(mode="json"),
        )


def register_rlrmp_training_manifest_metadata_projection(
    registry: TrainingMethodRegistry,
) -> None:
    """Register RLRMP selector projection governance on a Feedbax registry."""

    registration = TrainingManifestMetadataProjectionRegistration(
        source_payload_kind=RUN_SPEC_KIND,
        source_payload_schema_id=RUN_SPEC_SCHEMA_ID,
        source_payload_schema_version=RUN_SPEC_SCHEMA_VERSION,
        projection_schema_id=RLRMP_TRAINING_MANIFEST_METADATA_PROJECTION_SCHEMA_ID,
        projection_schema_version=(RLRMP_TRAINING_MANIFEST_METADATA_PROJECTION_SCHEMA_VERSION),
        values_model=RlrmpNativeManifestMetadata,
        owner="rlrmp.train.native_manifest",
        package="rlrmp",
    )
    try:
        existing = registry.resolve_manifest_metadata_projection(
            registration.source_key,
            path="/rlrmp_training_manifest_metadata_projection",
        )
    except ValueError as exc:
        if "no manifest metadata projection registered" not in str(exc):
            raise
        registry.register_manifest_metadata_projection(registration)
        return
    if existing != registration:
        raise ValueError("conflicting RLRMP training-manifest metadata projection registration")


__all__ = [
    "RLRMP_NATIVE_MANIFEST_COMPANION_KEY",
    "RLRMP_NATIVE_MANIFEST_COMPANION_SCHEMA_ID",
    "RLRMP_NATIVE_MANIFEST_COMPANION_SCHEMA_VERSION",
    "RLRMP_TRAINING_MANIFEST_METADATA_PROJECTION_SCHEMA_ID",
    "RLRMP_TRAINING_MANIFEST_METADATA_PROJECTION_SCHEMA_VERSION",
    "NativeManifestTrainingDiagnostics",
    "RlrmpNativeManifestCompanion",
    "RlrmpNativeManifestMetadata",
    "register_rlrmp_training_manifest_metadata_projection",
]
