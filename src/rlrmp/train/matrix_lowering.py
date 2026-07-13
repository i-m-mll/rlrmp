"""Deterministic lowering from compact RLRMP matrix intent to training specs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from feedbax.contracts.run_matrix import (
    AuthoredTrainingRow,
    RowLowererIdentity,
    TrainingRowLoweringResult,
)
from feedbax.contracts.training import TrainingRunSpec
from feedbax.config.namespace import TreeNamespace
from pydantic import BaseModel, ConfigDict, model_validator

from rlrmp.paths import portable_repo_path
from rlrmp.runtime.spec_migrations import (
    TRAINING_AUTHORING_INTENT_KIND,
    TRAINING_AUTHORING_INTENT_SCHEMA_ID,
    TRAINING_AUTHORING_INTENT_SCHEMA_VERSION,
    accept_rlrmp_spec_payload,
)
from rlrmp.runtime.training_run_specs import (
    FEEDBAX_TRAINING_RUN_SPEC_KEY,
    RLRMP_RUN_SPEC_PAYLOAD_KEY,
    attach_composed_training_specs,
    feedbax_training_run_spec_from_payload,
)
from rlrmp.train.config_materialization import (
    _apply_smoke_overrides,
    _config_namespace,
    build_hps,
)
from rlrmp.train.run_spec_authoring import (
    build_graph_bundle,
    build_run_spec,
    build_training_run_graph_spec,
    derive_spec_dir,
)
from rlrmp.train.science_lowering import lower_training_science
from rlrmp.train.native_manifest import (
    RLRMP_NATIVE_MANIFEST_COMPANION_KEY,
    NativeManifestTrainingDiagnostics,
    RlrmpNativeManifestCompanion,
    RlrmpNativeManifestMetadata,
)
from rlrmp.train.training_configs import CsNominalGruConfig


RLRMP_TRAINING_AUTHORING_INTENT_SCHEMA_ID = TRAINING_AUTHORING_INTENT_SCHEMA_ID
RLRMP_TRAINING_AUTHORING_INTENT_SCHEMA_VERSION = TRAINING_AUTHORING_INTENT_SCHEMA_VERSION
RLRMP_TRAINING_ROW_LOWERER_ID = "rlrmp.train.cs_nominal_gru.authoring"
RLRMP_TRAINING_ROW_LOWERER_VERSION = "v2"
RLRMP_SCIENCE_LOWERER_VERSION = "v1"
RLRMP_TRAINING_ARCHITECTURE_CONTRACT = "rlrmp.heterogeneous_cs_architecture.v1"
RLRMP_TRAINING_ARCHITECTURES = (
    "gru",
    "time_constrained_free_gain",
    "linear_recurrence",
)
RLRMP_TRAINING_ARCHITECTURE_LOWERER_ID = "rlrmp.heterogeneous_cs_architecture"
RLRMP_TRAINING_ARCHITECTURE_LOWERER_VERSION = "v1"


class RlrmpTrainingAuthoringIntent(BaseModel):
    """Compact governed input for one canonical C&S nominal-GRU run.

    Matrix axes patch ``config.*`` fields before this model is validated.  The
    config model owns all authoring defaults, while this envelope rejects
    precompiled graph, task, method, worker, and callback payloads by
    construction.
    """

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["rlrmp.spec.training_authoring_intent"] = (
        RLRMP_TRAINING_AUTHORING_INTENT_SCHEMA_ID
    )
    schema_version: Literal["rlrmp.spec.training_authoring_intent.v1"] = (
        RLRMP_TRAINING_AUTHORING_INTENT_SCHEMA_VERSION
    )
    config: CsNominalGruConfig

    @model_validator(mode="after")
    def _validate_compact_authoring(self) -> "RlrmpTrainingAuthoringIntent":
        if self.config.run_spec is not None:
            raise ValueError(
                "/config/run_spec is a legacy compiled-input surface; compact matrix "
                "intent must be lowered from config fields"
            )
        if self.config.controller_architecture not in RLRMP_TRAINING_ARCHITECTURES:
            raise ValueError(
                "/config/controller_architecture must select a public authored architecture: "
                + "|".join(RLRMP_TRAINING_ARCHITECTURES)
            )
        return self


def is_rlrmp_training_authoring_intent(payload: Mapping[str, Any]) -> bool:
    """Return whether a resolved matrix base belongs to this compact spec family."""

    return payload.get("schema_id") == RLRMP_TRAINING_AUTHORING_INTENT_SCHEMA_ID


def _resolved_config(row: AuthoredTrainingRow) -> CsNominalGruConfig:
    accepted = accept_rlrmp_spec_payload(
        TRAINING_AUTHORING_INTENT_KIND,
        row.payload,
        path=f"training_matrix.rows[{row.row_id!r}].authored_intent",
    )
    intent = RlrmpTrainingAuthoringIntent.model_validate(accepted.payload)
    config = intent.config
    if row.seed is not None:
        config = config.model_copy(update={"seed": row.seed})
    return config


def _canonical_args(config: CsNominalGruConfig) -> argparse.Namespace:
    canonical_config = config.model_copy(update={"controller_architecture": "gru"})
    args = _config_namespace(canonical_config)
    args = _apply_smoke_overrides(args)
    return _config_namespace(args)


def _lowerer_identities(hps: TreeNamespace) -> list[RowLowererIdentity]:
    science = lower_training_science(hps)
    return [
        RowLowererIdentity(
            lowerer_id=RLRMP_TRAINING_ROW_LOWERER_ID,
            lowerer_version=RLRMP_TRAINING_ROW_LOWERER_VERSION,
        ),
        *(
            RowLowererIdentity(
                lowerer_id=lowerer_id,
                lowerer_version=RLRMP_SCIENCE_LOWERER_VERSION,
            )
            for lowerer_id in science.lowerer_ids
        ),
        RowLowererIdentity(
            lowerer_id=RLRMP_TRAINING_ARCHITECTURE_LOWERER_ID,
            lowerer_version=RLRMP_TRAINING_ARCHITECTURE_LOWERER_VERSION,
        ),
    ]


def _training_distribution(
    config: CsNominalGruConfig,
) -> Literal["nominal", "broad_epsilon_pgd"]:
    return "broad_epsilon_pgd" if config.broad_epsilon_pgd_training else "nominal"


def _dispatch_architecture(
    canonical: TrainingRunSpec,
    *,
    architecture: str,
    training_distribution: Literal["nominal", "broad_epsilon_pgd"],
) -> TrainingRunSpec:
    """Dispatch one canonical base through the public architecture providers."""

    if architecture == "gru":
        from rlrmp.train.heterogeneous_training_matrix import author_gru_training_base

        return author_gru_training_base(
            canonical,
            training_distribution=training_distribution,
        )
    if architecture == "time_constrained_free_gain":
        from rlrmp.train.static_linear_native import (
            author_static_linear_training_base_from_canonical,
        )

        return author_static_linear_training_base_from_canonical(
            canonical,
            training_distribution=training_distribution,
        )
    if architecture == "linear_recurrence":
        from rlrmp.train.linear_recurrent_native import (
            author_linear_recurrent_training_base_from_canonical,
        )

        return author_linear_recurrent_training_base_from_canonical(
            canonical,
            training_distribution=training_distribution,
        )
    raise ValueError(f"unsupported authored controller architecture {architecture!r}")


def lower_rlrmp_training_row(row: AuthoredTrainingRow) -> TrainingRowLoweringResult:
    """Lower one axis-patched compact row into the complete execution contract."""

    config = _resolved_config(row)
    architecture = config.controller_architecture
    args = _canonical_args(config)
    hps = build_hps(args)
    output_dir = Path(args.output_dir)
    spec_dir = Path(args.spec_dir) if args.spec_dir is not None else derive_spec_dir(output_dir)
    graph_bundle = build_graph_bundle(hps)
    graph_spec = build_training_run_graph_spec(hps, seed=int(args.seed))
    run_spec = build_run_spec(
        args,
        output_dir=output_dir,
        spec_dir=spec_dir,
        graph_bundle=graph_bundle,
    )
    composed = attach_composed_training_specs(
        run_spec,
        graph_spec=graph_spec,
        output_dir=output_dir,
        spec_dir=spec_dir,
    )
    canonical_spec = feedbax_training_run_spec_from_payload(composed)
    execution_spec = _dispatch_architecture(
        canonical_spec,
        architecture=architecture,
        training_distribution=_training_distribution(config),
    )
    generic_execution_payload = execution_spec.model_dump(mode="json", exclude_none=True)
    rlrmp_run_spec = {
        **composed[RLRMP_RUN_SPEC_PAYLOAD_KEY],
        FEEDBAX_TRAINING_RUN_SPEC_KEY: generic_execution_payload,
    }
    diagnostics = rlrmp_run_spec.get("training_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("lowered RLRMPRunSpec lacks training_diagnostics metadata")
    diagnostics_enabled = bool(diagnostics.get("enabled"))
    companion = RlrmpNativeManifestCompanion(
        training_spec_payload=rlrmp_run_spec,
        training_spec_payload_ref=portable_repo_path(spec_dir.with_suffix(".json")),
        manifest_metadata=RlrmpNativeManifestMetadata(
            training_diagnostics=NativeManifestTrainingDiagnostics(enabled=diagnostics_enabled),
            gru_postrun_candidate=(architecture == "gru" and diagnostics_enabled),
        ),
    )
    execution_spec = execution_spec.model_copy(
        update={
            "metadata": {
                **execution_spec.metadata,
                RLRMP_NATIVE_MANIFEST_COMPANION_KEY: companion.model_dump(
                    mode="json", exclude_none=True
                ),
            }
        }
    )
    return TrainingRowLoweringResult(
        execution_payload=execution_spec.model_dump(mode="json", exclude_none=True),
        lowerer_identities=_lowerer_identities(hps),
    )


__all__ = [
    "RLRMP_SCIENCE_LOWERER_VERSION",
    "RLRMP_TRAINING_ARCHITECTURES",
    "RLRMP_TRAINING_ARCHITECTURE_CONTRACT",
    "RLRMP_TRAINING_ARCHITECTURE_LOWERER_ID",
    "RLRMP_TRAINING_ARCHITECTURE_LOWERER_VERSION",
    "RLRMP_TRAINING_AUTHORING_INTENT_SCHEMA_ID",
    "RLRMP_TRAINING_AUTHORING_INTENT_SCHEMA_VERSION",
    "RLRMP_TRAINING_ROW_LOWERER_ID",
    "RLRMP_TRAINING_ROW_LOWERER_VERSION",
    "RlrmpTrainingAuthoringIntent",
    "is_rlrmp_training_authoring_intent",
    "lower_rlrmp_training_row",
]
