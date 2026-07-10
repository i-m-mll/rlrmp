"""Registered, fail-closed parameter presets for reusable analyses."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

PRESET_SCHEMA_ID = "rlrmp.analysis_parameter_preset"
PRESET_SCHEMA_VERSION = "rlrmp.analysis_parameter_preset.v1"
_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "config" / "analysis_presets"


@dataclass(frozen=True)
class AnalysisParameterPreset:
    """Validated parameters and provenance for one named analysis preset."""

    preset_id: str
    source_issue: str
    parameters: Mapping[str, Any]
    content_sha256: str
    source_path: Path


@dataclass(frozen=True)
class AnalysisParameterPresetRegistration:
    """Registry entry locating one stable preset document."""

    preset_id: str
    document_relpath: str


_REGISTRY: dict[str, AnalysisParameterPresetRegistration] = {}


def register_analysis_parameter_preset(*, preset_id: str, document_relpath: str) -> None:
    """Register a preset document, rejecting identity collisions."""

    candidate = AnalysisParameterPresetRegistration(preset_id, document_relpath)
    existing = _REGISTRY.get(preset_id)
    if existing is not None and existing != candidate:
        raise ValueError(f"analysis parameter preset collision for {preset_id!r}")
    _REGISTRY[preset_id] = candidate


def registered_analysis_parameter_presets() -> Mapping[str, AnalysisParameterPresetRegistration]:
    """Return registered presets keyed by stable preset identity."""

    return MappingProxyType(dict(_REGISTRY))


@lru_cache(maxsize=None)
def load_analysis_parameter_preset(preset_id: str) -> AnalysisParameterPreset:
    """Load a registered preset and verify schema, identity, and content hash."""

    try:
        registration = _REGISTRY[preset_id]
    except KeyError as exc:
        raise KeyError(f"unregistered analysis parameter preset {preset_id!r}") from exc
    path = _CONFIG_ROOT / Path(registration.document_relpath).name
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load analysis parameter preset {preset_id!r}: {exc}") from exc
    if payload.get("schema_id") != PRESET_SCHEMA_ID:
        raise ValueError(f"analysis parameter preset {preset_id!r} has the wrong schema_id")
    if payload.get("schema_version") != PRESET_SCHEMA_VERSION:
        raise ValueError(f"analysis parameter preset {preset_id!r} has the wrong schema_version")
    if payload.get("preset_id") != preset_id:
        raise ValueError(f"analysis parameter preset {preset_id!r} has the wrong preset_id")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"analysis parameter preset {preset_id!r} has no parameter mapping")
    stored_hash = payload.get("content_sha256")
    semantic_payload = {key: value for key, value in payload.items() if key != "content_sha256"}
    computed_hash = hashlib.sha256(
        json.dumps(semantic_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if stored_hash != computed_hash:
        raise ValueError(
            f"analysis parameter preset {preset_id!r} has a stale content hash: "
            f"stored={stored_hash!r}, computed={computed_hash!r}"
        )
    source_issue = payload.get("source_issue")
    if not isinstance(source_issue, str) or not source_issue:
        raise ValueError(f"analysis parameter preset {preset_id!r} has no source_issue")
    return AnalysisParameterPreset(
        preset_id=preset_id,
        source_issue=source_issue,
        parameters=MappingProxyType(parameters),
        content_sha256=computed_hash,
        source_path=path,
    )


def _register_defaults() -> None:
    root = "src/rlrmp/config/analysis_presets"
    for preset_id in (
        "adversary_equivalence",
        "cs_game_card",
        "cs_gru_standard_materialization",
        "delayed_diagnostic_bundle",
        "diagnostic_provenance",
        "failure_decomposition",
        "gru_checkpoint_selection",
        "gru_perturbation_calibration",
        "gru_perturbation_response_norm_plots",
        "gru_pilot_figures",
        "gru_steady_state_perturbation_bank",
        "objective_comparator",
        "output_feedback_gamma_sweep",
        "output_feedback_interpolated_starts",
        "robust_bellman",
        "sisu_perturbation_comparison",
        "sisu_spectrum_diagnostics",
    ):
        register_analysis_parameter_preset(
            preset_id=preset_id,
            document_relpath=f"{root}/{preset_id}.json",
        )


_register_defaults()
