"""Typed registered model, graph, and migration presets."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import TypeVar, cast

from pydantic import BaseModel, ConfigDict


class PopulationStructureDefaults(BaseModel):
    """Fallback counts for optional population-structure partitions."""

    model_config = ConfigDict(extra="forbid")

    n_input_only: int
    n_readout_only: int
    n_recurrent_only: int
    n_input_readout: int


class CsLssGruPreset(BaseModel):
    """Typed construction defaults for the C&S LSS GRU graph."""

    model_config = ConfigDict(extra="forbid")

    schema_id: str
    schema_version: str
    preset_id: str
    delayed_pos_vel_indices: list[int]
    delayed_pos_vel_force_indices: list[int]
    population_defaults: PopulationStructureDefaults
    content_sha256: str


class LegacyPlantProcessForceNoisePreset(BaseModel):
    """Versioned defaults for the legacy force-noise migration."""

    model_config = ConfigDict(extra="forbid")

    delay: int
    noise_model: str
    noise_std: float
    noise_role: str
    noise_timing: str
    input_shape: list[int]


class FeedbaxGraphPreset(BaseModel):
    """Typed graph-construction and migration defaults."""

    model_config = ConfigDict(extra="forbid")

    schema_id: str
    schema_version: str
    preset_id: str
    graph_component_seed: int
    population_defaults: PopulationStructureDefaults
    legacy_plant_process_force_noise: LegacyPlantProcessForceNoisePreset
    content_sha256: str


ModelPreset = CsLssGruPreset | FeedbaxGraphPreset
ModelPresetT = TypeVar("ModelPresetT", bound=ModelPreset)
_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config" / "model_presets"
_MODEL_PRESET_REGISTRY: dict[str, tuple[str, type[ModelPreset], str, str]] = {
    "rlrmp.cs_lss_gru.default": (
        "cs_lss_gru.json",
        CsLssGruPreset,
        "rlrmp.model.cs_lss_gru_preset",
        "rlrmp.model.cs_lss_gru_preset.v1",
    ),
    "rlrmp.feedbax_graph.default": (
        "feedbax_graph.json",
        FeedbaxGraphPreset,
        "rlrmp.model.feedbax_graph_preset",
        "rlrmp.model.feedbax_graph_preset.v1",
    ),
}


@lru_cache(maxsize=None)
def _load_model_preset(preset_id: str) -> ModelPreset:
    try:
        filename, model_type, schema_id, schema_version = _MODEL_PRESET_REGISTRY[preset_id]
    except KeyError as exc:
        raise KeyError(f"unregistered model preset {preset_id!r}") from exc
    path = _CONFIG_ROOT / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored_hash = payload.get("content_sha256")
    semantic = {key: value for key, value in payload.items() if key != "content_sha256"}
    computed_hash = hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if stored_hash != computed_hash:
        raise ValueError(f"model preset {preset_id!r} has a stale content hash")
    if payload.get("schema_id") != schema_id:
        raise ValueError(f"model preset {preset_id!r} has an unsupported schema_id")
    if payload.get("schema_version") != schema_version:
        raise ValueError(f"model preset {preset_id!r} has an unsupported schema_version")
    preset = model_type.model_validate(payload)
    if preset.preset_id != preset_id:
        raise ValueError(f"model preset {preset_id!r} has the wrong preset_id")
    return preset


def load_model_preset(preset_id: str, model_type: type[ModelPresetT]) -> ModelPresetT:
    """Load one registered model preset as its exact schema type."""

    preset = _load_model_preset(preset_id)
    if not isinstance(preset, model_type):
        raise TypeError(f"model preset {preset_id!r} is not a {model_type.__name__}")
    return cast(ModelPresetT, preset)


def registered_model_presets() -> tuple[str, ...]:
    """Return stable registered model-preset identities."""

    return tuple(sorted(_MODEL_PRESET_REGISTRY))
