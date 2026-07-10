"""Typed registered runtime and execution parameter presets."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import TypeVar, cast

from pydantic import BaseModel, ConfigDict


class ModalRunnerPreset(BaseModel):
    """Typed defaults for the Modal/remote execution surface."""

    model_config = ConfigDict(extra="forbid")

    schema_id: str
    schema_version: str
    preset_id: str
    timeout_seconds: int
    n_train_batches: int
    batch_size: int
    n_replicates: int
    hidden_size: int
    checkpoint_interval_batches: int
    content_sha256: str


class EvaluationEnsemblePreset(BaseModel):
    """Typed defaults for replicate-ensemble evaluation."""

    model_config = ConfigDict(extra="forbid")

    schema_id: str
    schema_version: str
    preset_id: str
    n_replicates: int
    content_sha256: str


RuntimePreset = ModalRunnerPreset | EvaluationEnsemblePreset
RuntimePresetT = TypeVar("RuntimePresetT", bound=RuntimePreset)
_CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config" / "runtime_presets"
_RUNTIME_PRESET_REGISTRY: dict[str, tuple[str, type[RuntimePreset], str, str]] = {
    "rlrmp.modal_runner.default": (
        "modal_runner.json",
        ModalRunnerPreset,
        "rlrmp.runtime.modal_runner_preset",
        "rlrmp.runtime.modal_runner_preset.v1",
    ),
    "rlrmp.evaluation_ensemble.default": (
        "evaluation_ensemble.json",
        EvaluationEnsemblePreset,
        "rlrmp.runtime.evaluation_ensemble_preset",
        "rlrmp.runtime.evaluation_ensemble_preset.v1",
    ),
}


@lru_cache(maxsize=None)
def _load_runtime_preset(preset_id: str) -> RuntimePreset:
    try:
        filename, model_type, schema_id, schema_version = _RUNTIME_PRESET_REGISTRY[preset_id]
    except KeyError as exc:
        raise KeyError(f"unregistered runtime preset {preset_id!r}") from exc
    path = _CONFIG_ROOT / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored_hash = payload.get("content_sha256")
    semantic = {key: value for key, value in payload.items() if key != "content_sha256"}
    computed_hash = hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if stored_hash != computed_hash:
        raise ValueError(f"runtime preset {preset_id!r} has a stale content hash")
    if payload.get("schema_id") != schema_id:
        raise ValueError(f"runtime preset {preset_id!r} has an unsupported schema_id")
    if payload.get("schema_version") != schema_version:
        raise ValueError(f"runtime preset {preset_id!r} has an unsupported schema_version")
    preset = model_type.model_validate(payload)
    if preset.preset_id != preset_id:
        raise ValueError(f"runtime preset {preset_id!r} has the wrong preset_id")
    return preset


def load_runtime_preset(preset_id: str, model_type: type[RuntimePresetT]) -> RuntimePresetT:
    """Load one registered runtime preset as its exact schema type."""

    preset = _load_runtime_preset(preset_id)
    if not isinstance(preset, model_type):
        raise TypeError(f"runtime preset {preset_id!r} is not a {model_type.__name__}")
    return cast(RuntimePresetT, preset)


def registered_runtime_presets() -> tuple[str, ...]:
    """Return stable registered runtime-preset identities."""

    return tuple(sorted(_RUNTIME_PRESET_REGISTRY))
