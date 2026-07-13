"""Hash-verified authored defaults for training configuration schemas."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

from rlrmp.paths import REPO_ROOT


TRAINING_PRESET_RELPATH = "src/rlrmp/config/training_presets/defaults.json"
TRAINING_PRESET_PATH = REPO_ROOT / TRAINING_PRESET_RELPATH
TRAINING_PRESET_SCHEMA_ID = "rlrmp.training_authoring_presets"
TRAINING_PRESET_SCHEMA_VERSION = "rlrmp.training_authoring_presets.v1"
_TUPLE_FIELDS = {
    "amplitude_levels",
    "held_out_amplitudes_m",
    "held_out_directions_deg",
    "original_target_anchor_m",
    "seen_amplitudes_m",
    "seen_directions_deg",
    "sisu_levels",
}


@lru_cache(maxsize=1)
def load_training_presets() -> dict[str, dict[str, Any]]:
    """Load the authored preset document and fail closed on identity drift."""

    document = json.loads(TRAINING_PRESET_PATH.read_text(encoding="utf-8"))
    if document.get("schema_id") != TRAINING_PRESET_SCHEMA_ID:
        raise ValueError("training preset schema_id mismatch")
    if document.get("schema_version") != TRAINING_PRESET_SCHEMA_VERSION:
        raise ValueError("training preset schema_version mismatch")
    presets = document.get("presets")
    if not isinstance(presets, dict):
        raise ValueError("training preset document must contain a presets mapping")
    canonical = json.dumps(presets, sort_keys=True, separators=(",", ":")).encode()
    actual_hash = hashlib.sha256(canonical).hexdigest()
    if actual_hash != document.get("content_sha256"):
        raise ValueError(
            "training preset content hash mismatch: "
            f"expected {document.get('content_sha256')!r}, got {actual_hash!r}"
        )
    return {str(name): dict(values) for name, values in presets.items()}


def training_preset_value(preset: str, field: str) -> Any:
    """Return one authored preset value by schema and field name."""

    try:
        value = load_training_presets()[preset][field]
    except KeyError as exc:
        raise ValueError(f"training preset {preset!r} has no field {field!r}") from exc
    return tuple(value) if field in _TUPLE_FIELDS else value
