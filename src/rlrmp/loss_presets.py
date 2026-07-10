"""Typed, registered loss-preset documents."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class LossTopWeights(BaseModel):
    """Weights for the complete default loss-term vocabulary."""

    model_config = ConfigDict(extra="forbid")

    effector_pos: float
    effector_hold_pos: float
    effector_hold_vel: float
    effector_pos_mid: float
    effector_vel_mid: float
    effector_pos_late: float
    effector_vel_late: float
    effector_pos_running: float
    effector_vel_running: float
    effector_terminal_pos: float
    effector_terminal_vel: float
    effector_final_vel: float
    nn_output: float
    nn_hidden: float
    nn_hidden_derivative: float
    nn_output_jerk: float
    nn_output_pre_go: float
    delayed_pre_go_force_filter_hold: float
    delayed_pre_go_start_pos_hold: float
    delayed_pre_go_zero_vel_hold: float
    nn_hidden_derivative_pre_go: float
    goal_hit_in_window: float


class GoalHitSubweights(BaseModel):
    """Typed subweights for the goal-hit composite."""

    model_config = ConfigDict(extra="forbid")

    pos: float
    vel: float
    post_pos: float
    late_pos: float


class GoalHitParams(BaseModel):
    """Typed timing and smoothing parameters for goal-hit loss."""

    model_config = ConfigDict(extra="forbid")

    start_step_after_go: int
    end_step_after_go: int
    softmin_tau: float
    post_pos_sigma_t: float
    alpha_eps: float


class LateEffectorParams(BaseModel):
    """Typed late-window loss parameters."""

    model_config = ConfigDict(extra="forbid")

    start_step_after_go: int
    final_scale_factor: float


class MidEffectorParams(BaseModel):
    """Typed mid-window ramp parameters."""

    model_config = ConfigDict(extra="forbid")

    start_step_after_go: int
    end_step_after_go: int
    ramp_init_weight: float
    ramp_final_weight: float


class LossPreset(BaseModel):
    """Versioned schema for one registered RLRMP loss preset."""

    model_config = ConfigDict(extra="forbid")

    schema_id: str
    schema_version: str
    preset_id: str
    top_weights: LossTopWeights
    goal_hit_subweights: GoalHitSubweights
    goal_hit_params: GoalHitParams
    effector_pos_late_params: LateEffectorParams
    effector_vel_late_params: LateEffectorParams
    effector_pos_mid_params: MidEffectorParams
    effector_vel_mid_params: MidEffectorParams
    content_sha256: str


_LOSS_PRESET_REGISTRY = {
    "rlrmp.default_loss": (
        "default.json",
        "rlrmp.loss_preset",
        "rlrmp.loss_preset.v1",
    ),
}
_CONFIG_ROOT = Path(__file__).resolve().parent / "config" / "loss_presets"


@lru_cache(maxsize=None)
def load_loss_preset(preset_id: str = "rlrmp.default_loss") -> LossPreset:
    """Load a typed registered loss preset and verify its semantic hash."""

    try:
        filename, schema_id, schema_version = _LOSS_PRESET_REGISTRY[preset_id]
    except KeyError as exc:
        raise KeyError(f"unregistered loss preset {preset_id!r}") from exc
    path = _CONFIG_ROOT / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored_hash = payload.get("content_sha256")
    semantic = {key: value for key, value in payload.items() if key != "content_sha256"}
    computed_hash = hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if stored_hash != computed_hash:
        raise ValueError(
            f"loss preset {preset_id!r} has a stale content hash: "
            f"stored={stored_hash!r}, computed={computed_hash!r}"
        )
    if payload.get("schema_id") != schema_id:
        raise ValueError(f"loss preset {preset_id!r} has an unsupported schema_id")
    if payload.get("schema_version") != schema_version:
        raise ValueError(f"loss preset {preset_id!r} has an unsupported schema_version")
    preset = LossPreset.model_validate(payload)
    if preset.preset_id != preset_id:
        raise ValueError(f"loss preset {preset_id!r} has the wrong preset_id")
    return preset


def registered_loss_presets() -> tuple[str, ...]:
    """Return stable registered loss-preset identities."""

    return tuple(sorted(_LOSS_PRESET_REGISTRY))
