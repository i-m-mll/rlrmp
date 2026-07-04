"""Typed perturbation-row schema for analysis perturbation banks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast


PerturbationChannel = Literal[
    "initial_state",
    "command_input",
    "process_epsilon",
    "sensory_feedback",
    "delayed_observation",
    "target_stream",
]


@dataclass(frozen=True)
class _ChannelFieldRule:
    required: frozenset[str]
    optional: frozenset[str] = frozenset()
    families: frozenset[str] | None = None
    timing_required: frozenset[str] = frozenset()


_UNIVERSAL_REQUIRED = frozenset(
    {
        "perturbation_id",
        "channel",
        "family",
        "amplitude",
        "sign",
        "timing",
    }
)
_PULSE_TIMING_REQUIRED = frozenset({"start_time_index", "duration_steps"})

_CHANNEL_FIELD_RULES: dict[str, _ChannelFieldRule] = {
    "initial_state": _ChannelFieldRule(
        required=_UNIVERSAL_REQUIRED | {"axis"},
        optional=frozenset({"initial_position_case"}),
        families=frozenset({"initial_position_offset", "initial_velocity_offset"}),
    ),
    "process_epsilon": _ChannelFieldRule(
        required=_UNIVERSAL_REQUIRED | {"axis"},
        optional=frozenset({"epsilon_index", "epsilon_component"}),
        timing_required=_PULSE_TIMING_REQUIRED,
    ),
    "sensory_feedback": _ChannelFieldRule(
        required=_UNIVERSAL_REQUIRED | {"axis"},
        optional=frozenset(
            {
                "feedback_payload_index",
                "feedback_quantity",
                "force_filter_feedback_only",
                "channel_provenance",
            }
        ),
        timing_required=_PULSE_TIMING_REQUIRED,
    ),
    "delayed_observation": _ChannelFieldRule(
        required=_UNIVERSAL_REQUIRED | {"axis"},
        optional=frozenset(
            {
                "feedback_payload_index",
                "feedback_quantity",
                "force_filter_feedback_only",
                "channel_provenance",
            }
        ),
        timing_required=_PULSE_TIMING_REQUIRED,
    ),
    "command_input": _ChannelFieldRule(
        required=_UNIVERSAL_REQUIRED | {"axis"},
        optional=frozenset({"allow_zero_graph_effect"}),
        timing_required=_PULSE_TIMING_REQUIRED,
    ),
    "target_stream": _ChannelFieldRule(
        required=_UNIVERSAL_REQUIRED,
    ),
}

_CONSTRUCTOR_FIELDS = frozenset(
    {
        "perturbation_id",
        "channel",
        "family",
        "amplitude",
        "units",
        "axis",
        "basis",
        "sign",
        "timing",
        "adapter",
        "description",
    }
)
_OPTIONAL_FIELDS = frozenset(
    {
        "epsilon_component",
        "epsilon_index",
        "initial_position_case",
        "calibration_role",
        "timing_bin",
        "semantic_family",
        "channel_provenance",
        "calibration_provenance",
        "feedback_payload_index",
        "feedback_quantity",
        "force_filter_feedback_only",
        "allow_zero_graph_effect",
    }
)
_WIRE_FIELDS = _CONSTRUCTOR_FIELDS | _OPTIONAL_FIELDS


@dataclass(frozen=True)
class PerturbationSpec:
    """Declarative perturbation row in the standard C&S bank."""

    perturbation_id: str
    channel: PerturbationChannel
    family: str
    amplitude: float
    units: str
    axis: str
    basis: str
    sign: int
    timing: Mapping[str, Any]
    adapter: str
    description: str
    epsilon_component: str | None = None
    epsilon_index: int | None = None
    initial_position_case: str | None = None
    calibration_role: str | None = None
    timing_bin: str | None = None
    semantic_family: str | None = None
    channel_provenance: Mapping[str, Any] | None = None
    calibration_provenance: Mapping[str, Any] | None = None
    feedback_payload_index: int | None = None
    feedback_quantity: str | None = None
    force_filter_feedback_only: bool | None = None
    allow_zero_graph_effect: bool | None = None
    extra: Mapping[str, Any] | None = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> PerturbationSpec:
        """Lift a wire perturbation row into the typed schema.

        Unknown top-level keys are retained in ``extra`` so rows with flattened
        calibration provenance round-trip through ``to_json()`` without losing
        load-bearing metadata.
        """

        channel = str(row.get("channel", "<missing>"))
        missing = [field for field in _CONSTRUCTOR_FIELDS if field not in row]
        if missing:
            field = sorted(missing)[0]
            raise ValueError(
                f"perturbation row for channel {channel!r} is missing required field {field!r}"
            )
        flattened_known = {"calibration_role"} if "calibration_mode" in row else set()
        extra = {
            key: value
            for key, value in row.items()
            if key not in _WIRE_FIELDS or key in flattened_known
        }
        return cls(
            perturbation_id=str(row["perturbation_id"]),
            channel=cast(PerturbationChannel, str(row["channel"])),
            family=str(row["family"]),
            amplitude=float(row["amplitude"]),
            units=str(row["units"]),
            axis=str(row["axis"]),
            basis=str(row["basis"]),
            sign=int(row["sign"]),
            timing=cast(Mapping[str, Any], row["timing"]),
            adapter=str(row["adapter"]),
            description=str(row["description"]),
            epsilon_component=_optional_str(row.get("epsilon_component")),
            epsilon_index=_optional_int(row.get("epsilon_index")),
            initial_position_case=_optional_str(row.get("initial_position_case")),
            calibration_role=(
                None
                if "calibration_role" in flattened_known
                else _optional_str(row.get("calibration_role"))
            ),
            timing_bin=_optional_str(row.get("timing_bin")),
            semantic_family=_optional_str(row.get("semantic_family")),
            channel_provenance=_optional_mapping(row.get("channel_provenance")),
            calibration_provenance=_optional_mapping(row.get("calibration_provenance")),
            feedback_payload_index=_optional_int(row.get("feedback_payload_index")),
            feedback_quantity=_optional_str(row.get("feedback_quantity")),
            force_filter_feedback_only=_optional_bool(row.get("force_filter_feedback_only")),
            allow_zero_graph_effect=_optional_bool(row.get("allow_zero_graph_effect")),
            extra=extra or None,
        )

    def validate(self) -> None:
        """Validate the row against the channel-specific declarative field rules."""

        channel = str(self.channel)
        rule = _CHANNEL_FIELD_RULES.get(channel)
        if rule is None:
            raise ValueError(f"unknown perturbation channel {channel!r}")
        if rule.families is not None and self.family not in rule.families:
            allowed = ", ".join(sorted(rule.families))
            raise ValueError(
                f"perturbation row for channel {channel!r} has unsupported family "
                f"{self.family!r}; expected one of {allowed}"
            )
        if not isinstance(self.timing, Mapping):
            raise ValueError(
                f"perturbation row for channel {channel!r} field 'timing' must be a mapping"
            )
        for field in sorted(rule.timing_required):
            if field not in self.timing:
                raise ValueError(
                    f"perturbation row for channel {channel!r} timing is missing "
                    f"required field {field!r}"
                )

    def to_json(self) -> dict[str, Any]:
        """Return the canonical wire row.

        The emitted shape intentionally matches the legacy bank row byte order:
        core fields first, optional fields in the historical order, then
        flattened ``calibration_provenance`` and ``extra`` keys at top level.
        That flattening is part of the wire contract for cached eval manifests.
        """

        row = {
            "perturbation_id": self.perturbation_id,
            "channel": self.channel,
            "family": self.family,
            "amplitude": float(self.amplitude),
            "units": self.units,
            "axis": self.axis,
            "basis": self.basis,
            "sign": int(self.sign),
            "timing": dict(self.timing),
            "adapter": self.adapter,
            "description": self.description,
        }
        if self.epsilon_component is not None:
            row["epsilon_component"] = self.epsilon_component
        if self.epsilon_index is not None:
            row["epsilon_index"] = int(self.epsilon_index)
        if self.initial_position_case is not None:
            row["initial_position_case"] = self.initial_position_case
        if self.calibration_role is not None:
            row["calibration_role"] = self.calibration_role
        if self.timing_bin is not None:
            row["timing_bin"] = self.timing_bin
        if self.semantic_family is not None:
            row["semantic_family"] = self.semantic_family
        if self.channel_provenance is not None:
            row["channel_provenance"] = dict(self.channel_provenance)
        if self.feedback_payload_index is not None:
            row["feedback_payload_index"] = int(self.feedback_payload_index)
        if self.feedback_quantity is not None:
            row["feedback_quantity"] = self.feedback_quantity
        if self.force_filter_feedback_only is not None:
            row["force_filter_feedback_only"] = bool(self.force_filter_feedback_only)
        if self.allow_zero_graph_effect is not None:
            row["allow_zero_graph_effect"] = bool(self.allow_zero_graph_effect)
        if self.calibration_provenance is not None:
            row.update(dict(self.calibration_provenance))
        if self.extra is not None:
            row.update(dict(self.extra))
        return row


def _optional_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"expected mapping or None, got {type(value).__name__}")
    return value


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


__all__ = [
    "PerturbationChannel",
    "PerturbationSpec",
]
