"""Shared metadata labels for C&S rerun materializations."""

from __future__ import annotations

from typing import Any, Literal, get_args


Discretization = Literal["euler", "zoh"]
Lane = Literal["deterministic_analytical", "released_stochastic"]

DISCRETIZATION_CHOICES: tuple[str, ...] = get_args(Discretization)
LANE_CHOICES: tuple[str, ...] = get_args(Lane)


DEFAULT_DISCRETIZATION: Discretization = "euler"
DEFAULT_LANE: Lane = "deterministic_analytical"

LANE_DESCRIPTIONS: dict[Lane, str] = {
    "deterministic_analytical": (
        "Deterministic analytical lane: exact recursions and deterministic "
        "rollouts/audits with no sampled sensory, motor/process, or "
        "signal-dependent control noise."
    ),
    "released_stochastic": (
        "Released-code stochastic lane: Euler plant plus sampled sensory, "
        "motor/process, and signal-dependent control noise. Bellman parity is "
        "not claimed unless a separate stochastic objective is derived."
    ),
}

DISCRETIZATION_DESCRIPTIONS: dict[Discretization, str] = {
    "euler": "Forward Euler C&S released-code discretization.",
    "zoh": "Zero-order-hold discretization retained as a sensitivity/historical variant.",
}


def validate_discretization(value: str) -> Discretization:
    """Return a typed discretization label or raise ``ValueError``."""

    if value not in DISCRETIZATION_CHOICES:
        raise ValueError(
            f"Unknown discretization {value!r}; expected one of {DISCRETIZATION_CHOICES}."
        )
    return value  # type: ignore[return-value]


def validate_lane(value: str) -> Lane:
    """Return a typed rerun lane label or raise ``ValueError``."""

    if value not in LANE_CHOICES:
        raise ValueError(f"Unknown lane {value!r}; expected one of {LANE_CHOICES}.")
    return value  # type: ignore[return-value]


def build_rerun_metadata(
    *,
    discretization: str = DEFAULT_DISCRETIZATION,
    lane: str = DEFAULT_LANE,
    materializer: str,
) -> dict[str, Any]:
    """Build JSON-serializable metadata shared by rerun manifests."""

    discretization_label = validate_discretization(discretization)
    lane_label = validate_lane(lane)
    return {
        "discretization": discretization_label,
        "discretization_description": DISCRETIZATION_DESCRIPTIONS[discretization_label],
        "lane": lane_label,
        "lane_description": LANE_DESCRIPTIONS[lane_label],
        "materializer": materializer,
        "phase_label_policy": (
            "Phase 1 and Phase 3 reruns must report both discretization and lane. "
            "Euler deterministic analytical results are canonical after the Euler card lands; "
            "ZOH results remain sensitivity/historical unless explicitly promoted."
        ),
    }


def metadata_cli_help() -> str:
    """Return compact help text for materializer CLI wrappers."""

    return (
        "Metadata only: records the plant discretization/lane in generated manifests "
        "and notes. It does not change plant internals or enable stochastic simulation."
    )


__all__ = [
    "DEFAULT_DISCRETIZATION",
    "DEFAULT_LANE",
    "DISCRETIZATION_CHOICES",
    "LANE_CHOICES",
    "build_rerun_metadata",
    "metadata_cli_help",
    "validate_discretization",
    "validate_lane",
]
