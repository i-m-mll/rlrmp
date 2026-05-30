"""Shared contracts for analytical bridge rollout artifacts.

The bridge plan compares several controller families on the same output-feedback
game.  This module intentionally stays small: it records what was run, validates
the common rollout array shapes, and serializes manifests for later certificate
and aggregation code.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from jaxtyping import Float

BridgeObjective = Literal["optimal", "robust", "diagnostic"]
BridgeArchitecture = Literal[
    "free_time_varying",
    "time_constrained_free_gain",
    "linear_recurrence",
    "gru",
    "reference",
]
BridgeTrainingDistribution = Literal[
    "none",
    "nominal",
    "synthetic_initial_state",
    "eigenspectrum_trajectory",
    "eigenspectrum_state",
    "observer_error",
    "mixed",
]
BridgeEvaluationLane = Literal["deterministic", "released_stochastic", "diagnostic"]
BridgeComponentStatus = Literal["available", "not_applicable", "missing"]

_RUN_ID_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]+")


def make_bridge_run_id(*parts: object) -> str:
    """Return a stable filesystem-safe run identifier from nonempty parts."""

    raw = "__".join(str(part).strip() for part in parts if str(part).strip())
    run_id = _RUN_ID_PATTERN.sub("_", raw).strip("._-")
    if not run_id:
        raise ValueError("run id requires at least one nonempty part")
    return run_id.lower()


@dataclass(frozen=True)
class BridgeRunSpec:
    """Metadata that defines one bridge cell before training/evaluation.

    Attributes:
        issue_id: Tracking issue for this run or smoke artifact.
        run_id: Stable identifier used in result paths and summaries.
        objective: Scientific objective being trained or diagnosed.
        architecture: Controller family under evaluation.
        controller_label: Human-readable controller/condition label.
        optimizer_label: Optimizer or fitting method label.
        training_distribution: Training state-distribution arm.
        evaluation_lane: Deterministic, released-stochastic, or diagnostic lane.
        reference_controller: Analytical/controller reference used for comparisons.
        seed: Random seed when one is meaningful.
        gamma_factor: Gamma factor used for finite-gamma diagnostics, if any.
        parameters: Small JSON-serializable parameter metadata.
        notes: Short free-text notes for later reviewers.
    """

    issue_id: str
    run_id: str
    objective: BridgeObjective
    architecture: BridgeArchitecture
    controller_label: str
    optimizer_label: str
    training_distribution: BridgeTrainingDistribution
    evaluation_lane: BridgeEvaluationLane
    reference_controller: str
    seed: int | None = None
    gamma_factor: float | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "BridgeRunSpec":
        """Build a run spec from a decoded manifest object."""

        return cls(**data)


@dataclass(frozen=True)
class BridgeArraySpec:
    """Shape metadata for a rollout array stored in an external artifact."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    role: str

    @classmethod
    def from_array(cls, name: str, array: np.ndarray, *, role: str) -> "BridgeArraySpec":
        """Create metadata from an in-memory array."""

        return cls(name=name, shape=tuple(array.shape), dtype=str(array.dtype), role=role)


@dataclass(frozen=True)
class BridgeRolloutBatch:
    """Common in-memory rollout arrays for bridge controllers.

    The first dimension is batch/trial.  `plant_states` and optional
    `estimator_states`/`hidden_states` use horizon + 1 samples, while
    `actions`, `observations`, and `step_costs` use horizon samples.
    """

    plant_states: Float[np.ndarray, "batch horizon_plus_one state"]
    actions: Float[np.ndarray, "batch horizon action"]
    observations: Float[np.ndarray, "batch horizon observation"] | None = None
    estimator_states: Float[np.ndarray, "batch horizon_plus_one estimator"] | None = None
    hidden_states: Float[np.ndarray, "batch horizon_plus_one hidden"] | None = None
    step_costs: Float[np.ndarray, "batch horizon"] | None = None
    total_costs: Float[np.ndarray, " batch"] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate common shape relationships."""

        plant_states = np.asarray(self.plant_states)
        actions = np.asarray(self.actions)
        if plant_states.ndim != 3:
            raise ValueError("plant_states must have shape (batch, horizon + 1, state)")
        if actions.ndim != 3:
            raise ValueError("actions must have shape (batch, horizon, action)")
        if plant_states.shape[0] != actions.shape[0]:
            raise ValueError("plant_states and actions must have the same batch size")
        horizon = actions.shape[1]
        if plant_states.shape[1] != horizon + 1:
            raise ValueError("plant_states must have exactly one more time sample than actions")
        self._validate_horizon_array("observations", self.observations, horizon, ndim=3)
        self._validate_horizon_plus_one_array("estimator_states", self.estimator_states, horizon)
        self._validate_horizon_plus_one_array("hidden_states", self.hidden_states, horizon)
        self._validate_horizon_array("step_costs", self.step_costs, horizon, ndim=2)
        if self.total_costs is not None:
            total_costs = np.asarray(self.total_costs)
            if total_costs.shape != (plant_states.shape[0],):
                raise ValueError("total_costs must have shape (batch,)")

    @property
    def batch_size(self) -> int:
        """Number of trials in the batch."""

        return int(np.asarray(self.plant_states).shape[0])

    @property
    def horizon(self) -> int:
        """Number of action/control steps."""

        return int(np.asarray(self.actions).shape[1])

    def array_specs(self) -> tuple[BridgeArraySpec, ...]:
        """Return metadata for arrays present in this rollout batch."""

        arrays: list[tuple[str, np.ndarray | None, str]] = [
            ("plant_states", self.plant_states, "plant_state"),
            ("actions", self.actions, "action"),
            ("observations", self.observations, "observation"),
            ("estimator_states", self.estimator_states, "estimator_state"),
            ("hidden_states", self.hidden_states, "hidden_state"),
            ("step_costs", self.step_costs, "cost"),
            ("total_costs", self.total_costs, "cost"),
        ]
        return tuple(
            BridgeArraySpec.from_array(name, np.asarray(array), role=role)
            for name, array, role in arrays
            if array is not None
        )

    def _validate_horizon_array(
        self,
        name: str,
        array: np.ndarray | None,
        horizon: int,
        *,
        ndim: int,
    ) -> None:
        if array is None:
            return
        values = np.asarray(array)
        if values.ndim != ndim:
            raise ValueError(f"{name} must have {ndim} dimensions")
        if values.shape[0] != self.batch_size or values.shape[1] != horizon:
            raise ValueError(f"{name} must have shape (batch, horizon, ...) or (batch, horizon)")

    def _validate_horizon_plus_one_array(
        self,
        name: str,
        array: np.ndarray | None,
        horizon: int,
    ) -> None:
        if array is None:
            return
        values = np.asarray(array)
        if values.ndim != 3:
            raise ValueError(f"{name} must have shape (batch, horizon + 1, dim)")
        if values.shape[0] != self.batch_size or values.shape[1] != horizon + 1:
            raise ValueError(f"{name} must have shape (batch, horizon + 1, dim)")


@dataclass(frozen=True)
class BridgeCertificateComponent:
    """Status row for one standard-certificate quantity."""

    name: str
    status: BridgeComponentStatus
    summary: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    @classmethod
    def available(cls, name: str, **summary: Any) -> "BridgeCertificateComponent":
        """Create an available certificate-component row."""

        return cls(name=name, status="available", summary=summary)

    @classmethod
    def not_applicable(cls, name: str, reason: str) -> "BridgeCertificateComponent":
        """Create an explicit not-applicable certificate-component row."""

        return cls(name=name, status="not_applicable", reason=reason)


@dataclass(frozen=True)
class BridgeRunManifest:
    """Serializable manifest tying one run spec to artifacts and summaries."""

    spec: BridgeRunSpec
    status: str
    arrays: tuple[BridgeArraySpec, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    certificate_components: tuple[BridgeCertificateComponent, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest."""

        return {
            "spec": self.spec.to_json_dict(),
            "status": self.status,
            "arrays": [asdict(array) for array in self.arrays],
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "certificate_components": [
                asdict(component) for component in self.certificate_components
            ],
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "BridgeRunManifest":
        """Build a manifest from decoded JSON."""

        return cls(
            spec=BridgeRunSpec.from_json_dict(data["spec"]),
            status=data["status"],
            arrays=tuple(
                BridgeArraySpec(
                    name=row["name"],
                    shape=tuple(row["shape"]),
                    dtype=row["dtype"],
                    role=row["role"],
                )
                for row in data.get("arrays", ())
            ),
            metrics=dict(data.get("metrics", {})),
            artifacts=dict(data.get("artifacts", {})),
            certificate_components=tuple(
                BridgeCertificateComponent(**row) for row in data.get("certificate_components", ())
            ),
        )


def write_bridge_manifest(manifest: BridgeRunManifest, path: Path) -> None:
    """Write a bridge manifest as stable, indented JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True) + "\n")


def read_bridge_manifest(path: Path) -> BridgeRunManifest:
    """Read a bridge manifest written by :func:`write_bridge_manifest`."""

    return BridgeRunManifest.from_json_dict(json.loads(path.read_text()))


__all__ = [
    "BridgeArchitecture",
    "BridgeArraySpec",
    "BridgeCertificateComponent",
    "BridgeComponentStatus",
    "BridgeEvaluationLane",
    "BridgeObjective",
    "BridgeRolloutBatch",
    "BridgeRunManifest",
    "BridgeRunSpec",
    "BridgeTrainingDistribution",
    "make_bridge_run_id",
    "read_bridge_manifest",
    "write_bridge_manifest",
]
