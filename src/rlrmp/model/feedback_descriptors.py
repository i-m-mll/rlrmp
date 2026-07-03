"""RLRMP-owned descriptors for controller-visible feedback bases."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from feedbax.contracts.descriptors import (
    ComponentDescriptor,
    ComponentSlice,
    DescriptorBasisIdentity,
    SelectorFallbackPolicyIdentity,
    ComponentSelectorSyntax,
    SelectorRoleIdentity,
    VariableDescriptor,
    descriptor_basis_from_descriptors,
    resolve_descriptor_view,
)
from feedbax.contracts.value_schema import ValueSchema


CONTROLLER_FEEDBACK_DESCRIPTOR_SCHEMA_VERSION = "rlrmp.controller_feedback_descriptors.v1"
CONTROLLER_FEEDBACK_NAMESPACE = "rlrmp.controller_feedback"
CONTROLLER_FEEDBACK_VARIABLE_ID = "rlrmp.controller_feedback.vector"
CONTROLLER_FEEDBACK_POSITION_ID = "rlrmp.controller_feedback.position"
CONTROLLER_FEEDBACK_VELOCITY_ID = "rlrmp.controller_feedback.velocity"
CONTROLLER_FEEDBACK_FORCE_FILTER_ID = "rlrmp.controller_feedback.force_filter"
TARGET_RELATIVE_FEEDBACK_BASIS_ID = "target_relative_delayed_feedback"
PROPRIOCEPTIVE_FEEDBACK_BASIS_ID = "target_relative_delayed_feedback_plus_force_filter"

COMPONENT_POSITION = "position"
COMPONENT_VELOCITY = "velocity"
COMPONENT_FORCE_FILTER = "force_filter"

DESCRIPTOR_PAYLOAD_KEY = "controller_feedback_descriptors"

_COMPONENT_DESCRIPTOR_IDS = {
    COMPONENT_POSITION: CONTROLLER_FEEDBACK_POSITION_ID,
    COMPONENT_VELOCITY: CONTROLLER_FEEDBACK_VELOCITY_ID,
    COMPONENT_FORCE_FILTER: CONTROLLER_FEEDBACK_FORCE_FILTER_ID,
}

_COMPONENT_SPECS = {
    COMPONENT_POSITION: {
        "slice": (0, 2),
        "axes": ("x", "y"),
        "label": "Target-relative delayed position",
        "units": "m",
        "description": (
            "Controller-visible delayed position in target-relative coordinates: "
            "target position minus delayed effector position."
        ),
        "transform": "target_minus_delayed_position",
        "source_path": "states.mechanics.vector[..., delayed_position_xy]",
    },
    COMPONENT_VELOCITY: {
        "slice": (2, 4),
        "axes": ("vx", "vy"),
        "label": "Negated delayed velocity",
        "units": "m/s",
        "description": (
            "Controller-visible delayed velocity as the negative delayed effector "
            "velocity in the target-relative feedback basis."
        ),
        "transform": "negate_delayed_velocity",
        "source_path": "states.mechanics.vector[..., delayed_velocity_xy]",
    },
    COMPONENT_FORCE_FILTER: {
        "slice": (4, 6),
        "axes": ("x", "y"),
        "label": "Delayed force/filter state",
        "units": "N",
        "description": (
            "Controller-visible delayed C&S force/filter coordinates appended to "
            "the proprioceptive feedback basis."
        ),
        "transform": "identity",
        "source_path": "states.mechanics.vector[..., delayed_force_filter_xy]",
    },
}


@dataclass(frozen=True)
class ResolvedFeedbackComponent:
    """Resolved descriptor component plus optional sliced values."""

    descriptor_id: str
    component_id: str
    label: str
    units: str | None
    slice: ComponentSlice
    axes: tuple[str, ...]
    source_path: str
    sign: str
    transform: str
    values: Any | None = None
    absolute_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class ResolvedFeedbackView:
    """Descriptor-backed view over a controller feedback vector."""

    payload: Mapping[str, Any]
    variable: VariableDescriptor
    components: tuple[ComponentDescriptor, ...]
    basis: DescriptorBasisIdentity
    descriptor_basis_hash: str
    values: Any | None = None
    start_index: int = 0

    @property
    def feedback_dim(self) -> int:
        """Return the feedback vector width covered by this basis."""

        shape = self.variable.value_schema.shape or []
        return int(shape[-1])

    @property
    def basis_id(self) -> str:
        """Return the basis identity."""

        return self.basis.basis_id

    def component(self, component_id: str) -> ResolvedFeedbackComponent:
        """Resolve a component by semantic ID and return its descriptor view."""

        descriptor_id = _COMPONENT_DESCRIPTOR_IDS[component_id]
        resolved = resolve_descriptor_view(
            descriptor_id=descriptor_id,
            variable=self.variable,
            components=list(self.components),
            basis=self.basis,
            descriptor_basis_hash=self.descriptor_basis_hash,
        )
        descriptor = resolved.component_descriptor
        if descriptor is None or resolved.slice is None:
            raise ValueError(f"descriptor {descriptor_id!r} is not a component")
        axes = tuple(descriptor.metadata.get("axes", ()))
        values = None
        if self.values is not None:
            values = self.values[
                ..., resolved.slice.start : resolved.slice.stop : resolved.slice.step
            ]
        absolute_indices = tuple(
            self.start_index + index
            for index in range(resolved.slice.start, resolved.slice.stop, resolved.slice.step)
        )
        return ResolvedFeedbackComponent(
            descriptor_id=descriptor.descriptor_id,
            component_id=descriptor.component_id,
            label=descriptor.label,
            units=descriptor.value_schema.units,
            slice=resolved.slice,
            axes=axes,
            source_path=descriptor.source_path,
            sign=descriptor.sign,
            transform=descriptor.transform,
            values=values,
            absolute_indices=absolute_indices,
        )

    def component_index(self, component_id: str, axis: str) -> int:
        """Return the basis-local index for ``component_id`` and ``axis``."""

        component = self.component(component_id)
        axis_index = _axis_position(axis, component.axes)
        return int(component.slice.start + axis_index)

    def iter_components(self) -> Iterable[ResolvedFeedbackComponent]:
        """Yield resolved components in basis order."""

        for component in self.components:
            yield self.component(component.component_id)


def controller_feedback_descriptor_payload(
    *,
    feedback_dim: int,
    basis_id: str | None = None,
    adoption: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the JSON descriptor payload for a 4D or 6D controller feedback basis."""

    component_ids = _component_ids_for_dim(feedback_dim)
    basis_id = basis_id or (
        PROPRIOCEPTIVE_FEEDBACK_BASIS_ID if feedback_dim == 6 else TARGET_RELATIVE_FEEDBACK_BASIS_ID
    )
    variable = _variable_descriptor(feedback_dim=feedback_dim, basis_id=basis_id)
    components = [_component_descriptor(component_id) for component_id in component_ids]
    basis = descriptor_basis_from_descriptors(
        basis_id=basis_id,
        variable=variable,
        components=components,
        metadata={
            "owner": "rlrmp",
            "schema_version": CONTROLLER_FEEDBACK_DESCRIPTOR_SCHEMA_VERSION,
        },
    )
    payload: dict[str, Any] = {
        "schema_version": CONTROLLER_FEEDBACK_DESCRIPTOR_SCHEMA_VERSION,
        "namespace": CONTROLLER_FEEDBACK_NAMESPACE,
        "basis_id": basis.basis_id,
        "descriptor_basis_hash": basis.descriptor_basis_hash,
        "variable": variable.model_dump(mode="json", exclude_none=True),
        "components": [
            component.model_dump(mode="json", exclude_none=True) for component in components
        ],
        "basis": basis.model_dump(mode="json", exclude_none=True),
        "component_ids": list(component_ids),
    }
    if adoption is not None:
        payload["adoption"] = dict(adoption)
    return payload


def adopt_controller_feedback_descriptor_payload(
    payload: Mapping[str, Any] | None,
    *,
    feedback_dim: int | None = None,
    basis_id: str | None = None,
    source: str,
) -> dict[str, Any]:
    """Return an explicit current or legacy-adopted descriptor payload."""

    if payload is not None:
        return dict(payload)
    if feedback_dim is None:
        raise ValueError("feedback_dim is required for explicit legacy descriptor adoption")
    return controller_feedback_descriptor_payload(
        feedback_dim=feedback_dim,
        basis_id=basis_id,
        adoption={
            "policy": "explicit_legacy_default",
            "source": source,
            "reason": (
                "historical payload had controller feedback width but no Feedbax "
                "component descriptor records"
            ),
        },
    )


def resolve_controller_feedback_view(
    payload: Mapping[str, Any] | None,
    *,
    feedback_dim: int | None = None,
    values: Any | None = None,
    start_index: int = 0,
    source: str = "current_spec",
) -> ResolvedFeedbackView:
    """Resolve a descriptor payload and optionally bind feedback-vector values."""

    if payload is None:
        if feedback_dim is None and values is not None:
            feedback_dim = _feedback_dim_from_values(values)
        payload = adopt_controller_feedback_descriptor_payload(
            None,
            feedback_dim=feedback_dim,
            source=source,
        )
    variable = VariableDescriptor.model_validate(payload["variable"])
    components = tuple(
        ComponentDescriptor.model_validate(component) for component in payload["components"]
    )
    basis = DescriptorBasisIdentity.model_validate(payload["basis"])
    descriptor_basis_hash = str(payload["descriptor_basis_hash"])
    resolve_descriptor_view(
        descriptor_id=variable.descriptor_id,
        variable=variable,
        components=list(components),
        basis=basis,
        descriptor_basis_hash=descriptor_basis_hash,
    )
    return ResolvedFeedbackView(
        payload=payload,
        variable=variable,
        components=components,
        basis=basis,
        descriptor_basis_hash=descriptor_basis_hash,
        values=values,
        start_index=int(start_index),
    )


def resolve_controller_feedback_view_from_gru_input(
    gru_input: Any,
    *,
    payload: Mapping[str, Any] | None = None,
    source: str = "gru_input_legacy_trailing_feedback_block",
) -> ResolvedFeedbackView:
    """Resolve the controller-feedback block carried at the tail of GRU inputs."""

    values = jnp.asarray(gru_input)
    if values.ndim < 1:
        raise ValueError("controller feedback input must have at least one feature dimension")
    input_dim = int(values.shape[-1])
    feedback_dim = _feedback_dim_from_input_dim(input_dim)
    start = input_dim - feedback_dim
    feedback_values = values[..., start:input_dim]
    payload = adopt_controller_feedback_descriptor_payload(
        payload,
        feedback_dim=feedback_dim,
        source=source,
    )
    return resolve_controller_feedback_view(
        payload,
        feedback_dim=feedback_dim,
        values=feedback_values,
        start_index=start,
        source=source,
    )


def controller_feedback_descriptor_from_container(
    container: Mapping[str, Any] | None,
    *,
    feedback_dim: int | None = None,
    source: str,
) -> dict[str, Any]:
    """Extract or explicitly adopt feedback descriptors from a run/manifest container."""

    if isinstance(container, Mapping):
        payload = container.get(DESCRIPTOR_PAYLOAD_KEY)
        if isinstance(payload, Mapping):
            return dict(payload)
        feedback = container.get("feedback")
        if isinstance(feedback, Mapping):
            payload = feedback.get(DESCRIPTOR_PAYLOAD_KEY)
            if isinstance(payload, Mapping):
                return dict(payload)
        scales = container.get("controller_feedback_scales")
        if isinstance(scales, Mapping):
            payload = scales.get(DESCRIPTOR_PAYLOAD_KEY)
            if isinstance(payload, Mapping):
                return dict(payload)
    return adopt_controller_feedback_descriptor_payload(
        None,
        feedback_dim=feedback_dim,
        source=source,
    )


def controller_feedback_axis_index(component_id: str, axis: str, *, feedback_dim: int = 6) -> int:
    """Return the basis-local axis index via the descriptor resolver."""

    view = resolve_controller_feedback_view(
        None,
        feedback_dim=feedback_dim,
        source="axis_index_default_descriptor",
    )
    return view.component_index(component_id, axis)


def controller_feedback_order_labels(*, feedback_dim: int = 4) -> list[str]:
    """Return legacy feedback-order labels derived from descriptor components."""

    view = resolve_controller_feedback_view(
        None,
        feedback_dim=feedback_dim,
        source="legacy_feedback_order_labels",
    )
    labels: list[str] = []
    for component in view.iter_components():
        for axis in component.axes:
            labels.append(f"{_legacy_component_prefix(component.component_id)}_{axis[-1]}")
    return labels


def component_id_from_descriptor_id(descriptor_id: str) -> str:
    """Return the semantic component ID for a descriptor ID."""

    for component_id, candidate in _COMPONENT_DESCRIPTOR_IDS.items():
        if descriptor_id == candidate:
            return component_id
    raise ValueError(f"unknown controller feedback descriptor_id {descriptor_id!r}")


def _variable_descriptor(*, feedback_dim: int, basis_id: str) -> VariableDescriptor:
    return VariableDescriptor(
        descriptor_id=CONTROLLER_FEEDBACK_VARIABLE_ID,
        namespace=CONTROLLER_FEEDBACK_NAMESPACE,
        label="Controller-visible feedback vector",
        description=(
            "The feedback vector consumed by the GRU controller. Components are "
            "ordered by descriptor basis identity rather than consumer-local row "
            "constants."
        ),
        value_schema=ValueSchema(
            id=f"{CONTROLLER_FEEDBACK_NAMESPACE}.{basis_id}.vector",
            label="controller feedback",
            kind="array",
            dtype="float32",
            shape=[feedback_dim],
            rank=1,
            units="mixed",
            frame="controller_visible_target_relative",
            origin="declared",
            metadata={"basis_id": basis_id},
        ),
        source_kind="model",
        source_path="states.net.input.controller_feedback",
        timing="per_timestep",
        role=SelectorRoleIdentity(
            namespace="feedbax.selector.role",
            name="model_input",
            version="v1",
        ),
        selector_syntax=ComponentSelectorSyntax(
            namespace="feedbax.selector.syntax",
            name="component_descriptor_id",
            version="v1",
            grammar="descriptor://{basis_id}/{descriptor_id}",
        ),
        fallback_policy=SelectorFallbackPolicyIdentity(
            namespace="feedbax.selector.fallback",
            name="controller_feedback_legacy_width",
            version="v1",
            policy="allow_declared",
        ),
        scope={
            "project": "rlrmp",
            "basis_id": basis_id,
            "feedback_dim": feedback_dim,
        },
        tags=["controller_feedback", "cs_gru", "semantic_component_basis"],
        metadata={
            "component_ids": list(_component_ids_for_dim(feedback_dim)),
            "source_timing": "controller step after feedback delay and before GRU update",
        },
    )


def _component_descriptor(component_id: str) -> ComponentDescriptor:
    spec = _COMPONENT_SPECS[component_id]
    start, stop = spec["slice"]
    axes = tuple(spec["axes"])
    return ComponentDescriptor(
        descriptor_id=_COMPONENT_DESCRIPTOR_IDS[component_id],
        variable_descriptor_id=CONTROLLER_FEEDBACK_VARIABLE_ID,
        component_id=component_id,
        label=str(spec["label"]),
        description=str(spec["description"]),
        slice=ComponentSlice(start=int(start), stop=int(stop)),
        value_schema=ValueSchema(
            id=f"{CONTROLLER_FEEDBACK_NAMESPACE}.{component_id}",
            label=str(spec["label"]),
            kind="array",
            dtype="float32",
            shape=[int(stop) - int(start)],
            rank=1,
            units=str(spec["units"]),
            frame="controller_visible_target_relative",
            origin="declared",
            metadata={"axes": list(axes), "component_id": component_id},
        ),
        source_kind="model",
        source_path=str(spec["source_path"]),
        timing="per_timestep",
        sign="native",
        transform=str(spec["transform"]),
        tags=["controller_feedback_component", component_id],
        metadata={
            "axes": list(axes),
            "native_units": str(spec["units"]),
            "basis_slice": [int(start), int(stop)],
            "semantic_component_id": component_id,
        },
    )


def _component_ids_for_dim(feedback_dim: int) -> tuple[str, ...]:
    if feedback_dim == 4:
        return (COMPONENT_POSITION, COMPONENT_VELOCITY)
    if feedback_dim == 6:
        return (COMPONENT_POSITION, COMPONENT_VELOCITY, COMPONENT_FORCE_FILTER)
    raise ValueError(
        f"controller feedback descriptors support only 4D or 6D bases, got {feedback_dim}"
    )


def _feedback_dim_from_values(values: Any) -> int:
    shape = getattr(values, "shape", None)
    if not shape:
        raise ValueError("cannot infer controller feedback dimension from value without shape")
    return _feedback_dim_from_input_dim(int(shape[-1]))


def _feedback_dim_from_input_dim(input_dim: int) -> int:
    if input_dim >= 6:
        return 6
    if input_dim >= 4:
        return 4
    raise ValueError(
        "controller feedback basis requires at least 4 trailing dimensions, "
        f"got input_dim={input_dim}"
    )


def _axis_position(axis: str, axes: Sequence[str]) -> int:
    normalized = {"vx": "vx", "vy": "vy", "x": "x", "y": "y"}.get(axis, axis)
    if normalized not in axes:
        raise ValueError(f"axis {axis!r} is not valid for descriptor axes {tuple(axes)!r}")
    return int(tuple(axes).index(normalized))


def _legacy_component_prefix(component_id: str) -> str:
    if component_id == COMPONENT_POSITION:
        return "pos"
    if component_id == COMPONENT_VELOCITY:
        return "vel"
    if component_id == COMPONENT_FORCE_FILTER:
        return "force_filter"
    raise ValueError(f"unknown controller feedback component {component_id!r}")
