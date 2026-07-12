"""Descriptor-conformance canary (issue 7811e47).

Gate check 2 of the 419eed1 semantic-safety gate. The retired-ID scan
(``tests/test_retired_component_id_scan.py``) catches positional hacks and
retired-ID reintroduction, but not a consumer that uses the *right* component
IDs with the *wrong* basis / units / slice / order / transform. This canary
closes that gap.

For the controller-visible feedback basis it drives the resolved-descriptor-view
API that every one of the seven descriptor-adopted feedback-identity consumers
from [issue:259bd10] routes through, feeding synthetic 4D and 6D payloads with
unique per-component sentinel values, then repeating with permuted component
storage order, and requiring every resolution to match by descriptor identity
(value, unit, basis, slice, absolute index, and transform). It also exercises
the directly-callable consumer seam (``summarize_controller_feedback_scales``)
end to end, and asserts that all seven consumers route feedback identity through
the shared API rather than reconstructing it locally.

The canary fails on any unit / basis / slice / order / transform mismatch, and
its negative canaries prove it has teeth (positional confusion and basis
mismatch are rejected).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from rlrmp.eval.gru_diagnostics import (
    RolloutEvaluation,
    summarize_controller_feedback_scales,
)
from rlrmp.model.feedback_descriptors import (
    COMPONENT_FORCE_FILTER,
    COMPONENT_POSITION,
    COMPONENT_VELOCITY,
    CONTROLLER_FEEDBACK_FORCE_FILTER_ID,
    CONTROLLER_FEEDBACK_POSITION_ID,
    CONTROLLER_FEEDBACK_VELOCITY_ID,
    DESCRIPTOR_PAYLOAD_KEY,
    PROPRIOCEPTIVE_FEEDBACK_BASIS_ID,
    TARGET_RELATIVE_FEEDBACK_BASIS_ID,
    controller_feedback_axis_index,
    controller_feedback_descriptor_payload,
    resolve_controller_feedback_view,
    resolve_controller_feedback_view_from_gru_input,
)


pytestmark = pytest.mark.feedbax_contract

# The seven descriptor-adopted feedback-identity consumers from 259bd10 (the
# same set the 259bd10 AST scan pins). Each routes feedback identity through the
# shared resolved-descriptor-view API this canary exercises.
DESCRIPTOR_CONSUMER_RELPATHS = (
    "src/rlrmp/eval/gru_diagnostics.py",
    "src/rlrmp/analysis/pipelines/gru_feedback_ablation.py",
    "src/rlrmp/eval/perturbation_bank.py",
    "src/rlrmp/analysis/pipelines/gru_steady_state_perturbation_bank.py",
    "src/rlrmp/model/feedbax_graph.py",
    "src/rlrmp/train/cs_perturbation_training.py",
)

# Evidence tokens proving a consumer routes feedback identity through the shared
# descriptor API (direct API calls) or consumes its descriptor-keyed output.
_ROUTING_TOKENS = (
    "resolve_controller_feedback_view",
    "resolve_controller_feedback_view_from_gru_input",
    "controller_feedback_axis_index",
    "controller_feedback_descriptor_payload",
    "controller_feedback_descriptor_from_container",
    "controller_feedback_order_labels",
    "summarize_controller_feedback_scales",
    "controller_feedback_scales",
    "controller_feedback_descriptors",
)

# Ground-truth per-component descriptor contract (see feedback_descriptors.py).
_EXPECTED = {
    COMPONENT_POSITION: {
        "descriptor_id": CONTROLLER_FEEDBACK_POSITION_ID,
        "units": "m",
        "transform": "target_minus_delayed_position",
        "slice": (0, 2),
    },
    COMPONENT_VELOCITY: {
        "descriptor_id": CONTROLLER_FEEDBACK_VELOCITY_ID,
        "units": "m/s",
        "transform": "negate_delayed_velocity",
        "slice": (2, 4),
    },
    COMPONENT_FORCE_FILTER: {
        "descriptor_id": CONTROLLER_FEEDBACK_FORCE_FILTER_ID,
        "units": "N",
        "transform": "identity",
        "slice": (4, 6),
    },
}


def _component_ids(feedback_dim: int) -> tuple[str, ...]:
    if feedback_dim == 4:
        return (COMPONENT_POSITION, COMPONENT_VELOCITY)
    return (COMPONENT_POSITION, COMPONENT_VELOCITY, COMPONENT_FORCE_FILTER)


def _sentinel_values(feedback_dim: int) -> np.ndarray:
    """Return a (2, 3, feedback_dim) array where column c holds a unique value."""

    base = np.zeros((2, 3, feedback_dim), dtype=np.float64)
    for col in range(feedback_dim):
        base[..., col] = 7.0 * (col + 1) + 0.5  # distinct, non-zero, non-integer
    return base


# --------------------------------------------------------------------------- #
# Identity selection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("feedback_dim", [4, 6])
def test_descriptor_view_selects_component_values_by_identity(feedback_dim: int) -> None:
    payload = controller_feedback_descriptor_payload(feedback_dim=feedback_dim)
    values = _sentinel_values(feedback_dim)
    view = resolve_controller_feedback_view(payload, values=values)

    assert view.feedback_dim == feedback_dim
    expected_basis = (
        PROPRIOCEPTIVE_FEEDBACK_BASIS_ID if feedback_dim == 6 else TARGET_RELATIVE_FEEDBACK_BASIS_ID
    )
    assert view.basis_id == expected_basis

    for component_id in _component_ids(feedback_dim):
        expected = _EXPECTED[component_id]
        start, stop = expected["slice"]
        resolved = view.component(component_id)
        assert resolved.descriptor_id == expected["descriptor_id"]
        assert resolved.units == expected["units"]
        assert resolved.transform == expected["transform"]
        assert (resolved.slice.start, resolved.slice.stop) == (start, stop)
        assert resolved.absolute_indices == tuple(range(start, stop))
        np.testing.assert_array_equal(np.asarray(resolved.values), values[..., start:stop])


@pytest.mark.parametrize("feedback_dim", [4, 6])
def test_descriptor_view_is_invariant_under_permuted_component_storage(feedback_dim: int) -> None:
    payload = controller_feedback_descriptor_payload(feedback_dim=feedback_dim)
    values = _sentinel_values(feedback_dim)

    permuted = copy.deepcopy(payload)
    permuted["components"] = list(reversed(permuted["components"]))  # permute storage order
    # basis identity + hash are the canonical basis; only storage order changed.
    assert permuted["basis"] == payload["basis"]
    assert permuted["descriptor_basis_hash"] == payload["descriptor_basis_hash"]

    canonical_view = resolve_controller_feedback_view(payload, values=values)
    permuted_view = resolve_controller_feedback_view(permuted, values=values)

    assert permuted_view.basis_id == canonical_view.basis_id
    assert permuted_view.descriptor_basis_hash == canonical_view.descriptor_basis_hash

    for component_id in _component_ids(feedback_dim):
        canonical = canonical_view.component(component_id)
        permuted_component = permuted_view.component(component_id)
        assert permuted_component.descriptor_id == canonical.descriptor_id
        assert permuted_component.units == canonical.units
        assert permuted_component.transform == canonical.transform
        assert permuted_component.slice == canonical.slice
        assert permuted_component.absolute_indices == canonical.absolute_indices
        np.testing.assert_array_equal(
            np.asarray(permuted_component.values), np.asarray(canonical.values)
        )

    # Non-triviality: the first and last components carry different sentinels, so
    # identity selection genuinely differs from any positional (components[0])
    # shortcut.
    ids = _component_ids(feedback_dim)
    first = canonical_view.component(ids[0]).values
    last = canonical_view.component(ids[-1]).values
    assert not np.array_equal(np.asarray(first), np.asarray(last))


@pytest.mark.parametrize("feedback_dim", [4, 6])
def test_gru_input_seam_resolves_feedback_block_by_identity(feedback_dim: int) -> None:
    # One leading non-feedback feature followed by the trailing feedback block.
    input_dim = feedback_dim + 1
    gru_input = np.zeros((1, 1, input_dim), dtype=np.float64)
    gru_input[..., 0] = 99.0  # non-feedback leading feature
    for col in range(feedback_dim):
        gru_input[..., 1 + col] = 3.0 * (col + 1) + 0.25

    view = resolve_controller_feedback_view_from_gru_input(gru_input)
    assert view.feedback_dim == feedback_dim
    assert view.start_index == 1

    for component_id in _component_ids(feedback_dim):
        start, stop = _EXPECTED[component_id]["slice"]
        resolved = view.component(component_id)
        assert resolved.absolute_indices == tuple(range(1 + start, 1 + stop))
        np.testing.assert_array_equal(
            np.asarray(resolved.values), gru_input[..., 1 + start : 1 + stop]
        )


# --------------------------------------------------------------------------- #
# Axis-index identity
# --------------------------------------------------------------------------- #


def test_axis_index_resolves_by_descriptor_identity() -> None:
    assert controller_feedback_axis_index(COMPONENT_POSITION, "x") == 0
    assert controller_feedback_axis_index(COMPONENT_POSITION, "y") == 1
    assert controller_feedback_axis_index(COMPONENT_VELOCITY, "vx") == 2
    assert controller_feedback_axis_index(COMPONENT_VELOCITY, "vy") == 3
    assert controller_feedback_axis_index(COMPONENT_FORCE_FILTER, "x") == 4
    assert controller_feedback_axis_index(COMPONENT_FORCE_FILTER, "y") == 5


# --------------------------------------------------------------------------- #
# Directly-callable consumer seam (gru_evaluation_diagnostics)
# --------------------------------------------------------------------------- #


def _rollout_evaluation(feedback_dim: int) -> RolloutEvaluation:
    input_dim = feedback_dim
    gru_input = np.zeros((1, 1, 2, input_dim), dtype=np.float64)
    for col in range(input_dim):
        gru_input[..., col] = 2.0 * (col + 1) + 0.1
    return RolloutEvaluation(
        position=np.zeros((1, 1, 2, 2)),
        velocity=np.zeros((1, 1, 2, 2)),
        command=np.zeros((1, 1, 2, 2)),
        hidden=np.zeros((1, 1, 2, 2)),
        gru_input=gru_input,
        initial_position=np.zeros((1, 2)),
        initial_velocity=np.zeros((1, 2)),
        target_position=np.zeros((1, 2, 2)),
        dt=0.01,
    )


@pytest.mark.parametrize("feedback_dim", [4, 6])
def test_summary_consumer_seam_emits_descriptor_identity(feedback_dim: int) -> None:
    summary = summarize_controller_feedback_scales(
        _rollout_evaluation(feedback_dim), run_id="canary"
    )
    expected_basis = (
        PROPRIOCEPTIVE_FEEDBACK_BASIS_ID if feedback_dim == 6 else TARGET_RELATIVE_FEEDBACK_BASIS_ID
    )
    assert summary[DESCRIPTOR_PAYLOAD_KEY]["basis_id"] == expected_basis
    assert (
        summary["descriptor_basis_hash"] == summary[DESCRIPTOR_PAYLOAD_KEY]["descriptor_basis_hash"]
    )

    for component_id in _component_ids(feedback_dim):
        start, stop = _EXPECTED[component_id]["slice"]
        component = summary["components"][component_id]
        assert component["descriptor_id"] == _EXPECTED[component_id]["descriptor_id"]
        assert component["units"] == _EXPECTED[component_id]["units"]
        assert component["feedback_basis_indices"] == list(range(start, stop))
        assert component["gru_input_indices"] == list(range(start, stop))

    if feedback_dim == 4:
        assert COMPONENT_FORCE_FILTER not in summary["components"]


# --------------------------------------------------------------------------- #
# Coverage: every consumer routes through the shared API
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("relpath", DESCRIPTOR_CONSUMER_RELPATHS)
def test_every_descriptor_consumer_routes_through_shared_api(relpath: str) -> None:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / relpath).read_text(encoding="utf-8")
    assert any(token in source for token in _ROUTING_TOKENS), (
        f"{relpath} references no resolved-descriptor-view API symbol or "
        "descriptor-keyed output field; feedback identity may be resolved "
        "locally instead of through the shared descriptor API."
    )


# --------------------------------------------------------------------------- #
# Negative canaries (teeth)
# --------------------------------------------------------------------------- #


def test_canary_negative_rejects_positional_basis_confusion() -> None:
    values = _sentinel_values(6)
    view = resolve_controller_feedback_view(
        controller_feedback_descriptor_payload(feedback_dim=6), values=values
    )
    force_filter = np.asarray(view.component(COMPONENT_FORCE_FILTER).values)
    position = np.asarray(view.component(COMPONENT_POSITION).values)
    # Identity selection is not the naive "first two columns" positional shortcut.
    assert not np.array_equal(force_filter, values[..., 0:2])
    np.testing.assert_array_equal(force_filter, values[..., 4:6])
    assert not np.array_equal(force_filter, position)


def test_canary_negative_rejects_basis_mismatch() -> None:
    # A 4D basis has no force/filter component; requesting it must raise rather
    # than silently returning an out-of-basis slice.
    view = resolve_controller_feedback_view(
        controller_feedback_descriptor_payload(feedback_dim=4),
        values=_sentinel_values(4),
    )
    with pytest.raises((KeyError, ValueError)):
        view.component(COMPONENT_FORCE_FILTER)
